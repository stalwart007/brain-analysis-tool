"""Fetch a caller-supplied URL without handing them the inside of the network.

WHY THIS EXISTS. Asking a researcher to open DevTools and copy `outerHTML` is
not a product. They should paste a link. Everything in this file is the cost of
making that safe, and none of it is visible to them.

THE THREAT. A naive `httpx.get(url)` inside this process is a server-side
request forgery primitive, and this deployment is close to the worst case for
one: the API runs on a private 6PN network with no public listener, next to a
volume-backed database, on a host that answers `http://[fdaa::3]:4280` for
instance metadata. `url="http://169.254.169.254/..."` or the 6PN address of the
API itself would let anyone holding an API key read things the internet cannot
reach. The dashboard proxy would happily relay it back to them.

THE DEFENCE, in the order it has to happen:

  1. Scheme and port allowlist. `file://`, `gopher://`, `redis://` and a port
     scan of the private network are all requests we simply never make.
  2. Resolve DNS ourselves and check EVERY address the name returns. A name
     with one public and one private A record is a known bypass; one bad
     address disqualifies the host.
  3. Connect to the ADDRESS WE VALIDATED, not to the name. This is the part
     that is usually missing. Validating a hostname and then handing the same
     hostname to the HTTP client leaves a window in which the attacker's DNS
     server answers publicly for our check and privately for the real
     connection — DNS rebinding. Pinning the IP closes the window because
     there is no second lookup.
  4. Re-run all of it on every redirect. A public URL that 302s to
     `http://[::1]:8000/v1/sessions` is the same attack with an extra step, so
     redirects are followed by hand rather than by the client.
  5. Never forward credentials, and cap time and bytes so a hostile server
     cannot hold a worker open or feed us a decompression bomb.

WHAT THIS IS NOT. It is not a renderer. A single-page app that ships an empty
`<div id="root">` will come back nearly empty, and `probe()` says so plainly
rather than letting a study run against a nav bar and report findings about it.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse, urlunparse

import httpx

#: Only the two schemes a landing page is ever served over.
ALLOWED_SCHEMES = frozenset({"http", "https"})

#: Anything else is a service on a host, not a web page. Blocking the port
#: range matters as much as blocking the address range: 127.0.0.1 is refused by
#: the IP check, but a public host that proxies to its own Redis is not.
ALLOWED_PORTS = frozenset({80, 443, 8080, 8443})

#: A hostile origin should not be able to hold a worker open.
CONNECT_TIMEOUT = 5.0
TOTAL_TIMEOUT = 15.0

#: Measured against real sites rather than guessed. Two megabytes sounded
#: generous and rejected linear.app outright: a modern marketing page inlines
#: its CSS, its critical JS and a base64 hero image, so the HTML is mostly not
#: prose. Eight covers everything tested; the read is streamed, so the cap
#: stops a hostile server rather than sizing a buffer.
MAX_BYTES = 8_000_000

#: Followed by hand, each one revalidated. Three is enough for the usual
#: apex → www → https chain and short enough to bound the work.
MAX_REDIRECTS = 3

#: A study reads prose. An octet-stream is either a download or a trap.
ALLOWED_CONTENT_PREFIXES = ("text/html", "application/xhtml", "text/plain")

#: Presented as a browser because many sites serve a challenge page or a
#: stripped response to unknown agents, and a challenge page would be analysed
#: as if it were the landing page. Honest about who we are in the comment; the
#: header is a compatibility measure, not a disguise for abuse.
USER_AGENT = (
    "Mozilla/5.0 (compatible; CogniSwarmBot/1.0; +https://cogniswarm-app.fly.dev) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class UnsafeURL(ValueError):
    """The URL is refused before any connection is attempted."""


class FetchFailed(RuntimeError):
    """The URL was allowed, but could not be retrieved."""


@dataclass(frozen=True)
class Fetched:
    """What a successful fetch yields."""

    #: Raw markup, ready for the `page` modality adapter.
    html: str
    #: Where we ended up, which is not always where we were pointed.
    final_url: str
    #: Redirect chain actually walked, for display.
    hops: tuple[str, ...]
    content_type: str
    bytes_read: int


def _reject_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> Optional[str]:
    """Return a reason to refuse this address, or None if it is safe.

    Deny by category rather than by a list of known-bad addresses. A blocklist
    of `169.254.169.254` and `127.0.0.1` misses `[::ffff:127.0.0.1]`,
    `2130706433`, `0.0.0.0`, Fly's `fdaa::/16` 6PN range and every other
    spelling of the same idea.
    """
    if ip.is_loopback:
        return "loopback address"
    if ip.is_private:
        # Covers RFC1918, RFC4193 unique-local (fc00::/7 — Fly's 6PN lives
        # in fdaa::/16, so this is what keeps the private backend private),
        # and the IPv4 carrier-grade NAT range.
        return "private address"
    if ip.is_link_local:
        # 169.254.0.0/16 and fe80::/10 — the cloud metadata service.
        return "link-local address"
    if ip.is_reserved or ip.is_multicast or ip.is_unspecified:
        return "reserved address"
    if isinstance(ip, ipaddress.IPv6Address):
        # ::ffff:127.0.0.1 is a loopback wearing an IPv6 costume; the flags
        # above do not see through the mapping, so unwrap it and re-check.
        # 2002:7f00:1::1 is the same trick via 6to4, and 64:ff9b::/96 (NAT64)
        # is caught by `is_reserved` above — all three encode an IPv4 address
        # the plain flags would wave through.
        mapped = ip.ipv4_mapped
        if mapped is not None:
            return _reject_ip(mapped)
        if ip.sixtofour is not None:
            return _reject_ip(ip.sixtofour)

    # The catch-all, and the one that actually decides. The named flags above
    # exist to produce a useful message; this is the rule. `is_private` alone
    # is NOT sufficient and testing showed it: 100.64.0.0/10 (carrier-grade
    # NAT, RFC 6598) reports private=False and reserved=False, so it sailed
    # through every specific check and the fetcher opened a connection to it.
    # That range is not hypothetical — Tailscale assigns out of 100.64/10, so
    # a tailnet host would have been reachable from this endpoint.
    if not ip.is_global:
        return "non-routable address"
    return None


async def _resolve(host: str, port: int) -> list[tuple[int, str]]:
    """Resolve to (family, address) pairs, or raise UnsafeURL.

    EVERY address must pass. A name that returns one public and one private
    record is a deliberate bypass, not a misconfiguration, and taking "any
    address is fine" would walk straight into it.
    """
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise UnsafeURL(f"could not resolve {host}") from exc
    if not infos:
        raise UnsafeURL(f"could not resolve {host}")

    out: list[tuple[int, str]] = []
    for family, _type, _proto, _canon, sockaddr in infos:
        addr = sockaddr[0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            raise UnsafeURL(f"{host} resolved to something that is not an address")
        reason = _reject_ip(ip)
        if reason is not None:
            raise UnsafeURL(
                f"{host} resolves to a {reason} — refusing to fetch it. "
                "Only publicly routable addresses are allowed."
            )
        out.append((family, addr))
    return out


def _validate(url: str) -> tuple[str, str, int, str]:
    """Split and check a URL. Returns (scheme, host, port, pinned_path_url)."""
    if len(url) > 2048:
        raise UnsafeURL("that URL is implausibly long")
    parsed = urlparse(url.strip())
    scheme = parsed.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise UnsafeURL(
            f"only http and https are supported, not {scheme or 'a bare string'}. "
            "Paste the full link, including https://"
        )
    host = parsed.hostname
    if not host:
        raise UnsafeURL("that URL has no host")
    if parsed.username or parsed.password:
        # https://trusted.com@169.254.169.254/ reads as trusted to a human.
        raise UnsafeURL("URLs with embedded credentials are not accepted")
    port = parsed.port or (443 if scheme == "https" else 80)
    if port not in ALLOWED_PORTS:
        raise UnsafeURL(f"port {port} is not allowed — use the standard web ports")
    return scheme, host, port, urlunparse(parsed)


async def fetch_page(url: str) -> Fetched:
    """Fetch `url` safely, following and revalidating up to MAX_REDIRECTS hops."""
    hops: list[str] = []
    current = url

    async with httpx.AsyncClient(
        follow_redirects=False,  # walked by hand so every hop is revalidated
        timeout=httpx.Timeout(TOTAL_TIMEOUT, connect=CONNECT_TIMEOUT),
        # A hostile server should not be able to keep sockets alive across
        # our own redirect walk.
        limits=httpx.Limits(max_connections=4, max_keepalive_connections=0),
        trust_env=False,  # ignore ambient proxy env vars
    ) as client:
        for _hop in range(MAX_REDIRECTS + 1):
            scheme, host, port, normalised = _validate(current)
            resolved = await _resolve(host, port)
            hops.append(normalised)

            # Connect to the address we just validated rather than to the name.
            # Re-resolving here is what DNS rebinding attacks rely on.
            family, addr = resolved[0]
            literal = f"[{addr}]" if family == socket.AF_INET6 else addr
            pinned = urlparse(normalised)._replace(netloc=f"{literal}:{port}")

            try:
                response = await client.get(
                    urlunparse(pinned),
                    headers={
                        # The origin still needs to know which vhost we want,
                        # and TLS still has to be verified against the NAME —
                        # `sni_hostname` drives both SNI and certificate
                        # validation, so pinning the IP does not silently
                        # downgrade us to an unverified connection.
                        "Host": host if port in (80, 443) else f"{host}:{port}",
                        "User-Agent": USER_AGENT,
                        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9",
                        "Accept-Language": "en-US,en;q=0.9",
                    },
                    extensions={"sni_hostname": host},
                )
            except httpx.HTTPError as exc:
                raise FetchFailed(f"could not reach {host}: {type(exc).__name__}") from exc

            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise FetchFailed(f"{host} sent a redirect with no destination")
                # Resolve relative Locations against the NAME-based URL, never
                # against the pinned IP form, or a relative hop would silently
                # stay pinned to an address the new host never claimed.
                current = str(httpx.URL(normalised).join(location))
                await response.aclose()
                continue

            if response.status_code >= 400:
                await response.aclose()
                raise FetchFailed(
                    f"{host} answered {response.status_code}. "
                    "The page may require a login, or may be blocking automated access."
                )

            ctype = response.headers.get("content-type", "").split(";")[0].strip().lower()
            if ctype and not ctype.startswith(ALLOWED_CONTENT_PREFIXES):
                await response.aclose()
                raise FetchFailed(
                    f"that URL serves {ctype}, not a web page. "
                    "For images or video, upload the file instead."
                )

            # Streamed with a running total: Content-Length is a claim by the
            # server, not a fact, so the cap is enforced on bytes actually read.
            chunks: list[bytes] = []
            total = 0
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > MAX_BYTES:
                    await response.aclose()
                    raise FetchFailed("that page is larger than 2 MB — too big to study")
                chunks.append(chunk)
            await response.aclose()

            raw = b"".join(chunks)
            html = raw.decode(response.encoding or "utf-8", errors="replace")
            return Fetched(
                html=html,
                final_url=normalised,
                hops=tuple(hops),
                content_type=ctype or "text/html",
                bytes_read=total,
            )

    raise FetchFailed(
        f"that URL redirected more than {MAX_REDIRECTS} times without settling"
    )
