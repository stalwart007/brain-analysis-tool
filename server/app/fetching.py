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
from typing import Optional, Sequence
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

#: What each kind of study asset is served as, and how much of it we will take.
#:
#: The caps differ by an order of magnitude because the downstream cost does.
#: An image goes to a vision model as base64, which inflates it by a third and
#: is billed; a video is never decoded here at all, only relayed to the browser
#: that will decode it, so it can be larger. The ceiling is enforced on bytes
#: ACTUALLY READ rather than on Content-Length, because Content-Length is a
#: claim by a server we have no reason to trust.
ASSET_KINDS: dict[str, tuple[tuple[str, ...], int]] = {
    "page": (("text/html", "application/xhtml", "text/plain"), MAX_BYTES),
    "image": (("image/png", "image/jpeg", "image/webp", "image/gif", "image/avif"), 12_000_000),
    "document": (("application/pdf",), 25_000_000),
    "video": (("video/",), 80_000_000),
    # `application/ogg` is the registered type for an .ogg container and is what
    # Wikimedia and a good deal of podcast hosting actually serve; matching only
    # `audio/` refused real audio files as "not something this platform can
    # study", which is a confusing thing to be told about an audio file.
    "audio": (("audio/", "application/ogg"), 60_000_000),
}

#: Every prefix any kind accepts. An octet-stream matches nothing and is
#: refused: it is either a download or a trap, and guessing its type from the
#: file extension is how a "video" turns out to be a script.
ALLOWED_CONTENT_PREFIXES: tuple[str, ...] = tuple(
    prefix for prefixes, _cap in ASSET_KINDS.values() for prefix in prefixes
)

#: Largest cap across all kinds — the bound the transport enforces when the
#: caller has not narrowed to one kind, since the type is only known after the
#: response headers arrive.
MAX_ANY_BYTES = max(cap for _prefixes, cap in ASSET_KINDS.values())


def kind_for_content_type(ctype: str) -> Optional[str]:
    """Which study modality a MIME type belongs to, or None.

    Content type decides, never the URL's extension. `…/promo.mp4` served as
    `text/html` is a page that will be analysed as a page, and `…/x?id=9`
    served as `image/png` is an image — the bytes are what the study consumes,
    so the bytes' declared type is what routes them.
    """
    ctype = (ctype or "").split(";")[0].strip().lower()
    for kind, (prefixes, _cap) in ASSET_KINDS.items():
        if ctype.startswith(prefixes):
            return kind
    return None

#: Presented as a browser because many sites serve a challenge page or a
#: stripped response to unknown agents, and a challenge page would be analysed
#: as if it were the landing page. Honest about who we are in the comment; the
#: header is a compatibility measure, not a disguise for abuse.
USER_AGENT = (
    "Mozilla/5.0 (compatible; CogniSwarmBot/1.0; +https://cogniswarm-app.fly.dev) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

#: The same identity without the browser costume, tried once when the costume
#: is what got us refused.
#:
#: Both directions of this are real and they pull opposite ways. Plenty of
#: marketing sites serve a challenge page or a stripped shell to anything that
#: does not look like a browser, which is why `USER_AGENT` exists. But some
#: origins — w3.org behind Cloudflare, measured — refuse a Chrome string coming
#: from a datacentre IP precisely BECAUSE it is transparently not Chrome, and
#: that refusal arrives as a 403 that reads exactly like "this page requires a
#: login". No single string wins on both, so a refusal is retried once with
#: the honest one before the failure is reported.
BOT_USER_AGENT = "CogniSwarmBot/1.0 (+https://cogniswarm-app.fly.dev)"

#: Statuses worth one retry under the other identity. Deliberately not 404 or
#: 5xx: those are about the resource or the origin, and re-asking as someone
#: else cannot change them.
_UA_RETRY_STATUSES = frozenset({400, 401, 403, 406, 418, 429})


class UnsafeURL(ValueError):
    """The URL is refused before any connection is attempted."""


class FetchFailed(RuntimeError):
    """The URL was allowed, but could not be retrieved."""


@dataclass(frozen=True)
class Fetched:
    """What a successful fetch yields."""

    #: Exactly what came off the wire. Text modalities decode it; image, video,
    #: audio and PDF need it intact, and decoding-then-re-encoding a JPEG to
    #: satisfy a `str`-typed field is how binary assets get silently corrupted.
    content: bytes
    #: Where we ended up, which is not always where we were pointed.
    final_url: str
    #: Redirect chain actually walked, for display.
    hops: tuple[str, ...]
    content_type: str
    bytes_read: int
    #: Which study modality this content belongs to, decided by content type.
    kind: Optional[str] = None
    #: What the origin SAID the body weighs, when probing rather than reading.
    #: Deliberately not folded into `bytes_read`: one is a measurement of bytes
    #: this process actually counted and the other is an unverified claim, and a
    #: size cap that trusts the claim is not a cap.
    declared_length: Optional[int] = None
    #: What the response said it was encoded as. Carried rather than assumed:
    #: decoding a Latin-1 or Shift-JIS page as UTF-8 replaces every non-ASCII
    #: character with U+FFFD, and the beats a study then runs on are the page's
    #: copy with its punctuation and accents shot through with question marks.
    encoding: Optional[str] = None

    @property
    def html(self) -> str:
        """Decoded text, for the `page` adapter.

        A property rather than a second stored field so there is exactly one
        copy of the payload in memory — these run to tens of megabytes for
        video, and holding both a bytes and a str view doubles that for no
        reason.
        """
        return self.content.decode(self.encoding or "utf-8", errors="replace")


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


def _status_hint(status: int) -> str:
    """What a failing status actually means, rather than one guess for all of them.

    Every 4xx and 5xx used to produce the same sentence — "The page may require a
    login, or may be blocking automated access." That is right for 401 and 403
    and wrong for everything else, and the most common failure by far is a plain
    404 from a URL with a typo in it. Telling someone their mistyped link is
    behind a login wall sends them to check credentials, change networks, or
    file a bug, when the fix is to look at the address. This is the same defect
    the audit backlog records for the `Accept` header — a message that blamed
    "blocking automated access" for a refusal we had caused ourselves, and sent
    the reader to fix nothing.

    Anything unrecognised falls through to a neutral statement rather than a
    speculative one: an unexplained status is better reported as unexplained.
    """
    if status == 404:
        # Deliberately does not mention authentication even to rule it out:
        # naming a theory in order to deny it still plants it, and this is the
        # status where the reader most needs to be looking at the address.
        return "That address does not exist on this host — check the URL for a typo."
    if status == 410:
        return "That address used to exist and has been removed."
    if status in (401, 407):
        return "That resource requires authentication, so it cannot be fetched from a link alone."
    if status == 403:
        return (
            "The host refused the request. It may be blocking automated access, "
            "or the resource may be private."
        )
    if status == 429:
        return "The host is rate-limiting us. Waiting and retrying usually works."
    if status in (405, 501):
        return "The host does not allow this kind of request for that address."
    if status == 451:
        return "The host is refusing for legal reasons."
    if 500 <= status < 600:
        return "That is a fault on the host's side, not with the link — retrying may work."
    if 400 <= status < 500:
        return "The host rejected the request as malformed or unacceptable."
    return "The host did not return a usable response."


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
    """Fetch `url` as a web page. Refuses anything that is not markup."""
    fetched = await fetch_url(url, kinds=("page",))
    return fetched


async def _guarded_request(
    url: str,
    *,
    allowed_prefixes: tuple[str, ...],
    cap: int,
    accept: str,
    method: str = "GET",
    body: Optional[bytes] = None,
    body_content_type: Optional[str] = None,
    user_agent: Optional[str] = None,
    on_wrong_type: Optional[object] = None,
    probe: bool = False,
) -> Fetched:
    """The guarded transport. Every byte this process pulls off the internet
    comes through here, and nothing else opens a socket.

    THE SSRF DEFENCE LIVES HERE AND ONLY HERE. Scheme and port allowlist,
    resolve every address ourselves, connect to the address we validated rather
    than to the name, revalidate on every redirect, forward no credentials, cap
    time and bytes. That is the reason this stayed a single function when the
    fetcher grew from HTML to five modalities, and it is the reason the JSON
    path added for YouTube is a POLICY WRAPPER over this rather than a second
    transport: adding a second fetch path would have doubled the number of
    places that have to get all of the above right, which is precisely how one
    of them ends up not.

    `on_wrong_type` is a callable taking the served content type and returning
    the refusal message, so the study-asset wrapper can keep its modality-aware
    wording without this function knowing what a modality is.
    """
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

            async def _send(agent: str):
                headers = {
                    # The origin still needs to know which vhost we want,
                    # and TLS still has to be verified against the NAME —
                    # `sni_hostname` drives both SNI and certificate
                    # validation, so pinning the IP does not silently
                    # downgrade us to an unverified connection.
                    "Host": host if port in (80, 443) else f"{host}:{port}",
                    "User-Agent": agent,
                    # Derived from what the caller will actually accept, not
                    # hardcoded. This header said `text/html,…` while the
                    # fetcher had grown to five modalities, and a correctly
                    # behaved origin honours it: Wikimedia answered 400 and
                    # w3.org 403 for an image and a PDF, which read as
                    # "blocking automated access" and were nothing of the
                    # kind — we asked for HTML and were served the refusal
                    # we requested.
                    "Accept": accept,
                    "Accept-Language": "en-US,en;q=0.9",
                }
                if body_content_type:
                    headers["Content-Type"] = body_content_type
                return await client.request(
                    method,
                    urlunparse(pinned),
                    content=body,
                    headers=headers,
                    extensions={"sni_hostname": host},
                )

            primary = user_agent or USER_AGENT
            try:
                response = await _send(primary)
                # One retry without the browser costume. Some origins refuse a
                # Chrome string from a datacentre IP precisely because it is
                # transparently not Chrome, and the refusal is indistinguishable
                # from a login wall in the message we would otherwise show.
                #
                # Skipped when the caller pinned a User-Agent: an API client
                # string like InnerTube's is not a costume to be dropped, it is
                # the credential that selects the response shape, and re-asking
                # as someone else returns a DIFFERENT document rather than the
                # same one unblocked.
                if response.status_code in _UA_RETRY_STATUSES and user_agent is None:
                    await response.aclose()
                    response = await _send(BOT_USER_AGENT)
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
                status = response.status_code
                await response.aclose()
                raise FetchFailed(f"{host} answered {status}. {_status_hint(status)}")

            ctype = response.headers.get("content-type", "").split(";")[0].strip().lower()
            if ctype and not ctype.startswith(allowed_prefixes):
                await response.aclose()
                if callable(on_wrong_type):
                    raise FetchFailed(on_wrong_type(ctype))
                raise FetchFailed(f"that URL serves {ctype}, which was not expected here.")

            if probe:
                # Everything the caller wanted is in the headers. Reading the
                # body would pull down up to eighty megabytes of video in order
                # to learn a content type — which is what this endpoint used to
                # do, twice, once here and once again in the relay.
                declared = response.headers.get("content-length")
                await response.aclose()
                return Fetched(
                    content=b"",
                    final_url=normalised,
                    hops=tuple(hops),
                    content_type=ctype or "application/octet-stream",
                    # A CLAIM by the server, not a measurement, and named as one
                    # by `declared_length` rather than being passed off as
                    # `bytes_read`. The cap is still enforced on real bytes when
                    # the relay actually pulls them.
                    bytes_read=0,
                    declared_length=int(declared) if (declared or "").isdigit() else None,
                    kind=kind_for_content_type(ctype),
                    encoding=response.encoding,
                )

            # Streamed with a running total: Content-Length is a claim by the
            # server, not a fact, so the cap is enforced on bytes actually read.
            chunks: list[bytes] = []
            total = 0
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > cap:
                    await response.aclose()
                    raise FetchFailed(
                        f"that file is larger than {cap // 1_000_000} MB, which is "
                        "the ceiling for this content type — too big to study."
                    )
                chunks.append(chunk)
            await response.aclose()

            return Fetched(
                content=b"".join(chunks),
                final_url=normalised,
                hops=tuple(hops),
                content_type=ctype or "text/html",
                bytes_read=total,
                kind=kind_for_content_type(ctype) or "page",
                encoding=response.encoding,
            )

    raise FetchFailed(
        f"that URL redirected more than {MAX_REDIRECTS} times without settling"
    )


async def fetch_url(
    url: str,
    kinds: Sequence[str] = tuple(ASSET_KINDS),
    max_bytes: Optional[int] = None,
    *,
    probe: bool = False,
) -> Fetched:
    """Fetch `url` safely as a STUDY ASSET, following up to MAX_REDIRECTS hops.

    `kinds` narrows what will be accepted. Every caller states it, because
    "whatever comes back" is how a study ends up analysing a 404 page: a URL
    that was meant to be a video and answers with HTML should fail as the wrong
    kind of thing rather than quietly become a page study of an error message.

    `probe` stops after the headers — see `probe_url`, which is the name to
    call it by.
    """
    allowed_prefixes = tuple(
        p for k in kinds for p in ASSET_KINDS.get(k, ((), 0))[0]
    )
    if not allowed_prefixes:
        raise UnsafeURL(f"no fetchable content kinds requested (got {list(kinds)!r})")
    cap = max_bytes or max(ASSET_KINDS[k][1] for k in kinds if k in ASSET_KINDS)
    # Wildcard families (`video/`, `audio/`) become `video/*` so the header is
    # a legal media range; concrete types pass through unchanged.
    accept = ", ".join(
        dict.fromkeys(p + "*" if p.endswith("/") else p for p in allowed_prefixes)
    )

    def _refusal(ctype: str) -> str:
        served = kind_for_content_type(ctype)
        if served:
            return (
                f"that URL serves {ctype}, which is a {served}, but this "
                f"study is expecting {' or '.join(kinds)}. Switch the "
                "content type and paste the link again."
            )
        return (
            f"that URL serves {ctype}, which is not something this "
            "platform can study. Supported: web pages, images, video, "
            "audio and PDFs."
        )

    return await _guarded_request(
        url,
        allowed_prefixes=allowed_prefixes,
        cap=cap,
        accept=accept,
        on_wrong_type=_refusal,
        probe=probe,
    )


async def probe_url(url: str, kinds: Sequence[str] = tuple(ASSET_KINDS)) -> Fetched:
    """Identify a URL without downloading its body.

    WHY. `/content/fetch` needs one fact about a hosted video — what it is —
    and used to establish it by pulling the entire file into memory, after
    which `/content/media` pulled the identical file into memory a second time
    so the browser could have it. Two full downloads and two copies of up to
    eighty megabytes resident, to answer a question the response headers had
    already answered before the first byte of body arrived.

    This is a real GET rather than a HEAD, because origins lie about HEAD far
    more than they lie about GET — S3-compatible hosts and CDNs routinely
    answer 403 or 405 to a HEAD for an object they will happily serve — and
    because the redirect walk has to run identically either way. The body is
    simply never read: the response is closed as soon as the content type is
    known, so the connection carries headers and nothing else.
    """
    return await fetch_url(url, kinds=kinds, probe=True)


#: What a JSON API answers with. Narrow on purpose: this path exists for one
#: caller (the YouTube manifest) and must not become a general-purpose way to
#: pull arbitrary documents back through the guard.
JSON_PREFIXES = ("application/json", "text/json")

#: A metadata document, not a media file. Two orders of magnitude below the
#: asset caps because an InnerTube player response is ~200 kB and anything
#: claiming to be far larger is not the document we asked for.
MAX_JSON_BYTES = 4_000_000


async def fetch_json(
    url: str,
    *,
    body: Optional[bytes] = None,
    user_agent: Optional[str] = None,
    max_bytes: int = MAX_JSON_BYTES,
) -> tuple[dict, str]:
    """POST (or GET) a JSON API through the same guard, returning (doc, final_url).

    The SSRF surface is identical to `fetch_url` because it IS `fetch_url`'s
    transport — only the accepted content type and the size cap differ. A
    `body` makes it a POST, which is what InnerTube requires.
    """
    import json as _json

    fetched = await _guarded_request(
        url,
        allowed_prefixes=JSON_PREFIXES,
        cap=max_bytes,
        accept="application/json",
        method="POST" if body is not None else "GET",
        body=body,
        body_content_type="application/json" if body is not None else None,
        user_agent=user_agent,
        on_wrong_type=lambda ctype: (
            f"expected JSON from that endpoint and got {ctype} — it is not the "
            "API it claims to be, or the request was intercepted."
        ),
    )
    try:
        doc = _json.loads(fetched.content)
    except ValueError as exc:
        raise FetchFailed("that endpoint answered with malformed JSON") from exc
    if not isinstance(doc, dict):
        raise FetchFailed("that endpoint answered with JSON that is not an object")
    return doc, fetched.final_url


#: A caption track is served as JSON or as one of two XML dialects depending on
#: the `fmt` asked for, and which one an origin honours is not knowable in
#: advance — so the accepted set spans all of them and the PARSER decides.
TEXT_DOCUMENT_PREFIXES = (
    "application/json",
    "text/json",
    "text/xml",
    "application/xml",
    "application/ttml+xml",
    "text/plain",
    "text/vtt",
)


async def fetch_text(
    url: str,
    *,
    prefixes: tuple[str, ...] = TEXT_DOCUMENT_PREFIXES,
    user_agent: Optional[str] = None,
    max_bytes: int = MAX_JSON_BYTES,
) -> str:
    """Fetch a small text document (a caption track) through the same guard.

    Separate from `fetch_json` only because the payload may legitimately be
    either JSON or XML and the caller parses whichever arrived. The transport,
    and therefore the entire SSRF surface, is identical.
    """
    fetched = await _guarded_request(
        url,
        allowed_prefixes=prefixes,
        cap=max_bytes,
        accept=", ".join(prefixes),
        user_agent=user_agent,
        on_wrong_type=lambda ctype: f"expected a text document and got {ctype}",
    )
    return fetched.html
