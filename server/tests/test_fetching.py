"""SSRF guard tests.

These are regression tests in the strongest sense: the fetcher exists to make a
URL box safe, and every case below is a way that box becomes a hole in the
private network if someone later "simplifies" the guard. They assert on the
REFUSAL, not on the error text, so they keep working if the wording changes.

The one that earns its place most is `test_carrier_grade_nat_is_refused`. That
range passed the original implementation — `is_private` and `is_reserved` both
report False for 100.64.0.0/10 — and the fetcher opened a connection to it.
Tailscale assigns out of that range.
"""

import asyncio
import ipaddress

import pytest

from app.fetching import (
    ALLOWED_PORTS,
    ALLOWED_SCHEMES,
    FetchFailed,
    UnsafeURL,
    _reject_ip,
    _validate,
    fetch_page,
)


def _refuses(url: str) -> str:
    """Run the fetcher and require an UnsafeURL. Returns the message."""
    with pytest.raises(UnsafeURL) as excinfo:
        asyncio.run(fetch_page(url))
    return str(excinfo.value)


# ── the address categories ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "addr",
    [
        "127.0.0.1",          # loopback
        "0.0.0.0",            # unspecified
        "10.0.0.1",           # RFC1918
        "172.16.5.4",         # RFC1918
        "192.168.1.1",        # RFC1918
        "169.254.169.254",    # cloud metadata
        "100.64.0.1",         # RFC6598 carrier-grade NAT / Tailscale
        "100.127.255.254",    # top of the CGNAT range
        "::1",                # IPv6 loopback
        "fe80::1",            # IPv6 link-local
        "fdaa:0:1::3",        # Fly 6PN unique-local — the private backend
        "fc00::1",            # ULA
        "::ffff:127.0.0.1",   # IPv4-mapped loopback
        "::ffff:10.0.0.1",    # IPv4-mapped RFC1918
        "2002:7f00:1::1",     # 6to4-encoded 127.0.0.1
        "64:ff9b::7f00:1",    # NAT64-encoded 127.0.0.1
        "224.0.0.1",          # multicast
        "255.255.255.255",    # broadcast
    ],
)
def test_non_routable_addresses_are_refused(addr):
    assert _reject_ip(ipaddress.ip_address(addr)) is not None, (
        f"{addr} was accepted — this is an SSRF hole"
    )


@pytest.mark.parametrize("addr", ["93.184.216.34", "1.1.1.1", "2606:4700::1111"])
def test_public_addresses_are_allowed(addr):
    assert _reject_ip(ipaddress.ip_address(addr)) is None


def test_carrier_grade_nat_is_refused():
    """The range that got through the first implementation.

    Kept as its own named test because the reason it slipped is subtle: both
    `is_private` and `is_reserved` are False for it, so a guard written from
    the obvious flags admits it. `is_global` is the property that matters.
    """
    ip = ipaddress.ip_address("100.64.0.1")
    assert ip.is_private is False and ip.is_reserved is False, (
        "the premise of this test changed — Python now classifies CGNAT "
        "differently, so re-check the guard rather than deleting this"
    )
    assert _reject_ip(ip) is not None


# ── URL shape ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "gopher://example.com/",
        "redis://example.com/",
        "ftp://example.com/",
        "javascript:alert(1)",
        "data:text/html,<h1>x</h1>",
        "example.com",  # no scheme at all
    ],
)
def test_only_http_schemes_are_accepted(url):
    with pytest.raises(UnsafeURL):
        _validate(url)


@pytest.mark.parametrize("port", [22, 25, 3306, 5432, 6379, 8000, 9200, 11211, 27017])
def test_service_ports_are_refused(port):
    with pytest.raises(UnsafeURL):
        _validate(f"http://example.com:{port}/")


def test_embedded_credentials_are_refused():
    """`https://trusted.com@169.254.169.254/` reads as trusted to a human."""
    with pytest.raises(UnsafeURL):
        _validate("http://trusted.com@169.254.169.254/")


def test_absurdly_long_urls_are_refused():
    with pytest.raises(UnsafeURL):
        _validate("https://example.com/" + "a" * 4000)


def test_allowlists_stay_narrow():
    """A guard is only as good as its lists; widening one should be deliberate."""
    assert ALLOWED_SCHEMES == {"http", "https"}
    assert ALLOWED_PORTS <= {80, 443, 8080, 8443}


# ── end to end, no network required ───────────────────────────────────────


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/",
        "http://127.0.0.1/",
        "http://[::1]/",
        "http://169.254.169.254/",
        "http://[fdaa:0:1::3]/",
        "http://100.64.0.1/",
        "http://10.0.0.1:8080/",
    ],
)
def test_fetch_refuses_before_connecting(url):
    """No socket is opened for any of these — the refusal is pre-connection.

    If one of these ever raises FetchFailed instead of UnsafeURL it means we
    tried to reach it, which is the bug.
    """
    _refuses(url)


def test_hostname_resolving_to_loopback_is_refused():
    """The name is public-looking; the address is not. Resolution decides."""
    msg = _refuses("http://localhost/")
    assert "loopback" in msg.lower()


def test_fetch_failed_is_distinct_from_unsafe():
    """Two different failures that deserve two different messages.

    `UnsafeURL` is "we will not", `FetchFailed` is "we could not". Collapsing
    them would report a refusal as an outage and send the researcher chasing a
    network problem that does not exist.
    """
    assert not issubclass(FetchFailed, UnsafeURL)
    assert not issubclass(UnsafeURL, FetchFailed)
