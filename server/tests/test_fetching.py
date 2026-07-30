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
    fetch_url,
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


# ── multi-modality routing ────────────────────────────────────────────────
#
# The fetcher used to accept HTML and nothing else, so a link to an ad image, a
# hosted MP4, a podcast or a PDF deck could not be studied from a URL at all —
# only by uploading the file. These cover the widened front door WITHOUT
# widening the SSRF guard, which is the part that must not move.


def test_content_type_decides_the_modality_not_the_extension():
    """`…/promo.mp4` served as text/html is a player page, and studying it as a
    video would mean decoding an HTML document. The bytes' declared type routes
    them; the URL's spelling never does."""
    from app.fetching import kind_for_content_type

    assert kind_for_content_type("image/png") == "image"
    assert kind_for_content_type("image/jpeg; charset=binary") == "image"
    assert kind_for_content_type("video/mp4") == "video"
    assert kind_for_content_type("audio/mpeg") == "audio"
    assert kind_for_content_type("application/pdf") == "document"
    assert kind_for_content_type("text/html") == "page"
    # Not a study asset in any modality.
    assert kind_for_content_type("application/octet-stream") is None
    assert kind_for_content_type("application/zip") is None
    assert kind_for_content_type("") is None


def test_every_modality_keeps_a_byte_ceiling():
    """A cap that exists for HTML and not for video is not a cap."""
    from app.fetching import ASSET_KINDS

    for kind, (prefixes, cap) in ASSET_KINDS.items():
        assert prefixes, f"{kind} accepts no content types"
        assert 0 < cap <= 100_000_000, f"{kind} cap is {cap}"


def test_media_kinds_are_refused_by_the_page_fetcher():
    """`fetch_page` must stay narrow. If it silently accepted an MP4, the page
    adapter would run its HTML regexes over binary and emit beats from it."""
    from app.fetching import ASSET_KINDS

    page_prefixes = ASSET_KINDS["page"][0]
    for kind in ("image", "video", "audio", "document"):
        for prefix in ASSET_KINDS[kind][0]:
            assert not prefix.startswith(tuple(page_prefixes))


def test_requesting_no_kinds_is_refused_rather_than_defaulting_open():
    """An empty allowlist must fail closed. Defaulting to "everything" here
    would make a typo in a caller's `kinds=` argument into a content-type
    bypass."""
    with pytest.raises(UnsafeURL):
        asyncio.run(fetch_url("https://example.com", kinds=()))


def test_player_pages_are_named_before_any_fetch():
    """youtube.com/watch answers 200 with an HTML shell whose extractable prose
    is the title and the comment policy. Fetching it SUCCEEDS, which is exactly
    why it has to be caught by name — the failure is silent otherwise, and
    produces a full appraisal tensor about page furniture."""
    from app.modality import is_player_url

    assert is_player_url("https://www.youtube.com/watch?v=abc") == "youtube.com"
    assert is_player_url("https://youtu.be/abc") == "youtu.be"
    assert is_player_url("https://vimeo.com/12345") == "vimeo.com"
    assert is_player_url("https://m.tiktok.com/x") == "tiktok.com"
    # a genuine media file on an unrelated host is not a player page
    assert is_player_url("https://cdn.example.com/spot.mp4") is None
    # and a lookalike host must not match by substring
    assert is_player_url("https://notyoutube.com/watch") is None
    assert is_player_url("https://youtube.com.evil.test/watch") is None


def test_the_relay_reruns_validation_rather_than_trusting_the_earlier_fetch():
    """/content/fetch and /content/media are separate requests, and a name can
    be re-pointed between them. The relay must refuse a private address on its
    own account."""
    for url in ("http://127.0.0.1/x.mp4", "http://169.254.169.254/latest.mp4"):
        with pytest.raises(UnsafeURL):
            asyncio.run(fetch_url(url, kinds=("video", "audio")))


def test_a_failing_status_is_diagnosed_rather_than_guessed_at():
    """One sentence for every 4xx and 5xx sent readers to fix the wrong thing.

    Every failing status used to produce "The page may require a login, or may
    be blocking automated access." That is right for 401 and 403 and wrong for
    everything else — and the overwhelmingly common case is a 404 from a URL
    with a typo in it. Telling someone their mistyped link is behind a login
    wall sends them to check credentials or change networks when the fix is to
    reread the address. It is the same defect the audit backlog records for the
    `Accept` header: a message that blamed "blocking automated access" for a
    refusal and sent the reader to fix nothing.
    """
    from app.fetching import _status_hint

    # the case that prompted this: a plain missing address
    missing = _status_hint(404)
    assert "does not exist" in missing
    assert "login" not in missing.lower()
    assert "blocking" not in missing.lower()

    # the two where the old sentence was actually correct keep saying so
    assert "authentication" in _status_hint(401)
    assert "blocking automated access" in _status_hint(403)

    # and the rest are distinguished rather than collapsed
    assert "rate-limiting" in _status_hint(429)
    assert "host's side" in _status_hint(503)
    assert _status_hint(410) != _status_hint(404)

    # an unrecognised status is reported as unexplained, never speculated about
    odd = _status_hint(418)
    assert "login" not in odd.lower() and "blocking" not in odd.lower()
