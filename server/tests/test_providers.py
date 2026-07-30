"""How a link describes itself, when the page cannot be read as content.

All offline. The network half is one function whose behaviour belongs to other
people's servers; what is ours is the scraping, the URL normalisation and the
merge policy, and each of those is silently wrong-able.

The two that earn their place most:

`test_relative_and_protocol_relative_images_are_absolutised` — an `og:image` of
`//cdn.example/x.jpg` handed to the fetcher unchanged is a URL with no scheme,
and `/hero.png` is a path. Both were plausible-looking strings that would have
produced confusing fetch failures rather than a clean absence.

`test_merge_is_per_field_not_per_document` — oEmbed and OpenGraph are good at
different things, and picking one document wholesale throws away the better
half of the other.
"""

import pytest

from app.providers import (
    LinkMeta,
    _absolute_image,
    extract_meta,
    merge,
    oembed_endpoint,
)

#: Shaped like real markup: both attribute orders, entities, a title tag that
#: disagrees with og:title. A hand-tidied fixture would test a format that does
#: not occur.
REAL_ISH = """
<!doctype html><html><head>
<title>Flickermood by Forss | Free Listening on SoundCloud</title>
<meta property="og:title" content="Flickermood"/>
<meta content="Stream Flickermood by Forss &amp; more on desktop and mobile." property="og:description"/>
<meta property="og:image" content="https://i1.sndcdn.com/artworks-000067273316-t500x500.jpg"/>
<meta property="og:type" content="music.song"/>
<meta property="og:site_name" content="SoundCloud"/>
<meta name="twitter:title" content="ignored, og wins"/>
</head><body><div id="root"></div></body></html>
"""


# ── scraping ───────────────────────────────────────────────────────────────


def test_opengraph_is_read_in_both_attribute_orders():
    meta = extract_meta(REAL_ISH, "https://soundcloud.com/forss/flickermood")
    assert meta.title == "Flickermood"
    # `content=` before `property=` — the reversed order, which a single
    # forward-only pattern misses entirely.
    assert meta.description.startswith("Stream Flickermood by Forss & more")
    assert meta.site_name == "SoundCloud"
    assert meta.og_type == "music.song"
    assert "opengraph" in meta.sources


def test_entities_are_unescaped_and_whitespace_collapsed():
    """Markup must never reach a twin. `&amp;` read literally becomes a beat
    containing the characters a-m-p, which the appraisal then rates."""
    meta = extract_meta(
        '<meta property="og:title" content="Tom &amp; Jerry\n\n   spaced"/>', "https://x.test/"
    )
    assert meta.title == "Tom & Jerry spaced"


def test_og_title_beats_the_title_tag():
    """A `<title>` on a page with proper cards is decoration — it carries the
    site name and SEO padding, which is not what the thing is called."""
    meta = extract_meta(REAL_ISH, "https://soundcloud.com/forss/flickermood")
    assert meta.title == "Flickermood"
    assert "SoundCloud" not in meta.title


def test_title_tag_is_the_floor_when_there_are_no_cards():
    meta = extract_meta("<html><head><title>  Plain   Page </title></head></html>", "https://x.test/")
    assert meta.title == "Plain Page"
    assert "title tag" in meta.sources


def test_malformed_markup_yields_an_empty_meta_rather_than_raising():
    for junk in ("", "<html", "<<>>", "not html at all"):
        assert extract_meta(junk, "https://x.test/").title == ""


# ── the `usable` gate ──────────────────────────────────────────────────────


def test_a_title_alone_is_not_enough_to_study():
    """THE gate. Every HTML page has a title, including consent walls and error
    pages, so accepting one would turn "we could not read this" into a
    confident study of the words "Access Denied"."""
    assert LinkMeta(url="u", title="Access Denied").usable is False
    assert LinkMeta(url="u", title="X", image_url="https://x.test/a.png").usable is True
    assert LinkMeta(url="u", title="X", description="d" * 80).usable is True
    # Just under the bar — a one-line description is a tagline, not content.
    assert LinkMeta(url="u", title="X", description="d" * 79).usable is False


def test_brief_reads_as_prose_not_as_a_field_dump():
    meta = LinkMeta(
        url="u", title="Do schools kill creativity?", author="Ken Robinson",
        site_name="TED", description="A talk about education.",
    )
    assert meta.as_brief() == (
        "Do schools kill creativity? by Ken Robinson on TED\n\nA talk about education."
    )


# ── image URL normalisation ────────────────────────────────────────────────


def test_relative_and_protocol_relative_images_are_absolutised():
    base = "https://site.test/blog/post"
    assert _absolute_image("//cdn.test/x.jpg", base) == "https://cdn.test/x.jpg"
    assert _absolute_image("/hero.png", base) == "https://site.test/hero.png"
    assert _absolute_image("thumb.png", base) == "https://site.test/blog/thumb.png"
    assert _absolute_image("https://other.test/a.png", base) == "https://other.test/a.png"


def test_non_http_image_schemes_are_refused_here_not_downstream():
    """Failing here is a clean absence. Passing a `data:` URL to the fetcher
    produces a scheme-allowlist error that reads as a network problem."""
    base = "https://site.test/"
    assert _absolute_image("data:image/png;base64,iVBOR", base) is None
    assert _absolute_image("javascript:alert(1)", base) is None
    assert _absolute_image("blob:https://site.test/abc", base) is None
    assert _absolute_image("", base) is None
    assert _absolute_image("   ", base) is None


def test_scraped_image_is_absolutised_through_extract():
    meta = extract_meta(
        '<meta property="og:image" content="/img/hero.png"/>', "https://site.test/a/b"
    )
    assert meta.image_url == "https://site.test/img/hero.png"


# ── oEmbed endpoint selection ──────────────────────────────────────────────


def test_discovery_in_the_markup_beats_the_registry():
    """A host that advertises its own endpoint knows better than a table
    compiled here, and the advertised URL already carries the right params."""
    html = (
        '<link rel="alternate" type="application/json+oembed" '
        'href="https://vimeo.com/api/oembed.json?url=SPECIFIC&amp;width=640">'
    )
    found = oembed_endpoint("https://vimeo.com/123", html)
    assert found is not None and "SPECIFIC" in found
    # The entity in the advertised href is unescaped, or the query breaks.
    assert "&amp;" not in found


def test_registry_covers_hosts_that_do_not_advertise():
    assert "dailymotion" in (oembed_endpoint("https://www.dailymotion.com/video/x8", "") or "")
    assert "spotify" in (oembed_endpoint("https://open.spotify.com/track/abc", "") or "")
    assert oembed_endpoint("https://unknown.test/thing", "") is None


def test_the_target_url_is_encoded_into_the_endpoint():
    """An unencoded `?` or `&` from the target silently truncates the query and
    the endpoint answers about the wrong resource."""
    found = oembed_endpoint("https://open.spotify.com/track/a?si=b&x=c", "")
    assert found is not None
    assert "%3Fsi%3Db%26x%3Dc" in found


def test_lookalike_host_does_not_match_the_registry():
    assert oembed_endpoint("https://vimeo.com.evil.test/1", "") is None
    assert oembed_endpoint("https://notvimeo.com/1", "") is None
    # A genuine subdomain still does.
    assert oembed_endpoint("https://player.vimeo.com/video/1", "") is not None


# ── merge policy ───────────────────────────────────────────────────────────


def test_merge_is_per_field_not_per_document():
    """oEmbed has the exact title and author because the host authored them;
    OpenGraph usually has the longer description and the bigger image. Taking
    either document whole discards the better half of the other."""
    structured = LinkMeta(url="u", title="Exact Title", author="Real Author", sources=["oembed"])
    scraped = LinkMeta(
        url="u", title="SEO padded title | Site", description="The long description.",
        image_url="https://x.test/big.png", site_name="Site", sources=["opengraph"],
    )
    out = merge(structured, scraped)
    assert out.title == "Exact Title"        # primary wins where it has a value
    assert out.author == "Real Author"
    assert out.description == "The long description."  # filled from secondary
    assert out.image_url == "https://x.test/big.png"
    assert out.site_name == "Site"
    assert out.sources == ["oembed", "opengraph"]


def test_merge_tolerates_either_side_missing():
    only = LinkMeta(url="u", title="T")
    assert merge(None, only) is only
    assert merge(only, None) is only
    assert merge(None, None) is None


def test_merge_does_not_duplicate_sources():
    a = LinkMeta(url="u", title="A", sources=["opengraph"])
    b = LinkMeta(url="u", description="d", sources=["opengraph"])
    assert merge(a, b).sources == ["opengraph"]
