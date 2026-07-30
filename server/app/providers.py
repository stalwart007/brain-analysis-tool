"""What a link says it is, when the page itself cannot be read as content.

WHY THIS EXISTS. The fetcher can already handle a page whose prose is the
content, a direct media file, an image and a PDF. Everything else — a Vimeo
page, a SoundCloud track, a Substack post that renders in the browser, a
LinkedIn post, a TikTok — used to be a dead end: either a 422 naming the
platform, or a "page" study of navigation furniture. That is the wrong shape of
answer. A researcher pasting a link is not asking us to classify their URL, they
are asking what the thing does to an audience, and "we do not support that host"
is a refusal to try.

THE OBSERVATION THAT MAKES THIS WORK. Almost every page on the web that
represents a piece of media already describes itself, in the markup, for
exactly this purpose — OpenGraph and Twitter cards exist so that a link
unfurls in a chat window. Measured across SoundCloud, Substack, Spotify,
Dailymotion and TED: `og:title`, `og:description` and `og:image` are present
and the image is a real, fetchable asset. That is a title, a pitch, and a hero
frame, which is enough for a genuine study of HOW A THING PRESENTS ITSELF.

WHAT THIS IS AND IS NOT. It is not a study of the video, the track or the
article, and nothing here pretends otherwise — `rung` carries that distinction
to the UI and the findings. It is a study of the listing: the thumbnail and the
copy that decide whether anybody clicks. That is a real question, and for a lot
of content it is the more commercially useful one.

ORDER OF PREFERENCE, and the reason for it:
  1. oEmbed, where the host publishes it. A structured document the host
     authored for consumers, so the title and author are exact rather than
     scraped.
  2. OpenGraph / Twitter cards, scraped from the markup we already fetched.
     Nearly universal, and free — the page is already in memory.
  3. The `<title>` tag, which is the floor.

Every request goes through `fetching`, like everything else in this codebase.
No socket is opened here.
"""

from __future__ import annotations

import html as _html
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import quote, urljoin, urlparse

from .fetching import FetchFailed, UnsafeURL, fetch_json

#: Hosts that publish oEmbed but do not advertise it in their markup, so
#: discovery cannot find it. Measured working from a server; the ones that
#: refused a datacentre address (TikTok times out, Instagram requires a token)
#: are deliberately absent — a registry entry that always fails is worse than
#: no entry, because it costs a round trip before the fallback runs.
OEMBED_ENDPOINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("vimeo.com", "player.vimeo.com"), "https://vimeo.com/api/oembed.json?url={url}"),
    (
        ("dailymotion.com", "dai.ly"),
        "https://www.dailymotion.com/services/oembed?format=json&url={url}",
    ),
    (("open.spotify.com",), "https://open.spotify.com/oembed?url={url}"),
    (("soundcloud.com",), "https://soundcloud.com/oembed?format=json&url={url}"),
    (("ted.com",), "https://www.ted.com/services/v1/oembed.json?url={url}"),
    (("flickr.com", "flic.kr"), "https://www.flickr.com/services/oembed?format=json&url={url}"),
)

#: `<link rel="alternate" type="application/json+oembed" href="…">`. Attribute
#: order is not fixed in the wild, so the tag is matched first and the href
#: pulled out of it separately rather than assuming a layout.
_OEMBED_LINK = re.compile(r"<link[^>]+application/json\+oembed[^>]*>", re.I)
_HREF = re.compile(r'href=["\']([^"\']+)["\']', re.I)

#: OpenGraph and Twitter card meta tags, in both attribute orders. Two patterns
#: rather than one clever one: `<meta property=… content=…>` and
#: `<meta content=… property=…>` both occur, and a single regex that tries to
#: accept either is the kind that also accepts things that are not meta tags.
_META_FWD = re.compile(
    r'<meta[^>]+(?:property|name)=["\']((?:og|twitter|article):[a-zA-Z:_-]+)["\'][^>]+content=["\']([^"\']*)["\']',
    re.I,
)
_META_REV = re.compile(
    r'<meta[^>]+content=["\']([^"\']*)["\'][^>]+(?:property|name)=["\']((?:og|twitter|article):[a-zA-Z:_-]+)["\']',
    re.I,
)
_TITLE_TAG = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
_DESCRIPTION_META = re.compile(
    r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']*)["\']', re.I
)

#: An og:image this small is a favicon or a tracking pixel, not a hero frame.
#: Not enforced here (we have not fetched it yet) but declared so the caller
#: can reject on the real dimensions.
MIN_HERO_PIXELS = 200


@dataclass
class LinkMeta:
    """How a URL describes itself."""

    url: str
    title: str = ""
    author: str = ""
    description: str = ""
    #: Absolute, and verified to be http(s) — a `data:` or protocol-relative
    #: og:image would otherwise be handed straight to the fetcher.
    image_url: Optional[str] = None
    #: `video`, `article`, `music.song`, … from og:type. Advisory only: the
    #: STUDY kind is decided by what we can actually retrieve, never by what a
    #: page claims about itself.
    og_type: str = ""
    site_name: str = ""
    provider: str = ""
    #: Seconds, when the page publishes it (og:video:duration). Rare but exact.
    duration_s: Optional[int] = None
    #: Which mechanisms actually contributed, for the receipt.
    sources: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """Enough to build a study from.

        A title alone is not. Every HTML page has a `<title>`, including error
        pages and consent walls, so treating one as sufficient would turn "we
        could not read this" into a confident study of the words "Access
        Denied".
        """
        return bool(self.image_url) or len(self.description) >= 80

    def as_brief(self) -> str:
        """The metadata as prose, for the twins to read as context."""
        bits = []
        if self.title:
            bits.append(self.title)
        if self.author:
            bits.append(f"by {self.author}")
        if self.site_name:
            bits.append(f"on {self.site_name}")
        head = " ".join(bits)
        return f"{head}\n\n{self.description}".strip() if self.description else head


def _clean(text: str) -> str:
    """Unescape entities and collapse whitespace. Markup never reaches a twin."""
    text = re.sub(r"<[^>]+>", " ", text or "")
    return " ".join(_html.unescape(text).split())


def _absolute_image(candidate: str, base: str) -> Optional[str]:
    """An og:image as an absolute http(s) URL, or None.

    Relative and protocol-relative values are both common and both would be
    passed to the fetcher as-is otherwise — one failing to parse, the other
    silently inheriting our scheme against a host that never agreed to it.
    `data:` and `javascript:` are refused outright rather than being handed to
    a fetcher whose scheme allowlist would refuse them anyway: failing here
    produces a clear absence instead of a confusing fetch error.
    """
    candidate = (candidate or "").strip()
    if not candidate:
        return None
    if candidate.startswith("//"):
        candidate = "https:" + candidate
    elif not candidate.lower().startswith(("http://", "https://")):
        if candidate.lower().startswith(("data:", "javascript:", "blob:")):
            return None
        candidate = urljoin(base, candidate)
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    return candidate


def extract_meta(html: str, url: str) -> LinkMeta:
    """Scrape OpenGraph, Twitter cards and the title tag. Never raises.

    Pure and offline — the markup is already in memory from the fetch that
    established this was a page, so this costs nothing and runs on every HTML
    response whether or not it is needed.
    """
    meta = LinkMeta(url=url)
    tags: dict[str, str] = {}
    for key, value in _META_FWD.findall(html or ""):
        tags.setdefault(key.lower(), value)
    for value, key in _META_REV.findall(html or ""):
        tags.setdefault(key.lower(), value)

    def pick(*keys: str) -> str:
        for k in keys:
            if tags.get(k):
                return _clean(tags[k])
        return ""

    meta.title = pick("og:title", "twitter:title")
    meta.description = pick("og:description", "twitter:description")
    meta.site_name = pick("og:site_name")
    meta.og_type = pick("og:type").lower()
    meta.author = pick("article:author", "twitter:creator")
    if meta.title or meta.description:
        meta.sources.append("opengraph")

    raw_image = tags.get("og:image") or tags.get("og:image:url") or tags.get("twitter:image")
    if raw_image:
        meta.image_url = _absolute_image(_html.unescape(raw_image), url)

    raw_duration = tags.get("og:video:duration") or tags.get("music:duration")
    if raw_duration and raw_duration.strip().isdigit():
        meta.duration_s = int(raw_duration.strip())

    # The floor. Only consulted when OpenGraph gave nothing, because a
    # `<title>` is decoration on a page that has proper cards.
    if not meta.title:
        found = _TITLE_TAG.search(html or "")
        if found:
            meta.title = _clean(found.group(1))[:300]
            if meta.title:
                meta.sources.append("title tag")
    if not meta.description:
        found = _DESCRIPTION_META.search(html or "")
        if found:
            meta.description = _clean(found.group(1))
            if meta.description:
                meta.sources.append("meta description")

    return meta


def oembed_endpoint(url: str, html: str = "") -> Optional[str]:
    """Where to ask this host for structured metadata, if anywhere.

    Discovery from the markup wins over the registry: a host that advertises
    its own endpoint knows better than a table compiled here, and the
    advertised URL already carries the correct parameters.
    """
    found = _OEMBED_LINK.search(html or "")
    if found:
        href = _HREF.search(found.group(0))
        if href:
            absolute = _absolute_image(_html.unescape(href.group(1)), url)
            if absolute:
                return absolute

    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    for hosts, template in OEMBED_ENDPOINTS:
        if any(host == h or host.endswith("." + h) for h in hosts):
            return template.format(url=quote(url, safe=""))
    return None


async def fetch_oembed(endpoint: str, base_url: str) -> Optional[LinkMeta]:
    """Ask an oEmbed endpoint what this is. None on any failure.

    Best-effort by design. oEmbed is a courtesy a host extends, not a contract
    it owes us, and a host that has taken its endpoint down should cost this
    call a round trip and nothing else — OpenGraph is already scraped and
    waiting.
    """
    try:
        doc, _final = await fetch_json(endpoint)
    except (UnsafeURL, FetchFailed):
        return None

    meta = LinkMeta(url=base_url, sources=["oembed"])
    meta.title = _clean(str(doc.get("title") or ""))
    meta.author = _clean(str(doc.get("author_name") or ""))
    meta.site_name = _clean(str(doc.get("provider_name") or ""))
    meta.description = _clean(str(doc.get("description") or ""))
    meta.og_type = str(doc.get("type") or "").lower()
    meta.image_url = _absolute_image(str(doc.get("thumbnail_url") or ""), base_url)
    raw = doc.get("duration")
    if isinstance(raw, (int, float)) and raw > 0:
        meta.duration_s = int(raw)
    return meta if (meta.title or meta.image_url) else None


def merge(primary: Optional[LinkMeta], secondary: Optional[LinkMeta]) -> Optional[LinkMeta]:
    """Field-by-field union, `primary` winning wherever it has a value.

    Not "whichever object is more complete". The two sources are good at
    different things and the useful result is per-field: oEmbed has the exact
    title and author because the host authored them, while OpenGraph usually
    carries the longer description and the larger hero image. Choosing one
    document wholesale discards the better half of the other.
    """
    if primary is None:
        return secondary
    if secondary is None:
        return primary
    for name in ("title", "author", "description", "site_name", "og_type"):
        if not getattr(primary, name):
            setattr(primary, name, getattr(secondary, name))
    if not primary.image_url:
        primary.image_url = secondary.image_url
    if primary.duration_s is None:
        primary.duration_s = secondary.duration_s
    for source in secondary.sources:
        if source not in primary.sources:
            primary.sources.append(source)
    return primary


async def describe_link(url: str, html: str) -> Optional[LinkMeta]:
    """Everything this URL says about itself, from every mechanism available."""
    scraped = extract_meta(html, url)
    endpoint = oembed_endpoint(url, html)
    structured = await fetch_oembed(endpoint, url) if endpoint else None
    combined = merge(structured, scraped)
    return combined if combined and (combined.title or combined.image_url) else None
