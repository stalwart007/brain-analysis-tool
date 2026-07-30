"""Turn ANY digital content into the beat sequence the neuro study consumes.

The study downstream is modality-agnostic: it wants an ordered list of beats
and it produces an appraisal tensor (twins × beats × 9 dimensions) over them.
What was missing was everything above that line — `segment_content` took a
`str`, so "content" meant "text I could paste".

This module is the adapter layer. Each modality becomes a `BeatSequence`, and
crucially each carries the AXIS it lives on, because that is what decides which
statistics downstream are meaningful:

    TEMPORAL    video, audio, animation — beats have real timestamps and a
                viewer experiences them in order at a rate they do not control.
                Peak-end, change points as moments, habituation, retention: all
                defined.

    SEQUENTIAL  article, deck, landing page — beats have an order but no clock.
                The reader sets the pace and can stop. Peak-end and retention
                are defined; anything phrased in seconds is not.

    SPATIAL     a static image, poster, banner — the "beats" are attention
                regions taken in a predicted scan order, but they are
                SIMULTANEOUSLY PRESENT. There is no ending to weight, no moment
                at which a change occurs, and no abandonment. Computing
                peak-END memory here would be reporting the valence of whatever
                region the model happened to list last.

That distinction is the whole reason this file exists rather than a `str`
conversion. A number computed on the wrong axis is not slightly wrong; it is a
category error wearing a decimal point.

SECURITY: nothing here fetches a URL, and `ContentAsset` has no url field.
Server-side fetching of caller-supplied URLs is an SSRF primitive — it would
let anyone with an API key read the metadata service, the private backend, or
anything else reachable from inside the network.

A landing page is the one case the browser cannot fetch for itself, because
cross-origin reads are blocked. That goes through `app.fetching`, which refuses
any URL not resolving to a publicly routable address, and returns extracted
prose rather than the page. It is a separate module and a separate endpoint on
purpose: there is exactly one place in this service that opens an outbound
connection to an address a caller chose, and it should stay countable.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import re
from enum import Enum
from html import unescape
from html.parser import HTMLParser
from typing import Literal, Optional

import openai

from .config import PROFILER_MODEL, TWIN_MODEL
from .oai import Refusal, async_client, parse_completion, response_format_for
from .schemas import (
    ContentAsset,
    ContentSegments,
    ImageHierarchy,
    VideoFrameRead,
)


class BeatAxis(str, Enum):
    TEMPORAL = "temporal"
    SEQUENTIAL = "sequential"
    SPATIAL = "spatial"


#: Which statistics each axis can support. Consumed by neuro.aggregate_content_study,
#: which reports an explicit reason for anything it withholds rather than
#: silently omitting it — an absent number and an inapplicable one look
#: identical in a UI, and only one of them is a finding.
AXIS_SUPPORTS: dict[BeatAxis, set[str]] = {
    BeatAxis.TEMPORAL: {"peak_end", "change_points", "retention", "trajectory", "isc"},
    BeatAxis.SEQUENTIAL: {"peak_end", "change_points", "retention", "trajectory", "isc"},
    # ISC survives on a spatial axis: it measures whether twins agree about
    # which regions pull attention, which is a real question about the image.
    # Everything that presumes an ordered experience does not.
    BeatAxis.SPATIAL: {"isc"},
}

AXIS_REASON: dict[str, str] = {
    "peak_end": (
        "peak-end memory needs an experience with an ending; the regions of a "
        "static image are present simultaneously, so the 'last' beat is an "
        "artefact of listing order, not something the viewer ended on"
    ),
    "change_points": (
        "a change point is a MOMENT at which the response shifts; the regions "
        "of a static image are not ordered in time, so there is no moment"
    ),
    "retention": (
        "abandonment is not defined for a static image — there is no next beat "
        "to fail to reach"
    ),
    "trajectory": (
        "an emotional trajectory is movement THROUGH a sequence; simultaneous "
        "regions have no path between them"
    ),
}


MAX_BEATS = 10
MIN_BEATS = 2
#: One vision call per frame, so this is a direct cost multiplier as well as a
#: latency one. Ten keyframes over a 30-second spot is roughly one every three
#: seconds, which is finer than the beat structure the study can resolve.
MAX_FRAMES = 10


class BeatSequence:
    """Ordered beats plus the axis they live on."""

    def __init__(
        self,
        beats: list[str],
        axis: BeatAxis,
        *,
        source: str,
        timestamps_ms: Optional[list[int]] = None,
        notes: Optional[str] = None,
    ) -> None:
        self.beats = beats
        self.axis = axis
        #: How the beats were derived — "vision", "llm", "structure",
        #: "paragraph_fallback". Persisted into the result, because a study run
        #: on mechanically split text is a different claim from one run on
        #: model-identified narrative beats and the reader must be able to tell.
        self.source = source
        self.timestamps_ms = timestamps_ms
        self.notes = notes

    def supports(self, statistic: str) -> bool:
        return statistic in AXIS_SUPPORTS[self.axis]

    def to_dict(self) -> dict:
        return {
            "beats": self.beats,
            "axis": self.axis.value,
            "source": self.source,
            "timestamps_ms": self.timestamps_ms,
            "notes": self.notes,
            "withheld": {
                stat: AXIS_REASON[stat]
                for stat in AXIS_REASON
                if not self.supports(stat)
            },
        }


class UnsupportedAsset(ValueError):
    """The asset cannot be turned into beats, and saying so beats guessing."""


# ── image ────────────────────────────────────────────────────────────────

_HIERARCHY_SYSTEM = """You are a visual-attention analyst. Given an image,
identify the 3-6 distinct elements a first-time viewer's eye lands on, IN THE
ORDER a typical viewer would encounter them — driven by visual salience (size,
contrast, faces, motion cues, colour pop) and reading order, not by importance
to the advertiser.

For each, give a short label plus one line describing what it shows and why it
pulls the eye. Do not invent elements that are not visible. If the image is
mostly empty or a single flat element, return fewer beats rather than padding."""


def _data_url(asset_b64: str, media_type: str) -> str:
    """Validate then wrap as a data URL.

    Decoded first because an invalid base64 blob would otherwise be discovered
    by the vision API rather than by us, turning a caller's typo into an opaque
    upstream error and a billed request.
    """
    try:
        base64.b64decode(asset_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise UnsupportedAsset("image data is not valid base64") from exc
    return f"data:{media_type};base64,{asset_b64}"


async def _beats_from_image(asset: ContentAsset) -> BeatSequence:
    if not asset.image_b64:
        raise UnsupportedAsset("an image asset needs image_b64")
    url = _data_url(asset.image_b64, asset.media_type or "image/png")
    completion = await async_client().chat.completions.create(
        # The profiler model, not the twin model: this is a single
        # judgement-heavy call whose output every twin then reasons over, so a
        # mistake here propagates to the whole run. Exactly the split the
        # scaling plan describes — low-volume and judgement-heavy stays on the
        # stronger model.
        model=PROFILER_MODEL,
        messages=[
            {"role": "system", "content": _HIERARCHY_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": asset.brief or "Analyse this image."},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            },
        ],
        temperature=0.2,
        response_format=response_format_for(ImageHierarchy),
    )
    parsed = parse_completion(completion, ImageHierarchy)
    beats = [f"{e.label} — {e.description}" for e in parsed.elements][:MAX_BEATS]
    if len(beats) < MIN_BEATS:
        raise UnsupportedAsset(
            "the image resolved to fewer than two distinct attention regions; "
            "there is no scan path to study"
        )
    return BeatSequence(
        beats,
        BeatAxis.SPATIAL,
        source="vision",
        notes=(
            "Beats are attention regions in predicted scan order. They are "
            "present simultaneously, so time-dependent statistics are withheld."
        ),
    )


# ── video ────────────────────────────────────────────────────────────────

_FRAME_SYSTEM = """You are describing one keyframe from a video, for an
audience-response study. In one or two sentences: what is on screen, what is
happening, and the emotional register. Describe only what is visible. Do not
speculate about what came before or after."""


async def _describe_frame(
    frame_b64: str, media_type: str, t_ms: int, semaphore: asyncio.Semaphore
) -> Optional[str]:
    url = _data_url(frame_b64, media_type)
    async with semaphore:
        try:
            completion = await async_client().chat.completions.create(
                model=TWIN_MODEL,
                messages=[
                    {"role": "system", "content": _FRAME_SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Keyframe at {t_ms / 1000:.1f}s."},
                            {"type": "image_url", "image_url": {"url": url}},
                        ],
                    },
                ],
                temperature=0.2,
                response_format=response_format_for(VideoFrameRead),
            )
            return parse_completion(completion, VideoFrameRead).description
        except (openai.OpenAIError, Refusal, RuntimeError, ValueError):
            # One unreadable frame must not lose the study. The beat is dropped
            # and the caller sees a shorter sequence, which is honest; failing
            # the run would throw away every frame that did work.
            return None


async def _beats_from_video(asset: ContentAsset) -> BeatSequence:
    """Keyframes plus optional transcript.

    The client extracts frames. That is deliberate rather than a shortcut:
    decoding caller-supplied video server-side means running a media decoder on
    untrusted input, which is a large and historically hostile attack surface,
    and it would put ffmpeg in a container that today has no native
    dependencies at all.
    """
    if not asset.frames:
        raise UnsupportedAsset(
            "a video asset needs keyframes — extract them client-side and send "
            "them with timestamps"
        )
    frames = sorted(asset.frames, key=lambda f: f.t_ms)[:MAX_FRAMES]
    semaphore = asyncio.Semaphore(4)
    described = await asyncio.gather(
        *(
            _describe_frame(f.image_b64, f.media_type or "image/jpeg", f.t_ms, semaphore)
            for f in frames
        )
    )

    # Transcript lines are aligned to the nearest keyframe so a beat carries
    # both what was on screen and what was said. Audio and picture are one
    # experience; splitting them into separate beats would double-count the
    # timeline and halve the apparent pace.
    lines = asset.transcript_cues or []
    beats: list[str] = []
    stamps: list[int] = []
    for frame, description in zip(frames, described):
        if description is None:
            continue
        spoken = [c.text for c in lines if abs(c.t_ms - frame.t_ms) <= 2500]
        seconds = frame.t_ms / 1000
        beat = f"[{seconds:.1f}s] {description}"
        if spoken:
            beat += f' — heard: "{" ".join(spoken)[:160]}"'
        beats.append(beat)
        stamps.append(frame.t_ms)

    if len(beats) < MIN_BEATS:
        raise UnsupportedAsset(
            "fewer than two keyframes could be read; supply more frames or "
            "check the image encoding"
        )
    return BeatSequence(
        beats,
        BeatAxis.TEMPORAL,
        source="vision",
        timestamps_ms=stamps,
        notes=f"{len(beats)} keyframes described"
        + (f", {len(lines)} transcript cues aligned" if lines else ", no transcript"),
    )


# ── audio ────────────────────────────────────────────────────────────────


async def _beats_from_audio(asset: ContentAsset) -> BeatSequence:
    """A transcript with cue timings is a temporal axis; without them it is
    only a sequence, and saying which is the difference between a retention
    curve in seconds and one in beats."""
    cues = asset.transcript_cues or []
    if cues:
        cues = sorted(cues, key=lambda c: c.t_ms)
        # Group into MAX_BEATS windows so the study resolves structure rather
        # than transcribing: 60 cues is not 60 beats, it is one conversation.
        per = max(1, len(cues) // MAX_BEATS + (1 if len(cues) % MAX_BEATS else 0))
        beats, stamps = [], []
        for i in range(0, len(cues), per):
            chunk = cues[i : i + per]
            t = chunk[0].t_ms
            beats.append(f"[{t / 1000:.1f}s] " + " ".join(c.text for c in chunk)[:400])
            stamps.append(t)
        if len(beats) >= MIN_BEATS:
            return BeatSequence(
                beats,
                BeatAxis.TEMPORAL,
                source="structure",
                timestamps_ms=stamps,
                notes=f"{len(cues)} transcript cues grouped into {len(beats)} beats",
            )
    if asset.text:
        segs = await _llm_segments(asset.text, "audio transcript")
        return BeatSequence(
            segs,
            BeatAxis.SEQUENTIAL,
            source="llm",
            notes=(
                "no cue timings supplied, so this is an ordered transcript "
                "rather than a timeline — retention is reported per beat, not "
                "per second"
            ),
        )
    raise UnsupportedAsset("an audio asset needs a transcript, with or without timings")


# ── page / document / text ───────────────────────────────────────────────

#: `header` and `footer` used to be in this list, and `div` still is. Measured
#: against real sites, that produced beats like "Products Solutions Developers
#: Resources Pricing Sign in Start now Contact sales" — Stripe's navigation bar,
#: read as though it were the opening moment of the page. Linear and Vercel gave
#: the same. A study over those beats reports how an audience feels about a
#: MENU, with confidence intervals, which is worse than reporting nothing.
#:
#: `div` stays because a great many pages wrap real copy in nothing else, but
#: chrome is now stripped first and blocks must read as prose (`_is_prose`).
_TAG_BLOCK = re.compile(
    r"<(h[1-6]|section|article|main|div|p|li)\b[^>]*>(.*?)</\1>",
    re.IGNORECASE | re.DOTALL,
)

#: Site furniture — but only the parts that are UNAMBIGUOUSLY furniture.
#:
#: `header` and `footer` were in this list and had to come out. On a real site
#: `<header>` is usually the nav bar, but it is also the correct HTML5 element
#: for a hero — headline, subhead, call to action — which is the single most
#: important beat on a landing page. Stripping it wholesale threw away the hero
#: to remove the menu, and a page study missing its opening is not a page study.
#: `<footer>` similarly carries real social proof.
#:
#: So only `nav`, `menu`, `dialog` and the matching ARIA roles are removed
#: structurally, and link-label soup is caught by `_is_prose` on its content
#: instead of by its container. Judging a block by what it SAYS survives a
#: hero in a `<header>`; judging it by its wrapper does not.
_CHROME = re.compile(
    r"""<(nav|menu|dialog)\b.*?</\1>
      |<(\w+)\b[^>]*?role\s*=\s*["'](?:navigation|menu|menubar|dialog|search)["'].*?</\2>""",
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)

#: Does this block read like something a person wrote to be read, or like a
#: list of link labels? Navigation survives tag-stripping as a run of short
#: Title Case fragments with no sentence in it, so that is what this tests.
def _is_prose(text: str, inner_html: str = "") -> bool:
    words = text.split()
    if len(words) < 8:
        return False

    # What is left once the links are removed. A menu is links and nothing
    # else; a hero is a headline that happens to sit beside two buttons.
    #
    # A link-DENSITY ratio was tried first and was wrong: at 0.6 it correctly
    # rejected Stripe's secondary nav and then also rejected Stripe's hero,
    # because "Financial infrastructure to grow your revenue" shares a wrapper
    # with "Start now" and "Contact sales". The ratio punishes a block for
    # having calls to action, which every landing page hero has by design.
    # Asking whether real copy exists OUTSIDE the links separates the two
    # cleanly and needs no threshold tuning.
    if inner_html:
        outside = _WS.sub(" ", _TAGS.sub(" ", _ANCHORS.sub(" ", inner_html))).strip()
        if len(outside) < 40:
            return False

    # A sentence mark is the strongest single signal, but a headline legitimately
    # has none — so a long block with normal lower-case flow also qualifies.
    has_sentence = any(mark in text for mark in (". ", "! ", "? ", ", "))
    lowercase = sum(1 for w in words if w[:1].islower())
    # Nav items are overwhelmingly capitalised; prose is mostly not.
    return has_sentence or lowercase / len(words) > 0.5
_TAGS = re.compile(r"<[^>]+>")
#: Whole anchor elements, removed to see what copy a block has of its own.
_ANCHORS = re.compile(r"<a\b[^>]*>.*?</a>", re.IGNORECASE | re.DOTALL)
_WS = re.compile(r"\s+")
_DROPPED = re.compile(
    r"<(script|style|noscript|svg|template)\b.*?</\1>", re.IGNORECASE | re.DOTALL
)

#: Content the visitor never sees. Dropping it is an ANALYSIS correctness fix
#: before it is a security one: a closed nav menu, a cookie modal, tab panels
#: behind other tabs and screen-reader-only text are all in the markup on
#: arrival, and counting them as beats means reporting an audience reaction to
#: something nobody looked at.
#:
#: It also happens to remove the classic prompt-injection vector, since text
#: planted for a scraper is planted where a human will not see it. That is a
#: side effect, not the defence — the real bound is that twin output is a
#: structured appraisal object per beat, so hostile copy can skew the numbers
#: it was measured on but cannot change what this service does.
#:
#: Regex over HTML is approximate and nesting will defeat it in places. It is
#: the same approximation the extractor above already makes, and the failure
#: mode is a beat that should have been dropped, not a crash.
_HIDDEN = re.compile(
    r"""<(div|section|p|li|span|aside|nav|header|footer)\b[^>]*?
        (?:\shidden(?:\s|=|>)|aria-hidden\s*=\s*["']true["']
          |style\s*=\s*["'][^"']*?(?:display\s*:\s*none|visibility\s*:\s*hidden)
        )[^>]*>.*?</\1>""",
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)


#: Elements whose closing tag ends a readable block of copy.
_BLOCK_TAGS = frozenset(
    {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "section", "article",
     "main", "div", "header", "footer", "blockquote", "figcaption", "td", "dd"}
)
#: Never rendered, never read.
_SKIP_TAGS = frozenset({"script", "style", "noscript", "svg", "template", "head"})
#: Site furniture, by element and by ARIA role.
_CHROME_TAGS = frozenset({"nav", "menu", "dialog"})
_CHROME_ROLES = frozenset({"navigation", "menu", "menubar", "dialog", "search", "banner"})
#: Void elements never get a closing tag; HTMLParser reports them via
#: `handle_startendtag` only when they are self-closed in the source, so the
#: stack has to know not to wait for `</img>`.
_VOID_TAGS = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
     "meta", "param", "source", "track", "wbr"}
)


class _BlockReader(HTMLParser):
    """Extract readable copy blocks from HTML with a real parser.

    WHY THIS REPLACED A REGEX. The previous extractor matched
    `<(div|section|…)>(.*?)</\\1>` non-greedily, which is wrong on any nested
    element of the same name: `<div><div>copy</div></div>` closes the OUTER
    match at the INNER `</div>`, so the block boundaries a modern page produces
    are essentially arbitrary. Measured against real sites, stripe.com yielded
    ZERO usable sections out of 11,162 characters of visible prose — the
    landing-page path simply did not work on one of the most-copied marketing
    pages on the web, and the failure surfaced as "this page builds its copy in
    the browser", which was not true and sent people to fix nothing.

    `html.parser` is stdlib, so this costs no dependency and the container
    keeps its no-native-code property.

    THE RULE for what becomes a block: a block-level element emits its text
    only if no block-level DESCENDANT already emitted. That is what keeps a
    `<section>` wrapping five `<p>`s from emitting the same copy a sixth time,
    while still letting a `<div>` that holds copy directly — which is most of
    the modern web — emit at all.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        # (tag, parts, non_link_parts, child_emitted)
        self._stack: list[tuple[str, list[str], list[str], list[bool]]] = []
        self._skip_depth = 0
        self._drop_depth = 0
        self._anchor_depth = 0

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _is_hidden(attrs: list[tuple[str, Optional[str]]]) -> bool:
        """Content present in the markup but never shown.

        Dropping it is an ANALYSIS correctness fix before it is a security one:
        a closed nav menu, a cookie modal, tab panels behind other tabs and
        screen-reader-only text are all in the document on arrival, and counting
        them as beats reports an audience reaction to something nobody looked at.
        """
        for key, value in attrs:
            key = key.lower()
            value = (value or "").lower()
            if key == "hidden":
                return True
            if key == "aria-hidden" and value == "true":
                return True
            if key == "style" and ("display:none" in value.replace(" ", "")
                                   or "visibility:hidden" in value.replace(" ", "")):
                return True
        return False

    @staticmethod
    def _is_chrome(tag: str, attrs: list[tuple[str, Optional[str]]]) -> bool:
        if tag in _CHROME_TAGS:
            return True
        for key, value in attrs:
            if key.lower() == "role" and (value or "").lower() in _CHROME_ROLES:
                return True
        return False

    # ── parser callbacks ──────────────────────────────────────────────────

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in _VOID_TAGS:
            return
        if self._skip_depth or tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._drop_depth or self._is_chrome(tag, attrs) or self._is_hidden(attrs):
            self._drop_depth += 1
            return
        if tag == "a":
            self._anchor_depth += 1
        if tag in _BLOCK_TAGS:
            self._stack.append((tag, [], [], [False]))

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _VOID_TAGS:
            return
        if self._skip_depth:
            self._skip_depth -= 1
            return
        if self._drop_depth:
            self._drop_depth -= 1
            return
        if tag == "a":
            self._anchor_depth = max(0, self._anchor_depth - 1)
        if tag not in _BLOCK_TAGS:
            return
        # Unbalanced markup is the norm, not the exception. Unwind to the
        # nearest matching frame rather than assuming the document is
        # well-formed; a stray `</div>` must not silently close a `<section>`.
        for depth in range(len(self._stack) - 1, -1, -1):
            if self._stack[depth][0] == tag:
                frames = self._stack[len(self._stack) : depth - 1 if depth else None]
                break
        else:
            return
        while len(self._stack) > depth:
            name, parts, own, child_emitted = self._stack.pop()
            text = _WS.sub(" ", "".join(parts)).strip()
            own_text = _WS.sub(" ", "".join(own)).strip()
            emitted = False
            if (
                not child_emitted[0]
                and _qualifies(name, text, own_text)
                and text not in self.blocks
            ):
                self.blocks.append(text)
                emitted = True
            if self._stack:
                parent = self._stack[-1]
                parent[1].append(" " + text)
                parent[2].append(" " + own_text)
                parent[3][0] = parent[3][0] or emitted or child_emitted[0]

    def handle_data(self, data: str) -> None:
        if self._skip_depth or self._drop_depth or not self._stack:
            return
        frame = self._stack[-1]
        frame[1].append(data)
        # Text outside an anchor is the page's own copy; text inside one is a
        # link label. Keeping them apart is what distinguishes a hero (a
        # headline that happens to sit beside two buttons) from a menu.
        if not self._anchor_depth:
            frame[2].append(data)

    def close_all(self) -> list[str]:
        """Flush frames left open by unterminated markup."""
        while self._stack:
            name, parts, own, child_emitted = self._stack.pop()
            text = _WS.sub(" ", "".join(parts)).strip()
            own_text = _WS.sub(" ", "".join(own)).strip()
            if (
                not child_emitted[0]
                and _qualifies(name, text, own_text)
                and text not in self.blocks
            ):
                self.blocks.append(text)
        return self.blocks


_HEADINGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})


def _qualifies(tag: str, text: str, own_text: str) -> bool:
    """Should this block become a beat?

    HEADINGS GET A LOWER BAR, and they have to. The prose heuristic below
    exists to tell copy from navigation, and its main instrument is length — a
    nav item is a short Title Case fragment. But a headline is ALSO short by
    design, and it is the single most important beat on a landing page:
    Stripe's "Financial infrastructure to grow your revenue" is six words and
    was being discarded for it, along with essentially every hero on the web.

    A heading does not need the heuristic, because the markup has already made
    the claim the heuristic is trying to infer: `<h1>` is not what anyone wraps
    a menu in. So headings qualify on length alone, and everything else still
    has to read like prose.
    """
    if tag in _HEADINGS:
        return len(text) >= 12 and len(text.split()) >= 3
    return len(text) >= 40 and _is_prose_text(text, own_text)


def _is_prose_text(text: str, own_text: str) -> bool:
    """Does this read like something a person wrote to be read?

    Navigation survives tag-stripping as a run of short Title Case fragments
    with no sentence in it, so that is what this tests.

    A link-DENSITY ratio was tried first and was wrong: at 0.6 it correctly
    rejected Stripe's secondary nav and then also rejected Stripe's hero,
    because "Financial infrastructure to grow your revenue" shares a wrapper
    with "Start now" and "Contact sales". The ratio punishes a block for having
    calls to action, which every landing page hero has by design. Asking
    whether real copy exists OUTSIDE the links separates the two cleanly and
    needs no threshold tuning.
    """
    words = text.split()
    if len(words) < 8:
        return False
    # A menu is links and nothing else; a hero is a headline beside buttons.
    if len(own_text) < 40:
        return False
    # A sentence mark is the strongest single signal, but a headline
    # legitimately has none — so a long block with normal lower-case flow also
    # qualifies.
    has_sentence = any(mark in text for mark in (". ", "! ", "? ", ", "))
    lowercase = sum(1 for w in words if w[:1].islower())
    return has_sentence or lowercase / len(words) > 0.5


def extract_blocks(html: str) -> list[str]:
    """Readable copy blocks from a page, in DOM order."""
    reader = _BlockReader()
    try:
        reader.feed(html)
    except Exception:  # noqa: BLE001 — malformed markup must not lose the page
        pass
    return reader.close_all()


def _beats_from_page(asset: ContentAsset) -> BeatSequence:
    """A landing page read in DOM order — which is the order a scroller meets it.

    Structural rather than model-driven on purpose: the section order of a page
    is a fact about the document, and asking a model to infer it would
    introduce variance into something that is not uncertain.
    """
    if not asset.text:
        raise UnsupportedAsset("a page asset needs its HTML or extracted text")
    blocks = extract_blocks(asset.text)
    if len(blocks) < MIN_BEATS:
        # Not HTML, or HTML with no prose — fall through to the text path
        # rather than fabricating structure.
        raise UnsupportedAsset(
            "could not find at least two content sections in this page; send "
            "the copy as text instead"
        )
    step = max(1, len(blocks) // MAX_BEATS + (1 if len(blocks) % MAX_BEATS else 0))
    grouped = [
        " ".join(blocks[i : i + step])[:400] for i in range(0, len(blocks), step)
    ]
    return BeatSequence(
        grouped[:MAX_BEATS],
        BeatAxis.SEQUENTIAL,
        source="structure",
        notes=f"{len(blocks)} page sections in DOM order, grouped into {len(grouped[:MAX_BEATS])} scroll beats",
    )


def _beats_from_document(asset: ContentAsset) -> BeatSequence:
    """Slides or pages, already separated by the client."""
    pages = [p.strip() for p in (asset.pages or []) if p.strip()]
    if len(pages) < MIN_BEATS:
        raise UnsupportedAsset("a document asset needs at least two non-empty pages")
    return BeatSequence(
        [p[:400] for p in pages[:MAX_BEATS]],
        BeatAxis.SEQUENTIAL,
        source="structure",
        notes=f"{len(pages)} pages/slides in document order",
    )


async def _llm_segments(content: str, content_type: str) -> list[str]:
    from .neuro import _SEGMENTER_SYSTEM  # single source of truth for the prompt

    completion = await async_client().chat.completions.create(
        model=TWIN_MODEL,
        messages=[
            {"role": "system", "content": _SEGMENTER_SYSTEM},
            {"role": "user", "content": f"Content type: {content_type}\n\n{content}"},
        ],
        temperature=0.2,
        response_format=response_format_for(ContentSegments),
    )
    segs = parse_completion(completion, ContentSegments).segments
    if not (MIN_BEATS <= len(segs) <= MAX_BEATS):
        raise UnsupportedAsset(
            f"the segmenter returned {len(segs)} beats, outside the "
            f"{MIN_BEATS}-{MAX_BEATS} range this study can interpret"
        )
    return segs


#: The floor for a text asset, in characters after stripping.
#:
#: The dashboard has enforced this since the panel was written (`assetReady`),
#: and the API did not — so the guard lived entirely in the client and any
#: direct caller walked straight past it. Measured against the deployed API:
#: `{"content": "hi"}` was accepted, segmented into beats, and dispatched to the
#: full twin swarm, which spends real OpenAI credit producing confident
#: statistics — synchrony, peak-end, a retention curve — about two characters.
#:
#: Twenty is the client's number, kept identical on purpose. A server floor that
#: disagreed with the button would either reject something the UI had just
#: enabled or accept something it had greyed out, and both read as a bug in
#: whichever half the reader is looking at.
MIN_TEXT_CHARS = 20


async def _beats_from_text(asset: ContentAsset) -> BeatSequence:
    text = (asset.text or "").strip()
    if not text:
        raise UnsupportedAsset("a text asset needs text")
    if len(text) < MIN_TEXT_CHARS:
        # Refused BEFORE the segmenter, which is the whole point: the segmenter
        # is an LLM call and the twins after it are many more, so the cheapest
        # place to decline is the first one.
        raise UnsupportedAsset(
            f"that is {len(text)} characters, and a study needs at least "
            f"{MIN_TEXT_CHARS} to have anything to read. Paste the script, the "
            "copy, or a link to the content."
        )
    try:
        segs = await _llm_segments(asset.text, asset.content_type or "text")
        return BeatSequence(segs, BeatAxis.SEQUENTIAL, source="llm")
    except (openai.OpenAIError, Refusal, RuntimeError, ValueError, UnsupportedAsset):
        # The old fallback silently truncated to `content[:400]` whenever the
        # text had no blank line — a single-paragraph ad, a headline, a
        # transcript — so the entire study ran on the first 400 characters and
        # one of the two "beats" could literally be the string "(end)". Split
        # on sentences across the WHOLE text, and record that the beats are
        # mechanical so the result cannot be mistaken for narrative structure.
        return BeatSequence(
            _sentence_beats(asset.text),
            BeatAxis.SEQUENTIAL,
            source="paragraph_fallback",
            notes=(
                "the segmenter was unavailable, so beats are a mechanical "
                "sentence split — they are not model-identified narrative beats"
            ),
        )


_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _sentence_beats(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE.split(text.strip()) if p.strip()]
    if len(parts) < MIN_BEATS:
        # Genuinely one sentence: split on length rather than claiming beats
        # that are not there, and cover the WHOLE string.
        mid = max(1, len(text) // 2)
        return [text[:mid].strip(), text[mid:].strip() or "(end)"]
    per = max(1, len(parts) // MAX_BEATS + (1 if len(parts) % MAX_BEATS else 0))
    return [" ".join(parts[i : i + per]) for i in range(0, len(parts), per)][:MAX_BEATS]


# ── entry point ──────────────────────────────────────────────────────────

Kind = Literal["text", "image", "video", "audio", "page", "document"]


def visible_text_length(html: str) -> int:
    """Roughly how much readable prose a page carries, after stripping.

    Used to tell two failures apart. Measured on the STRIPPED document, not the
    raw one: a modern marketing page is mostly inlined JavaScript, so counting
    characters before `_DROPPED` runs reports a JS-rendered shell as text-rich
    and produces exactly the wrong advice.
    """
    stripped = _CHROME.sub(" ", _HIDDEN.sub(" ", _DROPPED.sub(" ", html)))
    return len(_WS.sub(" ", _TAGS.sub(" ", stripped)).strip())


def preview_page(asset: ContentAsset) -> list[str]:
    """The sections a fetched page yields, without spending a model call.

    Shown to the researcher before the study runs so they can see the page was
    read the way they expect. `_beats_from_page` is purely structural — DOM
    order, no model — so this is exactly what the study will consume, not an
    approximation of it. Raises `UnsupportedAsset` when there is no prose to
    find, which is the single-page-app case worth naming explicitly.

    Returned in full. Truncating here for display would be a quiet data loss,
    because the caller submits these same strings back as the study body —
    blocks are already capped at 400 characters upstream. Shortening for the
    eye is the UI's job.
    """
    return _beats_from_page(asset).beats


# ── PDF / deck ───────────────────────────────────────────────────────────

#: A deck is beats; a hundred-page report is not, and asking a swarm to
#: experience one page by page would cost a hundred appraisal tensors to
#: measure something nobody reads that way. Truncation is reported rather than
#: silent, because "we studied the first 40 pages" and "we studied the
#: document" are different claims.
MAX_PDF_PAGES = 40


def pages_from_pdf(data: bytes) -> tuple[list[str], int]:
    """Extract per-page text from a PDF. Returns (pages, total_page_count).

    Page structure is kept rather than flattened to one blob: for a deck, a
    page IS a beat, and it is the only modality that arrives pre-segmented.
    Throwing that away and re-segmenting with a model would replace a fact
    about the document with an inference about it.

    A page with no extractable text is kept as an empty string and filtered by
    the caller, so page NUMBERS stay aligned with the document — an image-only
    slide silently shifting every subsequent page's index is worse than a gap.
    """
    from io import BytesIO

    import pypdf

    try:
        reader = pypdf.PdfReader(BytesIO(data))
        total = len(reader.pages)
        out: list[str] = []
        for page in reader.pages[:MAX_PDF_PAGES]:
            try:
                text = _WS.sub(" ", (page.extract_text() or "")).strip()
            except Exception:  # noqa: BLE001 — one bad page must not lose the file
                text = ""
            out.append(text)
        return out, total
    except Exception as exc:  # noqa: BLE001 — pypdf raises a wide variety
        raise UnsupportedAsset(
            f"that PDF could not be read ({type(exc).__name__}). It may be "
            "encrypted, or scanned images with no text layer — in which case "
            "export the slides as images and study them that way."
        ) from exc


#: Platforms that serve a player page rather than the media itself. Detected so
#: the refusal can name the actual problem: fetching youtube.com/watch returns
#: an HTML shell whose visible prose is the title and the comment policy, and
#: studying that as a "landing page" would produce a full appraisal tensor
#: about a page nobody came to read.
_PLAYER_HOSTS = (
    "youtube.com", "youtu.be", "vimeo.com", "tiktok.com", "instagram.com",
    "facebook.com", "twitter.com", "x.com", "linkedin.com", "loom.com",
)


def is_player_url(url: str) -> Optional[str]:
    """The host name if this is a known player page, else None."""
    from urllib.parse import urlparse

    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    for known in _PLAYER_HOSTS:
        if host == known or host.endswith("." + known):
            return known
    return None


async def extract_beats(asset: ContentAsset) -> BeatSequence:
    """Any supported asset → beats + the axis they live on.

    Raises UnsupportedAsset with a reason the caller can act on. Refusing is
    the correct behaviour: a study run on beats that do not represent the
    content is worse than no study, because it produces confident numbers about
    something the audience never saw.
    """
    if asset.kind == "image":
        return await _beats_from_image(asset)
    if asset.kind == "video":
        return await _beats_from_video(asset)
    if asset.kind == "audio":
        return await _beats_from_audio(asset)
    if asset.kind == "page":
        try:
            return _beats_from_page(asset)
        except UnsupportedAsset:
            if asset.text:
                return await _beats_from_text(asset)
            raise
    if asset.kind == "document":
        return _beats_from_document(asset)
    return await _beats_from_text(asset)
