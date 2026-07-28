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

SECURITY: nothing here fetches a URL. Server-side fetching of caller-supplied
URLs is an SSRF primitive — it would let anyone with an API key read the
metadata service, the private backend, or anything else reachable from inside
the network. Callers supply bytes or text; the browser can fetch its own.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import re
from enum import Enum
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

_TAG_BLOCK = re.compile(
    r"<(h[1-6]|section|article|header|footer|main|div|p|li)\b[^>]*>(.*?)</\1>",
    re.IGNORECASE | re.DOTALL,
)
_TAGS = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_DROPPED = re.compile(r"<(script|style|noscript|svg)\b.*?</\1>", re.IGNORECASE | re.DOTALL)


def _beats_from_page(asset: ContentAsset) -> BeatSequence:
    """A landing page read in DOM order — which is the order a scroller meets it.

    Structural rather than model-driven on purpose: the section order of a page
    is a fact about the document, and asking a model to infer it would
    introduce variance into something that is not uncertain.
    """
    if not asset.text:
        raise UnsupportedAsset("a page asset needs its HTML or extracted text")
    html = _DROPPED.sub(" ", asset.text)
    blocks: list[str] = []
    for _tag, inner in _TAG_BLOCK.findall(html):
        text = _WS.sub(" ", _TAGS.sub(" ", inner)).strip()
        # Drop chrome and fragments: a nav link is not a beat.
        if len(text) >= 40 and text not in blocks:
            blocks.append(text)
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


async def _beats_from_text(asset: ContentAsset) -> BeatSequence:
    if not asset.text:
        raise UnsupportedAsset("a text asset needs text")
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
