"""Turn a YouTube link into a study asset, without decoding a video.

WHY THIS EXISTS. Pasting a YouTube link is how most people have video, and
this platform refused it outright. The refusal was not lazy — it was written
against a real observation, that fetching `youtube.com/watch` returns an HTML
shell whose extractable prose is the title and the comment policy, and that
studying THAT as a landing page would produce a complete appraisal tensor
about a page nobody came to read. Correct diagnosis. Wrong conclusion: it
assumed the shell was all a server could get, and stopped there.

WHAT A SERVER CAN ACTUALLY GET, all of it public and unauthenticated, all of it
measured before this file was written:

  · The InnerTube player endpoint — the same API the YouTube apps call —
    returns title, channel, duration, the full description, the caption track
    list and the storyboard specification. The WEB client is refused from a
    datacentre address (`UNPLAYABLE`); the ANDROID and IOS clients are not,
    which is why those are the ones tried.
  · Storyboard spritesheets from i.ytimg.com: real frames sampled every two
    seconds across the whole video, packed into grids. These are the images
    YouTube itself shows when you scrub the progress bar. They fetch clean.
  · Caption tracks, when the video has them and the origin is willing. This is
    the one piece that is genuinely unreliable — `timedtext` rate-limits hard
    and answers 429 to a burst — so it is best-effort and its absence degrades
    the study rather than failing it.

WHAT THIS DELIBERATELY DOES NOT DO. It does not touch `streamingData` or try to
resolve a media URL. Those carry throttling parameters that have to be
descrambled by running YouTube's own player JavaScript, which is both fragile
and a different kind of act from reading a public preview asset. Storyboards
and captions are what the player surfaces to any viewer; that is the line.

NO DECODER RUNS HERE. The spritesheets are not cropped server-side — cropping a
WebP means running an image decoder over bytes a third party supplied, and this
container has no native dependencies by requirement. The sheet geometry is
computed here and the BROWSER does the cropping on a canvas, which is exactly
the boundary the uploaded-video path already respects.

EVERY REQUEST GOES THROUGH `fetching`. Not one socket is opened in this file.
YouTube is a public host today and could be a redirect to somewhere private
tomorrow, and a second transport is how an SSRF guard ends up with a hole in it.
"""

from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from html import unescape
from typing import Optional
from urllib.parse import parse_qs, urlparse

from .config import YOUTUBE_PROXY
from .fetching import (
    FetchFailed,
    UnsafeURL,
    allow_proxy_url,
    fetch_json,
    fetch_text,
    proxied_fetch_json,
)

#: The public web client key. This is not a secret and never was — it ships in
#: the JavaScript of every youtube.com page and identifies the API surface, not
#: the caller. Hardcoding it is what the YouTube apps do.
_INNERTUBE_KEY = "AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8"
_PLAYER_ENDPOINT = f"https://www.youtube.com/youtubei/v1/player?key={_INNERTUBE_KEY}"

#: The one destination the egress proxy may carry. Registered here, enforced in
#: `fetching`, and deliberately the only entry: everything else this module
#: fetches — storyboards, captions, stills — is measured to work fine from a
#: datacentre address, so proxying it would spend bandwidth to solve a problem
#: that does not exist.
allow_proxy_url(_PLAYER_ENDPOINT)

#: Client identities to try, in order, with the User-Agent each one must carry.
#:
#: Order is measured, not guessed. Against a residential address: ANDROID → OK,
#: IOS → OK, WEB → `UNPLAYABLE: Video unavailable`, TVHTML5 → `ERROR: YouTube is
#: no longer supported in this application`. The WEB client is the one that
#: looks most legitimate and the one that fails, which is worth writing down
#: because it is the obvious first thing to reach for.
_CLIENTS: tuple[tuple[str, dict, str], ...] = (
    (
        "ANDROID",
        {
            "clientName": "ANDROID",
            "clientVersion": "20.10.38",
            "androidSdkVersion": 30,
            "hl": "en",
            "gl": "US",
        },
        "com.google.android.youtube/20.10.38 (Linux; U; Android 11) gzip",
    ),
    (
        "IOS",
        {
            "clientName": "IOS",
            "clientVersion": "20.10.4",
            "deviceModel": "iPhone16,2",
            "hl": "en",
            "gl": "US",
        },
        "com.google.ios.youtube/20.10.4 (iPhone16,2; U; CPU iOS 17_5 like Mac OS X)",
    ),
)

#: Hosts a YouTube link is ever spelled with.
_HOSTS = frozenset(
    {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
     "youtu.be", "www.youtu.be", "youtube-nocookie.com", "www.youtube-nocookie.com"}
)

#: An id is exactly eleven characters of URL-safe base64. Checking the shape
#: matters: `/watch?v=<something else>` should be reported as a link we cannot
#: read rather than sent to the API to produce an opaque failure.
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

#: `1:23`, `01:23`, `1:02:03`, optionally bracketed, at the start of a line.
#: Anchored to line start because a timestamp in the middle of a sentence is a
#: reference to a moment, not a chapter heading for it.
_CHAPTER_RE = re.compile(
    r"^\s*[\[\(]?(?:(\d{1,2}):)?(\d{1,2}):(\d{2})[\]\)]?\s*[-–—:|]?\s*(.+?)\s*$"
)

#: Ceiling on beats regardless of what the video offers. The study resolves
#: structure, not a shot list: forty keyframes is not forty beats, it is one
#: film, and every extra frame is a vision call spent on resolution the
#: downstream statistics cannot use.
MAX_KEYFRAMES = 12


class NotYouTube(ValueError):
    """The URL is not a YouTube video link."""


class YouTubeUnavailable(RuntimeError):
    """It is a YouTube link, but the video cannot be read."""


@dataclass(frozen=True)
class StoryboardLevel:
    """One resolution tier of the scrub-bar filmstrip.

    A tier is a grid of tiles packed into sheets: `cols`×`rows` tiles per
    sheet, `frame_count` tiles in total across `sheets` of them, one tile every
    `interval_ms`. Everything the browser needs to crop frame *i* is derivable
    from these five numbers, which is the point — the geometry travels as
    integers and the pixels never touch this process.
    """

    width: int
    height: int
    frame_count: int
    cols: int
    rows: int
    interval_ms: int
    #: One URL per sheet, already signed and ready to relay.
    sheet_urls: tuple[str, ...]

    @property
    def per_sheet(self) -> int:
        return self.cols * self.rows

    def locate(self, frame_index: int) -> Optional[dict]:
        """Where frame `frame_index` physically is: which sheet, which tile."""
        if not 0 <= frame_index < self.frame_count:
            return None
        sheet, within = divmod(frame_index, self.per_sheet)
        if sheet >= len(self.sheet_urls):
            return None
        row, col = divmod(within, self.cols)
        return {
            "t_ms": frame_index * self.interval_ms,
            "sheet": sheet,
            "sheet_url": self.sheet_urls[sheet],
            "x": col * self.width,
            "y": row * self.height,
            "w": self.width,
            "h": self.height,
        }


@dataclass(frozen=True)
class CaptionTrack:
    language: str
    name: str
    #: True when YouTube generated it by speech recognition rather than a human
    #: writing it. Carried through to the receipt because an ASR transcript of a
    #: musical or heavily-accented video is a different quality of evidence, and
    #: a study that leans on it should say so.
    auto_generated: bool
    url: str


@dataclass
class VideoManifest:
    """Everything readable about a video without decoding it."""

    video_id: str
    title: str
    author: str
    duration_s: int
    description: str
    thumbnail_url: str
    view_count: Optional[int] = None
    chapters: list[dict] = field(default_factory=list)
    caption_tracks: list[CaptionTrack] = field(default_factory=list)
    storyboard: Optional[StoryboardLevel] = None
    #: Which client answered — `ANDROID`, `IOS`, or `oembed` when InnerTube
    #: refused and only the public preview was available.
    client: str = ""
    #: Why InnerTube refused, when it did. Carried so the receipt can say
    #: "this server is being rate-limited" instead of implying the video is
    #: private — they call for completely different actions from the reader.
    blocked_reason: str = ""
    #: Unsigned interior frames from the image CDN — the opening, quarter, half
    #: and three-quarter points. Populated only when the player API is out of
    #: reach, because when it is not the storyboard is strictly better.
    cdn_frames: list[dict] = field(default_factory=list)

    @property
    def watch_url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"


def parse_video_id(url: str) -> Optional[str]:
    """The eleven-character id, or None if this is not a YouTube video link.

    Every spelling YouTube itself emits is accepted — `watch?v=`, `youtu.be/`,
    `/shorts/`, `/embed/`, `/live/`, `/v/` — because a researcher pastes
    whatever the share button gave them, and refusing a Short for being spelled
    differently from a watch link is an arbitrary distinction to be on the
    wrong end of.
    """
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return None
    host = (parsed.hostname or "").lower()
    if host not in _HOSTS:
        return None

    if host.endswith("youtu.be"):
        candidate = parsed.path.lstrip("/").split("/")[0]
        return candidate if _ID_RE.match(candidate) else None

    if parsed.path == "/watch":
        candidate = (parse_qs(parsed.query).get("v") or [""])[0]
        return candidate if _ID_RE.match(candidate) else None

    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live", "v"}:
        return parts[1] if _ID_RE.match(parts[1]) else None
    return None


def is_youtube_url(url: str) -> bool:
    """True for any youtube.com/youtu.be link, even one we cannot read.

    Distinct from `parse_video_id` returning a value: a channel page or a
    playlist is a YouTube URL that is not a video, and the two want different
    messages. Conflating them tells someone who pasted a channel link that
    YouTube is unsupported.
    """
    try:
        return (urlparse(url.strip()).hostname or "").lower() in _HOSTS
    except ValueError:
        return False


def parse_chapters(description: str, duration_s: int) -> list[dict]:
    """Chapter markers a creator wrote into the description.

    These are the ONLY beat boundaries in the whole pipeline that are authored
    rather than inferred — the person who made the video saying where its parts
    begin. When they exist they beat anything a segmenter or a fixed sampling
    interval would produce, so they are looked for first.

    Validated rather than trusted. A list has to start at or near zero, run
    forward, and stay inside the runtime; a description that merely mentions
    three times in passing satisfies none of those and is rejected as a whole
    rather than yielding two plausible chapters and one absurd one.
    """
    found: list[tuple[int, str]] = []
    for line in (description or "").splitlines():
        match = _CHAPTER_RE.match(line)
        if not match:
            continue
        hours, minutes, seconds, label = match.groups()
        total = int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds)
        label = label.strip(" -–—:|")
        # A bare timestamp with no words after it is a reference, not a chapter.
        if not label or len(label) > 200:
            continue
        found.append((total, label))

    if len(found) < 2:
        return []
    # Must be in order and inside the runtime. Out-of-order stamps mean these
    # are citations scattered through prose, not a running order.
    if any(b[0] <= a[0] for a, b in zip(found, found[1:])):
        return []
    if duration_s and found[-1][0] >= duration_s:
        return []
    # A running order that starts at 4:12 is a list of highlights, not the
    # structure of the video — the first chapter of a video is its opening.
    if found[0][0] > 5:
        return []

    out: list[dict] = []
    for i, (start, label) in enumerate(found):
        end = found[i + 1][0] if i + 1 < len(found) else (duration_s or found[-1][0] + 60)
        out.append({"start_s": start, "end_s": end, "title": label})
    return out


def _storyboard_from_spec(spec: str, duration_s: int) -> Optional[StoryboardLevel]:
    """Parse YouTube's storyboard specification into concrete sheet URLs.

    The format is `base|level0|level1|…`, where the base carries `$L` (level
    index) and `$N` (sheet name, itself carrying `$M` for the sheet number),
    and each level is `w#h#count#cols#rows#interval#name#sigh`. The signature
    is appended as a query parameter and the sheet is refused without it.

    The highest level is taken because it is the largest tile — 160×90 against
    48×27, which is the difference between a frame a vision model can describe
    and a smudge.
    """
    parts = [p for p in (spec or "").split("|") if p]
    if len(parts) < 2:
        return None
    base = parts[0]
    level_index = len(parts) - 2
    fields = parts[-1].split("#")
    if len(fields) < 8:
        return None
    try:
        width, height, frame_count, cols, rows, interval = (int(f) for f in fields[:6])
    except ValueError:
        return None
    name, sigh = fields[6], fields[7]
    if not all((width, height, frame_count, cols, rows)):
        return None

    # `interval` is 0 on the lowest tier, which packs a fixed number of frames
    # across the whole runtime rather than sampling at a fixed rate. Deriving it
    # from the duration keeps every downstream timestamp honest instead of
    # placing every frame at t=0.
    if interval <= 0:
        if not duration_s:
            return None
        interval = max(1, round(duration_s * 1000 / frame_count))

    per_sheet = cols * rows
    sheet_count = -(-frame_count // per_sheet)  # ceil
    urls: list[str] = []
    for sheet in range(sheet_count):
        url = base.replace("$L", str(level_index)).replace("$N", name)
        url = url.replace("$M", str(sheet))
        url += ("&" if "?" in url else "?") + "sigh=" + sigh
        urls.append(url)

    return StoryboardLevel(
        width=width,
        height=height,
        frame_count=frame_count,
        cols=cols,
        rows=rows,
        interval_ms=interval,
        sheet_urls=tuple(urls),
    )


#: Successful manifests, by video id, with the time they were fetched.
#:
#: THE ONLY LEVER THAT ACTUALLY HELPS. The bot wall is not per-video and not
#: per-request — it is IP REPUTATION, earned by volume. Every extra call makes
#: the next one likelier to fail, which means the instinctive fixes (retry, try
#: more clients, back off and try again) all make the problem WORSE, and were
#: measured doing exactly that: a burst of matrix probing took this host from
#: "ANDROID answers with a storyboard and 33 caption tracks" to "every client
#: and every library refuses", including a bare urllib call that had succeeded
#: minutes earlier.
#:
#: So the fix is to ASK LESS. A researcher pasting the same link twice, or
#: re-running a study, or a second person studying the same video, now costs
#: zero requests instead of two.
_MANIFEST_CACHE: dict[str, tuple[float, "VideoManifest"]] = {}

#: An hour. Long enough to cover a working session on one piece of content,
#: short enough that a video's captions appearing, or its title changing, shows
#: up the same day. Only SUCCESSES are cached — a refusal is a fact about this
#: minute's rate limit, not about the video, and caching it would turn a
#: transient block into a persistent one.
_CACHE_TTL_S = 3600.0

#: Bounded so a long-lived process cannot grow this without limit. Small
#: because the access pattern is a handful of videos per session, not a corpus.
_CACHE_MAX = 256


def _cache_get(video_id: str) -> Optional[VideoManifest]:
    entry = _MANIFEST_CACHE.get(video_id)
    if entry is None:
        return None
    fetched_at, manifest = entry
    if time.monotonic() - fetched_at > _CACHE_TTL_S:
        _MANIFEST_CACHE.pop(video_id, None)
        return None
    return manifest


def _cache_put(video_id: str, manifest: VideoManifest) -> None:
    if len(_MANIFEST_CACHE) >= _CACHE_MAX:
        # Oldest out. Insertion-ordered dicts make this exact without a heap,
        # and the eviction rate here is effectively zero anyway.
        _MANIFEST_CACHE.pop(next(iter(_MANIFEST_CACHE)), None)
    _MANIFEST_CACHE[video_id] = (time.monotonic(), manifest)


async def fetch_manifest(video_id: str) -> VideoManifest:
    """Ask InnerTube what this video is, asking as few times as possible.

    A per-client failure is not fatal and a per-client REFUSAL is not either:
    the WEB client answers 200 with `playabilityStatus: UNPLAYABLE` from a
    datacentre address, which is a refusal wearing a success code.

    ONCE THE BOT WALL APPEARS, THIS STOPS ASKING. That is the opposite of the
    obvious design and it is what the measurements support: the wall is keyed to
    the address, so a second client from the same address meets the same wall,
    and the matrix confirms the ordering is one-way — ANDROID succeeds where IOS
    fails and never the reverse. Trying the rest would spend requests that make
    the NEXT researcher's link likelier to fail, to obtain a refusal we can
    already predict.
    """
    cached = _cache_get(video_id)
    if cached is not None:
        return cached

    last_reason = ""

    for name, context, agent in _CLIENTS:
        payload = json.dumps(
            {
                "videoId": video_id,
                "context": {"client": context},
                "contentCheckOk": True,
                "racyCheckOk": True,
            }
        ).encode()
        try:
            if YOUTUBE_PROXY:
                # The one call that needs to come from a clean address. Roughly
                # 200 kB, once an hour per video thanks to the cache — which is
                # why routing it costs fractions of a cent rather than the
                # dollars proxying the media would.
                doc, _final = await proxied_fetch_json(
                    _PLAYER_ENDPOINT, proxy=YOUTUBE_PROXY, body=payload, user_agent=agent
                )
            else:
                doc, _final = await fetch_json(
                    _PLAYER_ENDPOINT, body=payload, user_agent=agent
                )
        except (UnsafeURL, FetchFailed) as exc:
            last_reason = str(exc)
            continue

        status = (doc.get("playabilityStatus") or {})
        details = (doc.get("videoDetails") or {})
        if status.get("status") not in {"OK", None} or not details.get("title"):
            last_reason = status.get("reason") or status.get("status") or "no video details"
            if _is_bot_wall(last_reason):
                # The wall is on the ADDRESS, not the client or the video. The
                # next client shares the address, so it meets the same wall —
                # and spending the request makes the next caller's link likelier
                # to fail. Stop here and take the preview.
                break
            continue

        try:
            duration_s = int(details.get("lengthSeconds") or 0)
        except (TypeError, ValueError):
            duration_s = 0

        tracks_raw = (
            (doc.get("captions") or {})
            .get("playerCaptionsTracklistRenderer", {})
            .get("captionTracks", [])
        )
        tracks = [
            CaptionTrack(
                language=t.get("languageCode") or "?",
                name=((t.get("name") or {}).get("simpleText")
                      or ((t.get("name") or {}).get("runs") or [{}])[0].get("text")
                      or t.get("languageCode") or "?"),
                auto_generated=t.get("kind") == "asr",
                url=t.get("baseUrl") or "",
            )
            for t in tracks_raw
            if t.get("baseUrl")
        ]

        spec = (
            (doc.get("storyboards") or {})
            .get("playerStoryboardSpecRenderer", {})
            .get("spec", "")
        )
        description = details.get("shortDescription") or ""

        try:
            views: Optional[int] = int(details.get("viewCount"))
        except (TypeError, ValueError):
            views = None

        manifest = VideoManifest(
            video_id=video_id,
            title=details.get("title") or "(untitled)",
            author=details.get("author") or "",
            duration_s=duration_s,
            description=description,
            # `hqdefault` rather than `maxresdefault`, which 404s for anything
            # never published above 720p and would put a broken image in the
            # receipt for exactly the older videos most likely to lack a
            # storyboard too. The higher-resolution candidates travel alongside
            # in the envelope for the client to try first.
            thumbnail_url=f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
            view_count=views,
            chapters=parse_chapters(description, duration_s),
            caption_tracks=tracks,
            storyboard=_storyboard_from_spec(spec, duration_s),
            client=name,
        )
        # Only successes. A refusal describes this minute's rate limit, not the
        # video, and caching one would turn a transient block into an hour-long
        # one for a link that would have worked on the next try.
        _cache_put(video_id, manifest)
        return manifest

    # ── InnerTube refused. Fall back to oEmbed. ─────────────────────────
    #
    # MEASURED IN PRODUCTION, and the reason this branch exists at all. From a
    # residential address InnerTube answers for everything; from Fly's
    # datacentre IPs it answers "Sign in to confirm you're not a bot" for most
    # videos — two of three tested, consistently, across both clients and
    # retries. So the path that works in development is the path that fails
    # once deployed, which is the worst shape a failure can have.
    #
    # oEmbed is unauthenticated, is not subject to that check, and answered 200
    # for every video InnerTube had just refused. It gives a title, an author
    # and a thumbnail — no duration, no chapters, no captions, no filmstrip —
    # which is not a video study and is a real one. Taking it beats returning
    # a 502 for a public video.
    fallback = await _oembed_manifest(video_id, last_reason)
    if fallback is not None:
        return fallback

    raise YouTubeUnavailable(
        f"YouTube would not describe that video ({last_reason or 'no client answered'}), "
        "and its public preview is unavailable too. Private, age-restricted, "
        "members-only and deleted videos all look like this from a server."
    )


#: Interior frames every video exposes as plain, unsigned CDN URLs.
#:
#: THE ROUTE ROUND THE BOT WALL, and the reason a blocked video is still worth
#: studying. `hq1`, `hq2` and `hq3` are the frames YouTube generates at roughly
#: the quarter, half and three-quarter points, and `hqdefault` is the opening.
#: They need NO API call, carry no signature, sit on the image CDN rather than
#: behind the player API, and answer `Access-Control-Allow-Origin: *`. Measured
#: present for every video tested — including "Me at the zoo", which is 19
#: seconds long and has no storyboard at all.
#:
#: Four frames is not eight, and the fractions are approximate. But four real
#: frames spanning the video is a temporal study, and the thumbnail alone is a
#: picture of a thumbnail. That is the whole difference between this tier
#: existing and not.
CDN_FRAMES: tuple[tuple[str, float], ...] = (
    ("default", 0.0),
    ("1", 0.25),
    ("2", 0.5),
    ("3", 0.75),
)

#: Resolution tiers for those frames, best first, measured with a JPEG header
#: parse rather than assumed:
#:
#:     maxres  1280×720   modern uploads
#:     sd       640×480
#:     hq       480×360   the floor; present on a 2005 upload with nothing else
#:     mq       320×180
#:
#: THIS IS THE WHOLE POINT, and it inverts what looked like the quality
#: ordering. A storyboard tile is 160×90. `maxres` is 1280×720 — SIXTY-FOUR
#: TIMES the pixels. At 160×90 a vision model cannot read a caption, a price, a
#: logo or a face; at 720p it reads all of them. So these frames are not a
#: consolation prize for a blocked video, they are the highest-detail view of it
#: available anywhere, and the filmstrip's real advantage is only that there are
#: more of them.
#:
#: Tried in order by the CLIENT rather than resolved here: picking the tier
#: server-side costs up to three extra requests per frame against the host we
#: are already being rate-limited by, and the browser is going to fetch the
#: winner through the relay regardless.
CDN_RESOLUTIONS = ("maxres", "sd", "hq", "mq")


def cdn_frame_urls(video_id: str, duration_s: int = 0) -> list[dict]:
    """The unsigned interior frames, best resolution first, in running order.

    `fraction` is always exact — these are published at the opening, quarter,
    half and three-quarter points. `t_ms` is only filled in when the runtime is
    actually known: reaching this tier usually means the player API refused us,
    and a frame labelled `1:23` when nothing measured 1:23 is a fabricated
    clock. `-1` is the explicit unknown, and the modality layer reads it as a
    sequential axis.
    """
    frames: list[dict] = []
    for name, fraction in CDN_FRAMES:
        frames.append(
            {
                "kind": "image",
                "fraction": fraction,
                "t_ms": int(duration_s * 1000 * fraction) if duration_s else -1,
                # Candidates, best first. `default` has no `maxresdefault`
                # sibling problem — every tier spells it the same way.
                "urls": [
                    f"https://i.ytimg.com/vi/{video_id}/{tier}{name}.jpg"
                    for tier in CDN_RESOLUTIONS
                ],
            }
        )
    return frames


#: YouTube's own oEmbed endpoint. Public, unauthenticated, and — unlike
#: InnerTube — not gated behind the bot check that fires on datacentre IPs.
#: It also answers CORS (measured: reflects the requesting origin), where
#: InnerTube refuses the preflight outright with a 403.
_OEMBED = "https://www.youtube.com/oembed?url={url}&format=json"

#: What the bot check says when it fires. Matched so the failure can be
#: reported as what it is — a transient block on THIS SERVER — rather than as a
#: property of the video. The old message named four permission states, none of
#: which applied, and sent people to check the settings of a public video.
_BOT_WALL = ("not a bot", "sign in to confirm")


def _is_bot_wall(reason: str) -> bool:
    lowered = (reason or "").lower()
    return any(marker in lowered for marker in _BOT_WALL)


async def _oembed_manifest(video_id: str, blocked_reason: str) -> Optional[VideoManifest]:
    """A metadata-only manifest from YouTube's public oEmbed. None if that fails too."""
    from urllib.parse import quote

    watch = f"https://www.youtube.com/watch?v={video_id}"
    try:
        doc, _final = await fetch_json(_OEMBED.format(url=quote(watch, safe="")))
    except (UnsafeURL, FetchFailed):
        return None
    title = str(doc.get("title") or "").strip()
    if not title:
        return None
    return VideoManifest(
        video_id=video_id,
        title=title,
        author=str(doc.get("author_name") or "").strip(),
        # oEmbed carries no runtime. Left at zero rather than guessed, because
        # every timestamp downstream is derived from it and an invented
        # duration would place beats on a clock that does not exist.
        duration_s=0,
        description="",
        thumbnail_url=str(doc.get("thumbnail_url") or f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"),
        client="oembed",
        blocked_reason=blocked_reason,
        cdn_frames=cdn_frame_urls(video_id),
    )


def pick_caption_track(tracks: list[CaptionTrack], prefer: str = "en") -> Optional[CaptionTrack]:
    """The best track available: human-written over ASR, preferred language over not.

    Human-written first even when it is in another language. An ASR track is a
    machine's guess at what was said; a manual track is what someone typed. For
    a study that reads the words as content rather than as a search index, that
    difference is worth more than the language matching.
    """
    if not tracks:
        return None
    def rank(t: CaptionTrack) -> tuple[int, int]:
        return (0 if not t.auto_generated else 1, 0 if t.language.startswith(prefer) else 1)
    return sorted(tracks, key=rank)[0]


def _parse_json3(raw: str) -> list[dict]:
    doc = json.loads(raw)
    cues: list[dict] = []
    for event in doc.get("events") or []:
        segs = event.get("segs") or []
        text = "".join(s.get("utf8", "") for s in segs).strip()
        # `\n` inside a cue is line wrapping in the caption renderer, not a
        # sentence break — joining with a space keeps the words readable.
        text = " ".join(text.split())
        if not text:
            continue
        cues.append({"t_ms": int(event.get("tStartMs") or 0), "text": text})
    return cues


def _parse_timedtext_xml(raw: str) -> list[dict]:
    """Both XML dialects: srv1's `<text start=… dur=…>` and srv3's `<p t=… d=…>`."""
    root = ET.fromstring(raw)
    cues: list[dict] = []
    for node in root.iter():
        if node.tag not in {"text", "p"}:
            continue
        start = node.get("start") or node.get("t")
        if start is None:
            continue
        try:
            # srv1 gives seconds as a float, srv3 milliseconds as an int.
            t_ms = int(float(start) * 1000) if node.tag == "text" else int(float(start))
        except ValueError:
            continue
        text = "".join(node.itertext())
        text = " ".join(unescape(text).split())
        if text:
            cues.append({"t_ms": t_ms, "text": text})
    return cues


async def fetch_captions(track: CaptionTrack) -> list[dict]:
    """A caption track as timed cues. Returns [] rather than raising.

    BEST-EFFORT BY DESIGN, and this is the one place in the pipeline where that
    is the right call. `timedtext` rate-limits hard — measured 429 on every
    format from an address that had made a handful of requests minutes earlier —
    and its availability is a property of the moment rather than of the video.
    A study built from storyboard keyframes, chapters and the description is a
    real study; refusing to run it because the subtitles were throttled would
    trade a good result for none.

    The caller is told whether cues arrived, so the absence shows up as a
    stated limitation rather than as a quietly thinner beat.
    """
    for fmt, parse in (("json3", _parse_json3), ("srv3", _parse_timedtext_xml), ("", _parse_timedtext_xml)):
        url = track.url + (f"&fmt={fmt}" if fmt else "")
        try:
            raw = await fetch_text(url)
        except (UnsafeURL, FetchFailed):
            continue
        try:
            cues = parse(raw)
        except (ValueError, ET.ParseError):
            continue
        if cues:
            return cues
    return []


def plan_keyframes(
    manifest: VideoManifest, want: int = 8
) -> list[dict]:
    """Which storyboard tiles to crop, and when each one happens.

    Sampled at BEAT CENTRES rather than at boundaries, for the same reason the
    uploaded-video path does: the first frame after a cut is often black and
    the last is often a logo card, and a vision model asked to describe either
    will describe it as if it were the content.

    When the creator wrote chapters, those decide the sampling instead of an
    even split — one frame from the middle of each authored section. That is
    the difference between "what does this video look like every 20 seconds"
    and "what does each part of this video look like".
    """
    # The four high-resolution frames go in FIRST, always, whether or not a
    # filmstrip exists. They are 1280×720 against the filmstrip's 160×90, so a
    # study built only from tiles was reading a smudge at four of the moments it
    # could have been reading the actual frame. Cheap, too: no API call, no
    # signature, no rate limiter.
    high_res = cdn_frame_urls(manifest.video_id, manifest.duration_s)

    board = manifest.storyboard
    if board is None or board.frame_count <= 0:
        return [{**f, "label": "", "source": "cdn"} for f in high_res]

    want = max(2, min(want, MAX_KEYFRAMES))
    duration_ms = (manifest.duration_s or 0) * 1000
    if duration_ms <= 0:
        duration_ms = board.frame_count * board.interval_ms

    if manifest.chapters and 2 <= len(manifest.chapters) <= MAX_KEYFRAMES:
        targets_ms = [
            (c["start_s"] + c["end_s"]) * 500  # midpoint in ms: (a+b)/2*1000
            for c in manifest.chapters
        ]
        labels = [c["title"] for c in manifest.chapters]
        source = "chapters"
    else:
        targets_ms = [duration_ms * (i + 0.5) / want for i in range(want)]
        labels = [""] * want
        source = "even"

    frames: list[dict] = []
    seen: set[int] = set()
    for target, label in zip(targets_ms, labels):
        index = min(board.frame_count - 1, max(0, round(target / board.interval_ms)))
        # Two chapters inside one storyboard interval would crop the same tile
        # twice and spend two vision calls describing one picture.
        while index in seen and index + 1 < board.frame_count:
            index += 1
        if index in seen:
            continue
        seen.add(index)
        tile = board.locate(index)
        if tile is None:
            continue
        frames.append({**tile, "label": label, "target_ms": int(target)})

    for f in frames:
        f["source"] = source
        f["kind"] = "tile"

    # Interleave. The filmstrip supplies DENSITY and the CDN frames supply
    # DETAIL, and taking either alone throws away the other. Tiles that land
    # near a high-resolution frame are dropped rather than shown beside it —
    # two beats a second apart describing the same shot at two resolutions is a
    # duplicated beat, and every beat costs a vision call.
    merged = list(high_res)
    span = max(1, (manifest.duration_s or 0) * 1000)
    near = span * 0.06
    for tile in frames:
        if manifest.duration_s and any(
            abs(tile["t_ms"] - hi["t_ms"]) < near for hi in high_res if hi["t_ms"] >= 0
        ):
            continue
        merged.append(tile)

    # Back into running order, and capped. Unknown timestamps sort first, which
    # only happens when the duration is unknown — in which case there are no
    # tiles to interleave anyway.
    merged.sort(key=lambda f: f["t_ms"] if f["t_ms"] >= 0 else 0)
    return merged[:MAX_KEYFRAMES]


#: The rungs, best first. Each is a REAL study on a genuinely different
#: evidence base, and the difference between them is not cosmetic:
#:
#:   video  — picture and (when captions arrived) speech, on a real clock.
#:   audio  — the transcript alone, still on a real clock, but blind.
#:   text   — the creator's own running order. This one is NOT a study of the
#:            video. It is a study of how the video describes itself, which is
#:            a legitimate question and a different question, and conflating
#:            the two would be the single most misleading thing this file could
#:            do. It is labelled everywhere it surfaces.
#:
#: Written as data rather than as branches in the route so the frontend, the
#: receipt and the refusal all read the same table.
LADDER = {
    "video": "keyframes from the scrub-bar filmstrip, described by a vision model",
    "audio": "the published transcript, on its own timings",
    "text": "the creator's own chapter list and description — NOT the video itself",
    "cdn_frames": (
        "the four full-resolution stills YouTube publishes at the opening, "
        "quarter, half and three-quarter points"
    ),
    "metadata": "the thumbnail and title only — NOT the video itself",
}


def choose_rung(manifest: VideoManifest, cues: list[dict], frames: list[dict]) -> str:
    """Which study this video can actually support, best first.

    Ordered by how close the evidence is to what an audience experienced.
    Keyframes plus optional speech beat speech alone, and both beat the
    creator's description of their own video — which is a study of the pitch,
    not of the thing.

    The bottom rung is the thumbnail. It exists because InnerTube is blocked
    from datacentre addresses often enough that it is the COMMON case in
    production, not an edge case, and a thumbnail study is a legitimate answer
    to "does this earn a click" even though it is no answer at all to "what
    does this video do to an audience". Both facts are stated in the receipt.
    """
    # Any filmstrip tile means density AND detail; only high-resolution images
    # means the four published frames and nothing between them. Both are video
    # studies, and the receipt should not claim a filmstrip we never got.
    if len(frames) >= 2:
        return "video" if any(f.get("kind") == "tile" for f in frames) else "cdn_frames"
    if len(cues) >= 4:
        return "audio"
    if len(manifest.chapters) >= 2:
        return "text"
    # The CDN frames. Below a real filmstrip because there are four of them and
    # their positions are approximate, and far above the thumbnail rung because
    # four frames spanning the video is a study of the video.
    if manifest.cdn_frames:
        return "cdn_frames"
    if manifest.thumbnail_url:
        return "metadata"
    return "none"


def manifest_envelope(manifest: VideoManifest, cues: list[dict], frames: list[dict]) -> dict:
    """What the browser gets: geometry, text, and an honest account of both.

    `note` is written here rather than in the route because this is the only
    place that knows what actually arrived. Every degradation the ladder allows
    has to be visible in the receipt — a study running on eight low-resolution
    tiles and no transcript is a real study and a THINNER one, and the person
    reading the output is entitled to know which they are looking at.
    """
    board = manifest.storyboard
    have_text = bool(cues)
    rung = choose_rung(manifest, cues, frames)
    parts: list[str] = []

    # The cause comes FIRST when there is one, because it is the only line that
    # tells the reader whether to do something. "YouTube is rate-limiting this
    # server" is a transient condition they can retry or ignore; the old
    # message named four permission states instead and sent people to check the
    # settings of a video that was public all along.
    if manifest.client == "oembed":
        parts.append(
            "YouTube refused to describe this video to our server"
            + (
                # Deliberately does NOT say "try again". The block is on this
                # server's address and is earned by request VOLUME, so retrying
                # deepens it — measured: a burst of probing took this host from
                # "ANDROID answers with a storyboard and 33 caption tracks" to
                # every client refusing. Telling someone to retry would make
                # their next link worse and everyone else's too. The reliable
                # route is the one that does not involve YouTube's API at all.
                " — it blocks datacentre addresses like ours, which is about "
                "where we are calling from and not about this video. For the "
                "full study, download the video and drop the file in above; "
                "keyframes are read in your browser"
                if _is_bot_wall(manifest.blocked_reason)
                else f" ({manifest.blocked_reason})"
            )
            + ", so only its public preview was available"
        )

    tiles = [f for f in frames if f.get("kind") == "tile"]
    images = [f for f in frames if f.get("kind") == "image"]
    if images and tiles and board:
        parts.append(
            f"{len(frames)} keyframes — {len(images)} at full resolution from "
            f"YouTube's published stills, plus {len(tiles)} at "
            f"{board.width}×{board.height} from its scrub-bar filmstrip"
        )
    elif images:
        parts.append(
            f"{len(images)} keyframes at full resolution — the stills YouTube "
            "publishes at the opening, quarter, half and three-quarter points"
        )
    elif frames and board:
        parts.append(
            f"{len(frames)} keyframes at {board.width}×{board.height} from "
            f"YouTube's own scrub-bar filmstrip"
        )
    elif board is None and manifest.client != "oembed":
        # Not a failure and worth saying so plainly: YouTube does not generate
        # a filmstrip for very short clips, so this is a property of the video
        # rather than something that went wrong or can be retried.
        #
        # Suppressed on the oEmbed path, where the filmstrip is absent because
        # we were never told about it — attributing that to the video's length
        # would be inventing a second, wrong explanation on top of the real one
        # already printed above.
        parts.append(
            "YouTube publishes no scrub-bar filmstrip for this video — usually "
            "because it is too short — so there are no keyframes to read"
        )
    if manifest.client == "oembed":
        pass  # captions were never offered either; the cause is already stated
    elif have_text:
        parts.append(f"{len(cues)} transcript cues")
    elif manifest.caption_tracks:
        parts.append(
            "captions exist but YouTube rate-limited the download — beats carry "
            "picture without speech"
        )
    else:
        parts.append("no captions published — beats carry picture without speech")
    if manifest.chapters:
        parts.append(f"{len(manifest.chapters)} creator-written chapters")

    return {
        "rung": rung,
        "rung_basis": LADDER.get(rung, "nothing readable"),
        # The higher-resolution thumbnails, best first. Tried in order by the
        # client because `maxresdefault` is absent for anything never published
        # above 720p and answers 404 rather than a smaller image.
        "thumbnail_candidates": [
            f"https://i.ytimg.com/vi/{manifest.video_id}/maxresdefault.jpg",
            f"https://i.ytimg.com/vi/{manifest.video_id}/sddefault.jpg",
            f"https://i.ytimg.com/vi/{manifest.video_id}/hqdefault.jpg",
        ],
        "video_id": manifest.video_id,
        "title": manifest.title,
        "author": manifest.author,
        "duration_s": manifest.duration_s,
        "view_count": manifest.view_count,
        "watch_url": manifest.watch_url,
        "thumbnail_url": manifest.thumbnail_url,
        "description": manifest.description[:4000],
        "chapters": manifest.chapters,
        "captions": {
            "cues": cues,
            "available": [
                {"language": t.language, "name": t.name, "auto": t.auto_generated}
                for t in manifest.caption_tracks
            ],
        },
        "keyframes": frames,
        "cdn_frames": manifest.cdn_frames,
        "storyboard": (
            {
                "width": board.width,
                "height": board.height,
                "interval_ms": board.interval_ms,
                "frame_count": board.frame_count,
                "sheets": len(board.sheet_urls),
            }
            if board
            else None
        ),
        "client": manifest.client,
        "note": " · ".join(parts),
        # Kept alongside `rung` because they answer different questions: `rung`
        # is which study to run, `degraded` is whether to say so loudly.
        "degraded": rung != "video",
    }
