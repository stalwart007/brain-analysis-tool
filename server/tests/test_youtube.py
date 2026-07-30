"""YouTube ingest: URL parsing, chapter detection, storyboard geometry.

Everything here is offline and pure. The network-facing half of `app.youtube`
is one function (`fetch_manifest`) whose behaviour is YouTube's to change, and
a test that asserts on it is a test that fails on a Tuesday for reasons nobody
in this repo controls. What IS ours is the arithmetic that turns a storyboard
specification into tile coordinates, and the judgement about what a video can
support — both of which are silently wrong-able, which is what tests are for.

The case that earns its place most is `test_prose_timestamps_are_not_chapters`.
A description mentioning two times in passing produced a two-chapter running
order, which then decided where every keyframe was sampled — so a video with no
chapters at all was studied at moments chosen by a sentence about something
else, and nothing in the output said so.
"""

import asyncio

import pytest

from app.youtube import (
    MAX_KEYFRAMES,
    CaptionTrack,
    VideoManifest,
    _parse_json3,
    _parse_timedtext_xml,
    _is_bot_wall,
    _storyboard_from_spec,
    choose_rung,
    is_youtube_url,
    manifest_envelope,
    parse_chapters,
    parse_video_id,
    pick_caption_track,
    plan_keyframes,
)

#: The real specification returned for dQw4w9WgXcQ, verbatim apart from a
#: shortened `sqp`. The shape is load-bearing — `$L`, `$N` and `$M` are
#: substituted, the tile grid is read out of the `#`-separated fields, and the
#: LEVEL INDEX is positional — so a synthetic or trimmed string would test the
#: parser against a format of this file's own invention. All three tiers are
#: kept for that reason: dropping one silently renumbers the rest.
def asyncio_run(coro):
    """`asyncio.run` under a name that reads as a helper at the call sites."""
    return asyncio.run(coro)


REAL_SPEC = (
    "https://i.ytimg.com/sb/dQw4w9WgXcQ/storyboard3_L$L/$N.jpg?sqp=-oaymwENSDf"
    "|48#27#100#10#10#0#default#rs$AOn4CLDgtWGAnaqZ"
    "|80#45#108#10#10#2000#M$M#rs$AOn4CLBf8GkpJjLT0"
    "|160#90#108#5#5#2000#M$M#rs$AOn4CLClA1jTU48sH"
)


# ── URL parsing ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ?t=42", "dQw4w9WgXcQ"),
        ("https://m.youtube.com/watch?v=dQw4w9WgXcQ&list=PLxyz", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/live/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("  https://www.youtube.com/watch?v=dQw4w9WgXcQ  ", "dQw4w9WgXcQ"),
    ],
)
def test_every_spelling_youtube_emits_is_accepted(url, expected):
    """A researcher pastes whatever the share button gave them."""
    assert parse_video_id(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/@SomeChannel",
        "https://www.youtube.com/playlist?list=PLxyz",
        "https://www.youtube.com/results?search_query=cats",
        "https://www.youtube.com/watch?v=tooshort",
        "https://www.youtube.com/watch",
    ],
)
def test_youtube_urls_without_a_video_yield_no_id(url):
    """Still YouTube, still nothing to study — the two facts are separate."""
    assert parse_video_id(url) is None
    assert is_youtube_url(url) is True


def test_lookalike_host_is_not_youtube():
    """The check that stops `youtube.com.evil.test` reaching the InnerTube path.

    Suffix matching on the hostname is the classic way this goes wrong, and it
    would let an attacker choose the host the manifest call is sent to.
    """
    assert is_youtube_url("https://youtube.com.evil.test/watch?v=dQw4w9WgXcQ") is False
    assert parse_video_id("https://youtube.com.evil.test/watch?v=dQw4w9WgXcQ") is None
    assert is_youtube_url("https://notyoutube.com/watch?v=dQw4w9WgXcQ") is False


def test_video_id_shape_is_checked():
    """Eleven URL-safe base64 characters, or it is not an id.

    Without this, `?v=../../etc` is passed to the API as a video id.
    """
    assert parse_video_id("https://www.youtube.com/watch?v=../../../etc/passwd") is None
    assert parse_video_id("https://www.youtube.com/watch?v=abc!def@hij") is None


# ── chapters ───────────────────────────────────────────────────────────────


def test_chapters_are_read_and_given_end_times():
    chapters = parse_chapters(
        "Welcome!\n0:00 Intro\n1:30 The problem\n4:12 The fix\n7:00 Wrap up\nthanks",
        duration_s=500,
    )
    assert [c["title"] for c in chapters] == ["Intro", "The problem", "The fix", "Wrap up"]
    assert chapters[0]["start_s"] == 0
    # Each chapter ends where the next begins; the last runs to the end.
    assert chapters[0]["end_s"] == 90
    assert chapters[-1]["end_s"] == 500


def test_hour_long_stamps_parse():
    chapters = parse_chapters("0:00 Start\n1:02:03 Later\n", duration_s=7200)
    assert [c["start_s"] for c in chapters] == [0, 3723]


def test_prose_timestamps_are_not_chapters():
    """THE regression. Times mentioned in a sentence are not a running order.

    Read as chapters they would decide where every keyframe is sampled, so a
    video with no chapters gets studied at moments chosen by a sentence about
    something else — and the output would not say so anywhere.
    """
    assert parse_chapters("See 4:12 for the demo, and 1:02 for setup.", 500) == []


def test_out_of_order_and_overrunning_stamps_are_rejected():
    # Descending: a highlight list, not a structure.
    assert parse_chapters("0:00 A\n5:00 B\n2:00 C", 600) == []
    # Past the end of the video: these are timestamps for something else.
    assert parse_chapters("0:00 A\n1:00 B\n99:00 C", 300) == []


def test_a_running_order_must_start_at_the_beginning():
    """A list starting at 4:12 is highlights. A video's first chapter is its opening."""
    assert parse_chapters("4:12 Middle bit\n6:00 Another bit", 600) == []


def test_bare_timestamp_with_no_label_is_skipped():
    assert parse_chapters("0:00\n1:00\n2:00", 300) == []


# ── storyboard geometry ────────────────────────────────────────────────────


def test_highest_level_is_taken_and_sheets_are_counted():
    board = _storyboard_from_spec(REAL_SPEC, duration_s=213)
    assert board is not None
    # The 160×90 tier, not the 48×27 one — the difference between a frame a
    # vision model can describe and a smudge.
    assert (board.width, board.height) == (160, 90)
    assert (board.cols, board.rows) == (5, 5)
    assert board.frame_count == 108
    # 108 frames at 25 per sheet is five sheets, the last one partial.
    assert len(board.sheet_urls) == 5
    assert "storyboard3_L2/M0.jpg" in board.sheet_urls[0]
    assert "storyboard3_L2/M4.jpg" in board.sheet_urls[4]
    # Refused without the signature.
    assert all("sigh=rs$AOn4CLClA1jTU48sH" in u for u in board.sheet_urls)


def test_tile_coordinates_walk_the_grid_then_the_sheet():
    board = _storyboard_from_spec(REAL_SPEC, duration_s=213)
    first = board.locate(0)
    assert (first["x"], first["y"], first["sheet"]) == (0, 0, 0)
    assert first["t_ms"] == 0

    # Frame 6 in a 5-wide grid is row 1, column 1.
    sixth = board.locate(6)
    assert (sixth["x"], sixth["y"], sixth["sheet"]) == (160, 90, 0)
    assert sixth["t_ms"] == 12_000

    # Frame 25 is the first tile of the SECOND sheet, back at the origin.
    across = board.locate(25)
    assert (across["x"], across["y"], across["sheet"]) == (0, 0, 1)
    assert "M1.jpg" in across["sheet_url"]


def test_out_of_range_frames_have_no_location():
    board = _storyboard_from_spec(REAL_SPEC, duration_s=213)
    assert board.locate(-1) is None
    assert board.locate(board.frame_count) is None


def test_zero_interval_tier_derives_its_spacing_from_the_duration():
    """The lowest tier packs N frames across the runtime instead of sampling.

    Left at zero, every frame reports t=0 and the whole timeline collapses.
    """
    single = "https://x.test/sb/$L/$N.jpg?a=b|48#27#100#10#10#0#default#rs$SIG"
    board = _storyboard_from_spec(single, duration_s=200)
    assert board.interval_ms == 2000
    assert board.locate(50)["t_ms"] == 100_000


def test_malformed_specs_are_refused_rather_than_guessed():
    assert _storyboard_from_spec("", 100) is None
    assert _storyboard_from_spec("https://x.test/only-a-base", 100) is None
    assert _storyboard_from_spec("https://x.test/$L|48#27#100", 100) is None
    # A zero-width tile would make every crop empty.
    assert _storyboard_from_spec("https://x.test/$L/$N|0#27#100#10#10#0#d#rs$S", 100) is None
    # No duration and no interval: nothing to derive a clock from.
    assert _storyboard_from_spec("https://x.test/$L/$N|48#27#100#10#10#0#d#rs$S", 0) is None


# ── keyframe planning ──────────────────────────────────────────────────────


def _manifest(**over) -> VideoManifest:
    base = dict(
        video_id="dQw4w9WgXcQ",
        title="T",
        author="A",
        duration_s=213,
        description="",
        thumbnail_url="https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg",
        storyboard=_storyboard_from_spec(REAL_SPEC, duration_s=213),
    )
    base.update(over)
    return VideoManifest(**base)


def test_frames_are_sampled_at_beat_centres_not_boundaries():
    """Never t=0. The first frame after a cut is often black, and a vision
    model asked to describe it describes black as if it were the content."""
    frames = plan_keyframes(_manifest(), want=8)
    assert len(frames) == 8
    assert frames[0]["t_ms"] > 0
    assert all(f["source"] == "even" for f in frames)
    # Monotonic and inside the runtime.
    stamps = [f["t_ms"] for f in frames]
    assert stamps == sorted(stamps)
    assert stamps[-1] <= 213_000


def test_chapters_decide_the_sampling_when_the_creator_wrote_them():
    """Authored structure beats an even split — one frame per named section."""
    chapters = [
        {"start_s": 0, "end_s": 60, "title": "Intro"},
        {"start_s": 60, "end_s": 140, "title": "Middle"},
        {"start_s": 140, "end_s": 213, "title": "End"},
    ]
    frames = plan_keyframes(_manifest(chapters=chapters), want=8)
    assert len(frames) == 3
    assert [f["label"] for f in frames] == ["Intro", "Middle", "End"]
    assert all(f["source"] == "chapters" for f in frames)
    # Midpoint of the first chapter is 30s, which is frame 15 at 2s spacing.
    assert frames[0]["t_ms"] == 30_000


def test_two_chapters_inside_one_interval_do_not_crop_the_same_tile_twice():
    """Otherwise two vision calls describe one picture and one beat is lost."""
    chapters = [
        {"start_s": 0, "end_s": 1, "title": "A"},
        {"start_s": 1, "end_s": 2, "title": "B"},
    ]
    frames = plan_keyframes(_manifest(chapters=chapters), want=8)
    assert len({f["t_ms"] for f in frames}) == len(frames)


def test_no_storyboard_means_no_frames_rather_than_an_exception():
    assert plan_keyframes(_manifest(storyboard=None)) == []


def test_frame_count_is_capped():
    frames = plan_keyframes(_manifest(), want=999)
    assert len(frames) <= MAX_KEYFRAMES


# ── the ladder ─────────────────────────────────────────────────────────────


def test_rung_prefers_picture_then_speech_then_the_creators_own_words():
    board_frames = plan_keyframes(_manifest(), want=4)
    cues = [{"t_ms": i * 1000, "text": "x"} for i in range(8)]
    chapters = [{"start_s": 0, "end_s": 5, "title": "A"}, {"start_s": 5, "end_s": 9, "title": "B"}]

    assert choose_rung(_manifest(), cues, board_frames) == "video"
    assert choose_rung(_manifest(), cues, []) == "audio"
    assert choose_rung(_manifest(chapters=chapters), [], []) == "text"
    # The floor: a thumbnail and nothing else. Not an edge case — measured from
    # Fly, InnerTube answers "Sign in to confirm you're not a bot" for most
    # videos, so this is the rung production actually lands on.
    assert choose_rung(_manifest(storyboard=None), [], []) == "metadata"
    # Nothing readable at all, not even a still, is refused rather than studied.
    assert choose_rung(_manifest(storyboard=None, thumbnail_url=""), [], []) == "none"


def test_the_bot_wall_is_named_as_a_block_on_us_not_on_the_video():
    """THE production regression.

    InnerTube refuses datacentre addresses with "Sign in to confirm you're not a
    bot". That was reported as "Private, age-restricted, members-only and
    region-blocked videos all look like this" — four permission states, none of
    which applied, for a video that was public. It sent the reader to check
    settings they did not need to change, on a condition that clears by itself.
    """
    assert _is_bot_wall("Sign in to confirm you’re not a bot") is True
    assert _is_bot_wall("SIGN IN TO CONFIRM you are not a BOT") is True
    # A real permission failure must NOT be softened into "just retry".
    assert _is_bot_wall("This video is private") is False
    assert _is_bot_wall("") is False


def test_a_metadata_rung_manifest_reports_the_cause_before_anything_else():
    """The receipt leads with why it degraded, because that is the only line
    that tells the reader whether there is anything to do."""
    blocked = _manifest(
        storyboard=None, client="oembed", blocked_reason="Sign in to confirm you’re not a bot"
    )
    envelope = manifest_envelope(blocked, [], [])
    assert envelope["rung"] == "metadata"
    assert envelope["degraded"] is True
    assert envelope["note"].startswith("YouTube refused to describe this video")
    assert "rate-limiting us" in envelope["note"]
    # The unrelated explanations must not also appear and confuse the cause.
    assert "too short" not in envelope["note"]
    assert "no captions published" not in envelope["note"]


def test_a_single_keyframe_is_not_a_temporal_study():
    """One frame is a picture. Two is a sequence. The floor matters because
    every temporal statistic downstream presumes an ordering to measure."""
    tiny = _storyboard_from_spec(
        "https://x.test/sb/$L/$N.jpg?a=b|160#90#1#1#1#2000#M$M#rs$SIG", duration_s=2
    )
    one = plan_keyframes(_manifest(storyboard=tiny, duration_s=2))
    assert len(one) == 1
    assert choose_rung(_manifest(storyboard=tiny), [], one) != "video"


def test_one_chapter_is_not_a_running_order():
    """A single named section says nothing about structure, so sampling falls
    back to an even split rather than putting every frame in one place."""
    frames = plan_keyframes(_manifest(chapters=[{"start_s": 0, "end_s": 213, "title": "All"}]))
    assert len(frames) == 8
    assert all(f["source"] == "even" for f in frames)


# ── caption tracks ─────────────────────────────────────────────────────────


def test_human_written_captions_beat_machine_ones_even_across_languages():
    """An ASR track is a machine's guess at what was said; a manual track is
    what someone typed. For a study that reads the words AS CONTENT, that is
    worth more than the language matching."""
    tracks = [
        CaptionTrack("en", "English (auto)", auto_generated=True, url="u1"),
        CaptionTrack("de", "Deutsch", auto_generated=False, url="u2"),
    ]
    assert pick_caption_track(tracks).url == "u2"


def test_preferred_language_wins_among_equals():
    tracks = [
        CaptionTrack("ja", "Japanese", auto_generated=False, url="u1"),
        CaptionTrack("en", "English", auto_generated=False, url="u2"),
    ]
    assert pick_caption_track(tracks).url == "u2"
    assert pick_caption_track([]) is None


def test_json3_cues_are_joined_and_unwrapped():
    raw = """{"events":[
      {"tStartMs":0,"segs":[{"utf8":"Hello"},{"utf8":" there"}]},
      {"tStartMs":1500,"segs":[{"utf8":"line one\\nline two"}]},
      {"tStartMs":3000,"segs":[{"utf8":"  "}]}
    ]}"""
    cues = _parse_json3(raw)
    # The blank cue is dropped; the wrapped one becomes a single line, because
    # `\n` inside a cue is the caption renderer wrapping, not a sentence break.
    assert cues == [
        {"t_ms": 0, "text": "Hello there"},
        {"t_ms": 1500, "text": "line one line two"},
    ]


def test_both_xml_caption_dialects_parse_to_the_same_shape():
    """srv1 counts in seconds, srv3 in milliseconds. Reading one as the other
    puts the whole transcript at the wrong point on the timeline."""
    srv1 = (
        '<transcript><text start="1.5" dur="2">Hello &amp; welcome</text>'
        '<text start="4" dur="1">Bye</text></transcript>'
    )
    assert _parse_timedtext_xml(srv1) == [
        {"t_ms": 1500, "text": "Hello & welcome"},
        {"t_ms": 4000, "text": "Bye"},
    ]

    srv3 = '<timedtext><body><p t="1500" d="2000">Hello</p><p t="4000">Bye</p></body></timedtext>'
    assert _parse_timedtext_xml(srv3) == [
        {"t_ms": 1500, "text": "Hello"},
        {"t_ms": 4000, "text": "Bye"},
    ]


# ── the oEmbed fallback ────────────────────────────────────────────────────


def test_innertube_refusal_falls_back_to_the_public_preview(monkeypatch):
    """THE production fix, exercised without a network.

    Measured from the deployment host: InnerTube answers "Sign in to confirm
    you're not a bot" for most videos, while oEmbed answers 200 for all of
    them. Before this fallback existed, that combination produced a 502 for a
    public video — so the feature worked in development and failed in
    production, which is the worst shape a failure can take.
    """
    import app.youtube as yt

    calls: list[str] = []

    async def fake_fetch_json(url, *, body=None, user_agent=None, max_bytes=0):
        calls.append(url)
        if "youtubei" in url:
            # What the bot wall actually returns: HTTP 200, with the refusal
            # inside `playabilityStatus`. A status-code check would miss it.
            return (
                {
                    "playabilityStatus": {
                        "status": "LOGIN_REQUIRED",
                        "reason": "Sign in to confirm you’re not a bot",
                    },
                    "videoDetails": {},
                },
                url,
            )
        return (
            {
                "title": "Me at the zoo",
                "author_name": "jawed",
                "thumbnail_url": "https://i.ytimg.com/vi/jNQXAC9IVRw/hqdefault.jpg",
            },
            url,
        )

    slept: list[float] = []

    async def no_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(yt, "fetch_json", fake_fetch_json)
    # Patched so the suite does not actually wait out the backoff. The DELAYS
    # are asserted instead, which is the part that could regress.
    monkeypatch.setattr(yt.asyncio, "sleep", no_sleep)
    manifest = asyncio_run(yt.fetch_manifest("jNQXAC9IVRw"))

    assert manifest.client == "oembed"
    assert manifest.title == "Me at the zoo"
    assert manifest.author == "jawed"
    assert manifest.thumbnail_url.endswith("hqdefault.jpg")
    # Every client, THEN the backoff retries, before settling for the preview.
    # Measured from the production host, the wall is rate-based rather than
    # per-video — so giving up on the first refusal permanently downgrades a
    # study that a retry three seconds later would have served in full.
    assert sum(1 for c in calls if "youtubei" in c) == len(yt._CLIENTS) + len(yt._RETRY_DELAYS)
    assert slept == list(yt._RETRY_DELAYS)
    # Duration is NOT invented. Every downstream timestamp derives from it.
    assert manifest.duration_s == 0
    assert yt._is_bot_wall(manifest.blocked_reason)

    envelope = yt.manifest_envelope(manifest, [], [])
    assert envelope["rung"] == "metadata"
    assert "rate-limiting us" in envelope["note"]


def test_a_real_permission_failure_is_not_softened_into_retry_advice(monkeypatch):
    """A private video and a rate-limited server look similar and are not.

    Telling someone to retry a video they do not have access to is a loop.
    """
    import app.youtube as yt

    async def fake_fetch_json(url, *, body=None, user_agent=None, max_bytes=0):
        if "youtubei" in url:
            return ({"playabilityStatus": {"status": "ERROR", "reason": "This video is private"}, "videoDetails": {}}, url)
        return ({"title": "Private thing", "thumbnail_url": "https://i.ytimg.com/vi/x/hqdefault.jpg"}, url)

    monkeypatch.setattr(yt, "fetch_json", fake_fetch_json)
    manifest = asyncio_run(yt.fetch_manifest("jNQXAC9IVRw"))
    note = yt.manifest_envelope(manifest, [], [])["note"]
    assert "This video is private" in note
    assert "rate-limiting" not in note


def test_everything_failing_still_raises(monkeypatch):
    """The fallback must not turn a dead link into a confident empty study."""
    import app.youtube as yt

    async def fake_fetch_json(url, *, body=None, user_agent=None, max_bytes=0):
        raise yt.FetchFailed("nope")

    monkeypatch.setattr(yt, "fetch_json", fake_fetch_json)
    with pytest.raises(yt.YouTubeUnavailable):
        asyncio_run(yt.fetch_manifest("jNQXAC9IVRw"))


def test_a_permanent_refusal_is_not_retried(monkeypatch):
    """Backoff is for the RATE LIMITER only.

    A private, deleted or region-blocked video answers identically however many
    times it is asked, so retrying spends five seconds of an interactive
    request to reach the same refusal. The reason string is what separates the
    transient case from the permanent one.
    """
    import app.youtube as yt

    slept: list[float] = []

    async def no_sleep(seconds):
        slept.append(seconds)

    async def fake_fetch_json(url, *, body=None, user_agent=None, max_bytes=0):
        if "youtubei" in url:
            return ({"playabilityStatus": {"status": "ERROR", "reason": "This video is private"},
                     "videoDetails": {}}, url)
        return ({"title": "T", "thumbnail_url": "https://i.ytimg.com/vi/x/hqdefault.jpg"}, url)

    monkeypatch.setattr(yt, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(yt.asyncio, "sleep", no_sleep)
    yt_manifest = asyncio_run(yt.fetch_manifest("jNQXAC9IVRw"))

    assert yt_manifest.client == "oembed"
    assert slept == []   # no backoff burned on a permanent condition
