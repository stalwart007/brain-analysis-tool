"""The modality layer, and the axis rules that keep it honest.

The whole point of this layer is that a number computed on the wrong axis is
not slightly wrong, it is a category error. These tests are mostly about what
the system REFUSES to compute.
"""

from __future__ import annotations

import base64

import pytest

from app.modality import (
    AXIS_SUPPORTS,
    BeatAxis,
    BeatSequence,
    UnsupportedAsset,
    _beats_from_document,
    _beats_from_page,
    _sentence_beats,
    extract_beats,
)
from app.schemas import ContentAsset


# ── axis rules ────────────────────────────────────────────────────────────


def test_a_static_image_refuses_time_dependent_statistics():
    """The regions of an image are SIMULTANEOUS. Peak-end would report the
    valence of whichever region the model happened to list last, change points
    would name a moment that does not exist, and abandonment has no next beat
    to fail to reach."""
    spatial = BeatSequence(["a", "b", "c"], BeatAxis.SPATIAL, source="vision")
    assert not spatial.supports("peak_end")
    assert not spatial.supports("change_points")
    assert not spatial.supports("retention")
    assert not spatial.supports("trajectory")
    # ISC survives: whether twins agree about which regions pull the eye is a
    # real question about the image.
    assert spatial.supports("isc")


def test_temporal_and_sequential_support_the_ordered_statistics():
    for axis in (BeatAxis.TEMPORAL, BeatAxis.SEQUENTIAL):
        seq = BeatSequence(["a", "b", "c"], axis, source="llm")
        for stat in ("peak_end", "change_points", "retention", "trajectory", "isc"):
            assert seq.supports(stat), f"{axis} should support {stat}"


def test_withheld_statistics_carry_a_reason_not_just_absence():
    """An absent number and an inapplicable one look identical in a UI, and
    only one of them is a finding."""
    spatial = BeatSequence(["a", "b"], BeatAxis.SPATIAL, source="vision")
    withheld = spatial.to_dict()["withheld"]
    assert set(withheld) == {"peak_end", "change_points", "retention", "trajectory"}
    assert all(len(reason) > 30 for reason in withheld.values())
    assert BeatSequence(["a"], BeatAxis.TEMPORAL, source="llm").to_dict()["withheld"] == {}


def test_every_axis_declares_its_supported_statistics():
    """A new axis with no entry would silently support nothing."""
    assert set(AXIS_SUPPORTS) == set(BeatAxis)


# ── structural adapters (no LLM) ──────────────────────────────────────────


def test_a_landing_page_becomes_scroll_beats_in_dom_order():
    html = """
      <script>tracker('should not appear')</script>
      <header><h1>Stop paying for twelve subscriptions every single month</h1></header>
      <section><p>One workspace replaces notes, tasks, calendar and email for you.</p></section>
      <section><p>Pricing starts at thirty dollars a month with a free trial week.</p></section>
      <footer><p>Trusted by four thousand teams who switched over this year.</p></footer>
    """
    seq = _beats_from_page(ContentAsset(kind="page", text=html))
    assert seq.axis is BeatAxis.SEQUENTIAL
    assert seq.source == "structure"
    assert len(seq.beats) >= 2
    joined = " ".join(seq.beats)
    assert "should not appear" not in joined, "script contents leaked into beats"
    assert "subscriptions" in joined


def test_a_page_with_no_prose_is_refused_rather_than_invented():
    with pytest.raises(UnsupportedAsset):
        _beats_from_page(ContentAsset(kind="page", text="<div><a>Home</a></div>"))


def test_a_deck_becomes_one_beat_per_slide():
    seq = _beats_from_document(
        ContentAsset(kind="document", pages=["Slide one", "Slide two", "Slide three"])
    )
    assert seq.axis is BeatAxis.SEQUENTIAL
    assert len(seq.beats) == 3


def test_a_one_page_document_is_refused():
    with pytest.raises(UnsupportedAsset):
        _beats_from_document(ContentAsset(kind="document", pages=["only one"]))


# ── the fallback must not truncate the content ────────────────────────────


def test_the_sentence_fallback_covers_the_whole_text():
    """The old fallback was `[content[:200], content[200:400] or "(end)"]` for
    anything without a blank line — a single-paragraph ad, a headline, a
    transcript. The entire study then ran on the first 400 characters and one
    of the two beats could literally be the string "(end)"."""
    text = " ".join(f"Sentence number {i} carries its own distinct meaning." for i in range(40))
    beats = _sentence_beats(text)
    assert 2 <= len(beats) <= 10
    # every sentence must survive somewhere
    assert "number 39" in " ".join(beats)
    assert "(end)" not in " ".join(beats)
    # and nothing is silently dropped
    assert len(" ".join(beats)) >= len(text) * 0.9


def test_a_single_sentence_still_yields_two_beats_without_losing_text():
    beats = _sentence_beats("One unbroken clause with no terminator")
    assert len(beats) == 2
    assert "".join(b.replace(" ", "") for b in beats).startswith("Oneunbroken")


# ── refusals carry actionable reasons ─────────────────────────────────────


def test_assets_missing_their_payload_are_refused_with_a_reason():
    import asyncio

    cases = [
        ContentAsset(kind="image"),
        ContentAsset(kind="video"),
        ContentAsset(kind="audio"),
        ContentAsset(kind="text"),
    ]
    for asset in cases:
        with pytest.raises(UnsupportedAsset) as exc:
            asyncio.run(extract_beats(asset))
        assert len(str(exc.value)) > 20, "a refusal must say what to do about it"


def test_invalid_base64_is_caught_before_it_reaches_the_vision_api():
    """Otherwise a caller's typo becomes an opaque upstream error and a billed
    request."""
    import asyncio

    from app.modality import _data_url

    with pytest.raises(UnsupportedAsset):
        _data_url("not!valid!base64!", "image/png")
    # and a valid one round-trips
    ok = base64.b64encode(b"\x89PNG\r\n").decode()
    assert _data_url(ok, "image/png").startswith("data:image/png;base64,")
