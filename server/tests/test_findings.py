"""The findings engine: harvest → FDR control → rank → citation-validated synthesis.

The model half needs an API key, so it is not exercised here. Everything that
DEFENDS the output — evidence extraction, multiplicity control, and the
citation validation that stops a fluent model from inventing a measurement —
is pure and is tested exhaustively, because that is the half a wrong answer
would come through.
"""

import asyncio

import pytest

from app.findings import (
    Evidence,
    build_findings,
    control,
    harvest,
    rank,
    validate_report,
)
from app.schemas import Finding, FindingsReport


def _content_result(**over) -> dict:
    base = {
        "twin_count": 12,
        "cognitive_load": "low",
        "segments": ["Opening shot", "The problem", "The product", "The offer"],
        "curves": {
            "attention": [0.8, 0.75, 0.4, 0.5],
            "valence": [0.3, -0.2, 0.4, 0.35],
            "arousal": [0.6, 0.5, 0.55, 0.5],
            "novelty": [0.7, 0.4, 0.6, 0.3],
            "goal_relevance": [0.5, 0.8, 0.7, 0.6],
            "social_resonance": [0.4, 0.3, 0.5, 0.35],
            "threat": [0.1, 0.5, 0.1, 0.1],
            "reward_anticipation": [0.3, 0.4, 0.8, 0.75],
            "cognitive_effort": [0.2, 0.35, 0.5, 0.3],
        },
        "curve_cis": {"attention": [[0.7, 0.9], [0.65, 0.85], [0.3, 0.5], [0.4, 0.6]]},
        "isc": {
            "overall": 0.62, "ci": [0.4, 0.78], "p_value": 0.002,
            "n_twins": 12, "grip": "locked-in",
        },
        "change_points": [2],
        "memory": {
            "peak_agreement": 0.75, "n_twins": 12, "encoding_strength": 0.71,
            "encoding_strength_ci": [0.6, 0.8], "remembered_affect": 0.28,
            "remembered_affect_ci": [0.1, 0.45], "peak_index": 0,
        },
        "retention": {"completion_rate": 0.66, "worst_beat": 2},
        "trajectory": {"dynamism": 1.2, "net_valence_shift": 0.05},
        "share_intent_mean": 0.42,
        "share_intent_ci": [0.3, 0.55],
    }
    base.update(over)
    return base


# ───────────────────────────── harvest


def test_harvest_pulls_statistics_with_their_uncertainty():
    """A number harvested without its interval is the whole failure mode."""
    evidence = {e.id: e for e in harvest("content", _content_result())}
    assert "isc" in evidence
    isc = evidence["isc"]
    assert isc.value == pytest.approx(0.62)
    assert isc.ci == [0.4, 0.78]
    assert isc.p_value == pytest.approx(0.002)
    assert isc.n == 12
    assert isc.interpretation  # never harvested bare


def test_harvest_skips_blocks_the_study_did_not_compute():
    """A spatial asset has no retention and no peak-end. Absent must mean
    absent, not zero — a zero would be rendered as a measured finding."""
    result = _content_result(retention=None, memory=None, trajectory=None)
    ids = {e.id for e in harvest("content", result)}
    assert "isc" in ids
    assert "completion" not in ids
    assert "encoding" not in ids
    assert "dynamism" not in ids


def test_harvest_derives_beat_comparisons_against_the_simultaneous_band():
    """The weakest beat is only a FINDING if its band clears the curve mean.

    Every curve has a lowest point; saying so is not a discovery. Beat 3 here
    sits at 0.40 with a band of [0.30, 0.50] against a curve mean of 0.6125, so
    it genuinely separates.
    """
    evidence = {e.id: e for e in harvest("content", _content_result())}
    weak = evidence["weakbeat"]
    assert "#3" in weak.label
    assert weak.supported is True
    assert "The product" in weak.interpretation


def test_beat_is_not_a_finding_when_its_band_overlaps_the_mean():
    flat = _content_result(
        curves={**_content_result()["curves"], "attention": [0.60, 0.61, 0.59, 0.60]},
        curve_cis={"attention": [[0.4, 0.8]] * 4},
    )
    evidence = {e.id: e for e in harvest("content", flat)}
    assert evidence["weakbeat"].supported is False
    assert "without being distinguishable" in evidence["weakbeat"].interpretation


def test_harvest_covers_every_study_kind():
    """A kind with no probe table would silently produce an empty report."""
    from app.findings import PROBES

    for kind in (
        "content", "swarm", "compare", "walk", "price",
        "objection", "virality", "optimize", "sequence", "room",
    ):
        assert PROBES.get(kind), f"{kind} has no probes"


def test_harvest_of_a_swarm_reads_segment_evidence():
    result = {
        "twin_count": 20, "mean_intent": 0.62, "intent_ci": [0.5, 0.72],
        "mean_engagement": 0.7, "engagement_ci": [0.6, 0.8],
        "polarization": 0.52, "bimodality": 0.61, "twins_failed": 0,
        "segments": {"k": 2, "gap_vs_k1": 2.4, "silhouette": 0.6},
    }
    evidence = {e.id: e for e in harvest("swarm", result)}
    assert evidence["segk"].to_dict()["display"] == "2"
    assert evidence["seggap"].value == pytest.approx(2.4)
    # the gap-vs-noise number must ride along with k, or two coloured groups
    # render identically whether they cleared the null by 0.2 SEs or 4
    assert "standard errors" in evidence["seggap"].interpretation


# ───────────────────────────── multiplicity control


def test_control_corrects_across_the_whole_family():
    """One p=0.04 among twenty tests is what a 5% test looks like on noise.

    Uncorrected it is "significant" and gets written up. Under BH the bar for
    the smallest of twenty p-values is 0.05/20 = 0.0025, so it does not
    survive — which is the entire point, since a study that runs twenty tests
    finds one of these every time.

    (Four tests all at 0.04 would legitimately ALL be rejected by BH, because
    p₍₄₎ = 0.04 ≤ 4α/4. BH controls the false-discovery *rate*, not the
    family-wise error rate, and that difference is deliberate — see the note on
    `benjamini_hochberg`.)
    """
    evidence = [Evidence(id="lucky", label="the one that fired", value=0.5, p_value=0.04)]
    evidence += [
        Evidence(id=f"n{i}", label=f"stat {i}", value=0.5, p_value=0.3 + i * 0.03)
        for i in range(19)
    ]
    bh = control(evidence)
    assert bh["n_tests"] == 20
    assert not any(e.supported for e in evidence)
    assert "once corrected for 20 simultaneous tests" in evidence[0].support_reason


def test_control_keeps_a_genuinely_strong_result():
    evidence = [Evidence(id="strong", label="isc", value=0.7, p_value=0.0001)] + [
        Evidence(id=f"n{i}", label="noise", value=0.1, p_value=0.6) for i in range(6)
    ]
    control(evidence)
    assert evidence[0].supported is True
    assert not any(e.supported for e in evidence[1:])


def test_control_falls_back_to_the_interval_when_there_is_no_test():
    clear = Evidence(id="a", label="lift", value=0.3, ci=[0.1, 0.5])
    spans = Evidence(id="b", label="lift", value=0.1, ci=[-0.2, 0.4])
    control([clear, spans])
    assert clear.supported is True and "excludes zero" in clear.support_reason
    assert spans.supported is False and "spans zero" in spans.support_reason


def test_zero_width_interval_is_not_treated_as_certainty():
    """Unanimity among 12 quantised twins is an absence of observed variation,
    not proof. Calling it supported would make agreement look like evidence."""
    ev = Evidence(id="u", label="share", value=0.5, ci=[0.5, 0.5])
    control([ev])
    assert ev.supported is False
    assert "no observed variation" in ev.support_reason


def test_descriptive_rows_are_never_called_findings():
    ev = Evidence(id="d", label="dynamism", value=1.2)
    control([ev])
    assert ev.supported is False
    assert "descriptive" in ev.support_reason


# ───────────────────────────── ranking


def test_ranking_promotes_evidence_relevant_to_the_question():
    evidence = harvest("content", _content_result())
    control(evidence)
    ordered = rank(evidence, "why do people stop watching, and where?")
    top = [e.id for e in ordered[:4]]
    assert any(t in top for t in ("completion", "worstbeat", "weakbeat"))


def test_ranking_is_deterministic():
    a = harvest("content", _content_result())
    b = harvest("content", _content_result())
    control(a), control(b)
    assert [e.id for e in rank(a, "q")] == [e.id for e in rank(b, "q")]


def test_ranking_never_drops_evidence():
    """Ordering decides emphasis, not inclusion — a filter here would let the
    ranking quietly hide a contradicting measurement from the synthesis."""
    evidence = harvest("content", _content_result())
    control(evidence)
    assert len(rank(evidence, "pricing anxiety")) == len(evidence)


def _row(id: str, label: str, interp: str) -> Evidence:
    ev = Evidence(id=id, label=label, value=0.5, interpretation=interp, where="w")
    ev.supported, ev.magnitude = True, 0.5
    return ev


def test_question_terms_reach_the_instrument_vocabulary():
    """The register gap, which is the one that actually occurs.

    Nobody asks about `threat` or `reward_anticipation`; they ask whether the
    pricing makes people anxious. Before the concept map, the row that answers
    that question scored no higher than the row about cascade size, so the
    ordering carried no information about the question at all.
    """
    rows = [
        _row("csize", "expected cascade size", "a typical share reaches 3.2 people"),
        _row("isc", "attention synchrony (ISC)", "the audience attends together"),
        _row("threat", "threat at beat 3", "unease peaks in the pricing block"),
        _row("elast", "price elasticity of demand", "demand falls per price rise"),
    ]
    top2 = {e.id for e in rank(rows, "does the pricing create anxiety for buyers?")[:2]}
    assert top2 == {"threat", "elast"}


def test_ranking_survives_english_spelling_changes_at_the_suffix():
    """"dropped" must reach drop-off and "buyers" must reach intent. Each of
    these is a regular English spelling change at the suffix boundary, and each
    one silently produced a stem that matched nothing before it was undone."""
    rows = [
        _row("novel", "mean novelty", "how fresh this felt"),
        _row("drop", "steepest drop-off beat", "where retention falls fastest"),
    ]
    assert rank(rows, "where do viewers get dropped?")[0].id == "drop"

    rows2 = [
        _row("novel", "mean novelty", "how fresh this felt"),
        _row("intent", "mean intent", "stated intent to act"),
    ]
    assert rank(rows2, "what do buyers intend to do?")[0].id == "intent"


def test_relevance_is_a_fraction_so_padding_a_question_cannot_flatten_it():
    """A hit COUNT rewards long questions for being long: six terms matching two
    scored the same as two matching two, so every row's relevance rose together
    and the ordering it was meant to sharpen went flat."""
    rows = [
        _row("threat", "threat at beat 3", "unease peaks here"),
        _row("isc", "attention synchrony (ISC)", "the audience attends together"),
    ]
    short = rank(rows, "anxiety?")
    padded = rank(rows, "anxiety, and also whatever else might possibly matter here")
    assert short[0].id == padded[0].id == "threat"


def test_a_question_of_pure_stopwords_does_not_divide_by_zero():
    rows = [_row("a", "mean intent", "x"), _row("b", "mean novelty", "y")]
    assert len(rank(rows, "what does this all about?")) == 2
    assert len(rank(rows, "")) == 2


# ───────────────────────────── citation validation


def _finding(**over) -> Finding:
    base = dict(
        headline="Beat 3 loses the audience",
        detail="Attention collapses.",
        evidence_ids=["weakbeat"],
        direction="risk",
        confidence="high",
        answers_question=True,
        recommended_action="Re-cut beat 3.",
    )
    base.update(over)
    return Finding(**base)


def test_uncited_findings_are_dropped():
    """The enforcement half of the citation rule. A model asked to cite will
    mostly cite, and 'mostly' is not a property you can put behind a number
    someone is about to act on."""
    evidence = harvest("content", _content_result())
    control(evidence)
    report = FindingsReport(
        answer="…",
        findings=[
            _finding(evidence_ids=["weakbeat"]),
            _finding(headline="Invented", evidence_ids=["does_not_exist"]),
            _finding(headline="Also invented", evidence_ids=[]),
        ],
        what_would_change_the_answer="…",
        limits="…",
    )
    out = validate_report(report, evidence)
    assert len(out["findings"]) == 1
    assert out["findings"][0]["headline"] == "Beat 3 loses the audience"
    assert out["dropped_uncited"] == 2


def test_partially_cited_findings_keep_only_resolvable_ids():
    evidence = harvest("content", _content_result())
    control(evidence)
    report = FindingsReport(
        answer="…",
        findings=[_finding(evidence_ids=["isc", "ghost", "weakbeat"])],
        what_would_change_the_answer="…",
        limits="…",
    )
    out = validate_report(report, evidence)
    assert out["findings"][0]["evidence_ids"] == ["isc", "weakbeat"]


def test_findings_built_only_on_unsupported_evidence_are_downgraded():
    """A confident finding standing entirely on rows that did not clear FDR is
    the exact overclaim this module exists to prevent, and it reads as
    authoritative unless it is marked."""
    evidence = [
        Evidence(id="weak", label="a thing", value=0.1, p_value=0.9),
        Evidence(id="alsoweak", label="another", value=0.1, p_value=0.8),
    ]
    control(evidence)
    report = FindingsReport(
        answer="…",
        findings=[_finding(evidence_ids=["weak", "alsoweak"], confidence="high")],
        what_would_change_the_answer="…",
        limits="…",
    )
    out = validate_report(report, evidence)
    assert out["findings"][0]["confidence"] == "low"
    assert "consistent with chance" in out["findings"][0]["downgraded"]


def test_a_finding_with_one_supported_row_keeps_its_confidence():
    evidence = [
        Evidence(id="strong", label="isc", value=0.7, p_value=0.00001),
        Evidence(id="weak", label="noise", value=0.1, p_value=0.9),
    ]
    control(evidence)
    report = FindingsReport(
        answer="…",
        findings=[_finding(evidence_ids=["strong", "weak"], confidence="high")],
        what_would_change_the_answer="…",
        limits="…",
    )
    out = validate_report(report, evidence)
    assert out["findings"][0]["confidence"] == "high"


# ───────────────────────────── end to end (no API key)


def test_build_findings_returns_evidence_even_when_synthesis_fails():
    """A completed study is worth returning even if the interpretation on top
    of it could not run. Without an API key the model call raises, and the
    evidence table is useful on its own."""
    out = asyncio.run(
        build_findings("content", _content_result(), question="where do we lose people?")
    )
    assert out["report"] is None
    assert "error" in out
    assert len(out["evidence"]) > 5
    assert out["question"] == "where do we lose people?"
    assert out["multiplicity"]["n_tests"] >= 1
    assert out["n_supported"] >= 1
    # every row carries its own provenance, not just a number
    assert all("interpretation" in row and "supported" in row for row in out["evidence"])


def test_build_findings_on_an_empty_result_refuses_rather_than_inventing():
    out = asyncio.run(build_findings("content", {}))
    assert out["report"] is None
    assert "no harvestable quantitative claims" in out["error"]
    assert out["evidence"] == []


# ───────────────────────────── the white room


def _room_result(analysis_over=None, **over) -> dict:
    """Shaped like what `boardroom.stream_deliberation` actually emits: the
    math lives under `analysis`, and `instrument_check` sits beside it.

    Written against the REAL key names, which differ from the obvious guesses in
    ways that fail silently — `room_mean_gap` not `room_gap`, `gap_ci` not
    `room_gap_ci`, `conformity_ratio` not `ratio`, `eta_squared` not
    `effect_size`, `absence` not `counterfactuals`, and `per_member_gap` is a
    list of floats aligned with `names` rather than a list of dicts. A probe
    pointing at a name that does not exist harvests nothing and looks exactly
    like a study with no findings, so these fixtures are deliberately literal.
    """
    analysis = {
        "motion": "Ship the migration in Q3 or defer to Q1.",
        "names": ["Priya", "Dan"],
        "roster": [
            {"name": "Priya", "final_public": 0.80, "final_private": 0.49},
            {"name": "Dan", "final_public": 0.60, "final_private": 0.65},
        ],
        "falsification": {
            "refused": None,
            "names": ["Priya", "Dan"],
            "per_member_gap": [0.31, -0.05],
            "room_mean_gap": 0.13,
            "gap_ci": [0.04, 0.22],
            "p_value": 0.01,
        },
        "influence": {"refused": None, "names": ["Priya", "Dan"], "n_obs_per_member": 12},
        "centrality": {
            "centrality": [0.72, 0.28],
            "spectral_gap": 0.44,
            "consensus_forecast": 0.71,
        },
        "absence": [
            {"name": "Priya", "delta_room_mean": -0.22},
            {"name": "Dan", "delta_room_mean": 0.03},
        ],
        "conformity": {"conformity_ratio": 2.4, "observed_log_contraction": 0.48},
        "order_effect": {"eta_squared": 0.04, "eta_p_value": 0.6},
        # separation is the server's own categorical; only "clear …" licenses an
        # influence claim, and convergence_* is spread REMAINING (lower = harder)
        "placebo": {
            "ran": True,
            "separation": "clear — the real arm converged further",
            "convergence_real": 0.30,
            "convergence_placebo": 0.92,
        },
    }
    analysis.update(analysis_over or {})
    base = {
        "motion_preview": analysis["motion"],
        "cast": [
            {"name": "Priya", "role": "CFO", "stake": "owns the budget",
             "openness": 0.2, "seniority": 0.9},
            {"name": "Dan", "role": "PM", "stake": "owns the date",
             "openness": 0.7, "seniority": 0.3},
        ],
        "instrument_check": {"falsification_below_resolution": False, "usable": True},
        "analysis": analysis,
    }
    base.update(over)
    return base


def test_room_harvest_reads_the_deliberation_not_the_stimulus():
    ev = harvest("room", _room_result())
    ids = {e.id for e in ev}
    # the room-level controls and the headline gap
    assert {"roomgap", "conform", "placebogap", "orderfx"} <= ids
    # and the per-member row, named — an unnamed boardroom finding is useless
    assert any(e.id.startswith("fals_") and "Priya" in e.label for e in ev)


def test_a_failed_placebo_arm_withholds_every_influence_claim():
    """The control that decides whether the study measured anything.

    A language model is agreeable. If members shown statements from a DIFFERENT
    room converge just as hard, what looked like persuasion was disposition, and
    no claim about who moved whom is admissible — so those rows must not reach
    the synthesis at all. Emitting them with a caveat is not enough: the caveat
    is prose the model may drop, while a missing evidence id is a citation it
    structurally cannot make.
    """
    dead = _room_result(
        analysis_over={
            "placebo": {
                "ran": True,
                "separation": "none — the placebo converged at least as hard",
                "convergence_real": 0.30,
                "convergence_placebo": 0.29,
            }
        }
    )
    ids = {e.id for e in harvest("room", dead)}
    assert not any(i.startswith("cent_") or i.startswith("cf_") for i in ids)
    # the gap and the controls are still reported — that is the finding now
    assert {"roomgap", "placebogap"} <= ids

    alive = _room_result()
    live_ids = {e.id for e in harvest("room", alive)}
    assert any(i.startswith("cent_") for i in live_ids)
    assert any(i.startswith("cf_") for i in live_ids)


def test_room_evidence_survives_a_refused_influence_fit():
    """The fit refuses when it is not identifiable, and a refusal must not
    become an exception or a fabricated network."""
    r = _room_result(
        analysis_over={
            "influence": {
                "refused": "4 observations per member, 6 parameters",
                "names": [],
            }
        }
    )
    ids = {e.id for e in harvest("room", r)}
    assert not any(i.startswith("cent_") or i.startswith("cf_") for i in ids)
    assert "roomgap" in ids


def test_room_context_block_names_the_cast():
    from app.findings import _context_block

    # Deliberately NOT setting a top-level "motion": boardroom emits
    # `motion_preview` there and keeps the full text under `analysis`, so this
    # pins that the block finds it without one.
    block = _context_block("room", _room_result(), "")
    assert "migration" in block
    assert "Priya" in block and "CFO" in block
    assert "MOTION PUT TO THE ROOM" in block
    # the declared/fitted distinction must be stated to the model
    assert "declared" in block.lower()


def test_an_unresolvable_gap_is_withheld_not_caveated():
    """The most quotable thing this study could say must clear its floor first.

    Measured on a deliberately hostile cast, mean |public − private| came out
    0.013 with members conditioned on themselves alone and 0.022 once each was
    told who outranked them — against a reporting granularity of about 0.05.
    That is not a small effect, it is no measurement, and a 0.02 gap shipped
    with an interval reads exactly like a real one. So the run's own instrument
    check vetoes the row rather than the synthesis being asked to caveat it: a
    caveat is prose a model may drop, a missing evidence id is a citation it
    cannot make.
    """
    unresolvable = _room_result(
        instrument_check={"falsification_below_resolution": True, "usable": True}
    )
    ids = {e.id for e in harvest("room", unresolvable)}
    assert "roomgap" not in ids
    assert not any(i.startswith("fals_") for i in ids)
    # the rest of the instrument is untouched — one unresolvable output must not
    # discard a study whose other four are fine
    assert {"conform", "placebogap", "orderfx"} <= ids
    assert any(i.startswith("cent_") for i in ids)

    resolvable = _room_result(
        instrument_check={"falsification_below_resolution": False, "usable": True}
    )
    live = {e.id for e in harvest("room", resolvable)}
    assert "roomgap" in live
    assert any(i.startswith("fals_") for i in live)
