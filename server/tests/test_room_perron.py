"""Known-answer tests for the room's centrality, on the four shapes an
influence matrix can actually take.

Kept separate from `test_room.py` because these pin one specific correction and
its reasoning: uniqueness of the stationary distribution turns on the number of
CLOSED COMMUNICATING CLASSES, not on strong connectivity. Irreducibility is
sufficient and not necessary, and testing for it refused the single most
valuable configuration this instrument can find — one member who listens to
nobody while everyone listens to them.

Before the fix, that room (reducible, one closed class) returned
`centrality: None` with the note "read this as a room that has split", which was
both a refusal and a wrong description of a room that had not split at all. Two
further consequences of the same confusion: the consensus forecast was gated on
`primitive` so it was withheld, and the Cesàro average — which exists only to
tame a PERIODIC iterate — was applied to a perfectly convergent one, averaging
the transient into the limit and reporting 0.9777 for a seat whose true weight
is exactly 1.0.
"""

import pytest

from app.room import influence_centrality


def test_a_room_of_equals_converges_to_the_plain_mean():
    """DeGroot's baseline: if everyone weights everyone equally the consensus is
    the unweighted mean of the opening ballots, and centrality is flat."""
    n = 4
    w = [[1.0 / n] * n for _ in range(n)]
    r = influence_centrality(w, innate=[0.1, 0.3, 0.7, 0.9])
    assert r["centrality"] == pytest.approx([0.25] * 4, abs=1e-9)
    assert r["consensus_forecast"] == pytest.approx(0.5, abs=1e-9)
    assert r["converges"] is True
    assert r["irreducible"] is True


def test_an_oracle_everyone_defers_to_carries_the_whole_room():
    """Reducible, ONE closed class — the case the irreducibility test threw away.

    Member 0 listens only to themselves; everyone else reaches them. There is
    exactly one closed class, {0}, so the stationary distribution is unique and
    sits entirely on it: the room's consensus is the oracle's own opinion,
    whatever anybody else believed at the ballot. That is a finding, not a
    degenerate case, and it is the one a real boardroom most needs told.
    """
    w = [
        [1.00, 0.00, 0.00, 0.00],
        [0.50, 0.50, 0.00, 0.00],
        [0.40, 0.20, 0.40, 0.00],
        [0.30, 0.30, 0.20, 0.20],
    ]
    r = influence_centrality(w, innate=[0.9, 0.2, 0.3, 0.4])
    assert r["centrality"] == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-6)
    # exactly the oracle's own position — not a blend, and not the 0.885 the
    # Cesàro average produced by folding in the transient
    assert r["consensus_forecast"] == pytest.approx(0.9, abs=1e-4)
    assert r["converges"] is True
    assert r["irreducible"] is False  # reducible, and still unique
    assert r["closed_groups"] == [[0]]
    assert "listen to nobody outside themselves" in (r["note"] or "")


def test_two_cliques_that_ignore_each_other_have_no_single_centrality():
    """TWO closed classes ⇒ genuinely non-unique. Here the refusal is correct,
    and 'the room has split into cliques' is the right description."""
    w = [
        [0.5, 0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.5],
        [0.0, 0.0, 0.5, 0.5],
    ]
    r = influence_centrality(w, innate=[0.2, 0.3, 0.8, 0.9])
    assert r["centrality"] is None
    assert r["consensus_forecast"] is None
    assert r["converges"] is False
    assert sorted(r["closed_groups"]) == [[0, 1], [2, 3]]
    assert "split into cliques" in (r["note"] or "")


def test_a_periodic_room_reports_its_average_and_refuses_a_forecast():
    """Two members who each adopt only the other's position oscillate forever.

    A unique stationary distribution exists, so centrality is reportable as the
    Cesàro limit — but opinions cycle rather than settle, so there is no
    consensus to forecast and `converges` must say so. This is the one case the
    running mean is FOR.
    """
    r = influence_centrality([[0.0, 1.0], [1.0, 0.0]], innate=[0.2, 0.8])
    assert r["centrality"] == pytest.approx([0.5, 0.5], abs=1e-6)
    assert r["converges"] is False
    assert r["consensus_forecast"] is None


def test_centrality_needs_no_innate_vector_to_rank_members():
    """`innate` is keyword-only and optional: the ranking is a property of W
    alone, and the forecast is the only thing that needs the ballots. Absent
    them the field is None rather than invented."""
    w = [[0.6, 0.4], [0.2, 0.8]]
    r = influence_centrality(w)
    assert r["centrality"] is not None
    assert r["consensus_forecast"] is None


def test_no_matrix_is_a_refusal_not_a_crash():
    r = influence_centrality(None)
    assert r["centrality"] is None
    assert "nothing to rank" in (r["note"] or "")
