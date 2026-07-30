"""Message Sequence Optimiser — order effects in a multi-message campaign.

Given N messages that will land one after another (a drip series, an onboarding
sequence, the section order of a pitch), find the ORDERING that ends with the
most intent. The interesting part is that this cannot be brute-forced and
cannot be answered by scoring messages individually, so the estimator is built
in three honest layers:

1. **Two-way additive decomposition (position vs content).**
   Every twin walks one ordering and reports intent after each message, so each
   observation is a per-step gain g = intent[p] − intent[p−1] (with intent[−1]
   = 0, which makes the gains sum exactly to the twin's end-state intent — the
   quantity being optimised). We fit

       g(message m at position p) ≈ μ + content[m] + position[p]

   by alternating mean-centering (mean polish: the classic two-way additive fit
   Tukey's median polish is the robust cousin of). This is what stops "message
   4 scored highest" being confused with "message 4 went last" — primacy and
   recency live in `position`, intrinsic strength lives in `content`, and they
   are reported separately.

   Note the consequence, which is the reason layer 2 exists at all: an additive
   model has NO order effects. Σ content is the same for every permutation and
   Σ position is the same for every permutation, so it predicts an identical
   end state for all N! orderings. Everything that makes ordering matter is
   interaction, which is exactly what the precedence matrix measures.

2. **Precedence advantage matrix.**
   P[i][j] = mean position-adjusted gain of message j over walks where i came
   before j, minus the same over walks where it did not. Adjusting for the
   position effect first is essential — otherwise P[i][j] mostly measures that
   j happened to sit late in the orderings where i preceded it. `support`
   reports the thinner of the two cells, so a number resting on a single walk
   is never mistaken for a measurement.

   P is directional but NOT antisymmetric: P[i][j] is about what happens to j,
   P[j][i] is about what happens to i, and both can be positive (two messages
   that help each other whichever way round they go). The *decision* between
   the two orders is the net advantage A[i][j] = P[i][j] − P[j][i] — what you
   gain in j minus what you give up in i — which is antisymmetric by
   construction and is what the objective is built on.

3. **Linear ordering problem.**
   With A in hand, the best ordering maximises Σ_{p<q} A[o[p]][o[q]] — the
   linear ordering problem, equivalently maximum acyclic subgraph, which is
   NP-hard. (Maximising over A and over P are the same problem: the two
   objectives differ by 2× and a constant, since every unordered pair
   contributes one direction or the other. A is used because it is signed —
   zero means "no preference" and reversing an ordering negates the score.)
   We use greedy insertion seeded by net outflow plus pairwise-swap local
   search. That is a heuristic with no approximation guarantee: we report the
   objective value, the number of improving swaps, and the objective of the
   ordering you submitted, so the result can be judged rather than trusted. The
   recommendation is the best of {heuristic solution, submitted ordering, every
   ordering actually walked}, so it can never be worse than something observed.

Design of the sample: the walked orderings are the submitted one, its exact
reverse, then seeded random permutations. Submitted+reverse is not decoration —
together they place every unordered pair {i,j} on both sides of the precedence
relation, which is the minimum condition for any P[i][j] to be estimable.

Honest limits, stated in the result: a disengaged twin stops the walk, so late
positions are estimated from the twins who survived that far (survivorship);
N! is explored to a fraction reported as `explored_fraction`; and the objective
is in precedence-advantage units summed over ordered pairs, not in probability.

`sequence_result` is pure — twin walks in, result dict out, zero network calls.
"""

from __future__ import annotations

import asyncio
import math
import random
from typing import AsyncIterator, Optional, Sequence

import openai

from .analytics import bayesian_ab, bootstrap_ci
from .config import LOAD_TO_TEMPERATURE, SWARM_CONCURRENCY, TWIN_MODEL
from .oai import Refusal, async_client, parse_completion, response_format_for
from .schemas import PersonaSeed, SequenceRequest, TwinSequenceWalk
from .studies import _clamp01, _persona_system

# Seed for the ordering design. Fixed so a re-run of the same request walks the
# same orderings and the two runs are comparable.
_DESIGN_SEED = 97
# Mean-polish passes. The two-way additive fit converges geometrically and is
# exact in one pass for a balanced design; the loop exits on convergence, and
# 12 is only the guard for the ragged designs disengagement produces.
_POLISH_PASSES = 12
# Convergence tolerance for the polish — well below the precision anything
# downstream is rounded to (4 decimal places).
_POLISH_TOL = 1e-9
# Local-search cap. n ≤ 10 means a full swap pass is ≤ 45 candidate orderings at
# O(n²) each; 50 passes can never be reached in practice, it just guarantees
# termination if float noise ever makes an "improvement" cycle.
_MAX_LOCAL_PASSES = 50
# Improvements must beat this to be accepted — pure float-noise guard.
_EPS = 1e-12
# Bootstrap replicates for the lift interval. Same budget virality.py spends on
# its twin bootstrap: enough for a 2.5/97.5 percentile at these sample sizes and
# cheap enough to stay inside one request.
_LIFT_REPS = 200
_LIFT_SEED = 5701
# Permutation replicates for the gain's null distribution. The null is the
# expensive one to get right, so it gets the same budget as the bootstrap.
_PERM_REPS = 200
_PERM_SEED = 88_003
# Repeated split-halves for the cross-fitted point estimate. One split is an
# unbiased but very noisy estimate of the honest gain — it throws away half the
# data for selection and the other half for evaluation. Averaging over repeated
# random splits (and both role assignments within each) recovers most of the
# precision while keeping selection and evaluation disjoint in every term.
_SPLIT_REPS = 12
# A twin "commits" when it finishes the sequence still wanting to act. 0.5 is
# the midpoint of the 0..1 intent scale AND the threshold written into the twin
# prompt ("above 0.5 means you would actually act"), so it is the twins' own
# convention rather than an analyst's constant.
_COMMIT_THRESHOLD = 0.5


# ─────────────────────────────── pure math ───────────────────────────────


def assign_orderings(
    n_personas: int, n_reps: int, n_orderings: int, seed: int = _DESIGN_SEED
) -> list[int]:
    """Assign an ordering to each (persona, repetition) slot.

    Slots are enumerated repetition-major — slot `rep * n_personas + p` belongs
    to persona `p` — so the caller must build its walker list the same way.

    Two properties matter, and the obvious arithmetic schemes get at most one:

    · **Flat coverage.** Every ordering is walked within one of every other,
      including ordering 0, the submitted baseline that every comparison is
      measured against. `(p + rep) % L` failed this — it covered a contiguous
      block and gave the baseline a single walk, so its interval was None.

    · **No persona/ordering aliasing.** `(p + rep * P) % L` confines persona p
      to the coset p + <P> in Z_L, so personas split into gcd(P, L) groups on
      disjoint ordering sets — at P = L = 6, persona p walks only ordering p.
      Every between-ordering number becomes a persona comparison in disguise.

    Round-robin gives the first; shuffling under a fixed seed gives the second.
    Randomisation is not a stylistic choice here — with R repetitions and
    L > R orderings no persona *can* reach every ordering, so an unbiased
    contrast requires the pairing to be random rather than systematic.
    """
    if n_orderings <= 0 or n_personas <= 0 or n_reps <= 0:
        return []
    assignment = [i % n_orderings for i in range(n_personas * n_reps)]
    random.Random(seed).shuffle(assignment)
    return assignment


def sample_orderings(n: int, k: int, seed: int = _DESIGN_SEED) -> list[list[int]]:
    """The experimental design: submitted, reversed, then random permutations.

    Deduplicated, and bounded by n! — asking for 24 orderings of 3 messages can
    only ever return 6, and quietly returning duplicates would inflate the
    apparent coverage of the sample space.
    """
    base = list(range(n))
    out: list[list[int]] = [base[:]]
    if n > 1:
        out.append(list(reversed(base)))
    seen = {tuple(o) for o in out}
    rng = random.Random(seed)
    # Bounded rejection sampling: with n! close to k the tail draws collide
    # constantly, so cap the attempts instead of spinning.
    attempts = 0
    while len(out) < k and attempts < max(50, k * 50):
        attempts += 1
        candidate = base[:]
        rng.shuffle(candidate)
        key = tuple(candidate)
        if key not in seen:
            seen.add(key)
            out.append(candidate)
    return out[: max(1, k)]


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def walk_gains(walk: dict) -> dict:
    """One twin walk → per-step gains, truncated at disengagement.

    A twin that disengages at position d never saw positions d+1…; those
    messages produced no observation and must not be scored as zero gain. The
    gains that remain sum to the twin's end-state intent by construction.
    """
    ordering = list(walk.get("ordering") or [])
    steps = list(walk.get("steps") or [])
    steps = steps[: len(ordering)]

    disengaged_at: Optional[int] = None
    for i, s in enumerate(steps):
        if s.get("disengaged"):
            disengaged_at = i
            steps = steps[: i + 1]
            break

    gains: list[tuple[int, int, float]] = []  # (message, position, gain)
    previous = 0.0
    terminal = 0.0
    fatigue_end = 0.0
    for position, step in enumerate(steps):
        intent = _clamp01(step.get("intent", 0.0))
        gains.append((ordering[position], position, intent - previous))
        previous = intent
        terminal = intent
        fatigue_end = _clamp01(step.get("fatigue", 0.0))
    return {
        "ordering": ordering,
        "gains": gains,
        "terminal_intent": terminal,
        "disengaged_at": disengaged_at,
        "fatigue_end": fatigue_end,
        "steps_seen": len(steps),
    }


def additive_effects(
    observations: Sequence[tuple[int, int, float]], n_messages: int, n_positions: int
) -> tuple[float, list[Optional[float]], list[Optional[float]]]:
    """Fit g ≈ μ + content[m] + position[p] by alternating mean-centering.

    Both effect vectors are centered to mean zero each pass, which is the
    identifiability constraint that makes the split unique (μ carries the
    level). Cells with no observations come back as None — an unobserved
    message-position combination is missing data, and returning 0.0 for it
    would look like "no effect".
    """
    content: list[float] = [0.0] * n_messages
    position: list[float] = [0.0] * n_positions
    if not observations:
        return 0.0, [None] * n_messages, [None] * n_positions

    # Bucket once. Rescanning every observation per cell per pass turned the
    # bootstrap into seconds of blocking CPU inside an async generator, which
    # is the exact failure virality.simulate_cascades documents.
    by_m: list[list[int]] = [[] for _ in range(n_messages)]
    by_p: list[list[int]] = [[] for _ in range(n_positions)]
    for index, (m, p, _) in enumerate(observations):
        by_m[m].append(index)
        by_p[p].append(index)
    seen_m = [bool(b) for b in by_m]
    seen_p = [bool(b) for b in by_p]

    mu = _mean([g for _, _, g in observations])
    for _ in range(_POLISH_PASSES):
        delta = 0.0
        for m in range(n_messages):
            if not by_m[m]:
                continue
            value = _mean(
                [observations[i][2] - mu - position[observations[i][1]] for i in by_m[m]]
            )
            delta = max(delta, abs(value - content[m]))
            content[m] = value
        shift = _mean([content[m] for m in range(n_messages) if seen_m[m]])
        for m in range(n_messages):
            content[m] -= shift
        mu += shift

        for p in range(n_positions):
            if not by_p[p]:
                continue
            value = _mean(
                [observations[i][2] - mu - content[observations[i][0]] for i in by_p[p]]
            )
            delta = max(delta, abs(value - position[p]))
            position[p] = value
        shift = _mean([position[p] for p in range(n_positions) if seen_p[p]])
        for p in range(n_positions):
            position[p] -= shift
        mu += shift

        if delta < _POLISH_TOL:
            break

    return (
        mu,
        [content[m] if seen_m[m] else None for m in range(n_messages)],
        [position[p] if seen_p[p] else None for p in range(n_positions)],
    )


def precedence_matrix(
    infos: Sequence[dict], position_effects: Sequence[Optional[float]], n: int
) -> tuple[list[list[float]], list[list[int]]]:
    """P[i][j] = position-adjusted advantage of j when i precedes it.

    `infos` are `walk_gains` outputs (the caller computes them once; the
    bootstrap resamples them rather than re-deriving them 200 times).

    Row i, column j reads "what i does to j". That is NOT antisymmetric —
    P[j][i] is a statement about a different message's gains, and two messages
    can genuinely help each other in both directions. Use `net_advantage` for
    the ordering decision. `support[i][j]` is min(|with|, |without|): the
    number of walks the *thinner* side of the comparison rests on, which is
    what makes a cell estimated from one walk visible as such.
    """
    # message j → position-adjusted gains, split by whether i preceded it
    adjusted: list[list[tuple[set[int], float]]] = [[] for _ in range(n)]
    for info in infos:
        seen_before: set[int] = set()
        for m, p, g in info["gains"]:
            effect = position_effects[p] if p < len(position_effects) else None
            adjusted[m].append((set(seen_before), g - (effect or 0.0)))
            seen_before.add(m)

    matrix = [[0.0] * n for _ in range(n)]
    support = [[0] * n for _ in range(n)]
    for j in range(n):
        for i in range(n):
            if i == j:
                continue
            with_i = [v for preds, v in adjusted[j] if i in preds]
            without_i = [v for preds, v in adjusted[j] if i not in preds]
            support[i][j] = min(len(with_i), len(without_i))
            # An empty side makes the difference undefined, not zero-advantage;
            # 0.0 is the neutral element for the objective sum, and `support`
            # is what tells the reader the cell was never measured.
            if with_i and without_i:
                matrix[i][j] = _mean(with_i) - _mean(without_i)
    return matrix, support


def net_advantage(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    """A[i][j] = P[i][j] − P[j][i]: the net intent benefit of putting i before j
    rather than after it — what j gains minus what i gives up. Antisymmetric by
    construction, so a zero means the pair's order genuinely does not matter and
    reversing an ordering negates the objective."""
    n = len(matrix)
    return [[matrix[i][j] - matrix[j][i] for j in range(n)] for i in range(n)]


def ordering_objective(order: Sequence[int], matrix: Sequence[Sequence[float]]) -> float:
    """Σ over ordered pairs (earlier, later) of the pairwise advantage. The
    linear-ordering objective; pass the net-advantage matrix for a signed one."""
    return sum(
        matrix[order[a]][order[b]]
        for a in range(len(order))
        for b in range(a + 1, len(order))
    )


def solve_linear_ordering(
    matrix: Sequence[Sequence[float]], max_passes: int = _MAX_LOCAL_PASSES
) -> tuple[list[int], float, dict]:
    """Greedy insertion + pairwise-swap local search over the precedence matrix.

    HEURISTIC. The linear ordering problem is NP-hard; this returns a local
    optimum with no approximation guarantee and the caller must say so. Greedy
    seeds by net outflow (a message that improves everything it precedes
    belongs early), inserting each message at its best position in the partial
    order, then 2-opt style swaps run until a full pass finds no improvement.
    """
    n = len(matrix)
    if n <= 1:
        return list(range(n)), 0.0, {"passes": 0, "improving_swaps": 0}

    by_outflow = sorted(
        range(n),
        key=lambda i: sum(matrix[i]) - sum(matrix[j][i] for j in range(n)),
        reverse=True,
    )
    order: list[int] = []
    for m in by_outflow:
        best_pos, best_val = 0, None
        for pos in range(len(order) + 1):
            candidate = order[:pos] + [m] + order[pos:]
            val = ordering_objective(candidate, matrix)
            if best_val is None or val > best_val:
                best_val, best_pos = val, pos
        order.insert(best_pos, m)

    current = ordering_objective(order, matrix)
    improving = 0
    passes = 0
    while passes < max_passes:
        passes += 1
        improved = False
        for a in range(n):
            for b in range(a + 1, n):
                candidate = order[:]
                candidate[a], candidate[b] = candidate[b], candidate[a]
                val = ordering_objective(candidate, matrix)
                if val > current + _EPS:
                    order, current, improved = candidate, val, True
                    improving += 1
        if not improved:
            break
    return order, current, {"passes": passes, "improving_swaps": improving}


def _percentile_ci(values: Sequence[float], nd: int = 4) -> Optional[list[float]]:
    """95% percentile interval from bootstrap replicates. None below 20
    replicates, where the 2.5th percentile is not a percentile of anything."""
    if len(values) < 20:
        return None
    ordered = sorted(values)
    n = len(ordered)
    return [
        round(ordered[int(0.025 * n)], nd),
        round(ordered[min(n - 1, int(0.975 * n))], nd),
    ]


def _estimate(
    infos: Sequence[dict], n: int
) -> tuple[list[list[float]], list[list[int]], list[Optional[float]], list[Optional[float]], float]:
    """One estimation pass over `walk_gains` outputs: additive effects, then
    the precedence matrix. Factored out so the bootstrap re-runs the *whole*
    estimator per replicate rather than resampling a statistic fitted once —
    position effects and the matrix are both re-estimated from the resample."""
    observations = [obs for info in infos for obs in info["gains"]]
    mu, content, position = additive_effects(observations, n, n)
    matrix, support = precedence_matrix(infos, position, n)
    return matrix, support, content, position, mu


def _select_ordering(
    net: Sequence[Sequence[float]],
    submitted: list[int],
    walked: set[tuple[int, ...]],
) -> tuple[str, list[int], float, dict]:
    """Pick the best ordering under a given net-advantage matrix.

    Factored out because SELECTION and EVALUATION must be able to run on
    different data. While this was inline there was exactly one `net` in scope
    and no way to express the split, which is how the bias below survived.
    """
    solved, solved_objective, search_info = solve_linear_ordering(net)
    candidates: list[tuple[str, list[int], float]] = [
        ("heuristic", solved, solved_objective),
        ("submitted", submitted, ordering_objective(submitted, net)),
    ]
    for order in sorted(walked):
        candidates.append(("walked", list(order), ordering_objective(list(order), net)))
    source, recommended, objective = max(candidates, key=lambda c: c[2])
    # The heuristic's own score travels with the search metadata rather than as
    # a fifth return value — it belongs to the search, and every caller that
    # wants it already wants the rest of `search_info`.
    search_info = {**search_info, "heuristic_objective": round(solved_objective, 4)}
    return source, recommended, objective, search_info


def _split_gain(
    infos: Sequence[dict],
    submitted: list[int],
    walked: set[tuple[int, ...]],
    n: int,
    idx: list[int],
) -> Optional[float]:
    """Select on one half of the walks, evaluate the gain on the other.

    Both role assignments are averaged, so every walk contributes to selection
    once and to evaluation once and the estimate does not depend on which half
    was arbitrarily called "training".
    """
    half = len(idx) // 2
    if half < 2:
        return None
    parts = ([infos[i] for i in idx[:half]], [infos[i] for i in idx[half:]])
    out: list[float] = []
    for train, test in (parts, parts[::-1]):
        try:
            net_train = net_advantage(_estimate(train, n)[0])
            net_test = net_advantage(_estimate(test, n)[0])
        except (ValueError, ZeroDivisionError):
            continue
        _, pick, _, _ = _select_ordering(net_train, submitted, walked)
        out.append(
            ordering_objective(pick, net_test) - ordering_objective(submitted, net_test)
        )
    return _mean(out) if out else None


def _permute_labels(info: dict, rng: random.Random) -> dict:
    """One walk under the null that MESSAGE IDENTITY carries no information.

    Positions are held fixed and their gains untouched; only which message sat
    at each position is shuffled. That is the precise null the recommendation
    is claiming to beat — it preserves the position effects (primacy, recency,
    fatigue), the per-walk gain distribution, and the walk length, and destroys
    only the message→position association the ordering search exists to find.

    Shuffling the gains instead would have tested a different and much weaker
    null, one that any position effect at all would reject.
    """
    shuffled = list(info["ordering"])
    rng.shuffle(shuffled)
    return {
        **info,
        "ordering": shuffled,
        # gains are (message, position, gain); the position and the gain stay
        # welded together, the message label at that position is resampled.
        "gains": [(shuffled[p], p, g) for (_m, p, g) in info["gains"]],
    }


def sequence_result(
    walks: list[dict],
    messages: list[str],
    cognitive_load: str,
    orderings_designed: int,
    failed_walks: int = 0,
) -> dict:
    """Pure aggregation: twin walks in, ordering recommendation out.

    Each walk is {"ordering": [message indices in delivered order],
    "steps": [{"intent", "fatigue", "disengaged"}, … one per delivered
    message]}. Walks whose step count doesn't match their ordering are dropped
    the same way studies.py drops a price curve that missed a price.
    """
    n = len(messages)
    walks = [
        w
        for w in walks
        if w.get("ordering") and len(w.get("steps") or []) == len(w["ordering"])
    ]
    if not walks:
        raise RuntimeError("Sequence study produced no valid twin walks.")

    submitted = list(range(n))
    infos = [walk_gains(w) for w in walks]
    terminals = [i["terminal_intent"] for i in infos]
    matrix, support, content, position, mu = _estimate(infos, n)
    # Every objective below is computed on the NET advantage: ordering is a
    # pairwise decision, and the directional matrix alone double-counts pairs
    # that help each other in both directions.
    net = net_advantage(matrix)

    # ── position effects: primacy and recency, stated separately ────────
    observed_positions = [p for p in position if p is not None]
    middle = [position[p] for p in range(1, n - 1) if position[p] is not None]
    middle_mean = _mean(middle) if middle else (_mean(observed_positions) if observed_positions else 0.0)
    primacy = (position[0] - middle_mean) if n > 0 and position[0] is not None else None
    recency = (position[n - 1] - middle_mean) if n > 0 and position[n - 1] is not None else None

    # ── solve, then take the best of solved / submitted / anything walked ──
    walked = {tuple(i["ordering"]) for i in infos if len(i["ordering"]) == n}
    submitted_objective = ordering_objective(submitted, net)
    source, recommended, objective, search_info = _select_ordering(
        net, submitted, walked
    )

    # ── how much that ordering actually gains, without the selection bias ──
    #
    # THE BUG THIS REPLACES. `recommended` is the argmax over candidates scored
    # on the full sample; `naive_gain` is then its margin over `submitted`
    # measured on that same sample. Taking a maximum over ~40 candidates and
    # reporting its value on the data that chose it is the textbook winner's
    # curse, and the old bootstrap did nothing about it: it re-estimated the
    # matrix per replicate but held `recommended` FIXED, so it measured the
    # uncertainty of scoring one pre-chosen ordering and never the uncertainty
    # of the choosing. Measured on 150 studies with NO order effect whatsoever,
    # `gain_supported` fired 20.7% of the time against a nominal 2.5%, with a
    # mean "gain" of +0.75 and P(gain > 0) = 98.7%. The product was telling
    # researchers to reorder their campaign on the basis of noise, five times
    # more often than its own stated error rate.
    #
    # Two independent corrections, because they answer different questions:
    #
    #   SAMPLE SPLITTING gives an unbiased magnitude. Selection runs on one
    #   half of the walks and evaluation on the other, so the half that scores
    #   the winner had no say in picking it. Repeated over random splits (both
    #   role assignments each time) to recover the precision that splitting
    #   costs.
    #
    #   A PERMUTATION TEST gives a real p-value. The whole select-then-evaluate
    #   procedure is re-run on data where message identity has been shuffled
    #   within each walk, so the null distribution is of the PROCEDURE — the
    #   maximisation included — rather than of a statistic. That is the only
    #   way the winner's curse can appear in the null as well as the estimate,
    #   which is what makes the comparison fair.
    #
    # `naive_gain` is still reported, labelled as what it is. Dropping it would
    # hide the size of the correction, and the gap between it and `gain` is
    # itself the diagnostic for how hard the search was working.
    naive_gain = objective - submitted_objective

    split_rng = random.Random(_LIFT_SEED)
    order_idx = list(range(len(infos)))
    honest: list[float] = []
    for _ in range(_SPLIT_REPS):
        split_rng.shuffle(order_idx)
        g = _split_gain(infos, submitted, walked, n, order_idx)
        if g is not None:
            honest.append(g)
    gain = round(_mean(honest), 4) if honest else None

    # Bootstrap the WHOLE procedure — resample walks, then split, select and
    # evaluate inside the replicate. Resampling around a fixed winner would
    # reproduce the original error in a new place.
    boot_rng = random.Random(_LIFT_SEED + 1)
    replicates: list[float] = []
    if honest:
        for _ in range(_LIFT_REPS):
            resample = [infos[boot_rng.randrange(len(infos))] for _ in range(len(infos))]
            rep_walked = {tuple(i["ordering"]) for i in resample if len(i["ordering"]) == n}
            rep_idx = list(range(len(resample)))
            boot_rng.shuffle(rep_idx)
            g = _split_gain(resample, submitted, rep_walked or walked, n, rep_idx)
            if g is not None:
                replicates.append(g)
    gain_ci = _percentile_ci(replicates) if len(replicates) >= 20 else None

    # Permutation null over the same cross-fitted statistic.
    perm_rng = random.Random(_PERM_SEED)
    n_ge = 0
    n_perm = 0
    if gain is not None:
        for _ in range(_PERM_REPS):
            permuted = [_permute_labels(i, perm_rng) for i in infos]
            perm_walked = {tuple(i["ordering"]) for i in permuted if len(i["ordering"]) == n}
            perm_idx = list(range(len(permuted)))
            perm_rng.shuffle(perm_idx)
            g = _split_gain(permuted, submitted, perm_walked or walked, n, perm_idx)
            if g is None:
                continue
            n_perm += 1
            if g >= gain:
                n_ge += 1
    # Add-one estimator, so p is never reported as exactly zero at finite P.
    gain_p_value = round((1.0 + n_ge) / (1.0 + n_perm), 4) if n_perm else None
    # BOTH conditions. A significant p with a negative point estimate is a
    # fluke of the tail, and a positive estimate whose interval straddles zero
    # is not a finding — the old code required only the latter.
    gain_supported = bool(
        gain is not None
        and gain > 0
        and gain_p_value is not None
        and gain_p_value < 0.05
        and gain_ci
        and gain_ci[0] > 0
    )

    # ── per-ordering outcomes + posteriors over what was actually walked ──
    by_order: dict[tuple[int, ...], list[dict]] = {}
    for info in infos:
        by_order.setdefault(tuple(info["ordering"]), []).append(info)
    orderings: list[dict] = []
    arms: dict[str, tuple[int, int]] = {}
    for order, group in sorted(by_order.items()):
        label = "-".join(str(i + 1) for i in order)
        ends = [g["terminal_intent"] for g in group]
        commits = sum(
            1
            for g in group
            if g["disengaged_at"] is None and g["terminal_intent"] >= _COMMIT_THRESHOLD
        )
        ci = bootstrap_ci(ends)
        orderings.append(
            {
                "label": label,
                "ordering": list(order),
                "walks": len(group),
                "mean_terminal_intent": round(_mean(ends), 4),
                "terminal_intent_ci": list(ci) if ci else None,
                "commit_rate": round(commits / len(group), 4),
                "disengage_rate": round(
                    sum(1 for g in group if g["disengaged_at"] is not None) / len(group), 4
                ),
                "objective": round(ordering_objective(list(order), net), 4),
                "is_submitted": list(order) == submitted,
            }
        )
        arms[label] = (commits, len(group))
    # Posteriors compare the orderings we genuinely ran head-to-head. The
    # recommended ordering is model-derived and may not be among them — which
    # is exactly why the recommendation ships with its own bootstrap interval.
    bayes = bayesian_ab(arms) if arms else None

    disengaged = [i for i in infos if i["disengaged_at"] is not None]
    space = math.factorial(n)
    return {
        "run_id": "",
        "twin_count": len(walks),
        "failed_walks": failed_walks,
        "cognitive_load": cognitive_load,
        "messages": [m[:160] for m in messages],
        "submitted_ordering": submitted,
        "recommended_ordering": recommended,
        "recommended_from": source,
        "recommended_messages": [messages[i][:160] for i in recommended],
        "objective": round(objective, 4),
        "submitted_objective": round(submitted_objective, 4),
        # The headline gain is the CROSS-FITTED one: selection and evaluation
        # never saw the same walks. None when there were too few walks to split.
        "objective_gain": gain,
        "objective_gain_ci": gain_ci,
        "gain_p_value": gain_p_value,
        "gain_supported": gain_supported,
        # In-sample, selection-biased, kept only so the size of the correction
        # is visible. Never render this as the gain.
        "naive_objective_gain": round(naive_gain, 4),
        "selection_bias": (
            round(naive_gain - gain, 4) if gain is not None else None
        ),
        "gain_inference": (
            "cross-fitted: the ordering is selected on one half of the walks "
            "and scored on the other, averaged over repeated random splits. "
            "The p-value permutes message identity within each walk — holding "
            "position and its gain fixed — and re-runs the entire "
            "select-then-evaluate procedure, so the winner's curse is present "
            "in the null as well as in the estimate."
        ),
        "objective_units": (
            "sum over ordered pairs of the NET position-adjusted precedence "
            "advantage (what the later message gains minus what the earlier one "
            "gives up), in intent points per pair — not a probability, and not "
            "an end-state intent delta"
        ),
        "search": {
            **search_info,
            "heuristic": "greedy insertion by net outflow + pairwise-swap local search",
            "optimal": False,
            "note": (
                "The linear ordering problem is NP-hard; this is a local optimum "
                "with no approximation guarantee."
            ),
        },
        "exploration": {
            "orderings_designed": orderings_designed,
            "orderings_walked": len(by_order),
            "sample_space": space,
            "explored_fraction": round(len(by_order) / space, 6) if space else None,
            "pairs_estimated": sum(
                1 for i in range(n) for j in range(n) if i != j and support[i][j] > 0
            ),
            "pairs_total": n * (n - 1),
            "min_pair_support": min(
                (support[i][j] for i in range(n) for j in range(n) if i != j), default=0
            ),
        },
        "precedence_matrix": [[round(v, 4) for v in row] for row in matrix],
        "net_advantage_matrix": [[round(v, 4) for v in row] for row in net],
        "precedence_support": support,
        "position_effects": [round(p, 4) if p is not None else None for p in position],
        "content_effects": [round(c, 4) if c is not None else None for c in content],
        "content_ranking": [
            {"message_index": i, "preview": messages[i][:120], "content_effect": round(content[i], 4)}
            for i in sorted(
                (i for i in range(n) if content[i] is not None),
                key=lambda i: content[i],
                reverse=True,
            )
        ],
        "baseline_gain": round(mu, 4),
        "primacy_effect": round(primacy, 4) if primacy is not None else None,
        "recency_effect": round(recency, 4) if recency is not None else None,
        "mean_terminal_intent": round(_mean(terminals), 4),
        "terminal_intent_ci": list(bootstrap_ci(terminals) or []) or None,
        "disengage_rate": round(len(disengaged) / len(walks), 4),
        "mean_disengage_position": (
            round(_mean([i["disengaged_at"] for i in disengaged]), 2) if disengaged else None
        ),
        "mean_end_fatigue": round(_mean([i["fatigue_end"] for i in infos]), 4),
        "orderings": orderings,
        "bayesian": bayes,
        "method": (
            "two-way additive position/content decomposition (mean polish) · "
            "position-adjusted precedence matrix · net-advantage linear-ordering "
            "heuristic (greedy insertion + 2-opt) · 200-replicate bootstrap on "
            "the objective gain"
        ),
        "disclaimer": (
            "Order effects are estimated from a sampled fraction of the N! "
            "orderings, so the recommendation is an extrapolation from pairwise "
            "precedence, not a measured ordering. Twins who disengage stop the "
            "walk, so late positions are estimated from the survivors."
        ),
    }


# ───────────────────────── twin orchestration ────────────────────────────

_SEQUENCE_PREAMBLE = """You are one synthetic recipient of a sequence of
messages delivered one at a time, in the order shown. You read them in order
and you cannot see the later ones while reacting to an earlier one. After each
message report: `intent` — your 0..1 likelihood of taking the action being
asked for AT THAT MOMENT (above 0.5 means you would actually do it) — `fatigue`
(0 fresh .. 1 completely done with this), `disengaged` (true the moment you
would stop reading the rest of the sequence), and one line on why. Order is the
whole point: repetition wears you down, an ask that arrives too early feels
pushy, and a message that would land well cold can be ruined by what came
before it. Give exactly one step per message, in the delivered order."""


async def _sequence_twin(
    persona: PersonaSeed,
    messages: list[str],
    ordering: list[int],
    load: str,
    semaphore: asyncio.Semaphore,
) -> Optional[TwinSequenceWalk]:
    delivered = "\n\n".join(
        f"MESSAGE {position + 1}:\n{messages[m]}" for position, m in enumerate(ordering)
    )
    async with semaphore:
        try:
            completion = await async_client().chat.completions.create(
                model=TWIN_MODEL,
                messages=[
                    {"role": "system", "content": _SEQUENCE_PREAMBLE},
                    _persona_system(persona, load),
                    {"role": "user", "content": f"The sequence you receive:\n\n{delivered}"},
                ],
                temperature=LOAD_TO_TEMPERATURE[load],
                response_format=response_format_for(TwinSequenceWalk),
            )
            walk = parse_completion(completion, TwinSequenceWalk)
        except (openai.OpenAIError, Refusal, RuntimeError, ValueError):
            return None
    # A twin that skipped or invented steps can't be positioned, so it's dropped
    # rather than padded — same contract as the price curve in studies.py.
    return walk if len(walk.steps) == len(ordering) else None


async def stream_sequence(
    personas: list[PersonaSeed], request: SequenceRequest
) -> AsyncIterator[dict]:
    """The ordering study as SSE.

    Frames: start · ordering (the design, one per walked ordering) · agent ·
    agent_failed · matrix · solved · done.
    """
    n = len(request.messages)
    total = len(personas) * request.twins_per_persona
    # More orderings than twins would leave cells with a single walk behind
    # them, and a precedence estimate from one walk is not an estimate.
    designed = min(request.orderings_sampled, max(1, total))
    orderings = sample_orderings(n, designed)

    yield {
        "type": "start",
        "total": total,
        "n_messages": n,
        "orderings": len(orderings),
        "sample_space": math.factorial(n),
        "messages": [m[:160] for m in request.messages],
    }
    for index, order in enumerate(orderings):
        yield {
            "type": "ordering",
            "index": index,
            "ordering": order,
            "label": "-".join(str(i + 1) for i in order),
            "role": "submitted" if index == 0 else "reversed" if index == 1 else "random",
        }

    semaphore = asyncio.Semaphore(SWARM_CONCURRENCY)

    async def one(persona: PersonaSeed, order_index: int):
        order = orderings[order_index]
        walk = await _sequence_twin(
            persona, request.messages, order, request.cognitive_load, semaphore
        )
        return order_index, order, walk

    # Assign orderings so that the design is actually covered.
    #
    # This was `(p_index + rep) % len(orderings)`, which walks the contiguous
    # block {0 … P+R−2} and nothing else: with 6 personas × 4 repetitions the
    # 24 designed assignments landed on only 9 distinct orderings, and the walk
    # counts came out triangular — 1, 2, 3, 4, 4, 4, 3, 2, 1. Two consequences,
    # both bad. Fifteen designed orderings were reported as designed and never
    # run. And ordering index 0 — the SUBMITTED BASELINE that every comparison
    # is measured against — was always walked by exactly one twin, so its
    # terminal-intent interval was None and its Bayesian arm was a single
    # observation.
    #
    # Striding by the persona count fixed the coverage hole and introduced a
    # worse one. `(p_index + rep * P) % L` confines persona p to the coset
    # p + <P> in Z_L, so the personas split into gcd(P, L) groups walking
    # DISJOINT ordering sets. gcd(P, L) > 1 for nearly every panel size, and at
    # P = 6 with L = 6 it degenerates completely — persona p walks only
    # ordering p, all repetitions. Every between-ordering statistic
    # (mean_terminal_intent, commit_rate, disengage_rate, the Bayesian arms,
    # and therefore `recommended_from="walked"`) was then a *persona*
    # comparison wearing an ordering label. Strictly worse than the triangular
    # coverage it replaced, because the counts look flat and correct.
    #
    # The fix is the standard randomised block design: hand out ordering
    # indices round-robin so the counts stay flat, then shuffle the assignment
    # under the fixed design seed so persona and ordering are paired at random
    # rather than by arithmetic. Randomisation is what makes the contrast
    # unbiased; no deterministic stride can do it, because with L orderings and
    # R < L repetitions no persona can reach every ordering.
    assignment = assign_orderings(
        len(personas), request.twins_per_persona, len(orderings)
    )
    walkers = [
        persona for _rep in range(request.twins_per_persona) for persona in personas
    ]
    tasks = [
        asyncio.create_task(one(persona, order_index))
        for persona, order_index in zip(walkers, assignment)
    ]

    walks: list[dict] = []
    failed = 0
    for coro in asyncio.as_completed(tasks):
        order_index, order, walk = await coro
        if walk is None:
            failed += 1
            yield {"type": "agent_failed", "ordering_index": order_index}
            continue
        record = {
            "ordering": order,
            "steps": [
                {
                    "intent": _clamp01(s.intent),
                    "fatigue": _clamp01(s.fatigue),
                    "disengaged": bool(s.disengaged),
                }
                for s in walk.steps
            ],
        }
        walks.append(record)
        info = walk_gains(record)
        yield {
            "type": "agent",
            "ordering_index": order_index,
            "ordering": order,
            "terminal_intent": round(info["terminal_intent"], 3),
            "disengaged_at": info["disengaged_at"],
            "fatigue_end": round(info["fatigue_end"], 3),
            "gist": walk.gist[:120],
        }

    if not walks:
        raise RuntimeError("Sequence study produced no valid twin walks.")

    result = sequence_result(
        walks, request.messages, request.cognitive_load, len(orderings), failed_walks=failed
    )
    yield {
        "type": "matrix",
        "precedence": result["precedence_matrix"],
        "net_advantage": result["net_advantage_matrix"],
        "support": result["precedence_support"],
        "position_effects": result["position_effects"],
        "content_effects": result["content_effects"],
        "primacy_effect": result["primacy_effect"],
        "recency_effect": result["recency_effect"],
    }
    yield {
        "type": "solved",
        "ordering": result["recommended_ordering"],
        "from": result["recommended_from"],
        "objective": result["objective"],
        "submitted_objective": result["submitted_objective"],
        "gain": result["objective_gain"],
        "gain_ci": result["objective_gain_ci"],
        "gain_supported": result["gain_supported"],
        "search": result["search"],
    }
    yield {"type": "done", "result": result}
