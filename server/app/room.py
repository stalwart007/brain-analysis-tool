"""The white room — measuring what deliberation does to a room of characters.

Every other study in this product fans out to ISOLATED twins on purpose. This
one breaks the isolation: N characters sit at a table, each sees what the others
say, and each may move. The instrument is therefore not "what does the audience
think" but "what happens to what they think when they can hear each other", and
the two questions have different failure modes. The dominant one is that a
language model is agreeable, so a room of them will converge whether or not
anything persuasive was said. Almost every guard in this module exists to keep
that artefact from being reported as a finding.

THE DATA. Round 0 is a PRIVATE ballot — nobody has heard anything, so it is the
control arm of a within-subjects experiment. Rounds 1..T are deliberation, and
each member each round emits both a PUBLIC statement (with a position their
words convey) and a PRIVATE position never shown to anyone. Runs are replicated
with different seeded speaking orders, plus optional PLACEBO replicates in which
the statements a member is shown come from a DIFFERENT room.

INDEXING, because three agents code against it. Inside this module:

  · `innate[i]`             — member i's round-0 PRIVATE ballot. One vector per
                              room, or one per replicate (both accepted).
  · `trajectories[r][t][i]` — PUBLIC position, replicate r, deliberation round
                              t = 0..T−1 (t = 0 is the FIRST speaking round, not
                              the ballot), member i.
  · `private_by_round[…]`   — same shape, private positions. Functions that take
                              a 2-D `[t][i]` array accept it too; a 2-D input is
                              read as a single replicate.
  · `speaking_orders[r]`    — member indices in speaking order. A per-round list
                              `speaking_orders[r][t]` is also accepted.

Positions are on 0..1 (0 = fully against the motion, 1 = fully for it) and
arrive from a model roughly 0.05-quantised, which is why several gates below are
stated in units of that quantisation rather than as abstract tolerances.

`room_analysis` returns ONE dict. Its top-level keys, one line each:

  run_id              — "" here; the caller stamps it (see the audit backlog).
  motion              — the motion as put to the room, echoed.
  names               — member names, in the order every array below uses.
  n_members / n_rounds / n_replicates / n_placebo_replicates — the design.
  dropped_replicates  — replicates discarded for incomplete turns, with reasons.
  roster              — one row per member: declared disposition, fitted
                        susceptibility, innate/final positions, gap, centrality,
                        absence effect. The table the UI is built around.
  trajectory          — cross-replicate mean public and private position per
                        round, plus the round-0 ballot, for the line chart.
  influence           — `fit_social_influence`: W, susceptibility, fit quality,
                        and its refusal when the design cannot identify W.
  centrality          — `influence_centrality`: left Perron vector, spectral
                        gap, whether the room converges at all.
  equilibrium         — the Friedkin-Johnsen fixed point and its distance from
                        where the room actually landed.
  absence             — `counterfactual_absence`: leave-one-out landing points.
  falsification       — `preference_falsification`: public−private gap, per
                        member, per round, with a CI and an exact sign test.
  conformity          — `conformity_vs_bayes`: variance collapse against the
                        fastest rational pooling schedule the room allows.
  cascade             — `cascade_scores`: how much of a public position is
                        explained by whoever spoke first in the same round.
  coalitions          — camps in the final positions (public AND private), plus
                        power indices when a decision threshold was given.
  order_effect        — how much the landing point depends on the speaking order.
  arms                — `arm_summary` for the real arm and the placebo arm.
  placebo             — `placebo_contrast`: the control. Read this one first.
  consensus           — observed final mean/spread, convergence ratio, forecast.
  refusals            — every estimator that declined, with the reason.
  timing_ms           — wall time of the whole analysis.
  method              — one paragraph naming every estimator used.
  limits              — what this instrument cannot show. Render it.

Everything is pure, stdlib-only and deterministic: seeded permutations, no
`random` calls without an explicit seed, no numpy. Measured wall time for the
whole of `room_analysis`: **2 ms** at 3 members × 2 rounds × 1 replicate,
**89 ms** at the default 5 × 4 × 3 plus one placebo, and **443 ms** at the
schema's maximum 9 × 8 × 8 plus four placebo replicates. Two thirds of that is
the two design-matched nulls (the immovable-room susceptibility floor and the
rational-pooling calibration), which is the right place for the budget to go.
"""

from __future__ import annotations

import itertools
import math
import random
import time
from typing import Optional, Sequence

from .analytics import bootstrap_ci, mean, stdev

# ── gates, every one of them measured rather than chosen ──────────────────
#
# `fit_social_influence` carries the recovery grid the first two came from, and
# `conformity_vs_bayes` the calibration behind its null.
_MIN_OBS_PER_PARAM = 1.5  # below this nothing about W or λ survives
_ENTRY_OBS_PER_PARAM = 9.0  # below this W's ENTRIES lose to a uniform-row guess
_MIN_GRAM_EIGENVALUE = 1.0e-9  # numerical degeneracy only — measured not to discriminate
_NULL_DRAWS = 5  # design-matched immovable-room draws for the λ floor
_RATIONAL_DRAWS = 400  # design-matched rational-pooling draws for conformity
_FISTA_CAP = 800
_FISTA_TOL = 1.0e-10
_POSITION_QUANTUM = 0.05  # the grid a model's declared position lands on
_SD_FLOOR = 0.005  # a tenth of the quantum; used only to keep a log finite
_MIN_CASCADE_OBS = 6  # 3 regression parameters + 3
_PERM_DRAWS = 4000
_SEED = 17


# ══════════════════════════════ small numerics ══════════════════════════════


def _as_replicates(data: Sequence) -> list[list[list[float]]]:
    """Accept `[r][t][i]` or a bare `[t][i]`, always return `[r][t][i]`.

    Two of the public entry points below are naturally per-replicate and two are
    naturally room-level, and the orchestration layer has both shapes to hand.
    Guessing from the nesting depth is safe here because the innermost element is
    always a float and the middle one is always a list.
    """
    if not data:
        return []
    first = data[0]
    if not isinstance(first, (list, tuple)) or not first:
        return []
    if isinstance(first[0], (list, tuple)):
        return [[list(row) for row in rep] for rep in data]
    return [[list(row) for row in data]]


def _as_innate(innate: Sequence, n_replicates: int) -> list[list[float]]:
    """Accept one innate vector or one per replicate; return one per replicate."""
    if not innate:
        return [[] for _ in range(max(1, n_replicates))]
    if isinstance(innate[0], (list, tuple)):
        rows = [list(v) for v in innate]
        while len(rows) < n_replicates:
            rows.append(list(rows[-1]))
        return rows[: max(1, n_replicates)]
    return [list(innate) for _ in range(max(1, n_replicates))]


def _project_capped_simplex(v: Sequence[float]) -> list[float]:
    """Euclidean projection onto {a ≥ 0, Σa ≤ 1}.

    Exact, not iterative. KKT for the inequality-summed set says: clip at zero
    first; if the clipped vector already sums to ≤ 1 it IS the projection (the
    sum constraint is inactive). Otherwise the sum constraint binds with
    equality and the answer is the projection onto the probability simplex,
    which is Duchi et al.'s sort-and-shift.
    """
    clipped = [x if x > 0.0 else 0.0 for x in v]
    if sum(clipped) <= 1.0 + 1e-12:
        return clipped
    order = sorted(v, reverse=True)
    css = 0.0
    theta = 0.0
    for i, val in enumerate(order, start=1):
        css += val
        candidate = (css - 1.0) / i
        if val - candidate > 0.0:
            theta = candidate
    return [x - theta if x - theta > 0.0 else 0.0 for x in v]


def _jacobi_eigenvalues(a: list[list[float]], sweeps: int = 60) -> list[float]:
    """Eigenvalues of a small symmetric matrix by cyclic Jacobi rotations.

    Used only for the conditioning diagnostic on a ≤ 9 × 9 Gram matrix, where
    the cubic cost is irrelevant and the alternative (a characteristic
    polynomial) is numerically hopeless.
    """
    n = len(a)
    m = [list(row) for row in a]
    for _ in range(sweeps):
        off = 0.0
        for p in range(n - 1):
            for q in range(p + 1, n):
                off += m[p][q] * m[p][q]
        if off <= 1e-24:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(m[p][q]) <= 1e-18:
                    continue
                theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q])
                t = (1.0 if theta >= 0 else -1.0) / (
                    abs(theta) + math.sqrt(theta * theta + 1.0)
                )
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    mkp, mkq = m[k][p], m[k][q]
                    m[k][p] = c * mkp - s * mkq
                    m[k][q] = s * mkp + c * mkq
                for k in range(n):
                    mpk, mqk = m[p][k], m[q][k]
                    m[p][k] = c * mpk - s * mqk
                    m[q][k] = s * mpk + c * mqk
    return sorted(m[i][i] for i in range(n))


def _solve_normal(xtx: list[list[float]], xty: list[float]) -> Optional[list[float]]:
    """Gaussian elimination with partial pivoting. None if singular."""
    n = len(xty)
    aug = [list(xtx[i]) + [xty[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        inv = 1.0 / aug[col][col]
        for r in range(col + 1, n):
            factor = aug[r][col] * inv
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                aug[r][c] -= factor * aug[col][c]
    out = [0.0] * n
    for row in range(n - 1, -1, -1):
        acc = aug[row][n] - sum(aug[row][c] * out[c] for c in range(row + 1, n))
        out[row] = acc / aug[row][row]
    return out


def _ols_r2(rows: list[list[float]], y: list[float]) -> Optional[float]:
    """R² of an OLS fit of `y` on `rows` (intercept must be included by caller)."""
    if not rows or len(rows) != len(y):
        return None
    p = len(rows[0])
    xtx = [[sum(r[i] * r[j] for r in rows) for j in range(p)] for i in range(p)]
    xty = [sum(r[i] * yy for r, yy in zip(rows, y)) for i in range(p)]
    beta = _solve_normal(xtx, xty)
    if beta is None:
        return None
    ybar = mean(y)
    tss = sum((yy - ybar) ** 2 for yy in y)
    if tss <= 1e-15:
        return None
    rss = sum(
        (yy - sum(b * v for b, v in zip(beta, r))) ** 2 for r, yy in zip(rows, y)
    )
    return 1.0 - rss / tss


def _fista(
    gram: list[list[float]],
    xty: list[float],
    iterations: int = _FISTA_CAP,
    tol: float = _FISTA_TOL,
) -> list[float]:
    """Minimise ‖Za − y‖² over {a ≥ 0, Σa ≤ 1} by accelerated projected gradient.

    Projected gradient rather than an active-set NNLS because the feasible set is
    not the nonnegative orthant — it is the orthant intersected with Σa ≤ 1, and
    that intersection has an exact closed-form projection (above) while an
    active-set method would need the sum constraint handled separately. The
    objective is convex and the set is convex, so any stationary point is global;
    Nesterov acceleration with the Lipschitz step 1/L (L = λ_max of the Gram
    matrix) is what makes a per-member solve of a ≤ 9-dimensional problem cheap.

    THE CAP IS 800 ITERATIONS AND THAT IS MEASURED, not assumed. Against a
    6,000-iteration reference at tol 1e-14, an 800-iteration solve deviated by at
    most **9.8e-3** (mean 3.4e-4) across 25 rooms × every member × every entry, at
    the smallest design tested, and at most 2.4e-3 at the largest. The
    estimator's OWN recovery error is 0.10-0.20 per entry, so the solver's
    residual sits three orders of magnitude below the thing it is estimating —
    tightening it further buys nothing and costs the runtime budget that the
    design-matched null below actually needs. Measured cost per full fit: 6.8 ms
    at 5 members, 30 ms at 9.
    """
    n = len(xty)
    eigs = _jacobi_eigenvalues(gram)
    lipschitz = max(eigs[-1], 1e-12)
    step = 1.0 / lipschitz
    a = [0.0] * n
    y = list(a)
    t_k = 1.0
    for _ in range(iterations):
        grad = [
            sum(gram[i][j] * y[j] for j in range(n)) - xty[i] for i in range(n)
        ]
        nxt = _project_capped_simplex([y[i] - step * grad[i] for i in range(n)])
        t_next = (1.0 + math.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        beta = (t_k - 1.0) / t_next
        y = [nxt[i] + beta * (nxt[i] - a[i]) for i in range(n)]
        moved = max(abs(nxt[i] - a[i]) for i in range(n))
        a = nxt
        t_k = t_next
        if moved < tol:
            break
    return a


# ══════════════════════ 1. the influence matrix ══════════════════════


def fit_social_influence(
    trajectories: Sequence,
    innate: Sequence,
    *,
    names: Optional[Sequence[str]] = None,
) -> dict:
    """Friedkin-Johnsen influence weights and susceptibility, per member.

    THE MODEL.  x_i(t+1) = λ_i·Σ_j W_ij·x_j(t) + (1−λ_i)·u_i, with u the round-0
    private ballot, λ_i ∈ [0,1] the member's susceptibility to the room and W_i·
    a row on the simplex saying whose position it moves toward. λ = 0 is a member
    who never leaves its opening ballot; λ = 1 is pure DeGroot averaging with no
    anchor at all.

    THE REPARAMETRISATION that makes this a constrained least-squares problem
    rather than a bilinear one. Put a_ij = λ_i·W_ij. Then Σ_j a_ij = λ_i and

        x_i(t+1) − u_i = Σ_j a_ij·(x_j(t) − u_i)

    which is linear in a with the feasible set {a ≥ 0, Σa ≤ 1} — convex, with an
    exact projection. Recovering λ_i = Σ_j a_ij and W_ij = a_ij/λ_i afterwards is
    a bijection EXCEPT at λ_i = 0, and that exception is a real result rather
    than a numerical corner: a member who never moved supplies no information
    about whom it would have listened to. Those rows are reported as self-weight
    1.0 — the behaviourally equivalent statement — and named in
    `unidentified_rows` with the reason. Do not read them as "listens only to
    itself by choice".

    THE FIRST USABLE TRANSITION IS ROUND 1 → 2, NOT BALLOT → ROUND 1. The FJ
    recursion needs the state members were responding to. Round 0 is a private
    ballot: nobody saw it, so `u` is not a state anyone conditioned on, and
    pretending otherwise would manufacture one free observation per member per
    replicate out of an event that did not happen. Hence
    `n_obs_per_member = (rounds − 1) · replicates`.

    IDENTIFIABILITY, WHICH IS THE CRUX. Per member the unknowns are λ_i plus n
    weights, and the simplex constraint Σ W = 1 removes exactly one, so
    `params_per_member = n`. A 5-member room over 4 deliberation rounds gives
    3 observations per member in one replicate — 0.6 observations per parameter.
    There is no honest estimate there, and a regulariser would only make the
    underdetermination invisible. So observations are POOLED ACROSS REPLICATES
    (the members and their innate positions are the same room; only the speaking
    order changed) and the fit REFUSES below a measured floor.

    WHAT THE RECOVERY GRID ACTUALLY SAYS. FJ data simulated with a known W and λ
    (λ ~ U[0.15, 0.85], rows drawn on the simplex with a self-weight, σ = 0.03
    per-round noise, positions quantised to the 0.05 grid a model emits), 60
    trials per cell, members × rounds × replicates swept over 3/5/9 × 3/4/6/8 ×
    1/2/3/4/6/8. Absolute MAE is close to meaningless on its own, so every figure
    below is measured AGAINST AN UNINFORMATIVE GUESS — a uniform row 1/n for W,
    the prior mean 0.5 for λ. A ratio above 1.0 means the fit is WORSE than not
    fitting at all:

        obs/param       <1.0  1.0-1.5  1.5-2.5  2.5-4   4-6   6-9   ≥9
        MAE(W)/base     1.69    1.66     1.51    1.38  1.23  1.09  0.87
        rank ρ(W row)  +0.09   +0.14    +0.21   +0.28 +0.37 +0.44 +0.56
        MAE(λ)/base     1.67    1.40     1.20    1.01  0.81  0.69  0.54
        rank ρ(λ)      +0.30   +0.42    +0.50   +0.58 +0.69 +0.75 +0.83

    Read that honestly and it says three different things at three scales, and
    all three are reported:

      · Below 1.5 obs/param, NOTHING survives — the λ ordering is at ρ = +0.30
        to +0.42 and both levels are worse than guessing. This is the refusal
        floor, and the assignment's example lands under it: 5 members, 4 rounds,
        1 replicate is 0.6 obs/param, and the fit declines.
      · From 1.5 up, the λ ORDERING is recovered (ρ +0.50 rising to +0.83) while
        the λ LEVEL is not better than the prior mean until ≈2.5. So "the CFO
        moved less than anyone in the room" is supportable a long way before
        "the CFO's λ is 0.31" is.
      · W's INDIVIDUAL ENTRIES do not beat a uniform-row guess until 9
        obs/param, which the schema's own limits barely reach (9 members × 8
        rounds × 8 replicates is 6.2). So `w_entries_supported` is False for
        almost every real design, and `w_entry_caveat` says so. At 5 members ×
        8 rounds × 8 replicates (11.2) the ratio is 0.96 for a diffuse W and
        0.76 for a concentrated one — the fit finds a hierarchy far better than
        it finds a flat room, which is the useful direction.

    AND YET THE DERIVED QUANTITIES RECOVER, which is why this module still ships
    them. The FJ equilibrium x* is dominated by u and λ, and errors in W largely
    cancel inside the solve: MAE on x* was **0.0192 against a 0.0847 baseline**
    (predicting the ballot u) at 5 × 4 × 3, and 0.0053 at 5 × 8 × 8, with rank
    correlation 0.96-0.99 in every cell. Centrality sits in between — rank ρ(π)
    0.24 (diffuse W) / 0.36 (concentrated) at 5 × 4 × 3, rising to 0.45 / 0.69 at
    5 × 8 × 8, and the single most-central member is identified 31% / 38% of the
    time at the small design against a 20% chance rate, 49% / 57% at the large
    one. So: trust `equilibrium` and `counterfactual_absence`, read `centrality`
    as a weak ordering, and do not read a single W entry.

    THE NO-INFLUENCE NULL, and the fix it forced. On an immovable room (W = I,
    λ = 0) with NO noise the fit is exactly right: λ = 0.0000 on 300/300 rows,
    off-diagonal W 0.0000, every row correctly flagged unidentified. Add the
    0.05 quantisation and it starts manufacturing susceptibility, because the
    constrained fit can trade positive against negative regressors to reproduce
    a rounding step. Measured null λ on rooms where nobody moved:

        design (n × rounds × reps)   median   p90     p95
        5 × 4 × 3                     0.186   0.929  1.000
        5 × 6 × 4                     0.181   0.643  0.928
        5 × 8 × 8                     0.099   0.345  0.460
        9 × 4 × 5                     0.389   1.000  1.000
        9 × 8 × 8                     0.186   0.534  0.651

    Real rooms at those designs read a median λ of 0.53-0.59 with a p10 of
    0.22-0.26, so the bias does not swamp the signal — but a p90 of 1.000 means a
    reader shown λ = 0.93 has no way to know it is noise. So the floor is
    measured AT RUNTIME for the design in hand: `susceptibility_null` fits
    immovable rooms of the same shape, same ballot and same quantisation, with the
    noise scale taken from this fit's own residuals, and
    `susceptibility_supported[i]` is False for any member whose λ falls inside
    that null's p90. Measured behaviour of that gate, 40 rooms per cell — the
    share of member rows it lets through:

        design      5×4×3  5×6×4  5×8×8  9×4×5  9×6×8  9×8×8  3×8×8
        immovable    14%    14%     5%     1%    12%    10%     8%
        real room    28%    52%    66%     3%    29%    36%    87%

    The immovable column tracks the 10% a p90 cut implies, so the gate is honest;
    the real column is the power, and it rises with the design exactly as the
    recovery grid says it should. At 9 members × 4 rounds × 5 replicates the null
    reaches λ = 1.000 and nothing passes — which is the correct answer for that
    design, and the two measurements agree: the recovery grid puts it at 1.67
    obs/param, where MAE(λ) is 1.35× worse than guessing.

    THE COLLINEARITY DIAGNOSTIC IS REPORTED BUT NOT USED AS A GATE, because it
    was measured not to discriminate: the smallest normalised Gram eigenvalue was
    2.4e-4 on null rooms against 6.5e-4 on real ones (real p10 4.1e-4), which
    overlaps far too much to threshold. Only exact degeneracy is caught (the
    noiseless null reads −6.2e-18). `gram_min_eigenvalue` ships as a diagnostic
    for the genuine failure it does name — a room that has already converged has
    regressors that no longer differ, so who moved whom stops being separable
    however many rounds are added — and the λ null above does the discriminating.

    Returns `refused` populated and **no W** when the design cannot support one.
    """
    reps = _as_replicates(trajectories)
    if not reps:
        return _influence_refusal("No public position trajectories were supplied.")
    n_rounds = len(reps[0])
    n = len(reps[0][0]) if n_rounds else 0
    if n < 2 or n_rounds < 2:
        return _influence_refusal(
            f"A trajectory of {n_rounds} round(s) over {n} member(s) has no "
            "usable transition; the model needs at least two rounds and two "
            "members."
        )
    reps = [r for r in reps if len(r) == n_rounds and all(len(x) == n for x in r)]
    if not reps:
        return _influence_refusal("Trajectories were ragged across rounds or members.")
    innate_by_rep = _as_innate(innate, len(reps))
    if any(len(u) != n for u in innate_by_rep):
        return _influence_refusal(
            "The round-0 private ballot does not have one position per member."
        )

    labels = list(names) if names else [f"member_{i}" for i in range(n)]
    n_obs = (n_rounds - 1) * len(reps)
    obs_per_param = n_obs / n
    base = {
        "names": labels,
        "innate": [round(v, 4) for v in _column_means(innate_by_rep)],
        "n_obs_per_member": n_obs,
        "params_per_member": n,
        "obs_per_param": round(obs_per_param, 3),
        "n_replicates": len(reps),
        "n_rounds": n_rounds,
        "method": (
            "Friedkin-Johnsen fitted per member as constrained least squares on "
            "a = λ·W over {a ≥ 0, Σa ≤ 1} (accelerated projected gradient, exact "
            "capped-simplex projection), observations pooled across replicates"
        ),
    }
    if obs_per_param < _MIN_OBS_PER_PARAM:
        return {
            **base,
            "W": None,
            "susceptibility": None,
            "r2_per_member": None,
            "residual_rms": None,
            "unidentified_rows": [],
            "gram_min_eigenvalue": None,
            "susceptibility_null": None,
            "susceptibility_supported": None,
            "w_entries_supported": False,
            "w_entry_caveat": None,
            "refused": (
                f"{n_obs} observations per member against {n} free parameters "
                f"({obs_per_param:.2f} per parameter) is below the "
                f"{_MIN_OBS_PER_PARAM} at which anything was measured to be "
                "recoverable: under that, the susceptibility ORDERING sits at a "
                "rank correlation of +0.30 to +0.42 and both W and λ are further "
                "from the truth than a uniform-row guess (MAE ratios 1.66-1.69 "
                "and 1.40-1.67). Add replicates — they are the cheap axis, since "
                "each buys (rounds − 1) observations per member — or add rounds. "
                "A fit here would be a confident-looking interpolation of noise."
            ),
        }

    raw = _fit_fj(reps, innate_by_rep, n, n_rounds)
    weights: list[list[float]] = []
    unidentified: list[dict] = []
    for i in range(n):
        lam = raw["lambdas"][i]
        if lam <= 1e-6:
            weights.append([1.0 if j == i else 0.0 for j in range(n)])
            unidentified.append(
                {
                    "name": labels[i],
                    "index": i,
                    "reason": (
                        "susceptibility fitted at zero — this member never left "
                        "its opening ballot, so the data contain no evidence "
                        "about whom it would have listened to. The row shown is "
                        "self-weight 1.0, which is behaviourally equivalent and "
                        "not a measurement."
                    ),
                }
            )
            continue
        weights.append([round(v / lam, 4) for v in raw["a"][i]])
        if raw["min_eigs"][i] < _MIN_GRAM_EIGENVALUE:
            unidentified.append(
                {
                    "name": labels[i],
                    "index": i,
                    "reason": (
                        f"the regressors this member saw are numerically "
                        f"degenerate (smallest normalised Gram eigenvalue "
                        f"{raw['min_eigs'][i]:.2e}): every other member sat at "
                        "the same position all study, so who moved whom is not "
                        "separable however many rounds are added."
                    ),
                }
            )

    null = _null_susceptibility(
        n, n_rounds, len(reps), innate_by_rep, raw["residual_rms"]
    )
    supported = [
        bool(raw["lambdas"][i] > (null["p90"] if null["p90"] is not None else 0.0))
        for i in range(n)
    ]
    for i in range(n):
        if not supported[i] and raw["lambdas"][i] > 1e-6:
            unidentified.append(
                {
                    "name": labels[i],
                    "index": i,
                    "reason": (
                        f"susceptibility {raw['lambdas'][i]:.3f} falls inside the "
                        f"p90 = {null['p90']:.3f} of what this exact design fits "
                        "on a room where NOBODY moved. Not distinguishable from "
                        "immovable; the influence row is meaningless for it."
                    ),
                }
            )

    entries_ok = obs_per_param >= _ENTRY_OBS_PER_PARAM
    return {
        **base,
        "W": weights,
        "susceptibility": [round(v, 4) for v in raw["lambdas"]],
        "r2_per_member": raw["r2s"],
        "residual_rms": round(raw["residual_rms"], 4),
        "gram_min_eigenvalue": [float(f"{v:.3e}") for v in raw["min_eigs"]],
        "unidentified_rows": unidentified,
        "susceptibility_null": null,
        "susceptibility_supported": supported,
        "w_entries_supported": entries_ok,
        "w_entry_caveat": (
            None
            if entries_ok
            else (
                f"At {obs_per_param:.2f} observations per parameter the INDIVIDUAL "
                f"entries of W were measured to be worse than a uniform-row guess "
                f"(MAE ratio 1.5 at 1.5-2.5, 1.1 at 6-9, first beating the guess "
                f"at {_ENTRY_OBS_PER_PARAM:.0f}). Read the row ordering, the "
                "susceptibility, the equilibrium and the leave-one-out deltas — "
                "all of which were measured to recover here — and do not quote a "
                "single weight."
            )
        ),
        "refused": None,
    }


def _constrained_ls(gram: list[list[float]], xty: list[float]) -> list[float]:
    """The constrained solve, taking the exact route whenever it is available.

    If the UNCONSTRAINED least-squares solution already satisfies a ≥ 0 and
    Σa ≤ 1 then no constraint is active, KKT is satisfied, and that solution IS
    the constrained optimum — exactly, from one Gaussian elimination. Only when
    it is infeasible (or the normal equations are singular) does the iterative
    projected-gradient solve run.

    This is not an optimisation, it is a correctness fix. On noiseless FJ data
    with a well-separated interior optimum, 800 FISTA iterations left λ at 0.469
    against a true 0.500 — a 6% error introduced by the solver on data that
    determines the answer exactly, and one that would have been read as
    estimation error. The exact branch returns 0.500. Feasibility is checked with
    a 1e-9 tolerance and the projection is applied to the exact solution anyway,
    so a solution sitting a rounding error outside the set is snapped in rather
    than rejected into the slow path.
    """
    exact = _solve_normal(gram, xty)
    if exact is not None:
        if min(exact) >= -1e-9 and sum(exact) <= 1.0 + 1e-9:
            return _project_capped_simplex(exact)
    return _fista(gram, xty)


def _fit_fj(
    reps: list[list[list[float]]],
    innate_by_rep: list[list[float]],
    n: int,
    n_rounds: int,
) -> dict:
    """The per-member constrained solve, factored out so the null can reuse it."""
    a_hats: list[list[float]] = []
    lambdas: list[float] = []
    r2s: list[Optional[float]] = []
    min_eigs: list[float] = []
    sq_resid = 0.0
    n_resid = 0
    for i in range(n):
        rows: list[list[float]] = []
        targets: list[float] = []
        for r, rep in enumerate(reps):
            u_i = innate_by_rep[r][i]
            for t in range(n_rounds - 1):
                rows.append([rep[t][j] - u_i for j in range(n)])
                targets.append(rep[t + 1][i] - u_i)
        gram = [[sum(r[a] * r[b] for r in rows) for b in range(n)] for a in range(n)]
        xty = [sum(r[a] * y for r, y in zip(rows, targets)) for a in range(n)]
        scaled = [[gram[a][b] / len(rows) for b in range(n)] for a in range(n)]
        min_eigs.append(_jacobi_eigenvalues(scaled)[0])
        a_hat = _constrained_ls(gram, xty)
        a_hats.append(a_hat)
        lambdas.append(sum(a_hat))
        rss = sum(
            (y - sum(a_hat[j] * r[j] for j in range(n))) ** 2
            for r, y in zip(rows, targets)
        )
        sq_resid += rss
        n_resid += len(rows)
        ybar = mean(targets)
        tss = sum((y - ybar) ** 2 for y in targets)
        r2s.append(round(1.0 - rss / tss, 4) if tss > 1e-12 else None)
    return {
        "a": a_hats,
        "lambdas": lambdas,
        "r2s": r2s,
        "min_eigs": min_eigs,
        "residual_rms": math.sqrt(sq_resid / n_resid) if n_resid else 0.0,
    }


def _null_susceptibility(
    n: int,
    n_rounds: int,
    n_reps: int,
    innate_by_rep: list[list[float]],
    residual_rms: float,
) -> dict:
    """What susceptibility this exact design invents on a room where nobody moved.

    The null room is W = I, λ = 0: every member states its opening ballot every
    round and moves for nobody. The only thing that varies is the 0.05 grid the
    positions land on, driven at the scale of THIS fit's own residuals — so the
    floor is matched to the design, the ballot and the observed noise rather than
    taken from a table. Deterministic: the RNG is seeded from the design itself.
    """
    sigma = min(0.15, max(0.004, residual_rms))
    draws: list[float] = []
    rng = random.Random(_SEED * 1000 + n * 97 + n_rounds * 13 + n_reps)
    for _ in range(_NULL_DRAWS):
        sim = []
        for r in range(n_reps):
            u = innate_by_rep[min(r, len(innate_by_rep) - 1)]
            sim.append(
                [
                    [_quantise(u[i] + rng.gauss(0.0, sigma)) for i in range(n)]
                    for _ in range(n_rounds)
                ]
            )
        draws.extend(_fit_fj(sim, innate_by_rep, n, n_rounds)["lambdas"])
    draws.sort()
    if not draws:
        return {"median": None, "p90": None, "p95": None, "draws": 0, "sigma": None}
    return {
        "median": round(draws[len(draws) // 2], 4),
        "p90": round(draws[min(len(draws) - 1, int(0.90 * len(draws)))], 4),
        "p95": round(draws[min(len(draws) - 1, int(0.95 * len(draws)))], 4),
        "draws": len(draws),
        "sigma": round(sigma, 4),
        "note": (
            "Susceptibility fitted on immovable rooms of this exact shape, ballot "
            "and noise scale. A member whose λ sits below p90 did not measurably "
            "move; the fit would have produced that number from rounding alone."
        ),
    }


def _quantise(v: float) -> float:
    """Snap to the 0.05 grid a declared position lands on, clamped to [0, 1]."""
    return min(1.0, max(0.0, round(v / _POSITION_QUANTUM) * _POSITION_QUANTUM))


def _influence_refusal(reason: str) -> dict:
    return {
        "W": None,
        "susceptibility": None,
        "innate": None,
        "names": None,
        "n_obs_per_member": 0,
        "params_per_member": 0,
        "obs_per_param": 0.0,
        "r2_per_member": None,
        "residual_rms": None,
        "gram_min_eigenvalue": None,
        "unidentified_rows": [],
        "susceptibility_null": None,
        "susceptibility_supported": None,
        "w_entries_supported": False,
        "w_entry_caveat": None,
        "refused": reason,
        "method": "Friedkin-Johnsen constrained least squares (not run)",
    }


def _column_means(rows: Sequence[Sequence[float]]) -> list[float]:
    if not rows:
        return []
    n = len(rows[0])
    return [mean([r[i] for r in rows]) for i in range(n)]


# ══════════════════════ 2. who the consensus listens to ══════════════════════


def _reachable(adj: list[list[bool]], start: int, reverse: bool = False) -> set[int]:
    n = len(adj)
    seen = {start}
    stack = [start]
    while stack:
        v = stack.pop()
        for w in range(n):
            edge = adj[w][v] if reverse else adj[v][w]
            if edge and w not in seen:
                seen.add(w)
                stack.append(w)
    return seen


def _closed_classes(adj: list[list[bool]]) -> list[list[int]]:
    """The closed communicating classes of the influence digraph.

    A communicating class is a maximal set of members who can all reach each
    other. It is CLOSED when no edge leaves it — nobody in it listens to anybody
    outside. For a row-stochastic matrix the stationary distribution is unique
    **iff there is exactly one closed class**, and its support is that class.

    Strong connectivity is sufficient for that and NOT necessary, which is the
    distinction the previous version missed: it tested irreducibility and
    refused whenever it failed. The configuration that refusal threw away is the
    single most valuable one this instrument can find — one member who listens to
    nobody while everybody listens to them. That room has a perfectly unique
    centrality (all of it on the oracle), and calling it "a room that has split"
    was both a refusal and a wrong description.
    """
    n = len(adj)
    # i and j communicate when each reaches the other; O(n²) reachability is
    # ample at n ≤ 9 seats and keeps this readable.
    reach = [_reachable(adj, i) for i in range(n)]
    seen: set[int] = set()
    classes: list[list[int]] = []
    for i in range(n):
        if i in seen:
            continue
        cls = sorted(j for j in range(n) if j in reach[i] and i in reach[j])
        seen.update(cls)
        classes.append(cls)
    members = {j for cls in classes for j in cls}
    assert members == set(range(n))
    # closed ⇔ every edge out of the class lands back inside it
    closed = []
    for cls in classes:
        inside = set(cls)
        if all(j in inside for i in cls for j in range(n) if adj[i][j]):
            closed.append(cls)
    return closed


def _is_primitive(adj: list[list[bool]]) -> bool:
    """True when some power of the adjacency is strictly positive.

    Wielandt's bound says a primitive n × n nonnegative matrix has A^k > 0 for
    some k ≤ n² − 2n + 2, so the search terminates. At n ≤ 9 this is 65 boolean
    matrix products, which is nothing, and it is the definition rather than a
    proxy for it.
    """
    n = len(adj)
    bound = max(1, n * n - 2 * n + 2)
    power = [row[:] for row in adj]
    for _ in range(bound):
        if all(all(row) for row in power):
            return True
        power = [
            [any(power[i][k] and adj[k][j] for k in range(n)) for j in range(n)]
            for i in range(n)
        ]
    return all(all(row) for row in power)


def influence_centrality(
    w: Optional[Sequence[Sequence[float]]],
    *,
    innate: Optional[Sequence[float]] = None,
) -> dict:
    """Left Perron eigenvector of W — whose opinion the room's consensus weights.

    DeGroot's result is the reason this is the headline and not a decoration: if
    everyone averages (λ = 1) and W is primitive, the room converges to π'u where
    π is the LEFT eigenvector of W for eigenvalue 1. The consensus is a weighted
    average of the OPENING positions, and π is the weights. A member with high π
    does not need to talk more or move less; the room's landing point is simply
    more theirs.

    Computed by power iteration on π ← W'π with L1 renormalisation, which is the
    right method here for the reason it is usually the wrong one: it converges at
    the rate |λ₂|, and |λ₂| is itself the quantity we want to report.

    THE SPECTRAL GAP, 1 − |λ₂|, is how fast the room settles. It is measured on
    the deflated matrix B = W − 1π′, which annihilates both of W's dominant
    eigenvectors (B1 = 0 and π'B = 0) so its spectral radius is exactly |λ₂|.
    The estimate is the geometric mean of the growth factor ‖B^k v‖ over the last
    iterations rather than a Rayleigh quotient, because λ₂ of a stochastic matrix
    is routinely a complex pair — a Rayleigh quotient oscillates forever, while
    the growth rate converges to the modulus either way.

    REDUCIBLE AND PERIODIC ROOMS ARE ANSWERED, NOT PAPERED OVER. Uniqueness of
    the stationary distribution turns on the number of CLOSED communicating
    classes, not on irreducibility — strong connectivity is sufficient and not
    necessary — so the three cases are:

      · One closed class, aperiodic: π is unique and the plain iterate converges
        to it. This includes the reducible case that matters most here, the
        oracle everybody defers to and who listens to nobody: that room has all
        of its centrality on one seat, `converges` is True, and the note says
        the whole consensus is theirs. `irreducible` is reported separately from
        `converges` so the structure is still visible.
      · More than one closed class: two or more groups who do not listen outside
        themselves, so W has more than one stationary distribution and there is
        no such thing as "the" centrality. `centrality` is None, `converges`
        False, and `closed_groups` names the cliques — a room that has split has
        no shared consensus for anyone to be central in.
      · One closed class but periodic: a unique stationary π exists while
        opinions cycle rather than settle. π is returned as the Cesàro limit of
        the averaged iterates, which converges when the plain iteration does
        not, `converges` is False, and `consensus_forecast` is withheld because
        there is no limit to forecast.

    `consensus_forecast` is π'u, and it needs u. The positional contract is
    `influence_centrality(W)`, so `innate` is keyword-only and optional; without
    it the field is None rather than invented.
    """
    if not w or not w[0]:
        return {
            "centrality": None,
            "spectral_gap": None,
            "converges": False,
            "iterations": 0,
            "consensus_forecast": None,
            "primitive": False,
            "irreducible": False,
            "note": "No influence matrix was estimated, so there is nothing to rank.",
        }
    n = len(w)
    adj = [[w[i][j] > 1e-9 for j in range(n)] for i in range(n)]
    irreducible = len(_reachable(adj, 0)) == n and len(_reachable(adj, 0, True)) == n
    primitive = irreducible and _is_primitive(adj)
    closed = _closed_classes(adj)

    # Uniqueness turns on the number of CLOSED classes, not on irreducibility.
    # A room with one member who listens to nobody while everyone listens to
    # them is reducible and has a perfectly well-defined centrality — all of it
    # on that member. Refusing there discarded the most useful finding this
    # module produces.
    if len(closed) != 1:
        return {
            "centrality": None,
            "spectral_gap": None,
            "converges": False,
            "iterations": 0,
            "consensus_forecast": None,
            "primitive": False,
            "irreducible": False,
            "closed_groups": closed,
            "note": (
                f"The room contains {len(closed)} groups who do not listen "
                "outside themselves, so W has more than one stationary "
                "distribution and no single centrality exists. The groups are "
                f"{closed} (by seat index). Read this as a room that has split "
                "into cliques rather than as a failure to converge — there is "
                "no shared consensus for anyone to be central in."
            ),
        }

    # Determined BEFORE the iteration because it decides which iterate to keep.
    # `primitive` is the wrong test: it requires irreducibility, so a reducible
    # room with one aperiodic closed class — the oracle everybody defers to — was
    # being Cesàro-averaged even though its plain iterate converges exactly. The
    # average then dragged the transient mass into the answer and reported 0.978
    # on a seat whose true weight is 1.0.
    core = closed[0]
    if len(core) == n:
        settles = bool(primitive)
    elif len(core) > 1:
        settles = _is_primitive([[adj[i][j] for j in core] for i in core])
    else:
        settles = bool(adj[core[0]][core[0]])

    pi = [1.0 / n] * n
    running = [0.0] * n
    iterations = 0
    for k in range(1, 5001):
        nxt = [sum(w[i][j] * pi[i] for i in range(n)) for j in range(n)]
        total = sum(nxt)
        if total <= 1e-15:
            break
        nxt = [v / total for v in nxt]
        running = [running[j] + nxt[j] for j in range(n)]
        delta = max(abs(nxt[j] - pi[j]) for j in range(n))
        pi = nxt
        iterations = k
        if delta < 1e-13:
            break
    if not settles:
        # Cesàro average: the plain iterate cycles, its running mean does not.
        # Only for a genuinely PERIODIC chain — a settling one must keep its last
        # iterate, or the transient is averaged into the limit.
        total = sum(running) or 1.0
        pi = [v / total for v in running]

    gap: Optional[float] = None
    if primitive:
        deflated = [[w[i][j] - pi[j] for j in range(n)] for i in range(n)]
        rng = random.Random(_SEED)
        v = [rng.uniform(-1.0, 1.0) for _ in range(n)]
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        v = [x / norm for x in v]
        growth: list[float] = []
        for k in range(120):
            nv = [sum(deflated[i][j] * v[i] for i in range(n)) for j in range(n)]
            norm = math.sqrt(sum(x * x for x in nv))
            if norm <= 1e-14:
                growth.append(0.0)
                break
            v = [x / norm for x in nv]
            if k >= 80:
                growth.append(norm)
        if growth:
            positive = [g for g in growth if g > 0]
            lam2 = (
                math.exp(mean([math.log(g) for g in positive]))
                if positive
                else 0.0
            )
            gap = round(max(0.0, min(1.0, 1.0 - lam2)), 4)

    forecast = None
    if innate is not None and len(innate) == n:
        forecast = round(sum(pi[i] * innate[i] for i in range(n)), 4)

    return {
        "centrality": [round(v, 4) for v in pi],
        "spectral_gap": gap,
        "converges": bool(settles),
        "iterations": iterations,
        "consensus_forecast": forecast if settles else None,
        "primitive": bool(primitive),
        "irreducible": bool(irreducible),
        "closed_groups": closed,
        "note": (
            (
                "Left Perron vector by power iteration; spectral gap from the "
                "growth rate of the deflated matrix."
                + (
                    ""
                    if irreducible
                    else (
                        f" This room is not strongly connected: seats {core} "
                        "listen to nobody outside themselves while the rest "
                        "listen inward, so the whole consensus is theirs. The "
                        "centrality is still unique — one closed group means one "
                        "stationary distribution."
                    )
                )
            )
            if settles
            else (
                "W is irreducible but periodic: a unique stationary distribution "
                "exists and is reported as the Cesàro limit, but opinions cycle "
                "rather than settle, so there is no consensus to forecast."
            )
        ),
    }


# ══════════════════════ 3-4. equilibrium and absence ══════════════════════


def equilibrium(
    w: Optional[Sequence[Sequence[float]]],
    susceptibility: Optional[Sequence[float]],
    innate: Optional[Sequence[float]],
) -> Optional[list[float]]:
    """The Friedkin-Johnsen fixed point x* = (I − ΛW)⁻¹(I − Λ)u.

    Solved by the iteration x ← ΛWx + (I−Λ)u rather than by forming the inverse,
    and the reason is a limit case rather than a performance one: at Λ = I the
    matrix I − ΛW is SINGULAR (W has eigenvalue 1), so the closed form does not
    exist — while the iteration is just DeGroot averaging and still converges,
    to π'u, whenever W is primitive. Writing the solve as an iteration therefore
    covers the fully-open room and the mixed room with one piece of code.

    The two ends of the range are worth stating because they are the sanity
    checks: a fully STUBBORN room (all λ = 0) returns u exactly — nobody moves,
    so the equilibrium is the opening ballot — and a fully OPEN room (all λ = 1)
    converges to the DeGroot consensus π'u, identical for every member. Anything
    in between lands between them, and the spread that survives is the part of
    the disagreement that deliberation structurally cannot remove.

    Returns None when the iteration does not converge, which for a row-stochastic
    W and λ ≤ 1 means a periodic room whose positions cycle forever.
    """
    if not w or susceptibility is None or innate is None:
        return None
    n = len(w)
    if len(susceptibility) != n or len(innate) != n:
        return None
    x = list(innate)
    for _ in range(20000):
        nxt = [
            susceptibility[i] * sum(w[i][j] * x[j] for j in range(n))
            + (1.0 - susceptibility[i]) * innate[i]
            for i in range(n)
        ]
        delta = max(abs(nxt[i] - x[i]) for i in range(n))
        x = nxt
        if delta < 1e-12:
            return [round(v, 4) for v in x]
    return None


def counterfactual_absence(
    w: Optional[Sequence[Sequence[float]]],
    susceptibility: Optional[Sequence[float]],
    innate: Optional[Sequence[float]],
    names: Optional[Sequence[str]],
) -> list[dict]:
    """Leave-one-out: how far the room's landing point moves without each member.

    Remove member k, drop its row and column from W, renormalise the remaining
    rows back onto the simplex, and re-solve the equilibrium. The comparison is
    against the full-room equilibrium RESTRICTED TO THE SAME SURVIVORS, so the
    delta is "where the others end up", not an artefact of averaging over a
    different set of people.

    A survivor whose remaining weights sum to zero placed all of its trust in the
    member who left. Renormalising is impossible, so that row becomes self-weight
    1.0 — the member is left with nobody to listen to and stays on its opening
    ballot. Flagged per row in `orphaned` rather than silently smoothed.

    Sorted by |Δ| on the room mean, descending: the top row is the member whose
    presence changed the outcome most, which is a different question from who
    talked most and usually a different answer.
    """
    if not w or susceptibility is None or innate is None:
        return []
    n = len(w)
    if len(susceptibility) != n or len(innate) != n or n < 3:
        return []
    labels = list(names) if names and len(names) == n else [f"member_{i}" for i in range(n)]
    full = equilibrium(w, susceptibility, innate)
    if full is None:
        return []

    out: list[dict] = []
    for k in range(n):
        keep = [i for i in range(n) if i != k]
        sub_w: list[list[float]] = []
        orphaned: list[str] = []
        for i in keep:
            row = [w[i][j] for j in keep]
            total = sum(row)
            if total <= 1e-9:
                row = [1.0 if j == i else 0.0 for j in keep]
                orphaned.append(labels[i])
            else:
                row = [v / total for v in row]
            sub_w.append(row)
        sub_x = equilibrium(
            sub_w, [susceptibility[i] for i in keep], [innate[i] for i in keep]
        )
        if sub_x is None:
            out.append(
                {
                    "name": labels[k],
                    "index": k,
                    "delta_room_mean": None,
                    "abs_delta": None,
                    "note": "the room without this member does not settle",
                }
            )
            continue
        base = mean([full[i] for i in keep])
        without = mean(sub_x)
        per_member = [
            {"name": labels[i], "delta": round(sub_x[p] - full[i], 4)}
            for p, i in enumerate(keep)
        ]
        out.append(
            {
                "name": labels[k],
                "index": k,
                "delta_room_mean": round(without - base, 4),
                "abs_delta": round(abs(without - base), 4),
                "room_mean_with": round(base, 4),
                "room_mean_without": round(without, 4),
                "per_member_delta": per_member,
                "largest_single_shift": max(
                    per_member, key=lambda d: abs(d["delta"])
                ),
                "orphaned": orphaned,
            }
        )
    out.sort(key=lambda d: (d["abs_delta"] is None, -(d["abs_delta"] or 0.0)))
    return out


# ══════════════════════ 5. preference falsification ══════════════════════


def preference_falsification(
    public_by_round: Sequence,
    private_by_round: Sequence,
    names: Optional[Sequence[str]] = None,
) -> dict:
    """The gap between what members say and what they think (Kuran).

    A real boardroom cannot measure this about itself: everyone knows their own
    gap and nobody knows the room's. Here both series exist by construction, so
    the room-level mean gap is directly observable — and its per-round trajectory
    is the interesting object, because a gap that WIDENS as the room converges is
    the signature of a false consensus rather than a real one.

    THE RESAMPLING UNIT IS THE MEMBER, not the member-round. A member's gap at
    round 3 and at round 4 are the same person's disposition observed twice;
    treating them as independent would inflate the sample by the round count and
    shrink the interval by its square root. So the CI is a paired bootstrap over
    members via `analytics.bootstrap_ci`, which guards at n ≥ 8 and returns None
    below it. A 5-seat room therefore gets NO interval, and that is the correct
    output — `gap_ci` is None and `gap_ci_note` says why. Do not substitute a
    normal-approximation interval on five numbers.

    THE TEST IS AN EXACT SIGN-FLIP RANDOMISATION, which does work at n = 5: under
    the null that a member is as likely to overstate as to understate, every one
    of the 2ⁿ sign assignments of the per-member mean gaps is equally likely, so
    enumerating them gives an exact p with no distributional assumption. It is
    enumerated exactly up to 14 members (16,384 assignments) and drawn 4,000
    times, seeded, above that.

    BUT THE TEST HAS A FLOOR, and it is reported. The smallest two-sided p the
    enumeration can produce is 2/2ⁿ: 0.0625 at 5 members, 0.031 at 6, 0.016 at 7.
    A 5-member room CANNOT reach p < 0.05 no matter how large or consistent its
    gap, so `p_floor` ships alongside `p_value` and `supported` is False whenever
    the floor exceeds the level. A test that is structurally unable to fire is
    not evidence of absence.
    """
    pub = _as_replicates(public_by_round)
    prv = _as_replicates(private_by_round)
    if not pub or not prv:
        return _gap_refusal("No public or private position series were supplied.")
    n_rounds = min(len(pub[0]), len(prv[0]))
    n = len(pub[0][0]) if n_rounds else 0
    if n < 2 or n_rounds < 1:
        return _gap_refusal(
            f"{n_rounds} round(s) over {n} member(s) is not enough to compute a gap."
        )
    labels = list(names) if names and len(names) == n else [f"member_{i}" for i in range(n)]

    # gap[t][i], averaged over whatever replicates were supplied.
    per_round: list[list[float]] = []
    for t in range(n_rounds):
        row = []
        for i in range(n):
            vals = [
                rp[t][i] - rq[t][i]
                for rp, rq in zip(pub, prv)
                if len(rp) > t and len(rq) > t and len(rp[t]) > i and len(rq[t]) > i
            ]
            row.append(mean(vals) if vals else 0.0)
        per_round.append(row)

    per_member = [mean([per_round[t][i] for t in range(n_rounds)]) for i in range(n)]
    room_gap = mean(per_member)
    ci = bootstrap_ci(per_member, seed=_SEED)

    observed = abs(room_gap)
    if n <= 14:
        total = 1 << n
        hits = 0
        for mask in range(total):
            flipped = mean(
                [
                    (-v if (mask >> i) & 1 else v)
                    for i, v in enumerate(per_member)
                ]
            )
            if abs(flipped) >= observed - 1e-15:
                hits += 1
        p_value = hits / total
        p_floor = 2.0 / total
        exact = True
    else:
        rng = random.Random(_SEED)
        hits = 0
        for _ in range(_PERM_DRAWS):
            flipped = mean([v if rng.random() < 0.5 else -v for v in per_member])
            if abs(flipped) >= observed - 1e-15:
                hits += 1
        p_value = (1.0 + hits) / (1.0 + _PERM_DRAWS)
        p_floor = 1.0 / (1.0 + _PERM_DRAWS)
        exact = False

    widening = None
    if n_rounds >= 3:
        firsts = mean(per_round[0])
        lasts = mean(per_round[-1])
        widening = round(lasts - firsts, 4)

    return {
        "names": labels,
        "per_member_gap": [round(v, 4) for v in per_member],
        "room_mean_gap": round(room_gap, 4),
        "gap_ci": list(ci) if ci else None,
        "gap_ci_note": (
            None
            if ci
            else (
                f"No interval: the paired bootstrap resamples MEMBERS and there "
                f"are {n}, below the n ≥ 8 guard in analytics.bootstrap_ci. The "
                "sign test below is exact at this size and is the evidence."
            )
        ),
        "gap_by_round": [round(mean(row), 4) for row in per_round],
        "gap_by_round_per_member": [[round(v, 4) for v in row] for row in per_round],
        "gap_widening": widening,
        "p_value": round(p_value, 4),
        "p_floor": round(p_floor, 4),
        "exact_test": exact,
        "supported": bool(p_value < 0.05 and p_floor <= 0.05),
        "direction": (
            "members talk more in favour than they believe"
            if room_gap > 0
            else (
                "members talk more against than they believe"
                if room_gap < 0
                else "no net gap"
            )
        ),
        "refused": None,
        "method": (
            "public − private per member per round; CI = paired member-level BCa "
            "bootstrap; p = exact sign-flip randomisation over 2^n assignments "
            "of the per-member mean gaps"
        ),
    }


def _gap_refusal(reason: str) -> dict:
    return {
        "names": None,
        "per_member_gap": None,
        "room_mean_gap": None,
        "gap_ci": None,
        "gap_ci_note": None,
        "gap_by_round": None,
        "gap_by_round_per_member": None,
        "gap_widening": None,
        "p_value": None,
        "p_floor": None,
        "exact_test": False,
        "supported": False,
        "direction": None,
        "refused": reason,
        "method": "preference falsification (not run)",
    }


# ══════════════════════ 6. conformity against a Bayesian ══════════════════════


def conformity_vs_bayes(trajectories: Sequence, innate: Sequence) -> dict:
    """Does the room's agreement outrun the information available to produce it?

    A room that converges has either learned something or fallen in line, and the
    two look identical in the trajectory. What separates them is RATE: pooling
    independent information buys agreement at a known, unforgiving price, and
    conformity does not pay it.

    THE BENCHMARK, derived. Precision adds. A Bayesian who has pooled k
    independent signals of equal precision has posterior SD σ/√k, so pooling
    divides spread by √k and nothing else. Take the room's cross-member SD as
    tracking that individual posterior SD — a member reduces its distance from
    the room only as fast as its own uncertainty falls — and the benchmark for
    the contraction by round t is

        SD(t)/SD(0) = 1/√k(t),      k(t) = 1 + t·(n−1)

    i.e. each round each member absorbs ONE independent argument from each of the
    other n−1. Divide the observed contraction by the benchmark's, per round:

        ratio(t) = ln(SD(0)/SD(t)) / (½·ln(1 + t·(n−1)))

    with 1.0 meaning the room agreed exactly as fast as pooling justifies.

    THE HEADLINE IS max_t ratio(t), NOT ratio(T), and that is a correction rather
    than a preference. Measured: a DeGroot conformist (W uniform, λ = 0.85)
    collapses its whole disagreement in ROUND ONE and then sits still, so its
    endpoint ratio decays as the benchmark keeps accruing — 1.34 at T = 4 but
    1.09 at T = 8, i.e. watching a conformist room for longer made it look more
    rational. That is the wrong reading: after eight rounds of talking, pooling
    would indeed justify the agreement it reached in the first. The claim
    conformity makes is about TIMING — the room agreed before it had heard the
    arguments — so the statistic is the worst round, and the verdict is phrased as
    a timing claim. `ratio_by_round` ships so the whole path is visible.

    THE ASSUMPTIONS, AND WHERE THIS IS AN APPROXIMATION. Three things are being
    assumed and all three are worth doubting.

      · Independence. k(t) counts every argument as fresh. Real deliberation
        recycles, so the true k is smaller and the true benchmark contraction
        slower — meaning this benchmark is GENEROUS to the room and errs toward
        calling conformity information rather than the reverse. That is the
        direction to err in.
      · The bridge from posterior SD to cross-member SD. It is an
        approximation, and the two limits show why no exact version is
        available from what a room produces. If public statements revealed
        their speakers' evidence exactly, every member would condition on the
        same set and a fully rational room would reach consensus in ONE round —
        so no observed collapse could ever be "too fast". If statements revealed
        nothing, no collapse at all would be warranted and any collapse would be
        conformity. Neither limit is measurable here. The 1/√k law is the middle
        case with the property the measurement needs: it is a SCHEDULE, so a
        room can be seen to outrun it.
      · Equal precision. Members are treated as equally informative. A room
        with one genuine expert pools faster than this allows and will read
        below 1.0, which is the safe direction.

    So the verdict wording is deliberately narrow: "faster than the most
    generous rational pooling schedule this room's speaking capacity allows".
    It is not a claim that the members are irrational, and it is not a claim
    about a specific member.

    THE RATIO ALONE IS NOT A VERDICT, BECAUSE ITS NOISE IS ENORMOUS AT ROOM SIZE.
    A cross-member SD computed on five numbers is a rough thing, and the ratio is
    a log ratio of two of them. Measured on synthetic agents that pool exactly by
    the benchmark (so the truth is 1.0 by construction) the ratio's own 5-95%
    spread was [0.35, 1.40] at 5 members × 4 rounds and [0.00, 2.98] at 3
    members — at three seats the estimator can read anything. A fixed band would
    therefore fire on noise at small n and never fire at large n.

    So the verdict is gated on a DESIGN-MATCHED NULL, computed at runtime. 400
    synthetic rational rooms are simulated with this room's own member count,
    round count, opening spread and 0.05 quantisation; each yields its own
    max-over-rounds ratio; and `p_value` is the fraction of them reaching the
    observed one. That is the same discipline the rest of this codebase applies —
    calibrate against a null with a known answer — and it makes the test
    automatically conservative exactly where the estimator is noisy.

    CALIBRATION, BOTH WAYS. 150 trials per cell, positions quantised to 0.05,
    across six designs (members × rounds × replicates). "fires" is the rate at
    which the verdict reads CONFORMITY:

      Synthetic agents that pool exactly by the benchmark — each holding the
      running mean of 1 + t·(n−1) independent signals, so the truth is 1.0 by
      construction:

        design      5×4×3  5×4×8  5×8×3  9×4×3  9×8×8  3×4×3
        ratio med    1.12   1.09   1.17   1.06   1.03   1.31
        fires          3%     2%     6%     3%     7%     4%

      That is the level: 2-7% against a nominal 5%, flat across designs, with the
      ratio landing on 1.0 as it must. The 3-member cell reads 1.31 because a
      cross-member SD of three numbers is a very rough thing — the estimator is
      biased upward there by the max-over-rounds, and the null absorbs it rather
      than letting it become a false verdict.

      Synthetic DeGroot conformists (W uniform, susceptibility λ, so a fraction
      1−λ of the opening disagreement survives each round):

        λ       5×4×3        5×4×8        9×4×3        9×8×8
        0.95   2.04 / 85%   2.06 / 95%   1.55 / 93%   1.55 / 98%
        0.85   1.57 / 48%   1.56 / 91%   1.21 / 12%   1.21 / 75%
        0.70   1.06 /  0%   1.05 /  0%   0.81 /  0%   0.81 /  0%
        0.30   0.33 /  0%   0.32 /  0%   0.25 /  0%   0.25 /  0%
        0.05   0.06 /  0%   0.05 /  0%   0.04 /  0%   0.04 /  0%

      So the measure reads > 1 on conformists and < 1 on anchored rooms as
      required, and the crossover sits at λ ≈ 0.72 — a room that discards about
      70% of its opening disagreement in the first round is doing exactly what
      pooling justifies, and one that discards 95% is not. Power is a function of
      replicates far more than of rounds (48% → 91% going from 3 to 8 replicates
      at 5 members, while going from 4 to 8 rounds changes nothing), which is
      worth knowing before spending a budget on either axis. Below λ = 0.70 at 9
      members the verdict is UNDER-CONVERGED in 63-100% of trials: at nine seats
      the benchmark is far more demanding, and a room that keeps 30% of its
      disagreement really has stopped short of what its own information supports.

    GATES. The baseline spread must be real: with SD(0) below one half of the
    0.05 position quantum there is nothing to collapse and the ratio is a
    division by rounding error, so the estimator refuses. Fewer than 3 members or
    2 rounds refuses. An exactly unanimous round floors its SD at 0.005 — one
    tenth of the quantum — to keep the logarithm finite, and sets `sd_floored` so
    the ratio is read as a lower bound.
    """
    reps = _as_replicates(trajectories)
    if not reps:
        return _conformity_refusal("No public position trajectories were supplied.")
    n_rounds = len(reps[0])
    n = len(reps[0][0]) if n_rounds else 0
    if n < 3 or n_rounds < 2:
        return _conformity_refusal(
            f"{n_rounds} round(s) over {n} member(s): the collapse rate needs at "
            "least 3 members and 2 deliberation rounds."
        )
    innate_by_rep = _as_innate(innate, len(reps))
    if any(len(u) != n for u in innate_by_rep):
        return _conformity_refusal(
            "The round-0 private ballot does not have one position per member."
        )

    sd0 = mean([stdev(u) for u in innate_by_rep])
    sd_path = [
        mean([stdev(rep[t]) for rep in reps if len(rep) > t]) for t in range(n_rounds)
    ]
    if sd0 < _POSITION_QUANTUM / 2.0:
        return _conformity_refusal(
            f"The opening ballot spread is {sd0:.4f}, inside the "
            f"{_POSITION_QUANTUM} quantisation of a declared position. The room "
            "started in agreement, so there is no collapse to attribute to "
            "anything."
        )

    floored = any(v < _SD_FLOOR for v in sd_path)
    ratios = _ratio_path(sd0, sd_path, n)
    ratio = max(ratios)
    worst_round = ratios.index(ratio) + 1

    null = _rational_null_ratios(n, n_rounds, sd0, len(reps))
    n_ge = sum(1 for v in null if v >= ratio - 1e-12)
    p_value = (1.0 + n_ge) / (1.0 + len(null))
    null.sort()
    null_median = null[len(null) // 2]
    null_p05 = null[int(0.05 * len(null))]
    null_p95 = null[min(len(null) - 1, int(0.95 * len(null)))]

    if p_value < 0.05:
        verdict = (
            f"CONFORMITY. By round {worst_round} the room had already agreed "
            f"{ratio:.2f}× further than the most generous rational pooling "
            "schedule its speaking capacity allows, and only "
            f"{100 * p_value:.1f}% of genuinely-pooling rooms of this size and "
            "length ever look this convergent. The agreement arrived before the "
            "arguments that would justify it."
        )
    elif ratio < null_p05:
        verdict = (
            f"UNDER-CONVERGED. At its fastest the room agreed only {ratio:.2f}× "
            "the benchmark rate, below the "
            f"[{null_p05:.2f}, {null_p95:.2f}] a genuinely-pooling room of this "
            "size spans. Members stopped short of the agreement the information "
            "in the room supports — the opposite failure from groupthink, and "
            "usually a sign of positions held for reasons the arguments cannot "
            "touch."
        )
    else:
        verdict = (
            f"Consistent with rational pooling: at its fastest the room agreed "
            f"{ratio:.2f}× the benchmark rate, and rooms that genuinely pool "
            f"reach that {100 * p_value:.0f}% of the time at "
            f"{n} members × {n_rounds} rounds. No conformity is demonstrated — "
            "and at this size the test's resolution is coarse, so nor is its "
            f"absence (a rational room's ratio spans [{null_p05:.2f}, "
            f"{null_p95:.2f}] here)."
        )

    return {
        "n_members": n,
        "n_rounds": n_rounds,
        "n_replicates": len(reps),
        "sd_innate": round(sd0, 4),
        "sd_by_round": [round(v, 4) for v in sd_path],
        "sd_final": round(sd_path[-1], 4),
        "sd_floored": bool(floored),
        "observed_log_contraction": round(math.log(sd0 / max(sd_path[-1], _SD_FLOOR)), 4),
        "benchmark_log_contraction": round(0.5 * math.log(1.0 + n_rounds * (n - 1)), 4),
        "benchmark_k_final": round(1.0 + n_rounds * (n - 1), 2),
        "ratio_by_round": [round(v, 3) for v in ratios],
        "conformity_ratio": round(ratio, 3),
        "worst_round": worst_round,
        "p_value": round(p_value, 4),
        "null_median": round(null_median, 3),
        "null_p05": round(null_p05, 3),
        "null_p95": round(null_p95, 3),
        "null_draws": len(null),
        "supported": bool(p_value < 0.05),
        "verdict": verdict,
        "refused": None,
        "method": (
            "observed cross-member SD contraction against the 1/√k Bayesian "
            "pooling law with k(t) = 1 + t·(n−1) — one fresh independent argument "
            "absorbed per speaker per round, the fastest schedule the room's "
            "speaking capacity allows — taken at its worst round, with a p-value "
            "from 400 design-matched synthetic rooms that pool rationally"
        ),
    }


def _ratio_path(sd0: float, sd_path: Sequence[float], n: int) -> list[float]:
    """Observed over benchmark log-contraction, per round."""
    out = []
    for t, sd in enumerate(sd_path, start=1):
        observed = math.log(sd0 / max(sd, _SD_FLOOR))
        benchmark = 0.5 * math.log(1.0 + t * (n - 1))
        out.append(observed / benchmark if benchmark > 1e-9 else 0.0)
    return out


def _rational_null_ratios(
    n: int, n_rounds: int, sd0: float, n_reps: int
) -> list[float]:
    """max-over-rounds ratios from rooms that pool exactly by the benchmark.

    Each synthetic member starts with one private signal of SD `sd0` about a
    latent truth and each round absorbs one fresh independent signal from each of
    the other n−1, holding the running mean as its position. Its position is
    therefore the mean of k(t) = 1 + t·(n−1) independent draws — the benchmark,
    realised — so the ratio is 1.0 in expectation and everything away from 1.0 in
    these draws is the estimator's own noise at this room's size.

    THE NULL MUST BE AVERAGED OVER THE SAME NUMBER OF REPLICATES AS THE
    OBSERVATION, and getting that wrong made the test inert. The observed SD path
    is the mean over R replicates, so its noise falls as √R; a one-replicate null
    is therefore much noisier than the statistic it is calibrating, its upper
    quantile is far too high, and nothing ever reaches it. Measured with the
    mismatch in place, a DeGroot conformist at λ = 0.85 was detected 1-8% of the
    time — a test that could not fire. With the replicate count matched it detects
    the same room 48% of the time at 5 × 4 × 3 and 91% at 5 × 4 × 8, and at
    λ = 0.95 it detects 77-98% everywhere, while the false-positive rate on
    genuinely-pooling rooms stays at 2-7% against the nominal 5%.

    The sum of the (n−1) fresh signals is drawn as ONE Gaussian of variance
    (n−1)σ², which is exact and turns 400 × R × T × n × (n−1) draws into
    400 × R × T × n. Positions are snapped to the same 0.05 grid the real ones
    arrive on, because that quantisation is part of the noise being calibrated
    against. Seeded from the design, so the p-value is reproducible.
    """
    rng = random.Random(_SEED * 7 + n * 31 + n_rounds * 5 + n_reps)
    spread = max(sd0, _POSITION_QUANTUM)
    step_sd = spread * math.sqrt(n - 1)
    reps = max(1, n_reps)
    out: list[float] = []
    for _ in range(_RATIONAL_DRAWS):
        bases: list[float] = []
        paths: list[list[float]] = []
        for _r in range(reps):
            own = [rng.gauss(0.0, spread) for _ in range(n)]
            totals = list(own)
            bases.append(stdev([_quantise(0.5 + v) for v in own]))
            path = []
            for t in range(1, n_rounds + 1):
                k = 1 + t * (n - 1)
                for i in range(n):
                    totals[i] += rng.gauss(0.0, step_sd)
                path.append(stdev([_quantise(0.5 + totals[i] / k) for i in range(n)]))
            paths.append(path)
        base = mean(bases)
        if base < _POSITION_QUANTUM / 2.0:
            continue
        averaged = [mean([p[t] for p in paths]) for t in range(n_rounds)]
        out.append(max(_ratio_path(base, averaged, n)))
    return out or [1.0]


def _conformity_refusal(reason: str) -> dict:
    return {
        "n_members": 0,
        "n_rounds": 0,
        "n_replicates": 0,
        "sd_innate": None,
        "sd_by_round": None,
        "sd_final": None,
        "sd_floored": False,
        "observed_log_contraction": None,
        "benchmark_log_contraction": None,
        "benchmark_k_final": None,
        "ratio_by_round": None,
        "conformity_ratio": None,
        "worst_round": None,
        "p_value": None,
        "null_median": None,
        "null_p05": None,
        "null_p95": None,
        "null_draws": 0,
        "supported": False,
        "verdict": None,
        "refused": reason,
        "method": "conformity vs Bayesian pooling (not run)",
    }


# ══════════════════════ 7. cascade / herding on predecessors ══════════════════


def _normalise_orders(
    orders: Optional[Sequence], n_reps: int, n_rounds: int, n: int
) -> Optional[list[list[list[int]]]]:
    """Accept `orders[r]` or `orders[r][t]`; return `orders[r][t]`."""
    if not orders:
        return None
    out: list[list[list[int]]] = []
    for r in range(n_reps):
        entry = orders[r] if r < len(orders) else orders[-1]
        if entry and isinstance(entry[0], (list, tuple)):
            rounds = [list(x) for x in entry]
            while len(rounds) < n_rounds:
                rounds.append(list(rounds[-1]))
            out.append(rounds[:n_rounds])
        else:
            out.append([list(entry) for _ in range(n_rounds)])
    if any(sorted(rd) != list(range(n)) for rep in out for rd in rep):
        return None
    return out


def cascade_scores(
    trajectories: Sequence,
    private_by_round: Sequence,
    speaking_orders: Sequence,
    names: Optional[Sequence[str]] = None,
    *,
    innate: Optional[Sequence] = None,
) -> dict:
    """How much of a public position is the people who spoke first, not the argument.

    Per member, regress its PUBLIC position at round t on two things: its own
    private position going into that round, and the mean public position of the
    members who spoke BEFORE it in that same round. The incremental R² of the
    second over the first alone is the cascade score — the share of a member's
    stated position that predecessors explain over and above what it already
    thought. High is herding: the member's words track whoever went first rather
    than its own updated belief.

    "Going into that round" is round t−1's private position. Round 0's baseline is
    the opening ballot, which is why the ballot is worth its own round — pass it
    as the keyword-only `innate` and the score is estimated on all T rounds
    instead of T−1. Without it the first speaking round is dropped, which at the
    default 4 rounds costs a quarter of the sample.

    WHY THE SPEAKING ORDER IS REQUIRED and not inferred. Within-round precedence
    is the entire identifying variation; a member's predecessors change from
    replicate to replicate, and that is what separates "responds to what was just
    said" from "correlated with the room". A member who happens to speak FIRST in
    every replicate has no predecessors anywhere and is unscoreable — reported as
    a refusal naming the reason, not as a zero. Rotating the order across
    replicates is what prevents that.

    Gated at 6 observations per member (3 regression parameters plus 3). Members
    below the gate are listed in `refusals` with their observation count.
    """
    pub = _as_replicates(trajectories)
    prv = _as_replicates(private_by_round)
    if not pub or not prv:
        return {
            "names": None,
            "per_member": [],
            "room_mean_cascade": None,
            "refusals": [],
            "refused": "No public or private position series were supplied.",
            "method": "cascade regression (not run)",
        }
    n_rounds = min(len(pub[0]), len(prv[0]))
    n = len(pub[0][0]) if n_rounds else 0
    labels = list(names) if names and len(names) == n else [f"member_{i}" for i in range(n)]
    orders = _normalise_orders(speaking_orders, len(pub), n_rounds, n)
    if orders is None:
        return {
            "names": labels,
            "per_member": [],
            "room_mean_cascade": None,
            "refusals": [],
            "refused": (
                "A speaking order per replicate (a permutation of every member "
                "index) is required: within-round precedence is the identifying "
                "variation, and there is nothing to estimate without it."
            ),
            "method": "cascade regression (not run)",
        }
    ballots = _as_innate(innate, len(pub)) if innate else None
    if ballots and any(len(b) != n for b in ballots):
        ballots = None

    per_member: list[dict] = []
    refusals: list[dict] = []
    for i in range(n):
        own: list[float] = []
        pred: list[float] = []
        target: list[float] = []
        for r, rep in enumerate(pub):
            if len(rep) < n_rounds or len(prv[r]) < n_rounds:
                continue
            for t in range(n_rounds):
                order = orders[r][t]
                pos = order.index(i)
                if pos == 0:
                    continue  # spoke first: no predecessors to herd on
                predecessors = order[:pos]
                if t >= 1:
                    prior = prv[r][t - 1][i]
                elif ballots is not None:
                    prior = ballots[r][i]
                else:
                    continue
                own.append(prior)
                pred.append(mean([rep[t][j] for j in predecessors]))
                target.append(rep[t][i])
        if len(target) < _MIN_CASCADE_OBS:
            refusals.append(
                {
                    "name": labels[i],
                    "index": i,
                    "n_obs": len(target),
                    "reason": (
                        f"{len(target)} usable observations against a 3-parameter "
                        f"regression (gate {_MIN_CASCADE_OBS}). A member with no "
                        "predecessors — one that spoke first every round — cannot "
                        "be scored at all; rotate the speaking order across "
                        "replicates."
                    ),
                }
            )
            per_member.append(
                {
                    "name": labels[i],
                    "index": i,
                    "n_obs": len(target),
                    "cascade_r2_gain": None,
                    "r2_private_only": None,
                    "r2_with_predecessors": None,
                    "predecessor_beta": None,
                }
            )
            continue
        base = _ols_r2([[1.0, o] for o in own], target)
        full_rows = [[1.0, o, p] for o, p in zip(own, pred)]
        full = _ols_r2(full_rows, target)
        beta = None
        xtx = [[sum(r[a] * r[b] for r in full_rows) for b in range(3)] for a in range(3)]
        xty = [sum(r[a] * y for r, y in zip(full_rows, target)) for a in range(3)]
        solved = _solve_normal(xtx, xty)
        if solved is not None:
            beta = round(solved[2], 4)
        gain = None
        if full is not None and base is not None:
            gain = round(max(0.0, full - base), 4)
        per_member.append(
            {
                "name": labels[i],
                "index": i,
                "n_obs": len(target),
                "cascade_r2_gain": gain,
                "r2_private_only": round(base, 4) if base is not None else None,
                "r2_with_predecessors": round(full, 4) if full is not None else None,
                "predecessor_beta": beta,
            }
        )

    scored = [m["cascade_r2_gain"] for m in per_member if m["cascade_r2_gain"] is not None]
    ranked = sorted(
        [m for m in per_member if m["cascade_r2_gain"] is not None],
        key=lambda d: -d["cascade_r2_gain"],
    )
    return {
        "names": labels,
        "per_member": per_member,
        "room_mean_cascade": round(mean(scored), 4) if scored else None,
        "most_herding": ranked[0]["name"] if ranked else None,
        "refusals": refusals,
        "refused": None
        if scored
        else "No member had enough predecessor observations to score.",
        "method": (
            "incremental R² of the mean public position of same-round "
            "predecessors over a member's own prior private position, pooled "
            "across replicates"
        ),
    }


# ══════════════════════ 8. camps and power ══════════════════════


def _bounded_confidence_camps(
    positions: Sequence[float], epsilon: float
) -> list[list[int]]:
    """Single-linkage camps on the 1-D position line, cutting gaps > epsilon."""
    order = sorted(range(len(positions)), key=lambda i: positions[i])
    camps: list[list[int]] = []
    current: list[int] = []
    for k, idx in enumerate(order):
        if not current:
            current = [idx]
            continue
        if positions[idx] - positions[order[k - 1]] > epsilon:
            camps.append(current)
            current = [idx]
        else:
            current.append(idx)
    if current:
        camps.append(current)
    return camps


def coalitions(
    final_public: Sequence[float],
    final_private: Optional[Sequence[float]],
    w: Optional[Sequence[Sequence[float]]] = None,
    threshold: Optional[float] = None,
    names: Optional[Sequence[str]] = None,
) -> dict:
    """Camps in the final positions, and — given a threshold — who holds the power.

    WHY NOT COMMUNITY DETECTION ON W. Modularity maximisation and its relatives
    exist to find structure in a graph whose geometry is unknown. Here the
    geometry is known and it is one-dimensional: a position is a scalar on 0..1.
    Clustering the influence graph would answer a different question (who listens
    to whom) with far more machinery and far more ways to be wrong, and it is
    already answered by `influence_centrality`. So camps are single-linkage
    (bounded-confidence) clusters on the position line, which on 1-D data are
    exact, order-independent and have no seed.

    ε defaults to 0.15 — three steps of the 0.05 grid a declared position lands
    on — so a "camp" is a set of members no two adjacent members of which differ
    by more than three quantisation steps. `within_span` and `between_gap` ship
    with the partition so a reader can see whether the split is a canyon or a
    rounding artefact.

    PUBLIC AND PRIVATE CAMPS ARE BOTH REPORTED, and the difference between them
    is the point: a room with one public camp and two private ones has a
    consensus that nobody in it holds.

    POWER INDICES, when `threshold` is given. Each member's final position
    becomes a yes/no vote at that cut. The voting game is weighted by
    `influence_centrality(W)` — the room's own listening structure, which is what
    actually determines where a consensus lands — with a simple 0.5 quota on the
    normalised weights; without W, equal weights, which makes both indices
    degenerate to 1/n and is reported as such rather than dressed up. Shapley-
    Shubik is computed from the subset form Σ|S|!(n−|S|−1)!/n! over pivotal S,
    and Banzhaf from swing counts. Both are EXACT ENUMERATIONS over 2^(n−1)
    subsets, which is 256 at the 9-seat maximum — no sampling, no seed.
    """
    n = len(final_public)
    if n < 2:
        return {
            "epsilon": None,
            "public_camps": [],
            "private_camps": [],
            "hidden_split": None,
            "vote": None,
            "shapley_shubik": None,
            "banzhaf": None,
            "refused": "A room needs at least two members to have camps.",
            "method": "camps (not run)",
        }
    labels = list(names) if names and len(names) == n else [f"member_{i}" for i in range(n)]
    eps = 3.0 * _POSITION_QUANTUM if threshold is None else 3.0 * _POSITION_QUANTUM

    def _describe(positions: Sequence[float]) -> list[dict]:
        camps = _bounded_confidence_camps(positions, eps)
        out = []
        centres = [mean([positions[i] for i in c]) for c in camps]
        for k, camp in enumerate(camps):
            vals = [positions[i] for i in camp]
            out.append(
                {
                    "members": [labels[i] for i in camp],
                    "indices": list(camp),
                    "size": len(camp),
                    "centre": round(mean(vals), 4),
                    "within_span": round(max(vals) - min(vals), 4),
                    "between_gap": (
                        round(centres[k] - centres[k - 1], 4) if k > 0 else None
                    ),
                }
            )
        return out

    public_camps = _describe(final_public)
    private_camps = _describe(final_private) if final_private and len(final_private) == n else []

    result = {
        "epsilon": eps,
        "public_camps": public_camps,
        "private_camps": private_camps,
        "n_public_camps": len(public_camps),
        "n_private_camps": len(private_camps) or None,
        "hidden_split": (
            bool(private_camps and len(private_camps) > len(public_camps))
            if private_camps
            else None
        ),
        "vote": None,
        "shapley_shubik": None,
        "banzhaf": None,
        "refused": None,
        "method": (
            "single-linkage (bounded-confidence) camps on the 1-D position line "
            f"at ε = {eps:.2f}; power indices by exact subset enumeration"
        ),
    }
    if threshold is None:
        return result

    votes = [1 if p >= threshold else 0 for p in final_public]
    centrality = influence_centrality(w) if w else {"centrality": None}
    weights = centrality.get("centrality")
    if not weights or len(weights) != n:
        weights = [1.0 / n] * n
        weight_source = (
            "equal weights (no identified influence matrix), so both indices are "
            "1/n by construction and say nothing about this room"
        )
    else:
        weight_source = "influence centrality from the fitted W"
    total = sum(weights) or 1.0
    weights = [v / total for v in weights]
    quota = 0.5

    ss = [0.0] * n
    bz = [0] * n
    others_of = [[j for j in range(n) if j != i] for i in range(n)]
    fact = [math.factorial(k) for k in range(n + 1)]
    for i in range(n):
        rest = others_of[i]
        for size in range(len(rest) + 1):
            for subset in itertools.combinations(rest, size):
                base = sum(weights[j] for j in subset)
                if base < quota <= base + weights[i]:
                    ss[i] += fact[size] * fact[n - size - 1] / fact[n]
                    bz[i] += 1
    swings = sum(bz) or 1
    result["vote"] = {
        "threshold": threshold,
        "votes": votes,
        "yes": sum(votes),
        "no": n - sum(votes),
        "carries_by_headcount": bool(sum(votes) * 2 > n),
        "carries_by_influence": bool(
            sum(weights[i] for i in range(n) if votes[i]) >= quota
        ),
        "weights": [round(v, 4) for v in weights],
        "weight_source": weight_source,
    }
    result["shapley_shubik"] = [
        {"name": labels[i], "index": round(ss[i], 4)} for i in range(n)
    ]
    result["banzhaf"] = [
        {"name": labels[i], "index": round(bz[i] / swings, 4), "swings": bz[i]}
        for i in range(n)
    ]
    result["power_note"] = (
        f"Exact enumeration over 2^(n−1) = {1 << (n - 1)} subsets; no sampling. "
        "The game is weighted voting with a 0.5 quota on normalised influence "
        "weights, not headcount majority — headcount with equal weights gives "
        "every member the same index and measures nothing."
    )
    return result


# ══════════════════════ 9. the order confound ══════════════════════


def order_effect(trajectories: Sequence, speaking_orders: Sequence) -> dict:
    """How much of where the room landed is who happened to speak first.

    THIS IS WHY THE REPLICATES EXIST. One run cannot tell a real convergence from
    an artefact of the seating plan, and a room that lands somewhere different
    every time the order changes has not reached a conclusion — it has reached
    the first speaker's.

    Two measurements, because they answer different questions.

      · `eta_squared` — the share of variance in FINAL positions explained by
        replicate identity, after removing each member's own average (members are
        paired across replicates, so the member effect is a nuisance, not signal).
        Its p-value permutes each member's final positions ACROSS replicates
        independently, which is the exact null "replicate identity carries
        nothing" with the member effect held fixed. Seeded Monte Carlo with the
        add-one estimator, so p is never reported as zero.
      · `first_speaker_pull` — the correlation across replicates between the
        first speaker's opening position and the room's landing point. Its test
        is an EXACT permutation over all R! assignments of first speakers to
        landing points, and it has a floor: 1/R! two-sided-doubled, which is
        0.333 at 3 replicates and 0.083 at 4. Below 5 replicates this test
        CANNOT reach 0.05, and `p_floor` says so.

    Refuses with a reason below 2 replicates: there is no order effect to measure
    when only one order was run, and reporting zero would read as evidence of
    absence.
    """
    reps = _as_replicates(trajectories)
    if len(reps) < 2:
        return {
            "n_replicates": len(reps),
            "landing_points": [round(mean(r[-1]), 4) for r in reps] if reps else [],
            "landing_sd": None,
            "landing_range": None,
            "eta_squared": None,
            "eta_p_value": None,
            "first_speaker_pull": None,
            "pull_p_value": None,
            "pull_test_exact": False,
            "p_floor": None,
            "verdict": None,
            "refused": (
                f"{len(reps)} replicate(s): the speaking order was never varied, "
                "so its effect is not identified. This is the confound the "
                "replicates exist to expose — run at least 2, ideally 5."
            ),
            "method": "order effect (not run)",
        }
    n_rounds = len(reps[0])
    n = len(reps[0][0])
    finals = [rep[-1] for rep in reps if len(rep) == n_rounds and len(rep[-1]) == n]
    if len(finals) < 2:
        return {
            "n_replicates": len(reps),
            "landing_points": [],
            "landing_sd": None,
            "landing_range": None,
            "eta_squared": None,
            "eta_p_value": None,
            "first_speaker_pull": None,
            "pull_p_value": None,
            "pull_test_exact": False,
            "p_floor": None,
            "verdict": None,
            "refused": "Replicates were ragged; final rounds do not align.",
            "method": "order effect (not run)",
        }
    r_count = len(finals)
    landings = [mean(f) for f in finals]

    # η²: member effects removed first, so what is left is replicate identity.
    member_means = [mean([finals[r][i] for r in range(r_count)]) for i in range(n)]
    centred = [[finals[r][i] - member_means[i] for i in range(n)] for r in range(r_count)]

    def _eta(rows: list[list[float]]) -> float:
        grand = mean([v for row in rows for v in row])
        between = sum(n * (mean(row) - grand) ** 2 for row in rows)
        total = sum((v - grand) ** 2 for row in rows for v in row)
        return between / total if total > 1e-15 else 0.0

    eta = _eta(centred)
    rng = random.Random(_SEED)
    hits = 0
    for _ in range(_PERM_DRAWS):
        shuffled = [[0.0] * n for _ in range(r_count)]
        for i in range(n):
            col = [centred[r][i] for r in range(r_count)]
            rng.shuffle(col)
            for r in range(r_count):
                shuffled[r][i] = col[r]
        if _eta(shuffled) >= eta - 1e-15:
            hits += 1
    eta_p = (1.0 + hits) / (1.0 + _PERM_DRAWS)

    orders = _normalise_orders(speaking_orders, r_count, n_rounds, n)
    pull = None
    pull_p = None
    floor = None
    exact_pull = False
    if orders is not None:
        firsts = [orders[r][0][0] for r in range(r_count)]
        opening = [reps[r][0][firsts[r]] for r in range(r_count)]
        pull = _pearson(opening, landings)
        if pull is not None:
            # Both Pearson denominators are permutation-invariant, so the
            # permutation ranking is decided by |Σ a_{p(i)}·b_i| alone. Dropping
            # the denominators turns a full correlation per permutation into a
            # dot product and is not an approximation.
            ma, mb = mean(opening), mean(landings)
            ca = [v - ma for v in opening]
            cb = [v - mb for v in landings]
            observed = abs(sum(x * y for x, y in zip(ca, cb)))
            if math.factorial(r_count) <= 5040:
                perms = itertools.permutations(range(r_count))
                total_perm = math.factorial(r_count)
                exact_pull = True
                floor = 2.0 / total_perm
            else:
                rng2 = random.Random(_SEED + 3)
                idx = list(range(r_count))

                def _draws():
                    for _ in range(_PERM_DRAWS):
                        rng2.shuffle(idx)
                        yield tuple(idx)

                perms = _draws()
                total_perm = _PERM_DRAWS
                floor = 1.0 / (1.0 + _PERM_DRAWS)
            ge = sum(
                1
                for perm in perms
                if abs(sum(ca[p] * cb[i] for i, p in enumerate(perm)))
                >= observed - 1e-12
            )
            pull_p = ge / total_perm if exact_pull else (1.0 + ge) / (1.0 + total_perm)

    sd = stdev(landings)
    verdict = (
        "the landing point moved with the speaking order — read the room's "
        "conclusion as one draw, not a result"
        if eta_p < 0.05 or sd > _POSITION_QUANTUM
        else "the landing point held across speaking orders"
    )
    return {
        "n_replicates": r_count,
        "landing_points": [round(v, 4) for v in landings],
        "landing_sd": round(sd, 4),
        "landing_range": round(max(landings) - min(landings), 4),
        "eta_squared": round(eta, 4),
        "eta_p_value": round(eta_p, 4),
        "first_speaker_pull": round(pull, 4) if pull is not None else None,
        "pull_p_value": round(pull_p, 4) if pull_p is not None else None,
        "pull_test_exact": exact_pull,
        "p_floor": round(floor, 4) if floor is not None else None,
        "verdict": verdict,
        "refused": None,
        "method": (
            "η² of replicate identity on member-centred final positions with a "
            "stratified permutation null; plus the exact R!-permutation test of "
            "the first speaker's pull on the landing point"
        ),
    }


def _pearson(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    if len(a) != len(b) or len(a) < 3:
        return None
    ma, mb = mean(a), mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    if da < 1e-12 or db < 1e-12:
        return None
    return num / (da * db)


# ══════════════════════ 10. the placebo control ══════════════════════


def arm_summary(
    trajectories: Sequence,
    private_by_round: Sequence,
    innate: Sequence,
    *,
    conformity: Optional[dict] = None,
) -> dict:
    """Build the summary `placebo_contrast` compares. Call this per arm.

    Exported so the orchestration layer does not have to hand-assemble the dict
    and risk the two arms being summarised differently — which would make the
    control meaningless in exactly the way that is hardest to notice. The fields:

      n_members, n_rounds, n_replicates — the design of this arm.
      sd_innate      — cross-member SD of the round-0 private ballot.
      sd_final       — cross-member SD of the last public round.
      convergence_ratio — sd_final / sd_innate. Lower = converged harder.
      mean_gap       — room mean public − private over all rounds.
      gap_final_round— the same on the last round only.
      conformity_ratio — from `conformity_vs_bayes`, or None if it refused.

    Pass `conformity` if you already have this arm's `conformity_vs_bayes` result;
    it is otherwise recomputed, and its design-matched null is the most expensive
    thing in this module.
    """
    reps = _as_replicates(trajectories)
    prv = _as_replicates(private_by_round)
    if not reps or not reps[0]:
        return {
            "n_members": 0,
            "n_rounds": 0,
            "n_replicates": 0,
            "sd_innate": None,
            "sd_final": None,
            "convergence_ratio": None,
            "mean_gap": None,
            "gap_final_round": None,
            "conformity_ratio": None,
        }
    n_rounds = len(reps[0])
    n = len(reps[0][0])
    innate_by_rep = _as_innate(innate, len(reps))
    sd0 = mean([stdev(u) for u in innate_by_rep if len(u) == n]) if innate_by_rep else 0.0
    sd_t = mean([stdev(rep[-1]) for rep in reps if len(rep) == n_rounds])
    gaps: list[float] = []
    final_gaps: list[float] = []
    for r, rep in enumerate(reps):
        if r >= len(prv) or len(prv[r]) < n_rounds:
            continue
        for t in range(n_rounds):
            row = [rep[t][i] - prv[r][t][i] for i in range(n)]
            gaps.append(mean(row))
            if t == n_rounds - 1:
                final_gaps.append(mean(row))
    conf = conformity if conformity is not None else conformity_vs_bayes(reps, innate)
    return {
        "n_members": n,
        "n_rounds": n_rounds,
        "n_replicates": len(reps),
        "sd_innate": round(sd0, 4),
        "sd_final": round(sd_t, 4),
        "convergence_ratio": (
            round(sd_t / sd0, 4) if sd0 > _SD_FLOOR else None
        ),
        "mean_gap": round(mean(gaps), 4) if gaps else None,
        "gap_final_round": round(mean(final_gaps), 4) if final_gaps else None,
        "conformity_ratio": conf.get("conformity_ratio"),
    }


def placebo_contrast(real_summary: dict, placebo_summary: Optional[dict]) -> dict:
    """THE CONTROL. Real deliberation against the scrambled-statement arm.

    In the placebo arm each member is shown statements from a DIFFERENT room —
    fluent, on-topic, and about nothing these characters said. If that arm
    converges as hard and shows the same public/private gap, then convergence and
    falsification are properties of the model's disposition to agree with
    whatever is in its context, and the deliberation measured nothing. This is
    the single most important number in the study and it is the one a room
    cannot produce about itself.

    Both `real_summary` and `placebo_summary` are `arm_summary` dicts; use that
    function rather than assembling them by hand.

    THE COMPARISON IS DESCRIPTIVE ON PURPOSE. Replicate counts here are 1-8, so
    a test between arms would have essentially no power and a non-significant
    result would be read as "the placebo differed", which is the exact error this
    control exists to prevent. So the arms' numbers are reported side by side with
    an explicit `separation` label and the replicate counts that produced them,
    and the verdict speaks plainly when the placebo matched the real arm. Where a
    difference is claimed it is claimed as a difference in point estimates from
    small samples, and the field says so.
    """
    if not placebo_summary or not placebo_summary.get("n_replicates"):
        return {
            "ran": False,
            "real": real_summary,
            "placebo": None,
            "convergence_real": real_summary.get("convergence_ratio"),
            "convergence_placebo": None,
            "convergence_delta": None,
            "gap_real": real_summary.get("mean_gap"),
            "gap_placebo": None,
            "gap_delta": None,
            "separation": None,
            "verdict": (
                "NO CONTROL WAS RUN. Convergence and the public/private gap in "
                "this study are uncontrolled: a language model agrees with "
                "whatever is in its context, so neither number can be "
                "attributed to this room's deliberation rather than to that "
                "disposition. Set placebo_replicates ≥ 1."
            ),
            "refused": "placebo_replicates = 0",
            "method": "placebo contrast (not run)",
        }

    cr = real_summary.get("convergence_ratio")
    cp = placebo_summary.get("convergence_ratio")
    gr = real_summary.get("mean_gap")
    gp = placebo_summary.get("mean_gap")
    c_delta = round(cp - cr, 4) if cr is not None and cp is not None else None
    g_delta = round(gr - gp, 4) if gr is not None and gp is not None else None

    # A separation worth naming has to clear the quantisation of the underlying
    # positions; below that the two arms differ by rounding.
    if c_delta is None:
        separation = None
        verdict = (
            "The arms cannot be compared: at least one has no measurable opening "
            "spread to converge from."
        )
    elif c_delta <= 0.0:
        separation = "none — the placebo converged at least as hard"
        verdict = (
            "THE DELIBERATION MEASURED NOTHING about these characters. The "
            f"placebo arm, shown statements from a different room entirely, "
            f"converged to a spread of {cp} against the real arm's {cr}. Whatever "
            "agreement this room reached, a room reading unrelated text reached "
            "it too, so it is a property of the model's agreeableness. Do not "
            "report the convergence, the influence matrix, or the coalitions as "
            "findings about this cast."
        )
    elif c_delta < _POSITION_QUANTUM:
        separation = "marginal — within the position quantisation"
        verdict = (
            f"The real arm converged only {c_delta} further than the placebo "
            f"({cr} vs {cp}), which is inside the 0.05 grid a declared position "
            "lands on. Treat the deliberation-specific part of the convergence "
            "as unresolved at this replicate count "
            f"({real_summary.get('n_replicates')} real, "
            f"{placebo_summary.get('n_replicates')} placebo)."
        )
    else:
        separation = "clear — the real arm converged further"
        verdict = (
            f"The real arm converged to {cr} against the placebo's {cp}, a "
            f"difference of {c_delta} on the spread ratio. Some of this room's "
            "agreement is specific to what its members said to each other. This "
            "is a difference in point estimates from "
            f"{real_summary.get('n_replicates')} and "
            f"{placebo_summary.get('n_replicates')} replicates, not a tested one."
        )

    gap_note = None
    if g_delta is not None:
        if abs(g_delta) < 0.02:
            gap_note = (
                f"The public/private gap is the same in both arms ({gr} vs {gp}): "
                "members say something other than what they think whether or not "
                "the statements they hear have anything to do with them. Read the "
                "gap as a disposition of the model, not as this room's politics."
            )
        else:
            gap_note = (
                f"The gap is {gr} in the real arm and {gp} in the placebo, a "
                f"difference of {g_delta}."
            )

    return {
        "ran": True,
        "real": real_summary,
        "placebo": placebo_summary,
        "convergence_real": cr,
        "convergence_placebo": cp,
        "convergence_delta": c_delta,
        "gap_real": gr,
        "gap_placebo": gp,
        "gap_delta": g_delta,
        "conformity_real": real_summary.get("conformity_ratio"),
        "conformity_placebo": placebo_summary.get("conformity_ratio"),
        "separation": separation,
        "gap_note": gap_note,
        "verdict": verdict,
        "refused": None,
        "method": (
            "descriptive side-by-side of convergence ratio and public/private "
            "gap between the real and scrambled-statement arms; not a test, "
            "because 1-8 replicates per arm cannot support one"
        ),
    }


# ══════════════════════ 11. the assembled result ══════════════════════

_LIMITS = (
    "These are language-model characters, not people. Nothing here measures how a "
    "human boardroom behaves, and the cast's declared dispositions were asserted "
    "by a model rather than observed. "
    "The influence matrix W is estimated from what the model produced, so at best "
    "it describes the transcript rather than the characters — and its INDIVIDUAL "
    "ENTRIES are not recoverable at any design this product can run: measured "
    "against a uniform-row guess, a fitted entry is further from the truth than "
    "the guess until roughly 9 observations per parameter, which 9 members over 8 "
    "rounds and 8 replicates does not reach. What was measured to recover is the "
    "susceptibility ORDERING (who moved more than whom), the equilibrium, and the "
    "leave-one-out absence deltas; read those and not a single weight. "
    "A room's landing point moves with the speaking order, so one conclusion is "
    "one draw — order_effect is there to say how much. "
    "Positions arrive on a ~0.05 grid, so any difference below that is rounding, "
    "and several tests here cannot reach p < 0.05 at small counts at all (a "
    "5-member sign test floors at 0.0625; a 3-replicate order test at 0.333) — "
    "where that happens the floor is reported next to the p-value. "
    "Above all, read the placebo contrast first: if the arm shown statements from "
    "a different room converged as hard, then every number in this result is "
    "about the model's agreeableness and not about this deliberation."
)

_METHOD = (
    "Friedkin-Johnsen influence weights by constrained least squares on the "
    "reparametrised a = λ·W over {a ≥ 0, Σa ≤ 1}, pooled across replicates and "
    "refused below a measured observations-per-parameter floor; left Perron "
    "centrality and spectral gap by power iteration on the deflated matrix; the "
    "FJ fixed point by iteration (so the singular Λ = I case still resolves); "
    "leave-one-out counterfactual equilibria; preference falsification with a "
    "member-level BCa bootstrap and an exact sign-flip randomisation test; "
    "cross-member variance collapse against the 1/√k Bayesian pooling law; "
    "cascade regression on same-round predecessors; single-linkage camps on the "
    "position line with exact Shapley-Shubik and Banzhaf enumeration; a "
    "stratified permutation test of the speaking-order confound; and a "
    "scrambled-statement placebo arm as the control."
)


def _turn_key(turn) -> dict:
    """Read one turn from a dict or a pydantic-ish object, tolerantly."""
    if isinstance(turn, dict):
        return turn
    return {
        k: getattr(turn, k)
        for k in (
            "replicate",
            "round",
            "member",
            "name",
            "member_index",
            "order",
            "arm",
            "public_position",
            "private_position",
            "public_statement",
            "private_note",
            "conceded_to",
        )
        if hasattr(turn, k)
    }


def room_analysis(
    turns: Sequence,
    cast: Sequence,
    motion: str,
    *,
    replicates_meta: Optional[Sequence[dict]] = None,
) -> dict:
    """Assemble every estimator above into one JSON-serialisable result.

    `turns` is a flat list of records — dicts, or objects with the same
    attributes — one per (replicate, round, member):

        replicate        int, 0-based.
        round            int. ROUND 0 IS THE PRIVATE OPENING BALLOT: its
                         `private_position` is the innate position and its
                         public fields are ignored. Rounds 1..T are deliberation.
        member           the member's name (or `member_index`, or `name`).
        order            0-based speaking position within its round. Optional;
                         if absent it is taken from `replicates_meta[r]["order"]`,
                         and failing that from the order the turns appear in.
        arm              "real" or "placebo". Optional; defaults from
                         `replicates_meta`, then to "real".
        public_position  0..1, the position the words convey.
        private_position 0..1, what the member actually believes.

    `cast` is the RoomMemberSpec list (or dicts with the same fields) in the
    canonical member order. `replicates_meta` is one dict per replicate with
    `arm`, `seed` and `order` (member names or indices); it is optional but
    without it the speaking order must be on the turns.

    A replicate missing any (round, member) cell is DROPPED whole and named in
    `dropped_replicates`, the same way a price study drops a curve that missed a
    price. Partial rounds would silently change which transitions the influence
    fit sees.

    The module docstring lists every key of the returned dict.
    """
    started = time.perf_counter()
    names = [
        (c.get("name") if isinstance(c, dict) else getattr(c, "name", None))
        or f"member_{i}"
        for i, c in enumerate(cast)
    ]
    n = len(names)
    index_of = {nm: i for i, nm in enumerate(names)}
    meta = list(replicates_meta or [])
    refusals: list[dict] = []

    records = [_turn_key(t) for t in turns]
    grouped: dict[tuple[int, int], list[dict]] = {}
    for rec in records:
        rep = int(rec.get("replicate", 0) or 0)
        rnd = int(rec.get("round", 0) or 0)
        grouped.setdefault((rep, rnd), []).append(rec)

    def _arm_of(rep: int) -> str:
        if rep < len(meta) and meta[rep].get("arm"):
            return str(meta[rep]["arm"])
        for rec in records:
            if int(rec.get("replicate", 0) or 0) == rep and rec.get("arm"):
                return str(rec["arm"])
        return "real"

    def _member_index(rec: dict) -> Optional[int]:
        if rec.get("member") in index_of:
            return index_of[rec["member"]]
        if rec.get("name") in index_of:
            return index_of[rec["name"]]
        idx = rec.get("member_index")
        if isinstance(idx, int) and 0 <= idx < n:
            return idx
        return None

    replicate_ids = sorted({rep for rep, _ in grouped})
    rounds_seen = sorted({rnd for _, rnd in grouped})
    delib_rounds = [r for r in rounds_seen if r >= 1]

    arms: dict[str, dict] = {}
    for arm in ("real", "placebo"):
        pub: list[list[list[float]]] = []
        prv: list[list[list[float]]] = []
        innate: list[list[float]] = []
        orders: list[list[list[int]]] = []
        for rep in replicate_ids:
            if _arm_of(rep) != arm:
                continue
            ok = True
            ballot = [None] * n
            for rec in grouped.get((rep, 0), []):
                i = _member_index(rec)
                if i is not None and rec.get("private_position") is not None:
                    ballot[i] = float(rec["private_position"])
            if any(v is None for v in ballot):
                refusals.append(
                    {
                        "what": f"replicate {rep}",
                        "reason": "the round-0 private ballot is missing a member",
                    }
                )
                continue
            rep_pub: list[list[float]] = []
            rep_prv: list[list[float]] = []
            rep_ord: list[list[int]] = []
            for rnd in delib_rounds:
                cells = grouped.get((rep, rnd), [])
                row_pub: list[Optional[float]] = [None] * n
                row_prv: list[Optional[float]] = [None] * n
                seats: list[tuple[int, int]] = []
                for slot, rec in enumerate(cells):
                    i = _member_index(rec)
                    if i is None:
                        continue
                    if rec.get("public_position") is not None:
                        row_pub[i] = float(rec["public_position"])
                    if rec.get("private_position") is not None:
                        row_prv[i] = float(rec["private_position"])
                    seat = rec.get("order")
                    seats.append((int(seat) if isinstance(seat, int) else slot, i))
                if any(v is None for v in row_pub) or any(v is None for v in row_prv):
                    ok = False
                    break
                seats.sort()
                order = [i for _, i in seats]
                if sorted(order) != list(range(n)):
                    order = _order_from_meta(meta, rep, names, n)
                rep_pub.append([float(v) for v in row_pub])
                rep_prv.append([float(v) for v in row_prv])
                rep_ord.append(order)
            if not ok or len(rep_pub) != len(delib_rounds) or not rep_pub:
                refusals.append(
                    {
                        "what": f"replicate {rep}",
                        "reason": "a deliberation round is missing a member's turn",
                    }
                )
                continue
            pub.append(rep_pub)
            prv.append(rep_prv)
            innate.append(ballot)
            orders.append(rep_ord)
        arms[arm] = {
            "public": pub,
            "private": prv,
            "innate": innate,
            "orders": orders,
        }

    real = arms["real"]
    placebo = arms["placebo"]
    dropped = [r for r in refusals if r["what"].startswith("replicate")]
    if not real["public"]:
        return {
            "run_id": "",
            "motion": motion,
            "names": names,
            "n_members": n,
            "n_rounds": 0,
            "n_replicates": 0,
            "n_placebo_replicates": len(placebo["public"]),
            "dropped_replicates": dropped,
            "roster": [],
            "trajectory": None,
            "influence": _influence_refusal("No complete real-arm replicate."),
            "centrality": influence_centrality(None),
            "equilibrium": None,
            "absence": [],
            "falsification": _gap_refusal("No complete real-arm replicate."),
            "conformity": _conformity_refusal("No complete real-arm replicate."),
            "cascade": {"refused": "No complete real-arm replicate."},
            "coalitions": {"refused": "No complete real-arm replicate."},
            "order_effect": {"refused": "No complete real-arm replicate."},
            "arms": {"real": None, "placebo": None},
            "placebo": placebo_contrast({}, None),
            "consensus": None,
            "refusals": refusals
            + [{"what": "whole study", "reason": "no complete real-arm replicate"}],
            "timing_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "method": _METHOD,
            "limits": _LIMITS,
        }

    n_rounds = len(real["public"][0])
    influence = fit_social_influence(real["public"], real["innate"], names=names)
    innate_mean = _column_means(real["innate"])
    centrality = influence_centrality(influence.get("W"), innate=innate_mean)
    eq = equilibrium(
        influence.get("W"), influence.get("susceptibility"), innate_mean
    )
    absence = counterfactual_absence(
        influence.get("W"), influence.get("susceptibility"), innate_mean, names
    )
    falsification = preference_falsification(real["public"], real["private"], names)
    conformity = conformity_vs_bayes(real["public"], real["innate"])
    cascade = cascade_scores(
        real["public"],
        real["private"],
        real["orders"],
        names,
        innate=real["innate"],
    )
    order = order_effect(real["public"], real["orders"])

    final_public = _column_means([rep[-1] for rep in real["public"]])
    final_private = _column_means([rep[-1] for rep in real["private"]])
    threshold = None
    if meta and isinstance(meta[0], dict):
        threshold = meta[0].get("decision_threshold")
    camps = coalitions(
        final_public, final_private, influence.get("W"), threshold, names
    )

    real_arm = arm_summary(
        real["public"], real["private"], real["innate"], conformity=conformity
    )
    placebo_arm = (
        arm_summary(placebo["public"], placebo["private"], placebo["innate"])
        if placebo["public"]
        else None
    )
    control = placebo_contrast(real_arm, placebo_arm)

    for label, block in (
        ("influence matrix", influence),
        ("preference falsification", falsification),
        ("conformity vs Bayes", conformity),
        ("cascade scores", cascade),
        ("order effect", order),
        ("placebo control", control),
    ):
        if isinstance(block, dict) and block.get("refused"):
            refusals.append({"what": label, "reason": block["refused"]})
    for row in cascade.get("refusals", []) or []:
        refusals.append({"what": f"cascade / {row['name']}", "reason": row["reason"]})
    for row in influence.get("unidentified_rows", []) or []:
        refusals.append(
            {"what": f"influence row / {row['name']}", "reason": row["reason"]}
        )

    absence_by_name = {row["name"]: row for row in absence}
    cascade_by_name = {row["name"]: row for row in cascade.get("per_member", [])}
    roster = []
    for i, nm in enumerate(names):
        spec = cast[i] if i < len(cast) else {}
        get = (lambda k: spec.get(k)) if isinstance(spec, dict) else (
            lambda k: getattr(spec, k, None)
        )
        lam = (influence.get("susceptibility") or [None] * n)[i]
        roster.append(
            {
                "name": nm,
                "role": get("role"),
                "archetype": get("archetype"),
                "declared_openness": get("openness"),
                "fitted_susceptibility": lam,
                "declared_vs_fitted": (
                    round(lam - get("openness"), 4)
                    if lam is not None and isinstance(get("openness"), (int, float))
                    else None
                ),
                "innate": round(innate_mean[i], 4),
                "final_public": round(final_public[i], 4),
                "final_private": round(final_private[i], 4),
                "gap": round(final_public[i] - final_private[i], 4),
                "moved": round(final_public[i] - innate_mean[i], 4),
                "centrality": (
                    (centrality.get("centrality") or [None] * n)[i]
                    if centrality.get("centrality")
                    else None
                ),
                "absence_delta": (absence_by_name.get(nm) or {}).get("delta_room_mean"),
                "cascade_r2_gain": (cascade_by_name.get(nm) or {}).get(
                    "cascade_r2_gain"
                ),
                "r2_of_fit": (influence.get("r2_per_member") or [None] * n)[i],
            }
        )

    trajectory = {
        "rounds": n_rounds,
        "ballot_private": [round(v, 4) for v in innate_mean],
        "public": [
            [round(v, 4) for v in _column_means([rep[t] for rep in real["public"]])]
            for t in range(n_rounds)
        ],
        "private": [
            [round(v, 4) for v in _column_means([rep[t] for rep in real["private"]])]
            for t in range(n_rounds)
        ],
        "public_mean_by_round": [
            round(mean(_column_means([rep[t] for rep in real["public"]])), 4)
            for t in range(n_rounds)
        ],
        "sd_by_round": conformity.get("sd_by_round"),
    }

    consensus = {
        "observed_final_mean": round(mean(final_public), 4),
        "observed_final_sd": round(stdev(final_public), 4),
        "innate_mean": round(mean(innate_mean), 4),
        "innate_sd": round(stdev(innate_mean), 4),
        "innate_replicate_sd": [
            round(stdev([rep[i] for rep in real["innate"]]), 4) for i in range(n)
        ],
        "convergence_ratio": real_arm.get("convergence_ratio"),
        "consensus_forecast": centrality.get("consensus_forecast"),
        "equilibrium_room_mean": round(mean(eq), 4) if eq else None,
        "equilibrium_vs_observed": (
            round(mean(eq) - mean(final_public), 4) if eq else None
        ),
    }

    return {
        "run_id": "",
        "motion": motion,
        "names": names,
        "n_members": n,
        "n_rounds": n_rounds,
        "n_replicates": len(real["public"]),
        "n_placebo_replicates": len(placebo["public"]),
        "dropped_replicates": dropped,
        "roster": roster,
        "trajectory": trajectory,
        "influence": influence,
        "centrality": centrality,
        "equilibrium": {
            "positions": eq,
            "room_mean": round(mean(eq), 4) if eq else None,
            "vs_observed_final": (
                round(mean(eq) - mean(final_public), 4) if eq else None
            ),
            "note": (
                None
                if eq
                else "The FJ iteration did not settle — the room's positions cycle."
            ),
        },
        "absence": absence,
        "falsification": falsification,
        "conformity": conformity,
        "cascade": cascade,
        "coalitions": camps,
        "order_effect": order,
        "arms": {"real": real_arm, "placebo": placebo_arm},
        "placebo": control,
        "consensus": consensus,
        "refusals": refusals,
        "timing_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "method": _METHOD,
        "limits": _LIMITS,
    }


def _order_from_meta(
    meta: Sequence[dict], rep: int, names: Sequence[str], n: int
) -> list[int]:
    if rep < len(meta):
        raw = meta[rep].get("order")
        if raw:
            index_of = {nm: i for i, nm in enumerate(names)}
            out = [
                index_of[v] if isinstance(v, str) and v in index_of else v
                for v in raw
            ]
            if sorted(x for x in out if isinstance(x, int)) == list(range(n)):
                return [int(x) for x in out]
    return list(range(n))
