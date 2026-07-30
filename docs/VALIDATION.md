# Validation guide

Everything added in this pass, and how to prove each piece yourself.

The statistical work is validated by **measurement, not assertion**: §3 gives
standalone scripts that simulate data with a known answer and count how often
the code gets it right. Those are the ones worth running — a passing unit test
says the code does what someone expected, while a measured false-discovery rate
says whether the number on the screen can be believed.

Prerequisites:

```bash
cd server && source .venv/bin/activate && pip install -r requirements.txt
```

`pypdf` is new — the install step matters if you have an existing venv.

---

## 1. What was implemented

### Deeper, query-specific analysis

| Thing | Where |
|---|---|
| **Findings engine** — harvest evidence from a study's own statistics in code → Benjamini-Hochberg across the family → rank against the question → model writes, constrained to cite evidence ids | [`server/app/findings.py`](../server/app/findings.py) (new, ~900 lines) |
| **Citation enforcement** — findings citing no resolvable id are deleted; findings standing only on FDR-failed rows are demoted to low confidence | `findings.validate_report` |
| **`research_question`** on every study request; never reaches the twins | `schemas.StudyRequest`, inherited by all 9 request models |
| **Wired into every study** at the single SSE seam, so a study added later cannot ship without it | `main._sse` |
| **Gap statistic** replaces silhouette for segment discovery; k = 1 admissible | `analytics.kmeans_segments` |
| **Cross-fitted ordering gain** + permutation null replaces the fixed-selection bootstrap | `sequence._select_ordering`, `_split_gain`, `_permute_labels` |
| **Miller-Madow** entropy debiasing | `analytics.shannon_entropy_normalised` |
| **Studentized sup-t simultaneous bands** replace per-beat pointwise intervals | `analytics.simultaneous_band`, used by `neuro.aggregate_content_study` |
| **Benjamini-Hochberg** FDR control | `analytics.benjamini_hochberg` |

### Links for every modality

| Thing | Where |
|---|---|
| Fetcher carries **any** media, routed by content type — page, image, video, audio, PDF | `fetching.fetch_url`, `ASSET_KINDS`, `kind_for_content_type` |
| Per-kind byte ceilings; `Accept` derived from requested kinds; honest-bot-UA retry on refusal | `fetching.fetch_url` |
| **Media relay** — video/audio bytes pass through to the browser, which decodes them | `POST /v1/content/media` |
| **PDF → pages**, which are beats the document already segmented | `modality.pages_from_pdf` |
| **Player pages named and refused** before any fetch | `modality.is_player_url` |
| **HTML extractor rewritten** on stdlib `html.parser` | `modality._BlockReader`, `extract_blocks` |
| Link input for every modality + provenance receipt | `dashboard/src/components/AssetInput.tsx` |

### UI

| Thing | Where |
|---|---|
| **Findings surface** — answer, ranked findings, clickable evidence chips, multiplicity ledger | [`FindingsPanel.tsx`](../dashboard/src/components/FindingsPanel.tsx) (new) |
| **Tensor explorer** — all 9 appraisal dimensions × beats, column read | [`TensorExplorer.tsx`](../dashboard/src/components/TensorExplorer.tsx) (new) |
| **Research question input** | `FindingsPanel.ResearchQuestion`, in all 8 study panels |
| Findings types on the wire contract | `dashboard/src/lib/api.ts` |

---

## 2. Automated suite

```bash
cd server && source .venv/bin/activate && python -m pytest -q
```

Expected: **378 passed** (was 335 before this pass — 43 new).

Targeted runs, grouped by what they defend:

```bash
python -m pytest tests/test_findings.py -v
```
20 tests. The load-bearing ones are `test_uncited_findings_are_dropped`,
`test_partially_cited_findings_keep_only_resolvable_ids` and
`test_findings_built_only_on_unsupported_evidence_are_downgraded` — these are
what stop a fluent model from stating a number that was never measured.

```bash
python -m pytest tests/test_analytics.py -k "kmeans or simultaneous or benjamini" -v
```
9 tests. `test_kmeans_refuses_structureless_noise` fails if the gap statistic is
reverted; `test_simultaneous_band_covers_the_whole_curve` measures coverage
across 120 simulated studies and requires ≥ 90%.

```bash
python -m pytest tests/test_sequence.py -k "null or order_effect" -v
```
3 tests. `test_gain_is_not_supported_under_a_true_null` is the regression for
the 20.7% false-positive rate; `test_real_order_effect_is_still_detected`
guards against "fixed" meaning "now detects nothing".

```bash
python -m pytest tests/test_modality.py -k "nested or wrapper or unbalanced or navigation or hidden" -v
python -m pytest tests/test_fetching.py -k "modality or kinds or player or relay or content_type" -v
```

---

## 3. Measure the statistics yourself

These are the real validation. Each simulates data whose truth is known and
counts how often the code is right. Run from `server/` with the venv active.

### 3.1 Segments invented from noise — was 86–90%, now 6–9%

```bash
python - <<'PY'
import random, sys; sys.path.insert(0, '.')
from app.analytics import kmeans_auto, kmeans_segments

def fdr(n, quant=None, trials=120):
    rng = random.Random(4); hits = 0
    for _ in range(trials):
        pts = []
        for _ in range(n):
            x, y = rng.random(), rng.random()
            if quant: x, y = round(x/quant)*quant, round(y/quant)*quant
            pts.append((x, y))
        hits += kmeans_auto(pts) is not None
    return hits / trials

print("FALSE DISCOVERY on pure noise — every 'segment' here is fabricated:")
for n in (8, 12, 20, 40):
    print(f"  n={n:3d}  uniform={fdr(n):.1%}   0.1-quantised={fdr(n,0.1):.1%}")

print("\nPOWER on a genuine two-camp split — must stay high:")
rng = random.Random(9)
for sep in (0.6, 0.4, 0.25):
    hits = 0
    for _ in range(40):
        a = [(rng.gauss(0.5-sep/2, 0.06), rng.gauss(0.5-sep/2, 0.06)) for _ in range(10)]
        b = [(rng.gauss(0.5+sep/2, 0.06), rng.gauss(0.5+sep/2, 0.06)) for _ in range(10)]
        r = kmeans_segments(a + b)
        hits += bool(r and r["k"] == 2)
    print(f"  separation {sep}: detected {hits/40:.0%}")
PY
```

Expected: FDR **6–9%** at every n (nominal ~5%), power **100%** at all three
separations. Before the fix this printed 85–90% FDR.

### 3.2 Ordering gain under a true null — was 20.7%, now 2%

```bash
python - <<'PY'
import random, sys; sys.path.insert(0, '.')
from app.sequence import sequence_result, assign_orderings, sample_orderings

def null_study(seed):
    """No order effect at all: intent is a random walk, message identity does
    nothing. Any 'gain' found is the search fitting noise."""
    rng = random.Random(seed)
    pool = sample_orderings(4, 8)
    walks = []
    for slot in assign_orderings(6, 4, len(pool)):
        order = pool[slot]; steps = []; intent = 0.0
        for _ in range(len(order)):
            intent = max(0.0, min(1.0, intent + rng.gauss(0.12, 0.18)))
            steps.append({"intent": intent, "fatigue": 0.2, "disengaged": False})
        walks.append({"ordering": list(order), "steps": steps})
    return sequence_result(walks, [f"m{i}" for i in range(4)], "low", len(pool))

res = [null_study(1000 + s) for s in range(40)]
print(f"gain_supported fired : {sum(bool(r['gain_supported']) for r in res)/40:.1%}   (was 20.7%)")
print(f"naive gain positive  : {sum(r['naive_objective_gain'] > 0 for r in res)/40:.1%}   (the bias, still visible)")
print(f"honest gain positive : {sum((r['objective_gain'] or 0) > 0 for r in res)/40:.1%}   (unbiased => ~50%)")
r = res[0]
print(f"\nexample: naive={r['naive_objective_gain']} honest={r['objective_gain']} "
      f"p={r['gain_p_value']} bias removed={r['selection_bias']}")
PY
```

Expected: `gain_supported` **0–8%**, naive gain positive **~95–100%** (that gap
*is* the bias), honest gain positive **~40–60%**.

### 3.3 Curve bands — pointwise covered 50–73%, sup-t covers 97–99%

```bash
python - <<'PY'
import random, sys; sys.path.insert(0, '.')
from app.analytics import simultaneous_band
TRUE = [0.3, 0.5, 0.7, 0.6, 0.4, 0.55, 0.65, 0.45]

def trial(seed, n_twins):
    rng = random.Random(seed); per_twin = []
    for _ in range(n_twins):
        base = rng.gauss(0, 0.10)          # twins have baselines => correlated
        per_twin.append([max(0, min(1, t + base + rng.gauss(0, 0.10))) for t in TRUE])
    return simultaneous_band([[tw[i] for tw in per_twin] for i in range(len(TRUE))],
                             iterations=600)

print("Coverage of the WHOLE 8-beat curve (target 95%):")
for n in (8, 12, 20, 40):
    sim = pt = 0
    for s in range(150):
        r = trial(s, n)
        sim += all(lo <= TRUE[i] <= hi for i, (lo, hi) in enumerate(r["band"]))
        pt  += all(lo <= TRUE[i] <= hi for i, (lo, hi) in enumerate(r["pointwise"]))
    print(f"  {n:3d} twins: sup-t {sim/150:5.1%} | pointwise {pt/150:5.1%}  <- what charts drew before")
PY
```

Expected: sup-t **95–99%**, pointwise **50–75%**. The pointwise row is the bug.

### 3.4 Multiple comparisons — 86.8% of null studies "found" something

```bash
python - <<'PY'
import random, sys; sys.path.insert(0, '.')
from app.analytics import benjamini_hochberg
rng = random.Random(1); raw = corrected = 0
for _ in range(300):
    ps = [rng.random() for _ in range(40)]       # 40 hypotheses, ALL null
    raw += any(p < 0.05 for p in ps)
    corrected += benjamini_hochberg(ps)["n_rejected"] > 0
print(f"uncorrected: {raw/300:.1%} of studies report >=1 finding")
print(f"BH-corrected: {corrected/300:.1%}")
# 5 real effects buried in 35 fixed nulls — fixed, not random, so the count is
# exactly 5 rather than 5 plus however many nulls happened to draw small.
print("power check, 5 real effects among 40:",
      benjamini_hochberg([1e-6]*5 + [0.4, 0.5, 0.6, 0.7, 0.8]*7)["n_rejected"], "/ 5")
PY
```

Expected: uncorrected **~86%**, corrected **~7%**, power **5 / 5**.

### 3.5 Page extraction on real sites — Stripe returned 0 sections before

```bash
python - <<'PY'
import asyncio, httpx, sys; sys.path.insert(0, '.')
from app.modality import preview_page, visible_text_length
from app.schemas import ContentAsset
SITES = ["https://stripe.com", "https://linear.app", "https://vercel.com",
         "https://tailwindcss.com", "https://www.python.org", "https://news.ycombinator.com"]
async def main():
    async with httpx.AsyncClient(follow_redirects=True, trust_env=False, timeout=25) as c:
        for url in SITES:
            try:
                r = await c.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; CogniSwarmBot/1.0)"})
                s = preview_page(ContentAsset.model_construct(kind="page", text=r.text))
                print(f"OK   {url:32s} {len(s):2d} sections | {s[0][:60]!r}")
            except Exception as e:
                print(f"NONE {url:32s} visible {visible_text_length(getattr(r,'text','')):,} -> {str(e)[:50]}")
asyncio.run(main())
PY
```

Expected: Stripe/Linear/Vercel/Tailwind/python.org all **7–10 sections**, first
one being the real hero. **Hacker News correctly returns NONE** — it is a table
of link titles, not prose, and inventing beats from it would be worse than
refusing.

---

## 4. API validation

Start the backend (needs `COGNISWARM_ALLOW_ANONYMOUS=1` or an API key in
`server/.env`, else every route 503s by design):

```bash
cd server && ./run.sh
```

Then, adjusting the port if you are not on 8000:

```bash
for u in "https://stripe.com" \
         "https://arxiv.org/pdf/1706.03762" \
         "https://picsum.photos/800/500" \
         "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4" \
         "https://upload.wikimedia.org/wikipedia/commons/c/c8/Example.ogg" \
         "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
         "http://169.254.169.254/latest/meta-data/"; do
  echo "── $u"
  curl -s -m 40 -X POST http://127.0.0.1:8000/v1/content/fetch \
    -H 'Content-Type: application/json' -d "{\"url\":\"$u\"}" | head -c 220; echo
done
```

| URL | Expected |
|---|---|
| stripe.com | `kind=page`, 9 sections, `final_url` may show a regional redirect |
| arXiv PDF | `kind=document`, `page_count=15`, 15 readable pages |
| picsum | `kind=image`, `image_b64` populated, `media_type=image/jpeg` |
| the MP4 | `kind=video`, `media_relay=/content/media`, **no** bytes inline |
| the .ogg | `kind=audio` (registered as `application/ogg`, which naive matching misses) |
| YouTube | **422** naming the player-page problem |
| 169.254.169.254 | **400** — SSRF guard, "resolves to a private address" |

Media relay — the bytes must come back intact and inert:

```bash
curl -s -D - -o /tmp/relay.mp4 -X POST http://127.0.0.1:8000/v1/content/media \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"}' \
  | grep -iE 'content-type|nosniff|x-upstream'
file -b /tmp/relay.mp4
```

Expected: `content-type: application/octet-stream` (deliberately **not** the
upstream type), `x-content-type-options: nosniff`, and `file` reporting
`ISO Media, MP4`. 991,017 bytes.

---

## 5. UI validation

Dashboard at `http://localhost:3100` (or `:3000` against your own backend).

**Content Lab → link ingestion**, no API credit spent:

| Step | Expected |
|---|---|
| *Landing page* → `https://stripe.com` → fetch | receipt: `✓ 9 sections`, final URL, redirect count, first 4 section previews |
| *Deck / document* → `https://arxiv.org/pdf/1706.03762` | receipt: `document`, ~2.2 MB, "15 readable page(s) of 15" |
| *Image* → `https://picsum.photos/800/500` | receipt: `image`, `image/jpeg`; Run button enables |
| *Video* → a direct `.mp4` | "downloading through the relay…" then "8 keyframes extracted in your browser" |
| *Landing page* → a YouTube watch URL | inline error naming the player-page problem |
| *Image* → a page URL | "That link is a web page (text/html), not an image" |

**Research question** — present on all 8 study panels, above the run controls.
The note under it states the twins never see it. Verify by running the same
study twice with different questions: the **numbers are identical**, only the
findings ordering and wording change.

**Findings surface and tensor explorer** appear after a study completes and
require an OpenAI call. In the findings block, check:

- header shows `N/M survived FDR`
- every finding carries at least one evidence chip; clicking one opens the
  measurement with its interval, `p`, `q` and `n`
- chips are `●` when supported, `○` when not
- a finding built only on `○` rows is marked *downgraded*
- "EVERY MEASUREMENT" expands the full ledger including rows no finding cited

In the tensor explorer: hover a **column** to read all 9 dimensions at that
beat, and confirm the highlight follows into the attention field, the
peak-index distribution and the retention curve — they share one cursor.

---

## 6. Not validated

- **No study has been run end-to-end** through the findings engine against a
  live model — that spends OpenAI credit and I did not spend yours. Everything
  under it (harvest, FDR, ranking, citation validation) is unit-tested with
  fixtures, and `test_build_findings_returns_evidence_even_when_synthesis_fails`
  covers the no-API-key path.
- **Frontend has no automated tests.** This is pre-existing — the repo has zero
  frontend/collector tests. `npx tsc --noEmit` and `npx next build` are clean;
  the UI checks above are manual.
- **Audio from a URL confirms reachability only.** Nothing here transcribes, so
  a transcript is still required. Inventing beats from a waveform would be
  fabricating the content rather than analysing it.
