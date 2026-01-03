````markdown
# CARE (Contrastive Anchored-REflection) for GRPO in EasyR1 — Implementation Roadmap

> **Goal:** Implement the **CARE method itself** (paper-aligned) inside EasyR1:
>
> - **ACS**: Anchored-Contrastive Subgroup (Shortest-Think anchor + cosine hard negatives)
> - **All-negative Rescue** (zero-sum pseudo-contrast)
> - **Region-weighted token objective** (answer-focused token credit)
> - **RGR**: Reflection-Guided Resampling (training-only, one-shot repair)
>
> **Hard constraint:** All runs must keep using the repo’s **current math reward** (do not add/override reward).

---

## 0) Scope & Invariants

- **Same model / data / verifier / optimizer / KL target** as current GRPO runs.
- **No custom reward:** CARE only changes *how rewards/logprobs are used* to build advantages and training signals.
- **Sampling budget:** keep per-prompt rollout count `G` unchanged. (RGR adds one extra decode; it is training-only.)
- **Inference unchanged:** no test-time reflection; evaluation/inference remains single decode.
- **No leakage:** do not train on held-out test sets (follow existing repo policy).

---

## 1) CARE Method Spec (what we must implement)

For each prompt `x`, sample `G` rollouts `{y_i}` and score them with the **existing** verifier reward `r_i` (e.g., `r_i ∈ {0,1}` from math reward).

### 1.1 If there is at least one positive rollout (success-containing group)

**(A) Anchor selection (Shortest-Think):**
- Let `P = { i | r_i == 1 }`.
- Choose anchor `i+` as:
  - minimal `<think>` length among `P`
  - tie-break: minimal `<answer>` length
- Anchor rollout is `y+ = y_{i+}`.

**(B) Hard negative selection (cosine-nearest failures):**
- Let `F = { i | r_i == 0 }`.
- Target hard-negative count: `K' = min(K, |F|)`.
- Compute a detached rationale embedding for each rollout:
  - extract `<think>` span hidden states from the final layer
  - mean-pool over tokens
  - L2 normalize
  - **detach() / stop-gradient** (selector must not backprop)
- For each failure `j ∈ F`, compute cosine distance to anchor:
  - `dcos(j) = 1 - <h_j, h_+>`
- Preselect top `M` failures by smallest `dcos` (nearest pool).
- Run **farthest-first** diversification within that pool (to reduce redundancy).
- Output `K'` negatives `{y^-_1..y^-_{K'}}`.

**(C) Subgroup formation:**
- Subgroup `S = { y+ } ∪ { y^-_1..y^-_{K'} }`.

**(D) Within-subgroup z-score advantages:**
- Compute subgroup mean and std on the *original verifier rewards*:
  - `mu = mean_{y ∈ S}(r_y)`
  - `sigma = std_{y ∈ S}(r_y) + eps`
- Raw subgroup advantages:
  - `A_raw(y) = (r_y - mu) / sigma` for `y ∈ S`
  - `A_raw(y) = 0` for `y ∉ S`

**(E) Negative-only scaling (failure penalty control):**
- For anchor: `A(y+) = A_raw(y+)`
- For negatives: `A(y^-) = - s * abs(A_raw(y^-))` where `s ∈ (0, 1]`

**(F) Update-size equalization (when K' < K):**
- Multiply all non-zero subgroup advantages by:
  - `eq = sqrt(K / K')` (only meaningful when `K' > 0`)
  - `A ← eq * A`

### 1.2 RGR (Reflection-Guided Resampling; training-only)

Triggered **only** when `P != ∅` (i.e., anchor exists).

- Select **exactly one** hard negative `y^-_t` from the chosen negatives.
- Create a reflected prompt by inserting a **structured repair cue** into the negative’s `<think>` (do not leak gold CoT; no hints required).
- Decode **one** reflected sample `y_ref` (same decoding hyperparams as rollout unless explicitly configured).
- Score `y_ref` with the same verifier reward `r_ref`:

**Safeguard logic:**
- If `r_ref == 1`:
  - **Replace** the original failure `y^-_t` in subgroup `S` with `y_ref`
- Else (`r_ref == 0`):
  - Keep as a negative, but apply **reduced penalty scaling** for this reflected failure:
    - `s_refl = s / 2` (default)
    - Only affects this one negative’s advantage shaping.

> Note: RGR must not change the reward function; it only adds a single extra training-time candidate and modifies advantage shaping/replacement.

### 1.3 If all rollouts are negative (all-negative group)

If `P == ∅`, trigger **All-negative Rescue** to avoid “zero-signal” steps.

**Rescue construction:**
- Choose pseudo-anchor `t` among failures with the largest `log π_old(y_t | x)` (highest likelihood under the old policy).
- Choose `K' = min(K, |F|)` representative negatives (can reuse cosine selector logic; there is no anchor embedding, so use a fallback strategy such as diversity among failures).
- Create a **pseudo reward** `r'` (zero-sum within selected subset):
  - `r'(t) = +δ`
  - for each selected negative `j`: `r'(j) = -δ / K'`
  - all other rollouts: `r'(i) = 0`

Then compute advantages using **the same pipeline** as subgroup normalization/scaling/equalization, but using `r'` instead of `r` for the rescue subset.

> Key requirement: Rescue injects a training signal **without modifying** the repo’s true verifier rewards.

### 1.4 Region-weighted token objective (token credit)

Convert sequence-level advantage `A_i` to per-token advantage by region weights:

- `<answer>` tokens weight `1.0`
- `<think>` tokens weight:
  - `γ+` if sample is verifier-positive
  - `0.0` if sample is verifier-negative
- Normalize per sample by total region weight:
  - `a_{i,t} = A_i * w_{i,t} / (sum_u w_{i,u} + eps_w)`

This ensures answer tokens drive learning, positive rationale is weakly reinforced, and **failed rationale does not receive credit**.

---

## 2) Implementation Plan (minimal, CARE-only)

### Phase 1 — Config & Wiring (CARE as a first-class variant)

**Update config surface** (e.g., `examples/config.yaml` + any schema):
- `grpo.variant=care` (new)
- CARE params:
  - `care.K` (default 4)
  - `care.M` (default 6)
  - `care.neg_scale_s` (default 0.5)
  - `care.eps` (default 1e-6)
  - `care.equalize=true`
  - `care.rescue.enable=true`
  - `care.rescue.delta` (default 0.1)
  - `care.token_weighting=region_weighted`
  - `care.gamma_pos` (default 0.005)
- RGR params:
  - `rgr.enable=true`
  - `rgr.template=structured`
  - `rgr.max_critique_tokens=64`
  - `rgr.s_refl` (default `care.neg_scale_s / 2`)
  - `rgr.sampling.temperature`, `rgr.sampling.top_p`, `rgr.sampling.max_tokens`

**Trainer wiring**
- Route `grpo.variant=care` to:
  - CARE subgroup builder + advantage estimator
  - region-weighted token objective adapter
  - optional RGR hook (training-only) called after reward, before advantage finalization

### Phase 2 — Negative Selector (cosine hard negatives; stop-grad)

Create a selector module, e.g.:
- `verl/algorithms/neg_selectors/cosine_hardneg.py`

Responsibilities:
- extract `<think>` token span indices for each rollout (robust to missing tags)
- compute final-layer hidden states (no-grad forward) → mean-pool → L2 norm → detach
- compute cosine distances to anchor embedding
- preselect nearest `M`
- farthest-first diversification inside preselected pool
- return `K'` indices

Edge cases:
- missing `<think>`: fallback to using whole sequence excluding prompt tokens (document behavior)
- `|F| == 0`: no negatives available → subgroup degenerates (skip CARE update for that prompt)

### Phase 3 — CARE Advantage Estimator (ACS + scaling + equalization + rescue)

Implement a CARE estimator, e.g.:
- `verl/algorithms/adv_estimators/care.py`

Responsibilities:
1) Identify positives/failures from verifier rewards.
2) If positives exist:
   - pick anchor (Shortest-Think)
   - pick `K'` hard negatives
   - build subgroup
   - compute z-score advantages on subgroup rewards
   - apply negative-only scaling `s`
   - apply equalization `sqrt(K/K')`
3) If all-negative:
   - perform rescue (pseudo reward `r'`)
   - compute rescue advantages (same normalization/scaling pipeline)
4) Return per-sample sequence advantages `A_i` plus any flags needed by token weighting and RGR.

### Phase 4 — RGR (Reflection-Guided Resampling; one-shot)

Implement as a hook called only in CARE mode, e.g.:
- `verl/hooks/rgr.py` (or adapt your existing reflex hook but keep semantics paper-aligned)

Responsibilities:
- trigger only when positives exist
- select exactly one hard negative (use same chosen negatives list)
- construct a reflected prompt by inserting a **structured repair cue** into negative’s `<think>`
- decode 1 sample, rescore with same verifier reward
- replace failure if success; else mark it as “reflected failure” to use `s_refl` penalty scaling

Integration requirement:
- RGR must run **post-reward** (we already have initial rewards) and **pre-advantage finalization** (so replacement/scaling affects advantages).

### Phase 5 — Region-weighted token objective

Implement token weighting as a small adapter in the loss pipeline, e.g.:
- `verl/algorithms/losses/region_weighted_tokens.py`

Responsibilities:
- parse `<think>` / `<answer>` spans
- assign weights:
  - answer tokens: 1
  - think tokens: γ+ for positive samples else 0
- normalize per sample and apply to token-level loss terms

Edge cases:
- missing `<answer>`: fallback to last N tokens or “all tokens treated as answer” (document behavior)
- keep a strict mode for debugging to ensure tags are present if desired

### Phase 6 — Instrumentation (minimal but required for debugging)

Emit JSONL logs (lightweight, one line per event):
- `care_events.jsonl`: subgroup formation and rescue
- `rgr_events.jsonl`: reflection outcomes

Minimum `care_events.jsonl` fields:
- `step`, `group`, `G`, `K`, `K_prime`
- `has_positive`, `anchor_idx`
- `neg_scale_s`, `equalize_factor`
- `rescue_triggered`, `rescue_delta`

Minimum `rgr_events.jsonl` fields:
- `step`, `group`, `triggered`, `passed`
- `replaced_original_failure`, `s_refl_used`
- `template`, `critique_len`, `final_len`

---

## 3) Minimal Run Command (CARE only)

```bash
RUN=runs/care/full
$BASE run.dir=$RUN \
  grpo.variant=care \
  care.K=4 care.M=6 \
  care.neg_scale_s=0.5 care.equalize=true \
  care.rescue.enable=true care.rescue.delta=0.1 \
  care.token_weighting=region_weighted care.gamma_pos=0.005 \
  rgr.enable=true rgr.template=structured \
  rgr.max_critique_tokens=64 \
  rgr.sampling.temperature=0.6 rgr.sampling.top_p=0.95 rgr.sampling.max_tokens=512 \
  care.instrument.enable=true \
  care.instrument.care_jsonl=$RUN/care_events.jsonl \
  care.instrument.rgr_jsonl=$RUN/rgr_events.jsonl
````

> This command must **not** set any custom reward; it relies on the repo’s current math reward.

---

## 4) Testing Plan (CARE-only; no ablations)

### Unit tests (fast)

* **Anchor selection**

  * chooses shortest `<think>` among positives
  * tie-break shortest `<answer>`
* **Negative selector**

  * embeddings are detached (no grad)
  * respects `K' = min(K, |F|)`
  * deterministic selection given fixed inputs
* **Z-score advantages**

  * subgroup-only non-zero advantages
  * stable with `sigma + eps`
* **Negative-only scaling**

  * negatives become non-positive and scaled by `s`
* **Equalization**

  * `eq = sqrt(K/K')` when `K' < K`
* **All-negative rescue**

  * triggers only when no positive
  * pseudo reward is zero-sum over selected subset
* **RGR**

  * triggers only when positives exist
  * exactly one resample
  * replace-on-success; reduced scaling on reflected failure

### Integration smoke test (tiny batch)

* run 10–50 steps on a tiny dataset slice
* verify:

  * `care_events.jsonl` and `rgr_events.jsonl` exist and are non-empty
  * no crash when `<think>`/`<answer>` tags are missing (fallback works)
  * training step completes and produces gradients

---

## 5) Final Review (merge gate)

### Design Review (must pass)

* [ ] **Reward integrity:** CARE does not add/override reward; it only reshapes advantages / selection / replacement.
* [ ] **Training-only reflection:** RGR is not invoked in evaluation/inference.
* [ ] **Stop-gradient selector:** hard-negative embeddings are detached; selection has no gradient path.
* [ ] **Budget awareness:** rollout `G` unchanged; RGR adds one decode only when positives exist.

### Code Review Checklist

* [ ] Clean integration point: post-reward, pre-advantage, pre-loss.
* [ ] Edge cases handled:

  * no failures (`|F|=0`) → skip CARE update safely
  * all-negative → rescue path; no RGR
  * missing tags → documented fallback
* [ ] Logging is buffered / non-blocking; JSONL lines are valid JSON.

### Acceptance Criteria (minimal)

* [ ] CARE run completes stably (no NaNs, no exploding KL/length) on at least one medium run.
* [ ] Logs confirm expected behavior:

  * non-zero `has_positive` rate
  * rescue triggers on all-negative prompts
  * RGR triggers only on success-containing prompts
* [ ] No evidence of reward modification or evaluation path change.

---

```
```
