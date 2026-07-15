# Panel-Level Group Ordering and Abnormality Experiment

This document describes the group experiment, its relationship to the original individual-lab
experiment in `Ordet_Abnomality.md`, the current state of the code, and open questions about
how causal methods might improve the analysis.

---

## Relationship to the Individual-Lab Experiment

The original experiment (`Ordet_Abnomality.md`) asks whether a single pooled Lab-MAE embedding `z`
can predict, for each individual lab test, (1) whether it will be ordered next day and (2)
whether it will be abnormal if ordered. It treats each lab independently and uses a per-lab
logistic regression linear probe on `z`.

The group experiment asks the same questions but at the **panel level**: instead of predicting
individual lab outcomes, it predicts whether a clinical panel (a set of labs co-ordered from the
same blood draw) will be ordered, and whether the panel draw will contain any abnormal results.
This mirrors how physicians actually order labs — not one test at a time, but as named clinical
panels reflecting a diagnostic hypothesis.

The same `z` embedding is used in both experiments. Neither experiment retrains the Lab-MAE model.

---

## Key Design: Dominant Draw Detection

The central challenge at the panel level is identifying which next-day labs belong to the
same panel order. `build_group_dataset.py` resolves this using **dominant draw detection**:

For each patient and each panel, next-day lab values that share an identical `nptime_last_*`
timestamp are assumed to have come from the same blood draw (the same tube/order). The draw
with the most panel labs present is selected as the "dominant draw." Ties are broken by earliest
timestamp.

This prevents a single spot-check (e.g., one INR result) from counting as a full Coagulation
panel order.

---

## Dataset Structure

`build_group_dataset.py` produces `data/group_dataset.csv`:
- One row per patient per panel (17 panels × 325,106 patients = ~5.5M rows)
- Reuses `z` embeddings from `order_abnormality_dataset.csv` (one `z` vector per patient,
  shared across all panel rows for the same patient)
- Key label columns:

| Column | Meaning |
|---|---|
| `ordered_any` | 1 if ≥1 panel lab was ordered next day |
| `ordered_majority` | 1 if >50% of panel labs were ordered next day |
| `abnormal_any` | 1 if any lab in the dominant draw was outside its reference range; NaN if not ordered |
| `abnormal_rate` | Fraction of ordered labs that were abnormal; NaN if not ordered |
| `n_ordered` | Number of panel labs found in the dominant draw |
| `panel_draw_time` | Timestamp of the dominant draw |
| `z_0 … z_63` | 64-dim pooled Lab-MAE embedding (same value for all panel rows of a patient) |

The default ordering target is `ordered_any` (≥1 lab ordered). `ordered_majority` is available
as a sensitivity analysis but is not suitable for CBC (25 labs; majority = 13+ labs ordered,
which rarely happens for routine draws).

---

## Panels

Panels span a range of clinical specificity, from high-frequency routine draws to targeted
single-biomarker tests:

**Routine multi-lab panels**

| Panel | N labs | Approx. ordering rate |
|---|---|---|
| BMP | 8 | ~76.6% |
| CBC | 25 | ~76.6% |
| LFT | 6 | ~50.8% |
| Coagulation | 4 | ~60.9% |
| ABG | 15 | — |

**Co-ordered multi-lab panels**

| Panel | N labs | Clinical indication |
|---|---|---|
| Electrolytes (Mg/Phos) | 2 | Add-on to BMP |
| Iron Studies | 4 | Anemia workup |
| Creatine Kinase | 2 | Myocardial injury / rhabdomyolysis |
| Pancreatic Enzymes | 2 | Pancreatitis evaluation |

**Single-biomarker targeted panels** (n_panel_labs = 1; ordered_any ≡ ordered_majority)

LDH, Troponin, HbA1c, CRP, TSH, Lipid Panel, Osmolality, Uric Acid

---

## Current Regression Models

All three tasks use the same linear probe architecture:

```python
make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, class_weight="balanced"),
)
```

`class_weight="balanced"` corrects for imbalanced base rates (relevant for targeted panels with
low ordering frequency). A stratified 75/25 train/test split is used independently for each panel.

### Task 1 — Panel Order Prediction

```
P(panel ordered | z)
```

- **Input**: 64-dim `z` (pooled embedding of current lab state, all-panel shared)
- **Target**: `ordered_any` (binary)
- **Metrics**: AUROC, Brier score
- **Results (initial 4-panel run)**:

| Panel | AUROC | Brier |
|---|---|---|
| BMP | 0.841 | — |
| CBC | 0.836 | — |
| LFT | 0.794 | — |
| Coagulation | 0.806 | — |

### Task 2 — Abnormality Prediction Among Ordered Panels

```
P(abnormal_any = 1 | ordered = 1, z)
```

- **Input**: 64-dim `z` (same embedding)
- **Training rows**: only panels where `ordered_any == 1` and `abnormal_any` is not NaN
- **Target**: `abnormal_any` (binary)
- **Metrics**: AUROC, AUPRC
- **Note**: AUPRC is more informative than AUROC when base rates are high. For BMP and CBC,
  `abnormal_any` approaches 0.92–0.96 among ordered rows, making Task 2 nearly trivial and AUPRC
  misleadingly high. LFT (~0.58) and Coagulation (~0.65) provide more discriminative signal.

The Task 2 model is also scored on **all** test rows (not just ordered rows) so that Task 3
can combine the two predictions.

### Task 3 — Joint Expected Yield

```
P(ordered AND abnormal | z) = P(ordered | z) × P(abnormal | ordered = 1, z)
```

```python
expected_abnormal = pred_order * pred_abnormal
```

- **Target**: `observed_and_abnormal = int(ordered == 1 and abnormal_any == 1)`
- **Metrics**: AUROC, AUPRC, Brier score
- This is the most clinically meaningful single number: the expected probability of observing
  an abnormal result on next day's panel, integrated over the ordering decision.

### Severity Decile Analysis

After Task 1, test rows are binned by `pred_order` into deciles. Within each decile, the
observed `abnormal_rate` (mean fraction of ordered labs that were abnormal) is computed for
the ordered subset. A monotone increase in `observed_abnormal_rate` across deciles confirms
that the embedding's ordering signal also encodes clinical severity, not just ordering tendency.

---

## Summary of Initial Results (BMP, CBC, LFT, Coagulation)

The initial 4-panel run (325,106 patients, `ordered_any` target) showed:

- Task 1 AUROC ranged from 0.794 (LFT) to 0.841 (BMP), consistent across routine and targeted
  panels. This is a strong result for a linear probe on a general-purpose embedding.
- Severity deciles confirmed monotone increase in abnormal rate with predicted order probability
  for all four panels, with a steeper gradient for targeted panels (LFT, Coagulation) than
  routine ones (BMP, CBC).
- Task 2/3 results for BMP and CBC are inflated due to near-ceiling `abnormal_any` base rates
  and should not be interpreted as meaningful discrimination.

---

## Current Code Status

| File | Status | Description |
|---|---|---|
| `build_order_abnormality_dataset.py` | Updated | Expanded from 8 to 76 target labs; covers all blood panels in `d_labitems_grouped.csv` |
| `run_order_abnormality_experiment.py` | Updated | Chunked CSV streaming to avoid OOM (76 labs × 325k patients = 24.7M rows); bar chart heights scale with number of labs; legends suppressed at >12 series |
| `build_group_dataset.py` | Updated | Expanded from 4 to 17 panels; all panels from `d_labitems_grouped.csv` (blood only) |
| `run_group_experiment.py` | Updated | `PANELS` list covers all 17 panels; cross-panel analysis removed |

The datasets need to be **rebuilt** to reflect the expanded lab and panel definitions:

```bash
cd order_abnormality_experiment

# 1. Rebuild individual-lab dataset (76 labs)
python build_order_abnormality_dataset.py --device mps   # or cpu / cuda

# 2. Rebuild group dataset (17 panels; reuses z from step 1)
python build_group_dataset.py

# 3. Run individual experiment
python run_order_abnormality_experiment.py

# 4. Run group experiment
python run_group_experiment.py
```

---

## Open Question: Causal Framing and IPW

### The Confounding Problem

The current models treat the ordering decision as a passive observation. But **ordering is a
clinical action**, not a random event. The physician orders a panel because `z` (or its
underlying clinical signal) already suggests something is wrong. This creates a structural
confound:

- `z` encodes overall acuity / clinical state.
- High acuity → clinician orders the panel (high `ordered_any`).
- High acuity → if the panel is drawn, results are more likely abnormal (high `abnormal_any`).

So when Task 2 fits `P(abnormal | ordered = 1, z)`, it is conditioning on a **collider-adjacent
selection**: the ordered subset is not a random sample of patients — it skews toward the sicker
patients for targeted panels (LFT, Coagulation), where ordering is more selective. The model
may be picking up residual acuity signal rather than a true relationship between `z` and
abnormality net of the ordering decision.

For BMP and CBC (76.6% ordering rate), the selection is mild. For LFT (50.8%) and Coagulation
(60.9%), the selection is more meaningful and the bias is larger.

### Where Causal Methods May Help

**Inverse Probability Weighting (IPW) for Task 2**

One natural correction is to reweight ordered rows by the inverse of their estimated ordering
probability:

```
weight_i = 1 / P(ordered = 1 | z_i)
```

This upweights patients who were unlikely to be ordered (but were), and downweights patients
who were very likely to be ordered. The weighted Task 2 model would estimate a
**marginal** abnormality probability closer to "what would the abnormal rate be if we ordered
this panel for everyone?" rather than "what is the abnormal rate among those who happened to
get the panel?"

In the Task 1 model we already have `pred_order = P(ordered | z)`, so the IPW weights are
directly available at no extra modeling cost.

**Augmented IPW (AIPW) / Doubly-Robust Estimation**

A doubly-robust estimator combines the outcome model (Task 2) with the propensity model (Task 1)
in a way that remains consistent if either model is correctly specified. This would give a
more stable estimate of the marginal abnormality risk.

**Potential Outcomes Framing**

Alternatively, the group experiment could be reframed as estimating the average treatment
effect (ATE) of "would order the panel" on "observing an abnormal result", using `z` as
the confounder set. This would quantify how much of the Task 3 signal reflects a causal
pathway (acuity → ordering decision → abnormal result) vs. a confounded associational signal.

### What Is Not Yet Clear

- Whether the selection bias in Task 2 is large enough to change the qualitative conclusions.
  The monotone severity decile curves suggest the embedding has a real signal; whether IPW
  would sharpen or qualitatively change these curves is unknown.
- How to define the counterfactual ordering decision at the panel level (vs. individual lab),
  given that panels are ordered as a unit.
- Whether AIPW is better applied to `abnormal_any` (binary) or `abnormal_rate` (continuous),
  and which is more clinically interpretable.

These questions are the natural next step after establishing the baseline discriminative
performance of the current linear probes.
