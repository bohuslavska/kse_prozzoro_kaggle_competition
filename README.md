# Prozorro.Sale — 2nd Place Solution by RANDOM FOREST FIRE

The **KSE – Prozorro. Predict Auction Participation** competition asked participants to predict bidder-participation demand for Ukraine's [Prozorro.Sale](https://prozorro.sale/) public e-auctions. Each lot had to be assigned one of four ordered demand classes, `target ∈ {0, 1, 2, 3}`, and submissions were evaluated using **Quadratic Weighted Kappa (QWK)** on a time-based test set.

**Final competition score: 0.7118 — 2nd place (Phase-2: Final Leaderboard)**
**Kaggle competiton link: https://www.kaggle.com/competitions/kse-prozorro-predict-auction-participation**

---

## Team

**RANDOM FOREST FIRE**

* Kateryna Bohuslavska (`katebohuslavska`)
* Bohdan Somriakov (`pontius2`)
* Michael Burenko (`michaelburenko`)
* Anastasiia Mozghova (`anastasiiathebrain`)

---

## Solution overview

Our final pipeline combined four main ideas:

1. **Treat the target as ordinal.** The four classes are buckets of the raw bid count:
   `0`, `1`, `2–4`, and `5+`. In our validation experiments, predicting the underlying
   count and learning three QWK-optimal thresholds worked better than direct four-class
   classification.
2. **Build broad, target-free features.** We combined auction metadata, text statistics,
   administrative geography, document-manifest aggregates, time features, frequency
   encodings, ratios, missingness indicators, and compressed text embeddings.
3. **Use the same temporal folds for every model.** LightGBM, XGBoost, and CatBoost produced
   aligned out-of-fold predictions over three expanding time windows.
4. **Blend complementary models.** We optimized non-negative weights using a 50/50 objective
   over pooled-OOF QWK and the most recent fold's QWK, then tuned the final thresholds on
   the most recent fold.

---

## The target

| Class | Raw bid count | Approximate share |
| ----: | ------------- | ----------------: |
|     0 | 0 bids        |              ~66% |
|     1 | 1 bid         |              ~14% |
|     2 | 2–4 bids      |              ~16% |
|     3 | 5+ bids       |               ~4% |

QWK penalizes disagreements according to the squared distance between the true and predicted
classes. Predicting class 2 instead of class 3 is therefore penalized less than predicting
class 0 instead of class 3. This made regression followed by threshold optimization a
natural formulation for the task.

---

## Feature engineering

The manually engineered features do not use the target. CatBoost additionally uses its own
ordered target-statistics mechanism internally to process categorical variables.

* **Base numeric features and log transforms.** We applied `log1p` to heavy-tailed money
  and area variables.
* **Text statistics.** For `title`, `description`, and `items_text`, we calculated length,
  character-class ratios, word and sentence counts, lexical diversity, hapax ratio, and
  Shannon entropy.
* **KOATUU hierarchy. We derived raion- and council-level prefixes, the administrative level represented by each code, and a settlement-level indicator.
* **Document-manifest aggregates.** We created document counts, distinct type and format
  counts, photo-related statistics, per-type presence flags, and pairwise combinations of
  the eight most frequent document types.
* **Time and seasonality.** Features included cyclical month encodings, quarter, season,
  a monotonic `ym_index`, and interactions between auction-duration variables.
* **Frequency, missingness, and ratio features.** We added frequency encodings for selected
  high-cardinality keys, missing-value indicators, and ratios such as documents per item
  and value per item.
* **Text embeddings.** Offline Qwen3-Embedding-4B vectors with 2,560 dimensions were reduced
  to 256 PCA components. PCA-256 gave the best validation trade-off in our experiments;
  increasing the dimension to 384 did not improve the result.

---

## PCA and frequency fit scope

`PCA_FREQ_TRAIN_ONLY` controls whether the PCA basis and frequency maps are fitted on:

* **`True`: the full training set only.** Competition test rows do not influence the
  representation. This was the configuration used for the **second-place submission** and
  it generalized best on the final leaderboard.
* **`False`: the training and test sets together.** This is a target-free transductive
  option, since test features are available at inference time, but it ties the
  representation to that particular test set. It did not perform as well in our final
  submissions.

---

## Validation

We used **three expanding temporal folds** defined by successive 8% quantile cutoffs. Because
all observations from the same month were kept together, the actual validation-fold sizes
were not exactly 8%.

For every fold, the model was trained on earlier months and evaluated on the next time
window. The final fold matched the original competition baseline's most recent holdout and
served as our main proxy for the unseen future test period.

The same fold masks were shared by all three models, which made their OOF predictions
directly blendable.

An OOF prediction for a training row is generated by a model fitted without that row. This
provides a more realistic validation signal than evaluating a model on its own training
observations.

Three decision thresholds were optimized to maximize QWK.

---

## Models

| Model        | Objective                          | Treatment of `organizer_id` and raw `koatuu` |
| ------------ | ---------------------------------- | -------------------------------------------- |
| **LightGBM** | `regression_l2` on `log1p(n_bids)` | Dropped to limit memorization                |
| **XGBoost**  | Native `count:poisson` on `n_bids` | Dropped to limit memorization                |
| **CatBoost** | `RMSE` on `log1p(n_bids)`          | Kept as categorical features                 |

CatBoost's ordered target statistics reduce the leakage risk associated with naive target
encoding of categorical variables. However, we did not run a separate ablation for the two
raw high-cardinality identifiers, so we cannot claim that keeping them was independently
beneficial.

Each model produced:

* aligned temporal OOF predictions for validation and blending;
* a full-training refit for test inference;
* test predictions averaged over seeds `42`, `7`, and `123`.

---

## Validation results from the published run

| Model / blend      | Pooled OOF QWK | Most recent fold QWK |
| ------------------ | -------------: | -------------------: |
| LightGBM           |         0.7336 |               0.7463 |
| XGBoost            |         0.7410 |               0.7495 |
| CatBoost           |         0.7174 |               0.7322 |
| Equal-weight blend |         0.7420 |               0.7506 |
| Optimized blend    |     **0.7430** |           **0.7517** |

The optimized convex blend used approximately:

* LightGBM: `0.319`
* XGBoost: `0.341`
* CatBoost: `0.340`

The thresholds obtained on the most recent fold were approximately:

`[0.6030, 1.3629, 2.7672]`

These are local validation results from this notebook run, not leaderboard scores. The final
competition score was **0.7118**, according to 🏁 Phase-2: Final Leaderboard.

---

## What helped

* Modeling the ordered target through regression and optimized thresholds.
* Time-aware validation rather than a random split.
* Combining three gradient-boosting implementations with aligned OOF predictions.
* Using PCA-256 embeddings alongside structured and document-level features in our
  strongest pipeline.
* Fitting PCA and frequency maps on the training set only for the final submission.

## What did not clearly help

* Increasing the embedding PCA dimension from 256 to 384.
* Fitting PCA and frequency maps on the training and test sets together.
* The contribution of keeping raw high-cardinality identifiers in CatBoost was not isolated
  in a separate ablation, so its individual benefit remains uncertain.
* We spent a substantial amount of time engineering historical bidder statistics for organizers, regions, and selling methods. Although some of them looked promising in local validation, they did not generalize reliably to the leaderboard and ultimately held the solution back. We gradually reduced the set from several lag features to two, and the best final submissions used none of them.

---

## Inputs

* `TRAIN_DATA/`: `train.csv`, `documents_manifest.csv`, `emb_train.npy`
* `NEW_TEST/`: `test_phase2.csv`, `test_phase2_documents_manifest.csv`,
  `emb_test_phase2.npy`, `test_phase2_sample_submission.csv`

## Outputs

Written to `RESULTS/` and `final_submission/`:

* `features_train.parquet`, `features_test.parquet`
* `train_meta.parquet`, `test_meta.parquet`, `meta.json`
* `lgb_*`, `xgb_*`, and `cb_*` OOF/test prediction caches
* `submission_final.csv` — optimized blend
* `submission_ensemble_equal.csv` — equal-weight blend
