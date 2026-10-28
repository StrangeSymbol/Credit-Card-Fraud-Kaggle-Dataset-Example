# Credit Card Fraud Detection: Imbalanced Learning, Cost-Based Thresholds & SHAP

An end-to-end fraud detection pipeline on the public [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset. The project focuses on the parts of fraud modelling that matter in practice: honest evaluation under extreme class imbalance, a chronological train/validation/test split, a decision threshold chosen to minimise **dollar loss** rather than F1, and model explanations with SHAP.

Everything lives in one notebook: [`fraud_detection_kaggle.ipynb`](fraud_detection_kaggle.ipynb)

## Headline results

Evaluated on a held-out, chronologically later test window (7.7 hours, 56,746 transactions, 74 frauds).

| Finding | Result |
|---|---|
| Best test PR-AUC | Random Forest **0.813** (95% CI 0.720-0.889), LightGBM **0.788** (0.699-0.872), Logistic Regression 0.746 (0.626-0.846). The supervised models' intervals overlap heavily |
| Lift over random | A random classifier scores 0.0013 PR-AUC, so LightGBM is roughly **600x** better |
| Supervised vs. unsupervised | Isolation Forest (no labels) reaches only 0.048 PR-AUC, even though its ROC-AUC is 0.955 |
| Imbalance handling | Square-root class weights, SMOTE and under-sampling all reach about 0.75-0.78 validation PR-AUC. Full class weighting collapsed to **0.005** (see lessons below) |
| Random vs. chronological split | Averaged over 10 random splits: 0.853 PR-AUC, versus 0.788 chronologically (+0.064) |
| Cost reduction vs. no model | **64%** at the cost-optimal threshold ($7,728 of fraud losses down to $2,795), catching 56 of 74 frauds with 16 false alarms |
| Cost reduction at F1-optimal threshold | Only **52%** (F1 treats a $5 false alarm the same as a ~$104 missed fraud) |

## Dataset

- European cardholders, two days in September 2013: 284,807 transactions, 492 frauds (0.172%).
- Features `V1`-`V28` are anonymised PCA components. Only `Time` and `Amount` are original.
- After dropping 1,081 exact duplicates: 283,726 transactions, 473 frauds (0.167%).
- Released under the Database Contents License (DbCL) v1.0.

Citation requested by the dataset authors: Dal Pozzolo, Caelen, Johnson and Bontempi, "Calibrating Probability with Undersampling for Unbalanced Classification," IEEE CIDM, 2015.

## Approach

1. **Load and clean.** The loader reads a local `creditcard.csv` or downloads it with `kagglehub`. Exact duplicates are dropped and rows sorted by time.
2. **EDA.** Fraud is bursty (43 frauds in a single hour). Fraud *rate* is highest in the low-volume overnight hours (up to ~1.5% vs. 0.17% overall), which raw counts hide.
3. **Features.** `log_amount` and an hour-of-day proxy are added to the 28 PCA components. Raw `Time` is excluded so the model can't key on position in the timeline.
4. **Chronological split.** Train on the first 60% of the timeline (342 frauds), validate on the next 20% (57 frauds), test on the final 20% (74 frauds).
5. **Class imbalance.** Five LightGBM variants are compared on validation PR-AUC: no handling, full class weights, square-root class weights with regularisation, random under-sampling and SMOTE.
6. **Model comparison.** Logistic Regression, Random Forest, LightGBM and Isolation Forest, each with a bootstrap 95% confidence interval on PR-AUC.
7. **Random vs. chronological split.** The same model recipe is trained both ways, with the random split averaged over 10 seeds.
8. **Cost-sensitive threshold.** A missed fraud costs its transaction `Amount`; a false alarm costs a flat $5. The threshold is tuned on validation and applied unchanged to test.
9. **SHAP.** Global feature importance, per-alert explanations and a waterfall plot.

## Key findings and lessons

- **Accuracy and ROC-AUC are misleading here.** The over-weighted model had a ROC-AUC of 0.81 but a PR-AUC of 0.005. Precision-recall metrics expose problems that ROC-AUC hides.
- **Extreme class weights overfit.** Weighting each fraud ~600x (the full class ratio) let the trees memorise individual training frauds. A square-root weight with stronger regularisation (`min_child_samples=50`, `reg_lambda=5`) fixed it and became the best validation strategy (PR-AUC 0.783).
- **Model choice is a statistical tie.** On the test set the three supervised models have heavily overlapping confidence intervals. LightGBM was selected on *validation* PR-AUC and kept for the downstream steps, rather than switching to Random Forest after seeing test results.
- **The optimal threshold is not special.** The validation cost curve is nearly flat between thresholds of ~0.05 and ~0.9, and the cost-optimal threshold (0.427) performs the same as the default 0.5 (63.8% vs. 63.9% savings). The clear result is that the F1-optimal threshold is worse in dollars.
- **A random split looks more optimistic.** It scored 0.853 PR-AUC vs. 0.788 for the chronological split. The two test sets have different fraud rates (about 0.17% vs. 0.13%) and few frauds, so the gap is indicative, not exact.
- **The model leans heavily on one component.** SHAP shows low `V14` values contribute up to ~+9 to +11 log-odds, with `V12`, `V10` and `V4` adding smaller confirming pushes. The effect saturates below about V14 = -4, as expected from tree splits.

## Limitations

- **Anonymised features.** PCA components prevent domain features (geo-velocity, customer baselines, merchant risk) and make explanations abstract.
- **Two days of data and few frauds.** There is no seasonality or drift analysis, and per-split fraud counts are small, so metrics are noisy.
- **Dated, heavily studied benchmark.** Strong scores here don't imply performance on today's fraud.
- **Simplified costs.** A flat $5 review cost and "loss = amount" ignore chargeback fees, recovery rates and the value of stopping small card-testing probes before larger fraud.
- **Single validation window.** The threshold and imbalance strategy were chosen from 57 validation frauds and would need re-tuning in production.
- **Hyperparameters were not tuned.** LightGBM uses sensible defaults, not a search.

## Possible extensions

- Add geo-velocity and per-customer features using a dataset with richer fields (Sparkov, IEEE-CIS).
- Calibrate probabilities (isotonic regression) and use an amount-aware alert rule.
- Time-aware cross-validation and hyperparameter tuning.
- An autoencoder as a second unsupervised baseline.
- Serve the model behind a FastAPI endpoint.

## Run it

```bash
git clone <your-repo-url>
cd <your-repo>
pip install numpy pandas scikit-learn lightgbm shap imbalanced-learn matplotlib seaborn kagglehub jupyter
jupyter notebook fraud_detection_kaggle.ipynb
```

Place `creditcard.csv` next to the notebook (downloaded from the Kaggle page), or configure Kaggle credentials so `kagglehub` can fetch it. Seeds are fixed, so results should be reproducible apart from small differences across library versions. The full run takes a few minutes.

## Tech stack

Python, pandas, NumPy, scikit-learn, LightGBM, imbalanced-learn (SMOTE), SHAP, matplotlib, seaborn.
