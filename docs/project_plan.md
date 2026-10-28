# Project plan and boundaries

## Title
Explainable Cardiovascular Disease Classification Using Routine Clinical and Lifestyle Indicators

## Research questions
1. How do three classifiers compare on recall, precision, F1, ROC-AUC and average precision?
2. Which predictors contribute most to predictions on validation data?
3. How does a reduced set of routine measurements compare with the full feature set?
4. How do calibration and classification errors vary across age and recorded gender groups?

## Assignment alignment
- Dataset (10): provenance, schema, quality, limitations; obtain lecturer approval.
- Research questions (10): connect questions to at least three recent reputable papers.
- Exploration (10): counts, mean, median, variance, standard deviation, missingness and flags.
- Visualisation (10): distributions, box plots, correlations and outcome comparisons.
- Models (20): three models, justified preprocessing, train-only CV and tuning, held-out evaluation, confusion matrices.
- Interpretation (15): answers to questions, feature importance, limitations, IEEE references.
- Viva (25): ten-minute overview and a reproducible code demonstration.

## Planned experiments
Start with the fixed baseline. Then compare full features with a prespecified reduced routine-measurement set, selecting any learned features within training CV. Test sensitivity to documented, justified data-quality rules; report exclusions and retained class distributions. Do not fabricate values or remove rows simply to improve scores. Choose any decision threshold on validation data and evaluate its sensitivity/specificity trade-off once on test data. Feature attributions describe model behaviour, not causal effects.

## Prototype and validation path
A FastAPI endpoint and web interface can demonstrate validated inputs, a versioned model and explanations. Initially show an experimental classification score, not a calibrated local clinical risk or treatment advice. An investor demonstration should include baseline comparisons, uncertainty, calibration and an explicit local validation plan. Clinical use needs representative prospective data, independent external validation, workflow evaluation and appropriate governance. The dataset alone cannot establish suitability for Nepal or India.

## Open provenance questions
Confirm dataset license, recruitment population, diagnostic definition, collection dates, measurement units and patient independence from the original source. The 70,000-row file has no follow-up time, so future-event prediction is out of scope.
