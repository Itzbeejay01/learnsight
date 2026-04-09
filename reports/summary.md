# LearnSight Model Training Report

- Run date (UTC): 2026-04-04T14:35:02+00:00
- Split: 80/20 stratified (test_size=0.2)
- CV: StratifiedKFold(n_splits=3)
- Best model: LightGBM
- Classes: A, B, C, D

## Performance Summary

| Model | Accuracy | F1 (weighted) | Precision (weighted) | Recall (weighted) | ROC-AUC (macro OvR) |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.9079 | 0.9074 | 0.9078 | 0.9079 | 0.9930 |
| XGBoost | 0.9097 | 0.9094 | 0.9096 | 0.9097 | 0.9883 |
| LightGBM | 0.9304 | 0.9301 | 0.9303 | 0.9304 | 0.9923 |

## Anti-overfitting Controls

- Held-out test set used once for final reporting
- Stratified cross-validation for hyperparameter selection
- Regularization constraints for tree-based models (depth, leaf sizes, subsampling, L1/L2)
- Robust scaling + median/mode imputations for stability
- Automatic drop of numeric features with |corr(feature, target)| >= 0.97 on the training split

## Best Hyperparameters (CV)

### Random Forest
- CV accuracy: 0.9009
```json
{
  "n_estimators": 300,
  "min_samples_split": 10,
  "min_samples_leaf": 4,
  "max_features": "sqrt",
  "max_depth": null,
  "class_weight": null
}
```

### XGBoost
- CV accuracy: 0.8993
```json
{
  "subsample": 0.85,
  "reg_lambda": 10.0,
  "reg_alpha": 0.8,
  "n_estimators": 550,
  "min_child_weight": 10,
  "max_depth": 4,
  "learning_rate": 0.08,
  "gamma": 0.1,
  "colsample_bytree": 0.65
}
```

### LightGBM
- CV accuracy: 0.9167
```json
{
  "subsample": 0.65,
  "reg_lambda": 1.0,
  "reg_alpha": 0.0,
  "num_leaves": 63,
  "n_estimators": 400,
  "min_split_gain": 0.0,
  "min_child_samples": 30,
  "max_depth": 6,
  "learning_rate": 0.06,
  "colsample_bytree": 0.65
}
```
