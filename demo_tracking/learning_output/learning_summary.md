# AutoFix Learning Pipeline Summary

**Generated**: 2025-11-16 19:09:40

## Training Data

- Total samples: 100
- Positive samples: 52 (52.0%)
- Negative samples: 48 (48.0%)

## Discovered Patterns

Total patterns: 18

### Top 10 Patterns by Support

| Pattern Type | Conditions | Success Rate | Support | Recommendation |
|-------------|------------|--------------|---------|----------------|
| confidence_calibration | confidence_range=(0.7, 0.8) | 9.8% | 41 | Confidence 0.7-0.8: overconfident (actual 9.8%) |
| severity_outcome | severity=low | 58.8% | 34 | LOW severity: needs manual review |
| severity_outcome | severity=medium | 45.5% | 33 | MEDIUM severity: needs manual review |
| severity_outcome | severity=high | 51.5% | 33 | HIGH severity: needs manual review |
| confidence_calibration | confidence_range=(0.9, 1.0) | 86.2% | 29 | Confidence 0.9-1.0: well calibrated |
| confidence_calibration | confidence_range=(0.8, 0.9) | 81.5% | 27 | Confidence 0.8-0.9: well calibrated |
| category_strategy | category=dead_code, strategy=ast | 88.9% | 9 | Prefer ast for dead_code (high success rate) |
| category_strategy | category=hardcoded_values, strategy=template | 55.6% | 9 | Avoid template for hardcoded_values (low success rate) |
| category_strategy | category=missing_docstrings, strategy=manual | 55.6% | 9 | Avoid manual for missing_docstrings (low success rate) |
| category_strategy | category=type_errors, strategy=ast | 55.6% | 9 | Avoid ast for type_errors (low success rate) |

## Calibration Quality

- **Brier Score**: 0.303 (0.0 = perfect)
- **Expected Calibration Error (ECE)**: 0.314
- **Maximum Calibration Error (MCE)**: 0.652
- **Overconfident predictions**: 100.0%
- **Underconfident predictions**: 0.0%
- **Recommended adjustment**: -0.314

## Recommendations

⚠️ **High overconfidence detected** - Consider:
1. Apply temperature scaling to confidence scores
2. Increase confidence threshold for auto-fix
3. Reduce confidence scores by ~0.314

## Next Steps

1. **Model Training**: Use `training_data.csv` to train/fine-tune confidence model
2. **Pattern Integration**: Integrate discovered patterns into autofix policy
3. **Calibration**: Apply isotonic regression or Platt scaling for better calibration
4. **Monitoring**: Set up continuous monitoring with periodic pipeline runs
5. **A/B Testing**: Test adjusted thresholds on holdout data

