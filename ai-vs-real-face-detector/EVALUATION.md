# Evaluation

Run evaluation from the project root with the existing checkpoint:

```powershell
.\venv\Scripts\python.exe src\evaluate.py
```

Optional arguments are `--checkpoint`, `--data-dir`, `--output-dir`, and
`--device cpu` (or `cuda`). The evaluator never trains or writes a model.

## Dataset selection

The evaluator first uses `data/test/real` and `data/test/fake` if present.
Otherwise it recreates the checkpoint's deterministic training/validation
split using the saved seed and validation ratio. Images may be nested under
`fake/stylegan2` or `fake/diffusion`; those source names receive separate
binary reports with all held-out real images as the negative class.

The selected test set must contain at least one image. For ROC-AUC and curve
plots it must contain both real and AI labels. If the repository has neither
an explicit test set nor enough images to create a validation set, evaluation
stops rather than reporting made-up values.

## Outputs

`evaluation_outputs/metrics.json` contains accuracy, precision, recall, F1,
specificity, ROC-AUC, and TP/TN/FP/FN. AI is the positive class and the
decision threshold is 0.50 on the AI probability. `predictions.csv` records
the source, true/predicted labels, and both class probabilities per image.

The folder also includes a confusion matrix, ROC curve, precision-recall
curve (when both classes are present), and `data_leakage_report.txt`. The
leakage report compares SHA-256 file hashes between the training and selected
test samples; it detects exact duplicates, not visually similar images.
