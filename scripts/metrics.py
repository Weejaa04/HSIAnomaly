"""Shared evaluation metrics for the HSI food anomaly benchmark."""

import numpy as np


def f1_acc_at_tnr95(scores, labels, tnr=0.95):
    """F1-score and accuracy at the operating point where TNR (specificity) = tnr.

    Finds the largest threshold whose FPR <= (1 - tnr) via the ROC curve,
    then computes accuracy and F1 at that threshold.

    Returns (f1, acc).
    """
    from sklearn.metrics import roc_curve

    scores = np.asarray(scores).ravel()
    labels = np.asarray(labels).ravel()
    target_fpr = 1.0 - tnr

    fpr, _, thresholds = roc_curve(labels, scores)
    ok = fpr <= target_fpr
    idx = int(np.where(ok)[0][-1]) if ok.any() else 0
    thresh = thresholds[idx]

    preds = (scores >= thresh).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return float(f1), float(acc)


def count_flops(model, *inputs):
    """Return forward-pass FLOPs in GFLOPs (2 * MACs / 1e9).

    Returns None if thop is unavailable or the forward cannot be profiled.
    """
    try:
        from thop import profile

        macs, _ = profile(model, inputs=inputs, verbose=False)
        return float(macs) * 2.0 / 1e9
    except Exception:
        return None
