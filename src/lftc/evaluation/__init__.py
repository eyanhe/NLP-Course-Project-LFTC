"""Full-label metrics, paired comparisons and prediction diagnostics.

Numerical dependencies load only when an evaluation function is called.
"""
from __future__ import annotations

from dataclasses import replace
import math
import re


def _task(task):
    value = getattr(task, 'value', task)
    if value == 'multi_label':
        value = 'multilabel'
    if value not in {'single_label', 'multilabel', 'pair_classification'}:
        raise ValueError(f'Unknown task: {value}')
    return value


def _arrays(labels, scores, task, score_kind):
    import numpy as np
    if score_kind not in {'probability', 'svm_margin', 'logit'}:
        raise ValueError(f'Unknown score kind: {score_kind}')
    task = _task(task)
    y, s = np.asarray(labels), np.asarray(scores, dtype=float)
    if s.ndim != 2 or not len(s) or not s.shape[1] or len(y) != len(s) or not np.isfinite(s).all():
        raise ValueError('Scores must be a finite nonempty N by declared-label matrix')
    if score_kind == 'probability' and ((s < 0).any() or (s > 1).any()):
        raise ValueError('Probability scores must be in [0, 1]')
    if task == 'multilabel':
        if y.shape != s.shape or not np.isin(y, [0, 1]).all():
            raise ValueError('Multilabel labels must be a full-width binary matrix')
    elif y.ndim != 1 or not np.isin(y, np.arange(s.shape[1])).all():
        raise ValueError('Single-label labels must be declared integer class indices')
    return y.astype(int), s, task


def _cutoff(score_kind, threshold):
    result = (.5 if score_kind == 'probability' else 0.) if threshold is None else float(threshold)
    if not math.isfinite(result):
        raise ValueError('Threshold must be finite')
    return result


def _multilabel_prf(y, pred):
    import numpy as np
    tp = ((y == 1) & pred.astype(bool)).sum(axis=0)
    support, predicted = y.sum(axis=0), pred.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp, dtype=float), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp, dtype=float), where=support != 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros_like(tp, dtype=float), where=(support + predicted) != 0)
    micro = float(2 * tp.sum() / (support.sum() + predicted.sum())) if support.sum() + predicted.sum() else 0.
    return precision, recall, f1, support, micro


def metric_bundle(labels, scores, task, score_kind, threshold=None, label_mask=None):
    """Use every declared score column; AP averages only nondegenerate labels."""
    import numpy as np
    from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                                 precision_recall_fscore_support)
    y, s, task = _arrays(labels, scores, task, score_kind)
    count = s.shape[1]
    multi = task == 'multilabel'
    cutoff = _cutoff(score_kind, threshold) if multi else None
    pred = (s >= cutoff).astype(int) if multi else s.argmax(axis=1)
    if multi:
        precision, recall, f1, support, micro = _multilabel_prf(y, pred)
    else:
        precision, recall, f1, support = precision_recall_fscore_support(
            y, pred, labels=np.arange(count), average=None, zero_division=0)
        micro = float(f1_score(y, pred, average='micro', zero_division=0))
    result = {'task': task, 'score_kind': score_kind, 'rows': len(y), 'label_count': count,
              'macro_f1': float(f1.mean()),
              'micro_f1': micro,
              'worst_label_f1': float(f1.min()),
              'per_label': [{'label_id': i, 'precision': float(precision[i]),
                             'recall': float(recall[i]), 'f1': float(f1[i]),
                             'support': int(support[i])} for i in range(count)]}
    if multi:
        valid = [i for i, n in enumerate(support) if 0 < n < len(y)]
        mask = valid if label_mask is None else list(label_mask)
        if (len(mask) != len(set(mask)) or any(not isinstance(i, (int, np.integer)) for i in mask)
                or not set(mask) <= set(valid)):
            raise ValueError('AP mask requires unique labels with positive and negative examples')
        aps = [float(average_precision_score(y[:, i], s[:, i])) if i in valid else None
               for i in range(count)]
        result.update(threshold=cutoff, ap_mask=mask,
                      macro_ap=float(np.mean([aps[i] for i in mask])) if mask else None,
                      masked_macro_f1=float(f1[mask].mean()) if mask else None,
                      subset_accuracy=float(np.all(y == pred, axis=1).mean()))
        for row, ap in zip(result['per_label'], aps):
            row['ap'] = ap
    else:
        result.update(accuracy=float((y == pred).mean()),
                      confusion_matrix=confusion_matrix(y, pred, labels=np.arange(count)).tolist())
    return result


def select_threshold(labels, scores, score_kind='probability', candidates=None):
    """Select a shared cutoff by macro F1. Call only with validation predictions."""
    import numpy as np
    y, s, _ = _arrays(labels, scores, 'multilabel', score_kind)
    default = _cutoff(score_kind, None)
    if candidates is None:
        values = np.unique(s)
        # Limit search cost while covering the observed validation score range.
        values = np.quantile(values, np.linspace(0, 1, min(101, len(values))))
        candidates = [default, *values.tolist(), float(np.nextafter(s.max(), math.inf))]
    candidates = sorted(set(float(v) for v in candidates))
    if not candidates or not all(math.isfinite(v) for v in candidates):
        raise ValueError('Threshold candidates must be nonempty and finite')
    return min(candidates, key=lambda v: (-float(_multilabel_prf(y, s >= v)[2].mean()),
                                          abs(v - default), v))


def holm_adjust(p_values):
    """Holm familywise p values, preserving list order or mapping keys."""
    mapping = isinstance(p_values, dict)
    keys = list(p_values) if mapping else None
    values = [p_values[k] for k in keys] if mapping else list(p_values)
    if any(not math.isfinite(float(v)) or not 0 <= v <= 1 for v in values):
        raise ValueError('p values must be finite and in [0, 1]')
    result = [0.] * len(values)
    previous = 0.
    for rank, index in enumerate(sorted(range(len(values)), key=lambda i: values[i])):
        previous = min(1., max(previous, (len(values) - rank) * values[index]))
        result[index] = previous
    return dict(zip(keys, result)) if mapping else result


def mcnemar_test(labels, predictions_a, predictions_b):
    """Exact two-sided McNemar test on aligned row predictions (A minus B)."""
    import numpy as np
    from scipy.stats import binomtest
    y, a, b = (np.asarray(v) for v in (labels, predictions_a, predictions_b))
    if y.ndim not in (1, 2) or not len(y) or y.shape != a.shape or y.shape != b.shape:
        raise ValueError('McNemar requires aligned nonempty labels and predictions')
    correct_a, correct_b = a == y, b == y
    if y.ndim == 2:
        correct_a, correct_b = correct_a.all(axis=1), correct_b.all(axis=1)
    wins = int((correct_a & ~correct_b).sum())
    losses = int((~correct_a & correct_b).sum())
    n = wins + losses
    return {'n': len(y), 'a_only_correct': wins, 'b_only_correct': losses,
            'both_correct': int((correct_a & correct_b).sum()),
            'both_wrong': int((~correct_a & ~correct_b).sum()), 'discordant': n,
            'difference': (wins - losses) / len(y),
            'p_value': float(binomtest(wins, n, .5).pvalue) if n else 1., 'method': 'exact_binomial'}


def bootstrap_difference(labels, scores_a, scores_b, task, score_kind='probability', *,
                         metric=None, threshold=None, label_mask=None, n_resamples=1000,
                         seed=42, confidence=.95):
    """Paired row bootstrap of metric(A)-metric(B), conditional on fitted models.

    AP retains the original partition's mask; degenerate resampled columns use
    AP=0 for no positives and AP=1 for all positives.
    """
    import numpy as np
    from sklearn.metrics import average_precision_score, f1_score
    y, a, task = _arrays(labels, scores_a, task, score_kind)
    _, b, _ = _arrays(labels, scores_b, task, score_kind)
    if a.shape != b.shape or not isinstance(n_resamples, int) or n_resamples < 1 or not 0 < confidence < 1:
        raise ValueError('Bootstrap requires equal score shapes, positive resamples and valid confidence')
    multi = task == 'multilabel'
    metric = metric or ('macro_ap' if multi else 'macro_f1')
    bundle_a = metric_bundle(y, a, task, score_kind, threshold, label_mask)
    bundle_b = metric_bundle(y, b, task, score_kind, threshold, label_mask)
    if metric not in {'macro_ap', 'macro_f1', 'micro_f1', 'accuracy', 'subset_accuracy'}:
        raise ValueError(f'Unsupported bootstrap metric: {metric}')
    if metric not in bundle_a or bundle_a[metric] is None:
        raise ValueError(f'Metric unavailable for this partition: {metric}')
    mask = bundle_a.get('ap_mask', [])
    cutoff = _cutoff(score_kind, threshold)
    pred_a, pred_b = ((a >= cutoff, b >= cutoff) if multi else (a.argmax(axis=1), b.argmax(axis=1)))
    def value(indices, scores, pred):
        truth, decisions = y[indices], pred[indices]
        if metric == 'macro_ap':
            return float(np.mean([0. if not truth[:, i].any() else
                                  1. if truth[:, i].all() else
                                  average_precision_score(truth[:, i], scores[indices, i]) for i in mask]))
        if metric in {'accuracy', 'subset_accuracy'}:
            correct = truth == decisions
            return float(correct.all(axis=1).mean() if multi else correct.mean())
        if multi:
            _, _, f1, _, micro = _multilabel_prf(truth, decisions)
            return float(f1.mean()) if metric == 'macro_f1' else micro
        return float(f1_score(truth, decisions, labels=None if multi else np.arange(a.shape[1]),
                              average='macro' if metric == 'macro_f1' else 'micro', zero_division=0))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_resamples):
        indices = rng.integers(0, len(y), size=len(y))
        draws.append(value(indices, a, pred_a) - value(indices, b, pred_b))
    lower, upper = np.quantile(draws, [(1-confidence)/2, (1+confidence)/2])
    difference = bundle_a[metric] - bundle_b[metric]
    return {'metric': metric, 'a': bundle_a[metric], 'b': bundle_b[metric],
            'difference': difference, 'lower': float(lower), 'upper': float(upper),
            'confidence': confidence, 'n_resamples': n_resamples, 'seed': seed,
            'unit': 'paired_row', 'conditional_on_fitted_models': True}


def diagnose_predictions(examples, scores, task, label_count, score_kind, threshold=None):
    """Summarize errors by visible text length and full-label support."""
    import numpy as np
    examples = tuple(examples)
    task = _task(task)
    multi = task == 'multilabel'
    labels = np.zeros((len(examples), label_count), dtype=int) if multi else []
    for i, row in enumerate(examples):
        if any(not 0 <= label < label_count for label in row.labels) or (not multi and len(row.labels) != 1):
            raise ValueError('Example labels must match the declared task and label count')
        if multi:
            labels[i, list(row.labels)] = 1
        else:
            labels.append(row.labels[0])
    y, s, _ = _arrays(labels, scores, task, score_kind)
    if s.shape[1] != label_count:
        raise ValueError('Scores must contain all declared label columns')
    pred = s >= _cutoff(score_kind, threshold) if multi else s.argmax(axis=1)
    correct = (y == pred).all(axis=1) if multi else y == pred
    lengths = np.asarray([len((row.text + ' ' + (row.text_pair or '')).split()) for row in examples])
    bins = []
    for lo, hi in ((0, 20), (21, 50), (51, 100), (101, 250), (251, None)):
        ids = np.flatnonzero((lengths >= lo) & (True if hi is None else lengths <= hi))
        bins.append({'min_words': lo, 'max_words': hi, 'rows': len(ids),
                     'errors': int((~correct[ids]).sum()),
                     'error_rate': float((~correct[ids]).mean()) if len(ids) else None})
    errors = []
    for i in np.flatnonzero(~correct):
        row = examples[int(i)]
        errors.append({'sample_id': row.sample_id, 'split': row.split, 'text': row.text,
                       'text_pair': row.text_pair, 'labels': list(row.labels),
                       'prediction': np.flatnonzero(pred[i]).tolist() if multi else [int(pred[i])],
                       'word_count': int(lengths[i]), 'scores': s[i].tolist()})
    return {'task': task, 'rows': len(y), 'metrics': metric_bundle(y, s, task, score_kind, threshold),
            'error_count': len(errors), 'error_rate': len(errors) / len(y), 'errors': errors,
            'label_support': y.sum(axis=0).tolist() if multi else np.bincount(y, minlength=label_count).tolist(),
            'length': {'unit': 'whitespace_words_both_texts', 'min_words': int(lengths.min()),
                       'max_words': int(lengths.max()), 'mean_words': float(lengths.mean()), 'bins': bins}}


def perturb_examples(examples, mode, *, words=(), max_words=128, seed=42):
    """Return copied examples for cue masking, truncation or word-order sensitivity."""
    import random
    if mode not in {'mask_cues', 'truncate', 'shuffle_words', 'lowercase'}:
        raise ValueError(f'Unknown perturbation: {mode}')
    if mode == 'truncate' and max_words < 1:
        raise ValueError('max_words must be positive')
    rng = random.Random(seed)
    cues = {str(word).casefold() for word in words}
    def edit(text):
        if text is None:
            return None
        if mode == 'lowercase':
            return text.casefold()
        if mode == 'mask_cues':
            return re.sub(r"\b[\w']+\b", lambda m: '[MASK]' if m[0].casefold() in cues else m[0], text)
        tokens = text.split()
        if mode == 'truncate':
            return ' '.join(tokens[:max_words])
        rng.shuffle(tokens)
        return ' '.join(tokens)
    return tuple(replace(row, text=edit(row.text), text_pair=edit(row.text_pair)) for row in examples)


__all__ = ['metric_bundle', 'select_threshold', 'bootstrap_difference', 'mcnemar_test',
           'holm_adjust', 'diagnose_predictions', 'perturb_examples']
