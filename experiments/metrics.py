"""Evaluation metrics for change-point detection (Ali & Ho, ICDM 2025).

Paper reference: §VI-A "Evaluation Protocol", Eq. 26–28.

Given ground-truth change points ``{t_i}`` and detected change points ``{tau_j}``
on a sequence of length ``T``, we compute three quantities:

    TPR = |{i : exists j s.t. tau_j in [t_i, t_i + Delta]}| / |{t_i}|         (Eq 26)
    FPR = |{tau_j : tau_j not in union_i [t_i, t_i + Delta]}| / (T - |{t_i}| * (Delta+1))   (Eq 27)
    ADD = mean over matched (i, j) of (tau_j - t_i)                           (Eq 28)

where ``Delta`` is the detection window (default 20, per paper §VI-A).

Why this design
===============

**Matching rule.** A ground-truth change at t_i is "detected" by the EARLIEST
tau_j falling in [t_i, t_i + Delta]. We greedily match each ground truth to
its earliest qualifying detection; each detection may only match one truth
(avoids double-counting when two changes are close). Detections in overlapping
windows go to whichever ground truth they can earliest serve.

**TPR denominator is the number of truths, not detections.** This follows
paper Eq 26. A detector that fires many times at one change still gets credit
for that one change — the TPR is about *coverage*, not *precision*.

**FPR denominator is the "opportunity space"**: the time indices NOT inside
any true change window. ``T - |{t_i}| * (Delta + 1)`` counts the points the
detector could have fired on without triggering a TP. We do NOT subtract
overlapping windows from this denominator; the paper doesn't, and for
``Delta=20, T=200`` overlaps are rare (Delta_min=40 in synthetic generators).

**Multiple detections inside one window collapse to 1 TP.** This prevents a
chattering detector from inflating its TPR by firing 10 times in the
detection window — the paper's metric rewards *first-pass* detection speed
(captured by ADD), not repeat-fire counts.

**Edge cases.**
  * Empty ground truth (true_cps == []): TPR is undefined (return None).
    FPR is well-defined: every detection is a false positive, denom = T.
  * Empty detections: TPR = 0, FPR = 0, ADD = None (nothing matched).
  * Single-window match: ADD is the delay of the matched detection.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def tpr_fpr_add(
    change_points: Sequence[int],
    true_cps: Sequence[int],
    T: int,
    Delta: int = 20,
) -> dict:
    """Compute TPR, FPR, and ADD for a single trial.

    Parameters
    ----------
    change_points : sequence of int
        Detected change-point time indices (tau_j). Need not be sorted.
    true_cps : sequence of int
        Ground-truth change points (t_i).
    T : int
        Sequence length. Used as the FPR denominator's upper bound.
    Delta : int, default 20
        Detection window (paper default).

    Returns
    -------
    dict with keys {'tpr', 'fpr', 'add', 'n_tp', 'n_fp', 'n_detections'}.
        tpr / add may be None under the edge cases documented in the module
        docstring. fpr is always a float (possibly 0.0).
    """
    dets = sorted(int(x) for x in change_points)
    truths = sorted(int(x) for x in true_cps)
    n_truths = len(truths)
    n_dets = len(dets)

    # Build the union of true windows as a boolean mask for FP classification.
    in_any_window = np.zeros(T + Delta + 1, dtype=bool)
    for t_i in truths:
        in_any_window[t_i : t_i + Delta + 1] = True

    # Greedy match: for each truth (earliest first) assign the earliest
    # unmatched detection that lies in its window.
    matched_det_idx: set[int] = set()
    delays: list[int] = []
    n_tp = 0
    for t_i in truths:
        for j, tau in enumerate(dets):
            if j in matched_det_idx:
                continue
            if tau < t_i:
                continue
            if tau <= t_i + Delta:
                matched_det_idx.add(j)
                delays.append(tau - t_i)
                n_tp += 1
                break
            else:
                # Detections are sorted — no later det can be in this window.
                break

    # FPs: detections not in ANY true window.
    n_fp = 0
    for j, tau in enumerate(dets):
        if j in matched_det_idx:
            continue
        if 0 <= tau < len(in_any_window) and in_any_window[tau]:
            # In some true window but lost the matching race (e.g., 2nd det
            # in the same window). By the "collapse duplicates" rule this
            # is NOT a FP — it's a redundant TP absorbed by the earlier det.
            continue
        n_fp += 1

    # TPR
    tpr: Optional[float]
    if n_truths == 0:
        tpr = None
    else:
        tpr = n_tp / n_truths

    # FPR denominator: "opportunity space" = time indices NOT inside any
    # true change-point window. Paper Eq 27 uses T - |{t_i}|*(Delta+1) which
    # assumes no overlaps; with Delta_min=40 > Delta=20 in synthetic data,
    # overlaps cannot happen so the formula is exact.
    fpr_denom = T - n_truths * (Delta + 1)
    if fpr_denom <= 0:
        fpr = 0.0 if n_fp == 0 else float("inf")
    else:
        fpr = n_fp / fpr_denom

    add: Optional[float]
    if not delays:
        add = None
    else:
        add = float(np.mean(delays))

    return {
        "tpr": tpr,
        "fpr": fpr,
        "add": add,
        "n_tp": n_tp,
        "n_fp": n_fp,
        "n_detections": n_dets,
    }


def aggregate(rows: list[dict], group_keys: Sequence[str]) -> list[dict]:
    """Aggregate per-trial metric rows by (detector, scenario) — mean +/- std.

    Nones in tpr/add are ignored in the mean/std (but counted in `n_trials`).
    """
    from collections import defaultdict

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = tuple(r[k] for k in group_keys)
        groups[key].append(r)

    out = []
    for key, grp in groups.items():
        agg = {k: v for k, v in zip(group_keys, key)}
        agg["n_trials"] = len(grp)
        for metric in ("tpr", "fpr", "add"):
            vals = [r[metric] for r in grp if r[metric] is not None]
            if vals:
                agg[f"{metric}_mean"] = float(np.mean(vals))
                agg[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            else:
                agg[f"{metric}_mean"] = None
                agg[f"{metric}_std"] = None
        agg["n_detections_mean"] = float(np.mean([r["n_detections"] for r in grp]))
        out.append(agg)
    return out
