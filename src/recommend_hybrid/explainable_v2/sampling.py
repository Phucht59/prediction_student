"""Grouped stratified sampling allocator for final selected case exports."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def perform_grouped_stratified_sampling(
    df,
    query_groups,
    student_to_queries,
    query_strata,
    panel_a_target=300,
    panel_b_target=150,
    seed=2026,
):
    """Perform Proportional Stratified Group Allocation for final selected cases.

    Uses largest-remainder rounding per stratum. Student disjointness enforced per stratum.

    Returns (panel_a_qids, panel_b_qids, sampling_audit_dict).
    """
    total_target = panel_a_target + panel_b_target

    student_strata = {}
    for sid, qids in student_to_queries.items():
        counts = pd.Series([query_strata[q] for q in qids]).value_counts()
        student_strata[sid] = counts.index[0]

    all_strata = sorted(set(student_strata.values()))
    strata_to_students = {s: [] for s in all_strata}
    for sid, stratum in student_strata.items():
        strata_to_students[stratum].append(sid)

    stratum_query_count = {}
    for stratum in all_strata:
        students = strata_to_students[stratum]
        total_qs = sum(len(student_to_queries[sid]) for sid in students)
        stratum_query_count[stratum] = total_qs

    total_queries = sum(stratum_query_count.values())
    pa_ratio = panel_a_target / total_target
    pb_ratio = panel_b_target / total_target

    # Normalize by total pool size to select panel_a_target out of total_queries
    pa_fracs = {s: stratum_query_count[s] * panel_a_target / total_queries for s in all_strata}
    pb_fracs = {s: stratum_query_count[s] * panel_b_target / total_queries for s in all_strata}

    def _round_quotas(fracs, total):
        floors = {k: int(v) for k, v in fracs.items()}
        remainders = sorted(fracs.keys(), key=lambda k: -(fracs[k] - floors[k]))
        deficit = total - sum(floors.values())
        for k in remainders[:deficit]:
            floors[k] += 1
        return floors

    pa_quotas = _round_quotas(pa_fracs, panel_a_target)
    pb_quotas = _round_quotas(pb_fracs, panel_b_target)

    rng = np.random.default_rng(seed)

    final_pa_qids = []
    final_pb_qids = []

    for stratum in all_strata:
        pa_q = pa_quotas[stratum]
        pb_q = pb_quotas[stratum]
        if pa_q + pb_q == 0:
            continue

        students = strata_to_students[stratum]
        shuffled = rng.permutation(students).tolist()

        pa_qs = []
        pb_qs = []
        pa_filled = 0
        pb_filled = 0

        for sid in shuffled:
            nq = len(student_to_queries[sid])
            if pa_filled < pa_q:
                pa_qs.extend(student_to_queries[sid])
                pa_filled += nq
            elif pb_filled < pb_q:
                pb_qs.extend(student_to_queries[sid])
                pb_filled += nq
            if pa_filled >= pa_q and pb_filled >= pb_q:
                break

        final_pa_qids.extend(pa_qs[:pa_q])
        final_pb_qids.extend(pb_qs[:pb_q])

    assert len(final_pa_qids) == panel_a_target, f"PA count {len(final_pa_qids)} != {panel_a_target}"
    assert len(final_pb_qids) == panel_b_target, f"PB count {len(final_pb_qids)} != {panel_b_target}"

    pa_set = set(final_pa_qids)
    pb_set = set(final_pb_qids)
    pa_sids_set = {sid for sid, qids in student_to_queries.items() if any(q in pa_set for q in qids)}
    pb_sids_set = {sid for sid, qids in student_to_queries.items() if any(q in pb_set for q in qids)}
    student_overlap = len(pa_sids_set & pb_sids_set)
    query_overlap = len(pa_set & pb_set)

    all_selected_qids = final_pa_qids + final_pb_qids

    stratum_breakdown = {}
    for stratum in all_strata:
        pa_in = sum(1 for q in final_pa_qids if query_strata.get(q) == stratum)
        pb_in = sum(1 for q in final_pb_qids if query_strata.get(q) == stratum)
        count_selected = pa_in + pb_in
        if count_selected == 0:
            continue
        pa_tgt = round(count_selected * pa_ratio, 4)
        pb_tgt = round(count_selected * pb_ratio, 4)
        pa_abs_dev = round(abs(pa_in - pa_tgt), 4)
        pb_abs_dev = round(abs(pb_in - pb_tgt), 4)
        pa_rel_dev = round(pa_abs_dev / max(1.0, pa_tgt), 4)
        pb_rel_dev = round(pb_abs_dev / max(1.0, pb_tgt), 4)
        stratum_breakdown[stratum] = {
            "pool_count": count_selected,
            "panel_a_target": pa_tgt,
            "panel_a_actual": pa_in,
            "panel_b_target": pb_tgt,
            "panel_b_actual": pb_in,
            "panel_a_absolute_deviation": pa_abs_dev,
            "panel_b_absolute_deviation": pb_abs_dev,
            "panel_a_relative_deviation": pa_rel_dev,
            "panel_b_relative_deviation": pb_rel_dev,
        }

    pool_order_qids = list(query_strata.keys())
    is_first_n = (all_selected_qids == pool_order_qids[:len(all_selected_qids)])

    stages_present = {query_groups.get_group(q).iloc[0]["stage"] for q in all_selected_qids}
    folds_present = {query_groups.get_group(q).iloc[0]["outer_fold"] for q in all_selected_qids}

    substantial = [
        max(r["panel_a_relative_deviation"], r["panel_b_relative_deviation"])
        for r in stratum_breakdown.values()
        if r.get("pool_count", 0) >= 9
    ]
    max_rel_dev = round(max(substantial), 4) if substantial else 0.0

    audit_dict = {
        "sampling_method": "PROPORTIONAL_STRATIFIED_GROUP_ALLOCATION",
        "is_first_n_truncation": is_first_n,
        "sampling_seed": seed,
        "panel_a_count": len(final_pa_qids),
        "panel_b_count": len(final_pb_qids),
        "panel_a_case_count": len(final_pa_qids),
        "panel_b_case_count": len(final_pb_qids),
        "final_selected_case_count": len(all_selected_qids),
        "max_relative_deviation": max_rel_dev,
        "student_overlap": student_overlap,
        "query_overlap": query_overlap,
        "panel_student_overlap_count": student_overlap,
        "panel_query_overlap_count": query_overlap,
        "all_stages_represented": len(stages_present) >= 4,
        "all_outer_folds_represented": len(folds_present) >= 3,
        "stratum_breakdown": stratum_breakdown,
    }

    return final_pa_qids, final_pb_qids, audit_dict
