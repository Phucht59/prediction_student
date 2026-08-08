from pathlib import Path

path = Path(r"C:\Users\tranh\Downloads\42_panel_a_release_gates.py")
if not path.exists():
    raise SystemExit(f"MISSING={path}")

text = path.read_text(encoding="utf-8")
old = '    data, _, _ = base._load_inputs()\n    if len(data) != EXPECTED_ROWS or data["case_id"].nunique() != EXPECTED_CASES:\n        raise RuntimeError("FINAL_EBM_INPUT_SHAPE_MISMATCH")\n    if int(data["eligible"].astype(bool).sum()) != EXPECTED_ELIGIBLE:\n        raise RuntimeError("FINAL_EBM_ELIGIBLE_COUNT_MISMATCH")\n    if getattr(base, "LOCKED_GRID_SELECTED_CONFIG_ID", None) != EXPECTED_CONFIG_ID:\n        raise RuntimeError("RUNNER_CONFIG_ID_NOT_LOCKED")\n    if int(base.EBM_PARAMS["interactions"]) != 3:\n        raise RuntimeError("FINAL_INTERACTION_BUDGET_MISMATCH")\n\n    data = data.reset_index(drop=True)\n    data["full_score"] = np.clip(\n        data["ebm_oof_score"].to_numpy(dtype=float) / 3.0,\n        0.0,\n        1.0,\n    )\n'
new = '    data, _, _ = base._load_inputs()\n    if len(data) != EXPECTED_ROWS or data["case_id"].nunique() != EXPECTED_CASES:\n        raise RuntimeError("FINAL_EBM_INPUT_SHAPE_MISMATCH")\n    if int(data["eligible"].astype(bool).sum()) != EXPECTED_ELIGIBLE:\n        raise RuntimeError("FINAL_EBM_ELIGIBLE_COUNT_MISMATCH")\n    if getattr(base, "LOCKED_GRID_SELECTED_CONFIG_ID", None) != EXPECTED_CONFIG_ID:\n        raise RuntimeError("RUNNER_CONFIG_ID_NOT_LOCKED")\n    if int(base.EBM_PARAMS["interactions"]) != 3:\n        raise RuntimeError("FINAL_INTERACTION_BUDGET_MISMATCH")\n\n    # _load_inputs() returns the training table and intentionally does not\n    # include OOF predictions. Load the frozen OOF artifact explicitly and\n    # align it by stable keys before computing release-gate metrics.\n    oof_path = (\n        root\n        / "artifacts/recommend_hybrid/explainable_v2/models/ebm_panel_a_v1"\n        / "panel_a_ebm_oof_predictions.parquet"\n    )\n    if not oof_path.exists():\n        raise RuntimeError(f"MISSING_OOF_PREDICTIONS={oof_path}")\n\n    oof = pd.read_parquet(oof_path)\n    required_oof = set(META_COLS) | {"ebm_oof_score"}\n    missing_oof = sorted(required_oof - set(oof.columns))\n    if missing_oof:\n        raise RuntimeError(f"OOF_COLUMNS_MISSING={missing_oof}")\n    if len(oof) != EXPECTED_ROWS:\n        raise RuntimeError(f"OOF_ROW_COUNT={len(oof)} expected={EXPECTED_ROWS}")\n\n    merge_keys = list(META_COLS)\n    if oof.duplicated(merge_keys).any():\n        raise RuntimeError("OOF_DUPLICATE_STABLE_KEYS")\n    if data.duplicated(merge_keys).any():\n        raise RuntimeError("TRAINING_TABLE_DUPLICATE_STABLE_KEYS")\n\n    data = data.merge(\n        oof[merge_keys + ["ebm_oof_score"]],\n        on=merge_keys,\n        how="left",\n        validate="one_to_one",\n    )\n    if data["ebm_oof_score"].isna().any():\n        raise RuntimeError(\n            "OOF_MERGE_MISSING_ROWS="\n            f"{int(data[\'ebm_oof_score\'].isna().sum())}"\n        )\n\n    data = data.reset_index(drop=True)\n    data["full_score"] = np.clip(\n        data["ebm_oof_score"].to_numpy(dtype=float) / 3.0,\n        0.0,\n        1.0,\n    )\n'
if old not in text:
    raise SystemExit("TARGET_BLOCK_NOT_FOUND")
text = text.replace(old, new, 1)

old2 = '    vote_keys = votes[list(META_COLS)].copy()\n    model_keys = data[list(META_COLS)].copy()\n    for col in META_COLS:\n        vote_keys[col] = vote_keys[col].astype(str)\n        model_keys[col] = model_keys[col].astype(str)\n    if not vote_keys.equals(model_keys):\n        raise RuntimeError("VOTE_AND_MODEL_ROW_ORDER_MISMATCH")\n'
new2 = '    vote_keys = votes[list(META_COLS)].copy()\n    model_keys = data[list(META_COLS)].copy()\n    for col in ("query_id", "case_id", "stage", "action_id"):\n        vote_keys[col] = vote_keys[col].astype(str)\n        model_keys[col] = model_keys[col].astype(str)\n    vote_keys["outer_fold"] = vote_keys["outer_fold"].astype(int)\n    model_keys["outer_fold"] = model_keys["outer_fold"].astype(int)\n    if not vote_keys.equals(model_keys):\n        raise RuntimeError("VOTE_AND_MODEL_ROW_ORDER_MISMATCH")\n'
if old2 not in text:
    raise SystemExit("VOTE_ALIGNMENT_BLOCK_NOT_FOUND")
text = text.replace(old2, new2, 1)

compile(text, str(path), "exec")
path.write_text(text, encoding="utf-8")
print("OOF_MERGE_PATCHED=TRUE")
print("VOTE_KEY_ALIGNMENT_HARDENED=TRUE")
print("PANEL_B_TOUCHED=FALSE")
print("MODEL_ARTIFACTS_MODIFIED=FALSE")
print("NEXT_ACTION=RERUN_RELEASE_GATES")
