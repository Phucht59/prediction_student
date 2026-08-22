"""P1: AP at 100% on students still present at 20% vs all 100% (survivorship)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .data import copy_locked_splits, ids_for_stage, inner_partitions, load_phase2_cache, scale_prepared
from .paths import REP, SERVING_CKPT, ensure
from .train import StageTrainer
from src.prediction.model.hybrid import Hybrid, HybridConfig
import torch


def ap_safe(y, p):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def main() -> None:
    ensure()
    copy_locked_splits()
    views, context = load_phase2_cache("oulad")
    ids20 = set(map(str, views["20pct"].record_id))
    ids100 = set(map(str, views["100pct"].record_id))
    both = ids20 & ids100
    dropped = ids20 - ids100
    rows = [
        {
            "n_20": len(ids20),
            "n_100": len(ids100),
            "n_both": len(both),
            "n_dropped_after_20": len(dropped),
            "prevalence_20": float(views["20pct"].target.mean()),
            "prevalence_100": float(views["100pct"].target.mean()),
            "prevalence_100_on_both": float(views["100pct"].target[np.isin(views["100pct"].record_id.astype(str), list(both))].mean())
            if both
            else float("nan"),
        }
    ]
    # try serving checkpoints fold 0/1/2 seed 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_rows = []
    for fold in (0, 1, 2):
        ckpt_path = SERVING_CKPT / f"c0_inner_fold{fold}_seed42.pt"
        if not ckpt_path.exists():
            continue
        fit, stop, valid = inner_partitions("oulad", context, fold)
        prepared = scale_prepared("oulad", views, context, fit)
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = HybridConfig(**{k: v for k, v in payload["config"].items() if k in HybridConfig.__dataclass_fields__})
        model = Hybrid(cfg).to(device)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        trainer = StageTrainer(prepared, "100pct", "full", seed=42)
        trainer.device = device
        valid100 = ids_for_stage(prepared.views["100pct"], valid)
        p = trainer._predict(model, valid100)
        y = np.asarray([int(prepared.views["100pct"].target[trainer.lookup[i]]) for i in valid100])
        recs = np.asarray(valid100)
        mask_both = np.isin(recs, list(both))
        eval_rows.append(
            {
                "fold": fold,
                "ap_100_all_valid": ap_safe(y, p),
                "ap_100_valid_also_in_20": ap_safe(y[mask_both], p[mask_both]),
                "n_valid_100": len(y),
                "n_valid_both": int(mask_both.sum()),
            }
        )
        del trainer
    pd.DataFrame(rows).to_csv(REP / "survival_counts.csv", index=False)
    pd.DataFrame(eval_rows).to_csv(REP / "survival_ap.csv", index=False)
    lines = [
        "# Survivorship vs extra VLE weeks",
        "",
        json.dumps(rows[0], indent=2),
        "",
        "Nếu AP 100% trên tập `both` (còn sống từ 20%) thấp hơn AP 100% trên mọi enrollment 100%, một phần ΔAP theo cutoff đến từ mẫu dễ hơn (Withdrawn sớm đã bị loại).",
        "",
    ]
    if eval_rows:
        ev = pd.DataFrame(eval_rows)
        lines += ["| fold | AP 100% all VALID | AP 100% ∩ still-in-20% | n_valid | n_both |", "|---:|---:|---:|---:|---:|"]
        for r in ev.itertuples():
            lines.append(
                f"| {r.fold} | {r.ap_100_all_valid:.4f} | {r.ap_100_valid_also_in_20:.4f} | {r.n_valid_100} | {r.n_valid_both} |"
            )
    else:
        lines.append("TODO — chưa load được serving checkpoint trên máy này.")
    (REP / "SURVIVAL.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote survival")


if __name__ == "__main__":
    main()
