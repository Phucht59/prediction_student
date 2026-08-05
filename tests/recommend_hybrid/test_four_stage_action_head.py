from __future__ import annotations

import numpy as np

from src.recommend_hybrid.final.actions import ACTION_COUNT
from src.recommend_hybrid.final.metrics import STAGE_ORDER, make_decisions
from src.recommend_hybrid.final.stage_aware_training import (
    FeatureStandardizer,
    FourStageActionData,
    calibrate_action_thresholds,
    grouped_outer_splits,
)


def _data() -> FourStageActionData:
    rng = np.random.default_rng(20260806)
    rows_per_stage = 90
    count = rows_per_stage * len(STAGE_ORDER)
    stages = np.repeat(np.asarray(STAGE_ORDER), rows_per_stage)
    student_ids = np.asarray([f"student-{index // 2}" for index in range(count)])
    record_ids = np.asarray([f"record-{index}" for index in range(count)])
    group = rng.normal(size=(count, 12)).astype(np.float32)
    action = rng.normal(size=(count, ACTION_COUNT, 6)).astype(np.float32)
    mask = np.ones((count, ACTION_COUNT), dtype=bool)
    target = np.zeros((count, ACTION_COUNT), dtype=np.int8)
    preferred = (group[:, 0] > 0).astype(int) + (
        np.arange(count) % (ACTION_COUNT - 1)
    )
    preferred %= ACTION_COUNT
    positive_group = group[:, 1] + 0.5 * group[:, 2] > -0.2
    target[np.flatnonzero(positive_group), preferred[positive_group]] = 1
    result = FourStageActionData(
        group_features=group,
        action_features=action,
        action_ids=np.tile(np.arange(ACTION_COUNT), (count, 1)),
        action_mask=mask,
        group_target=positive_group.astype(np.int8),
        action_target=target,
        student_ids=student_ids,
        record_ids=record_ids,
        stages=stages,
        group_feature_names=tuple(f"g{index}" for index in range(group.shape[1])),
        action_feature_names=tuple(f"a{index}" for index in range(action.shape[2])),
    )
    result.validate()
    return result


def test_grouped_outer_splits_have_no_student_overlap() -> None:
    data = _data()
    splits = grouped_outer_splits(data)
    assert len(splits) == 3
    covered = []
    for split in splits:
        train = set(data.student_ids[split.train_index])
        validation = set(data.student_ids[split.validation_index])
        test = set(data.student_ids[split.test_index])
        assert not train.intersection(validation)
        assert not train.intersection(test)
        assert not validation.intersection(test)
        covered.extend(split.test_index.tolist())
    assert sorted(covered) == list(range(len(data.group_target)))


def test_standardizer_uses_training_statistics_and_masks_invalid_actions() -> None:
    data = _data()
    train_index = grouped_outer_splits(data)[0].train_index
    train = data.subset(train_index)
    scaler = FeatureStandardizer.fit(train)
    transformed = scaler.transform(train)
    assert np.allclose(transformed.group_features.mean(axis=0), 0.0, atol=1e-5)
    masked = train.action_mask.copy()
    masked[:, -1] = False
    changed = FourStageActionData(
        group_features=train.group_features,
        action_features=train.action_features,
        action_ids=train.action_ids,
        action_mask=masked,
        group_target=train.group_target,
        action_target=np.where(masked, train.action_target, 0),
        student_ids=train.student_ids,
        record_ids=train.record_ids,
        stages=train.stages,
        group_feature_names=train.group_feature_names,
        action_feature_names=train.action_feature_names,
    )
    transformed_changed = scaler.transform(changed)
    assert np.all(transformed_changed.action_features[:, -1] == 0.0)


def test_calibration_returns_four_stage_thresholds() -> None:
    data = _data()
    direct = np.where(data.group_target > 0, 2.0, -1.5)
    action = np.full((len(data.group_target), ACTION_COUNT), -2.0)
    for row in range(len(data.group_target)):
        positives = np.flatnonzero(data.action_target[row] > 0)
        if len(positives):
            action[row, positives[0]] = 2.5
        else:
            action[row, row % ACTION_COUNT] = -0.5
    thresholds = calibrate_action_thresholds(
        direct_logits=direct,
        action_logits=action,
        data=data,
    )
    assert len(thresholds.stage_gate_probability) == 4
    decision = make_decisions(
        direct,
        action,
        data.action_mask,
        data.stages,
        thresholds,
    )
    assert decision.issued.shape == (len(data.group_target),)
    assert set(data.stages[decision.issued]).issubset(set(STAGE_ORDER))
