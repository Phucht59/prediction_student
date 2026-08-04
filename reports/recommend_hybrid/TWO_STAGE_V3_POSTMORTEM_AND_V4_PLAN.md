# Two-Stage V3 postmortem và kế hoạch V4 Action-Aware

## Kết quả V3 đã khóa

```text
Groups: 29.043
Positive groups: 9.304
Issued groups: 7.105
Issued positive groups: 4.850
False issue groups: 2.255
Correct issued actions: 4.569

Stage A precision: 0,6826
Stage A recall / coverage: 0,5213
Stage B conditional Precision@1: 0,9421
End-to-end Precision@1: 0,6431
Bootstrap 95% CI: [0,6327; 0,6540]
```

V3 cải thiện mạnh so với Hybrid-only deterministic:

```text
Hybrid-only end-to-end Precision@1: 0,2711
V3 end-to-end Precision@1:          0,6431
```

Tuy nhiên V3 không đạt release gate 0,80 và runtime vẫn bị khóa.

## Phân rã lỗi

End-to-end precision có thể phân rã thành:

```text
end-to-end precision
= recommendability precision
× conditional action precision

0,6431 ≈ 0,6826 × 0,9421
```

Với conditional Precision@1 giữ ở 0,9421, Stage A phải đạt ít nhất:

```text
0,80 / 0,9421 = 0,8492
```

Tức Stage A cần tăng từ 68,26% lên khoảng 84,92% tại coverage xấp xỉ 50%.

Ở số recommendation hiện tại, nếu giữ 4.569 correct actions thì số issued groups tối đa để đạt 80% là khoảng 5.711. V3 phát 7.105 recommendations, nên phải loại ít nhất khoảng 1.394 false hoặc incorrect issues mà gần như không làm mất positive coverage.

## Defect kiến trúc trong V3

V3 áp dụng:

```text
candidate binary loss → positive groups only
listwise action loss  → positive groups only
```

Với 19.739 negative groups, candidate action head không bị phạt trực tiếp khi tạo action logit cao. Vì vậy:

- Stage B học rất tốt cách chọn action trong positive groups;
- action logits không học đầy đủ ý nghĩa “không có action nào nên được phát”;
- direct Stage A head phải tự xử lý toàn bộ false-issue problem;
- action probability và margin không tạo được gate đủ mạnh trên negative groups.

Đây là lỗi thiết kế objective, không phải lỗi checkpoint CNN–BiLSTM, leakage hoặc threshold implementation.

## V4 Action-Aware correction

V4 giữ nguyên:

- 160.492 tham số residual CNN–BiLSTM frozen;
- 64-D student-state embedding;
- 32-D tabular-expert embedding;
- 29.043 groups;
- 82.847 candidates;
- silver labels;
- outer folds;
- release gates;
- claim boundary.

V4 thay objective của integrated heads:

```text
Direct group BCE
+ candidate BCE trên ALL_VALID_CANDIDATES
+ listwise loss trên positive groups
+ noisy-OR action-group loss
+ direct/action consistency loss
```

Negative groups có action target vector toàn 0 và trực tiếp tạo gradient kéo candidate probabilities xuống.

Recommendability cuối được tạo từ:

```text
direct group probability
+ action-derived probability = 1 - ∏(1 - p_action)
+ preregistered log-probability blend
```

Threshold được chọn từ inner OOF theo stage. Outer-test không được dùng để chọn model, blend hoặc threshold.

## Release interpretation

Nếu V4 main gates không đạt:

```text
TWO_STAGE_V4_EVIDENCE_BELOW_GATE
RECOMMENDATION_MODULE_NOT_COMPLETE
runtime_authorized = false
```

Không tiếp tục sửa labels hoặc giảm gates.

Nếu V4 main gates đạt:

```text
TWO_STAGE_V4_MAIN_EVALUATION_PASS_CONTROLS_PENDING
RECOMMENDATION_MODULE_SCIENTIFIC_EXECUTION_NOT_COMPLETE
runtime_authorized = false
```

Sau đó mới chạy authority-bound negative controls và runtime packaging.

Claim boundary giữ nguyên:

```text
OFFLINE_PREDICTIVE_RELEVANCE_NOT_CAUSAL_EFFECT
```
