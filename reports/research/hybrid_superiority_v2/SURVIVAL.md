# Survivorship vs extra VLE weeks

{
  "n_20": 26697,
  "n_100": 22522,
  "n_both": 22513,
  "n_dropped_after_20": 4184,
  "prevalence_20": 0.4238678503202607,
  "prevalence_100": 0.31689015185152297,
  "prevalence_100_on_both": 0.3167947408164172
}

Nếu AP 100% trên tập `both` (còn sống từ 20%) thấp hơn AP 100% trên mọi enrollment 100%, một phần ΔAP theo cutoff đến từ mẫu dễ hơn (Withdrawn sớm đã bị loại).

| fold | AP 100% all VALID | AP 100% ∩ still-in-20% | n_valid | n_both |
|---:|---:|---:|---:|---:|
| 0 | 0.9311 | 0.9311 | 4999 | 4999 |
| 1 | 0.9161 | 0.9159 | 4993 | 4991 |
| 2 | 0.9153 | 0.9155 | 5007 | 5004 |