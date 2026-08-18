# Class Imbalance Evidence

## 1. M?c ti?u
??nh gi? li?u x? l? m?t c?n b?ng c? c?i thi?n hi?u n?ng trong c?c protocol ?? ???c ki?m ch?ng, ??ng th?i kh?ng thay ??i frozen authority.

## 2. C?c k? thu?t kh?o s?t
NONE, Class Weight, SMOTE v? ADASYN. Resampling ch? thu?c train partition.

## 3. Ki?m so?t data leakage
Preprocessing fit tr?n train; validation v? outer test kh?ng resample.

## 4. Th? nghi?m sensitivity ban ??u
Frozen Hybrid embeddings + LogisticRegression l? supporting sensitivity evidence, kh?ng ph?i hu?n luy?n tr?c ti?p final Hybrid.

## 5. Direct Hybrid diagnostic experiment
52/52 job ho?n t?t nh?ng `INVALID_FOR_FINAL_AUTHORITY_COMPARISON`; kh?ng d?ng l?m headline conclusion.

## 6. Authority-equivalent verification
MAT reproduces exactly (0.9014601961315334). POR authority-policy replay reproduces exactly (0.8622587167738002); POR FIXED_NONE (0.8614773022644222) kh?ng ph?i authority v? fold 1 d?ng CLASS_WEIGHT. OULAD protocol-equivalent replay is 0.8942454181505014 versus 0.8940709888551659, delta 0.0001744292953355; forensic audit attributes residual to CUDA nondeterminism but exact control remains FAIL.

## 7. Ch?nh s?ch m?t c?n b?ng th?c t? c?a final models
MAT = MIXED_EFFECTIVE_LOSS; POR = MIXED_FOLD_SPECIFIC; OULAD = UNIFORM_NONE_STANDARD_BCE.

## 8. OULAD controlled class-weight challenge
The authority-equivalent `FIXED_CLASS_WEIGHT` challenge completed all 15 jobs (3 folds x 5 seeds). Against its record-aligned `FIXED_NONE` control, Macro-F1 changed from `0.894245` to `0.885983` (`-0.008262`), while Risk Recall increased from `0.784267` to `0.826008` and PR-AUC changed from `0.934926` to `0.934896`. `CLASS_WEIGHT_WINS=FALSE` and `PROMOTE_FINAL_V2=FALSE`; see `OULAD_CLASS_WEIGHT_CHALLENGE.md` and the stored run manifests, metrics, and OOF prediction archives.

## 9. K?t lu?n
X? l? m?t c?n b?ng kh?ng cho th?y l?i ?ch nh?t qu?n tr?n to?n b? b?i to?n. C?c k?t qu? sensitivity cho th?y SMOTE v? ADASYN kh?ng t? ??ng c?i thi?n hi?u n?ng v? trong nhi?u thi?t l?p c?n l?m gi?m F1/recall. Class Weight c? th? c? ?ch ? m?t s? tr??ng h?p c? th?, ?i?n h?nh l? m?t fold trong authority c?a Student-Por. ??i v?i OULAD, model final d?ng standard BCE m? kh?ng ?p d?ng t?i c?n b?ng l?p. V? v?y nghi?n c?u kh?ng ?p d?ng m?t k? thu?t c?n b?ng duy nh?t cho t?t c? b? d? li?u m? gi? ch?nh s?ch ???c x?c nh?n theo t?ng m? h?nh/protocol.
