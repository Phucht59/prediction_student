# Final Learning Recommendation Report

This report uses the final three-dataset CNN-BiLSTM prediction models. It does not retrain models.

## Prediction Metrics Recomputed From Final Models

| dataset | accuracy | precision_macro | recall_macro | f1_macro | rmse | r2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| student-mat | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| student-por | 0.9394 | 0.9444 | 0.9467 | 0.9413 | 0.2462 | 0.9607 |
| xapi | 0.8958 | 0.9007 | 0.9020 | 0.8993 | 0.3227 | 0.8147 |

## Recommendation Logic

- Student datasets: recommendations are based on predicted G3 class, confidence, G1/G2 trend, failures, absences, and studytime.
- xAPI: recommendations are based on predicted class, confidence, raised hands, visited resources, absence days, and discussion activity.
- The rules are deterministic and explainable; they should be reviewed by an academic advisor before real deployment.

## Sample At-Risk Recommendations

### student-mat
- row `10` true=0 pred=0 confidence=0.519: Dat lich hoc phu dao hang tuan va theo doi bai tap ngan theo tung chu de. / Uu tien on lai kien thuc nen tang vi diem G2 dang duoi nguong dat. / Can can thiep som vi diem G2 giam so voi G1. / Du doan chua chac chan cao; nen danh gia them bang bai kiem tra ngan.
- row `280` true=0 pred=0 confidence=0.531: Dat lich hoc phu dao hang tuan va theo doi bai tap ngan theo tung chu de. / Uu tien on lai kien thuc nen tang vi diem G2 dang duoi nguong dat. / Giam vang hoc va them buoi hoc bu vi so ngay vang cao. / Tang thoi gian tu hoc len muc toi thieu 2 buoi moi tuan. / Du doan chua chac chan cao; nen danh gia them bang bai kiem tra ngan.
- row `255` true=0 pred=0 confidence=0.574: Dat lich hoc phu dao hang tuan va theo doi bai tap ngan theo tung chu de. / Uu tien on lai kien thuc nen tang vi diem G2 dang duoi nguong dat. / Lap ke hoach bu lo hong do co lich su rot mon/that bai hoc tap. / Tang thoi gian tu hoc len muc toi thieu 2 buoi moi tuan.
- row `206` true=0 pred=0 confidence=0.801: Dat lich hoc phu dao hang tuan va theo doi bai tap ngan theo tung chu de. / Uu tien on lai kien thuc nen tang vi diem G2 dang duoi nguong dat. / Lap ke hoach bu lo hong do co lich su rot mon/that bai hoc tap.
- row `216` true=0 pred=0 confidence=0.926: Dat lich hoc phu dao hang tuan va theo doi bai tap ngan theo tung chu de. / Uu tien on lai kien thuc nen tang vi diem G2 dang duoi nguong dat. / Lap ke hoach bu lo hong do co lich su rot mon/that bai hoc tap. / Giam vang hoc va them buoi hoc bu vi so ngay vang cao.

### student-por
- row `528` true=0 pred=0 confidence=0.592: Dat lich hoc phu dao hang tuan va theo doi bai tap ngan theo tung chu de. / Uu tien on lai kien thuc nen tang vi diem G2 dang duoi nguong dat. / Tang thoi gian tu hoc len muc toi thieu 2 buoi moi tuan.
- row `78` true=1 pred=0 confidence=0.592: Dat lich hoc phu dao hang tuan va theo doi bai tap ngan theo tung chu de. / Uu tien on lai kien thuc nen tang vi diem G2 dang duoi nguong dat. / Lap ke hoach bu lo hong do co lich su rot mon/that bai hoc tap. / Tang thoi gian tu hoc len muc toi thieu 2 buoi moi tuan.
- row `476` true=0 pred=0 confidence=0.696: Dat lich hoc phu dao hang tuan va theo doi bai tap ngan theo tung chu de. / Uu tien on lai kien thuc nen tang vi diem G2 dang duoi nguong dat. / Can can thiep som vi diem G2 giam so voi G1. / Tang thoi gian tu hoc len muc toi thieu 2 buoi moi tuan.
- row `465` true=0 pred=0 confidence=0.830: Dat lich hoc phu dao hang tuan va theo doi bai tap ngan theo tung chu de. / Uu tien on lai kien thuc nen tang vi diem G2 dang duoi nguong dat. / Can can thiep som vi diem G2 giam so voi G1. / Lap ke hoach bu lo hong do co lich su rot mon/that bai hoc tap.
- row `436` true=0 pred=0 confidence=0.937: Dat lich hoc phu dao hang tuan va theo doi bai tap ngan theo tung chu de. / Uu tien on lai kien thuc nen tang vi diem G2 dang duoi nguong dat. / Lap ke hoach bu lo hong do co lich su rot mon/that bai hoc tap. / Tang thoi gian tu hoc len muc toi thieu 2 buoi moi tuan.

### xapi
- row `324` true=1 pred=0 confidence=0.659: Can can thiep muc cao: theo doi hang tuan va giao muc tieu hoc tap nho. / Tang tan suat phat bieu/dat cau hoi tren lop. / Tang so lan truy cap tai nguyen hoc tap moi tuan.
- row `181` true=0 pred=0 confidence=0.752: Can can thiep muc cao: theo doi hang tuan va giao muc tieu hoc tap nho. / Giam vang hoc vi StudentAbsenceDays dang o muc Above-7.
- row `331` true=0 pred=0 confidence=0.874: Can can thiep muc cao: theo doi hang tuan va giao muc tieu hoc tap nho. / Tang so lan truy cap tai nguyen hoc tap moi tuan. / Giam vang hoc vi StudentAbsenceDays dang o muc Above-7.
- row `133` true=0 pred=0 confidence=0.928: Can can thiep muc cao: theo doi hang tuan va giao muc tieu hoc tap nho. / Tang tan suat phat bieu/dat cau hoi tren lop. / Giam vang hoc vi StudentAbsenceDays dang o muc Above-7.
- row `479` true=0 pred=0 confidence=0.944: Can can thiep muc cao: theo doi hang tuan va giao muc tieu hoc tap nho. / Tang so lan truy cap tai nguyen hoc tap moi tuan. / Giam vang hoc vi StudentAbsenceDays dang o muc Above-7.

