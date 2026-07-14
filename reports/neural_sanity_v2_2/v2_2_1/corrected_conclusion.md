# Neural Sanity V2.2 closure correction

Budget 20 có dấu hiệu binding vì 12/25 early-stop runs chạy đến giới hạn 20 epoch. Tuy nhiên, median epoch được chọn để refit là 10. Với S3, 5/25 runs chạy đến giới hạn 40 và median selected epoch là 13.

`selected_epoch` là argmax internal-validation Macro-F1 và là số epoch dùng cho clean refit. `epochs_ran` là số epoch mà early-stopping phase thực sự chạy. `hit_epoch_cap` chỉ có nghĩa `epochs_ran >= max_epochs`; không được diễn giải nó như selected epoch.

Correction này chỉ sửa diễn giải báo cáo. Prediction, checkpoint, metric và kết luận so sánh S0–S5 không thay đổi; không có retraining.
