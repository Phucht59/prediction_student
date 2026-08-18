from src.database.repository import register_model, set_active_model
model_id=register_model("hybrid","Hybrid","CNN + BiLSTM Hybrid"); set_active_model(model_id); print("registered model=hybrid")
