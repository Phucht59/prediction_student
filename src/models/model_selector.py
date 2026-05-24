from src.models.basic_models import get_basic_models
from src.models.deep_models import build_cnn_bilstm, build_cnn_lstm


def get_basic_model(model_name: str, random_seed: int):
    models = get_basic_models(random_seed)
    if model_name not in models:
        available_names = ", ".join(models.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available_names}")
    return models[model_name]


def get_deep_model(model_name: str, input_shape: tuple[int, int], number_of_classes: int):
    if model_name == "cnn_lstm":
        return build_cnn_lstm(input_shape, number_of_classes)
    if model_name == "cnn_bilstm":
        return build_cnn_bilstm(input_shape, number_of_classes)
    raise ValueError(f"Unknown deep learning model '{model_name}'.")

