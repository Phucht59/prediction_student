from tensorflow.keras.layers import Bidirectional, Conv1D, Dense, Dropout, LSTM, MaxPooling1D
from tensorflow.keras.models import Sequential


def build_cnn_lstm(input_shape: tuple[int, int], number_of_classes: int) -> Sequential:
    model = Sequential(
        [
            Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            LSTM(32),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dense(number_of_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_cnn_bilstm(input_shape: tuple[int, int], number_of_classes: int) -> Sequential:
    model = Sequential(
        [
            Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Bidirectional(LSTM(32)),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dense(number_of_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

