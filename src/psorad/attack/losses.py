from __future__ import annotations

from typing import Protocol

import numpy as np
import paddle


class PredictionModel(Protocol):
    def predict(self, x: paddle.Tensor | np.ndarray) -> paddle.Tensor:
        ...


def to_paddle(tensor_image: np.ndarray) -> paddle.Tensor:
    return paddle.to_tensor(np.transpose(tensor_image, (2, 0, 1)), dtype="float32")


def to_flat_numpy(preds: paddle.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(preds, paddle.Tensor):
        return np.asarray(preds.numpy(), dtype=np.float64).flatten()
    return np.asarray(preds, dtype=np.float64).flatten()


class UnTargeted:
    def __init__(self, model: PredictionModel, true: int, to_paddle_input: bool = True):
        self.model = model
        self.true = int(true)
        self.to_paddle_input = bool(to_paddle_input)

    def _prepare_input(self, img: np.ndarray) -> paddle.Tensor | np.ndarray:
        if self.to_paddle_input:
            return to_paddle(img).unsqueeze(0)
        return np.expand_dims(img, axis=0)

    def _predict_vector(self, img: np.ndarray) -> np.ndarray:
        preds = self.model.predict(self._prepare_input(img))
        return to_flat_numpy(preds)

    def get_label(self, img: np.ndarray) -> int:
        pred_vec = self._predict_vector(img)
        return int(np.argmax(pred_vec))

    def __call__(self, img: np.ndarray) -> list[float]:
        pred_vec = self._predict_vector(img)
        y_pred = int(np.argmax(pred_vec))

        is_adversarial = float(y_pred != self.true)
        true_score = float(pred_vec[self.true])

        masked = pred_vec.copy()
        masked[self.true] = -np.inf
        other_max = float(np.max(masked))

        loss = true_score - other_max
        return [is_adversarial, loss]
