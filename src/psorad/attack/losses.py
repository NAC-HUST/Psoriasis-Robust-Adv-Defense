from __future__ import annotations

from typing import Protocol

import numpy as np
import torch


class PredictionModel(Protocol):
    def predict(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        ...


def to_pytorch(tensor_image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(tensor_image).permute(2, 0, 1)


def to_flat_numpy(preds: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(preds, torch.Tensor):
        return preds.detach().float().cpu().flatten().numpy()
    return np.asarray(preds, dtype=np.float64).flatten()


class UnTargeted:
    def __init__(self, model: PredictionModel, true: int, to_pytorch_input: bool = True):
        self.model = model
        self.true = int(true)
        self.to_pytorch_input = bool(to_pytorch_input)

    def _prepare_input(self, img: np.ndarray) -> torch.Tensor | np.ndarray:
        if self.to_pytorch_input:
            return to_pytorch(img).unsqueeze(0).to(dtype=torch.float32)
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
