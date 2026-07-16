from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch import nn

from psorad.attack.losses import UnTargeted
from psorad.attack.samoo_core.attack import Attack, AttackParams
from psorad.utils.image import center_crop_resize


def _safe_path_token(value: str) -> str:
    token = value.strip().replace(" ", "_")
    token = token.replace("/", "_").replace("\\", "_")
    return token or "unknown"


def _resolution_profile(height: int, width: int) -> dict[str, float | int]:
    total_pixels = int(height * width)
    small_resolution_threshold = 64 * 64

    if total_pixels <= small_resolution_threshold:
        return {
            "eps_ratio": 0.024,
            "iterations": 500,
            "pc": 0.1,
            "pm": 0.4,
            "pm_end": 0.1,
            "pop_size": 2,
            "query_budget": 1000,
            "zero_probability": 0.3,
            "p_size": 2.0,
            "tournament_size": 2,
        }

    return {
        "eps_ratio": 0.0023,
        "iterations": 2500,
        "pc": 0.1,
        "pm": 0.4,
        "pm_end": 0.1,
        "pop_size": 2,
        "query_budget": 5000,
        "zero_probability": 0.3,
        "p_size": 2.0,
        "tournament_size": 2,
    }


def _resolve_attack_hparams(
    *,
    height: int,
    width: int,
    eps: int | None,
    iterations: int | None,
    pc: float | None,
    pm: float | None,
    pm_end: float | None,
    pop_size: int | None,
    query_budget: int | None,
    zero_probability: float | None,
    p_size: float | None,
    tournament_size: int | None,
) -> dict[str, Any]:
    profile = _resolution_profile(height=height, width=width)
    total_pixels = int(height * width)

    profile_eps = max(1, min(int(round(float(profile["eps_ratio"]) * total_pixels)), total_pixels))
    resolved_eps = min(int(eps), total_pixels) if eps is not None else profile_eps
    eps_is_adaptive = eps is None

    resolved_pop_size = int(pop_size) if pop_size is not None else int(profile["pop_size"])
    resolved_query_budget = int(query_budget) if query_budget is not None else int(profile["query_budget"])

    if iterations is not None:
        resolved_iterations = int(iterations)
    elif resolved_query_budget > 0:
        resolved_iterations = max(2, int(math.ceil(resolved_query_budget / float(resolved_pop_size))))
    else:
        resolved_iterations = int(profile["iterations"])

    return {
        "eps": resolved_eps,
        "eps_is_adaptive": eps_is_adaptive,
        "iterations": resolved_iterations,
        "pc": float(pc) if pc is not None else float(profile["pc"]),
        "pm": float(pm) if pm is not None else float(profile["pm"]),
        "pm_end": float(pm_end) if pm_end is not None else float(profile["pm_end"]),
        "pop_size": resolved_pop_size,
        "query_budget": resolved_query_budget,
        "zero_probability": float(zero_probability) if zero_probability is not None else float(profile["zero_probability"]),
        "p_size": float(p_size) if p_size is not None else float(profile["p_size"]),
        "tournament_size": int(tournament_size) if tournament_size is not None else int(profile["tournament_size"]),
        "resolution_profile": "small" if total_pixels <= 64 * 64 else "large",
    }


def _apply_confidence_boost(
    *,
    hparams: dict[str, Any],
    probs_before: np.ndarray,
    true_label: int,
    image_height: int,
    image_width: int,
    eps_user_provided: bool,
    iterations_user_provided: bool,
    pop_size_user_provided: bool,
    query_budget_user_provided: bool,
) -> tuple[dict[str, Any], bool, str]:
    boosted = dict(hparams)

    if str(boosted.get("resolution_profile")) != "large":
        return boosted, False, "resolution=small"
    if probs_before.size == 0 or true_label < 0 or true_label >= probs_before.size:
        return boosted, False, "invalid_probability_vector"

    true_conf = float(probs_before[true_label])
    total_pixels = int(image_height * image_width)

    if true_conf >= 0.9999:
        target_eps_ratio = 0.010
        target_query_budget = 12000
        target_pop_size = 4
        reason = "very_high_confidence"
    elif true_conf >= 0.995:
        target_eps_ratio = 0.006
        target_query_budget = 8000
        target_pop_size = 4
        reason = "high_confidence"
    else:
        return boosted, False, "confidence_not_high"

    if not eps_user_provided:
        target_eps = max(1, min(int(round(target_eps_ratio * total_pixels)), total_pixels))
        boosted["eps"] = max(int(boosted["eps"]), target_eps)

    if not pop_size_user_provided:
        boosted["pop_size"] = max(int(boosted["pop_size"]), int(target_pop_size))

    if not query_budget_user_provided:
        boosted["query_budget"] = max(int(boosted["query_budget"]), int(target_query_budget))

    if not iterations_user_provided:
        pop_size = max(1, int(boosted["pop_size"]))
        query_budget = max(1, int(boosted["query_budget"]))
        boosted["iterations"] = max(2, int(math.ceil(query_budget / float(pop_size))))

    return boosted, True, reason


class BinaryModelAdapter:
    def __init__(self, model: nn.Module, device: torch.device, backbone: str):
        self.model = model
        self.device = device

        if backbone == "resnet50":
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device)
        elif backbone == "siglip":
            mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device=device)
            std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device=device)
        else:
            raise ValueError("backbone 仅支持 resnet50 或 siglip")

        self.mean = mean.view(1, 3, 1, 1)
        self.std = std.view(1, 3, 1, 1)

    def _prepare_tensor(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        tensor = x
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)

        if tensor.ndim == 3:
            if tensor.shape[-1] == 3:
                tensor = tensor.permute(2, 0, 1)
            tensor = tensor.unsqueeze(0)

        tensor = tensor.to(self.device, dtype=torch.float32)
        tensor = torch.clamp(tensor, 0.0, 1.0)
        return (tensor - self.mean) / self.std

    @torch.no_grad()
    def predict(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        x = self._prepare_tensor(x)
        logits = self.model(x)
        if not isinstance(logits, torch.Tensor):
            raise TypeError("model forward 必须返回 torch.Tensor")
        # 现在模型直接输出 (batch, num_classes)
        if logits.ndim == 1:
            # 如果还是旧的二分类输出 (batch,)，则转换为 (batch, 2)
            logits_binary = logits.reshape(-1)
            logits = torch.stack([-logits_binary, logits_binary], dim=1)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor | np.ndarray) -> np.ndarray:
        logits = self.predict(x)
        probs = torch.softmax(logits, dim=-1)
        return np.asarray(probs.detach().cpu().numpy(), dtype=np.float32).reshape(-1)

    @torch.no_grad()
    def predict_binary_logit(self, x: np.ndarray) -> float:
        tensor = self._prepare_tensor(x)
        logits = self.model(tensor)
        if not isinstance(logits, torch.Tensor):
            raise TypeError("model forward 必须返回 torch.Tensor")
        # 返回正类（class 1）的 logit
        if logits.ndim == 1:
            # 旧的二分类模式
            logit = logits.reshape(-1)[0]
        else:
            # 新的多分类模式，取 class 1 的 logit
            logit = logits[0, 1]
        return float(logit.item())


def _to_uint8_image(img: np.ndarray) -> np.ndarray:
    clipped = np.clip(img, 0.0, 1.0)
    return cast(np.ndarray, (clipped * 255.0).round().astype(np.uint8))


def _build_comparison_image(before: np.ndarray, after: np.ndarray) -> Image.Image:
    before_u8 = _to_uint8_image(before)
    after_u8 = _to_uint8_image(after)
    diff = np.abs(after - before)
    diff = np.clip(diff * 4.0, 0.0, 1.0)
    diff_u8 = _to_uint8_image(diff)

    before_img = Image.fromarray(before_u8)
    after_img = Image.fromarray(after_u8)
    diff_img = Image.fromarray(diff_u8)

    width, height = before_img.size
    title_h = 32
    canvas = Image.new("RGB", (width * 3, height + title_h), color=(255, 255, 255))
    canvas.paste(before_img, (0, title_h))
    canvas.paste(after_img, (width, title_h))
    canvas.paste(diff_img, (width * 2, title_h))

    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), "Before", fill=(0, 0, 0))
    draw.text((width + 10, 8), "After", fill=(0, 0, 0))
    draw.text((width * 2 + 10, 8), "|After-Before| x4", fill=(0, 0, 0))
    return canvas


def _pick_best_candidate(payload: dict[str, Any], y_true: int) -> tuple[int, np.ndarray, int]:
    front0_imgs = payload.get("front0_imgs", [])
    if len(front0_imgs) == 0:
        raise RuntimeError("攻击结果中未包含 front0_imgs。")

    adv_labels = payload.get("adversarial_labels", [])
    for idx, adv_label in enumerate(adv_labels):
        if int(adv_label) != y_true:
            return idx, np.asarray(front0_imgs[idx], dtype=np.float32), int(adv_label)

    return 0, np.asarray(front0_imgs[0], dtype=np.float32), int(adv_labels[0]) if len(adv_labels) > 0 else y_true


def _export_attack_artifacts(
    export_dir: Path,
    backbone: str,
    checkpoint_path: str,
    sample_index_subset: int,
    sample_index_global: int,
    attack_split: str,
    raw_npy_path: Path,
    payload: dict[str, Any],
    original_img: np.ndarray,
    true_label: int,
    pred_before: int,
    pred_after: int,
    logit_before: float,
    logit_after: float,
    selected_idx: int,
    selected_adv: np.ndarray,
    modified_pixel_count: int,
    modified_channel_count: int,
    prob_before_vec: np.ndarray,
    prob_after_vec: np.ndarray,
    detailed_log_lines: list[str],
) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)

    before_img = Image.fromarray(_to_uint8_image(original_img))
    after_img = Image.fromarray(_to_uint8_image(selected_adv))
    diff_img = Image.fromarray(_to_uint8_image(np.clip(np.abs(selected_adv - original_img) * 4.0, 0.0, 1.0)))
    compare_img = _build_comparison_image(original_img, selected_adv)

    before_path = export_dir / "before.png"
    after_path = export_dir / "after.png"
    diff_path = export_dir / "diff_x4.png"
    compare_path = export_dir / "before_after_diff.png"

    before_img.save(before_path)
    after_img.save(after_path)
    diff_img.save(diff_path)
    compare_img.save(compare_path)

    perturb = selected_adv - original_img
    l2 = float(np.linalg.norm(perturb.reshape(-1), ord=2))
    linf = float(np.max(np.abs(perturb)))

    success = bool(pred_after != true_label)
    queries = int(payload.get("queries", -1))
    prob_true_before = float(prob_before_vec[true_label]) if true_label < len(prob_before_vec) else float("nan")
    prob_true_after = float(prob_after_vec[true_label]) if true_label < len(prob_after_vec) else float("nan")
    prob_pred_before = float(prob_before_vec[pred_before]) if pred_before < len(prob_before_vec) else float("nan")
    prob_pred_after = float(prob_after_vec[pred_after]) if pred_after < len(prob_after_vec) else float("nan")
    prob_class1_before = float(prob_before_vec[1]) if len(prob_before_vec) > 1 else float(prob_before_vec[0])
    prob_class1_after = float(prob_after_vec[1]) if len(prob_after_vec) > 1 else float(prob_after_vec[0])

    summary = {
        "backbone": backbone,
        "checkpoint_path": checkpoint_path,
        "sample_index": sample_index_subset,
        "sample_index_subset": sample_index_subset,
        "sample_index_global": sample_index_global,
        "attack_split": attack_split,
        "true_label": true_label,
        "pred_before": pred_before,
        "pred_after": pred_after,
        "logit_before": logit_before,
        "logit_after": logit_after,
        "prob_true_before": prob_true_before,
        "prob_true_after": prob_true_after,
        "prob_pred_before": prob_pred_before,
        "prob_pred_after": prob_pred_after,
        "prob_class1_before": prob_class1_before,
        "prob_class1_after": prob_class1_after,
        "success": success,
        "queries": queries,
        "selected_front0_index": selected_idx,
        "l2": l2,
        "linf": linf,
        "modified_pixel_count": int(modified_pixel_count),
        "modified_channel_count": int(modified_channel_count),
        "probs_before": [float(v) for v in prob_before_vec.tolist()],
        "probs_after": [float(v) for v in prob_after_vec.tolist()],
        "raw_npy_path": str(raw_npy_path),
        "images": {
            "before": str(before_path),
            "after": str(after_path),
            "diff_x4": str(diff_path),
            "comparison": str(compare_path),
        },
    }

    json_path = export_dir / "summary.json"
    txt_path = export_dir / "summary.txt"
    detail_log_path = export_dir / "attack_log.txt"

    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    txt_path.write_text(
        "\n".join(
            [
                "SAMOO Attack Summary",
                f"backbone: {backbone}",
                f"checkpoint: {checkpoint_path}",
                f"sample_index_subset: {sample_index_subset}",
                f"sample_index_global: {sample_index_global}",
                f"attack_split: {attack_split}",
                f"true_label: {true_label}",
                f"pred_before: {pred_before}",
                f"pred_after: {pred_after}",
                f"logit_before: {logit_before:.6f}",
                f"logit_after: {logit_after:.6f}",
                f"prob_true_before: {prob_true_before:.6f}",
                f"prob_true_after: {prob_true_after:.6f}",
                f"prob_pred_before: {prob_pred_before:.6f}",
                f"prob_pred_after: {prob_pred_after:.6f}",
                f"prob_class1_before(compat): {prob_class1_before:.6f}",
                f"prob_class1_after(compat): {prob_class1_after:.6f}",
                f"success: {success}",
                f"queries: {queries}",
                f"selected_front0_index: {selected_idx}",
                f"l2: {l2:.6f}",
                f"linf: {linf:.6f}",
                f"modified_pixel_count: {modified_pixel_count}",
                f"modified_channel_count: {modified_channel_count}",
                f"raw_npy_path: {raw_npy_path}",
                f"comparison_image: {compare_path}",
            ]
        ),
        encoding="utf-8",
    )
    detail_log_path.write_text("\n".join(detailed_log_lines) + "\n", encoding="utf-8")


def _load_checkpoint(backbone: str, checkpoint_path: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)

    # 从权重推断 num_classes
    state_dict = ckpt["state_dict"]
    # 找最后一层的权重，推断输出维度
    num_classes = 2  # 默认二分类
    for key in state_dict:
        if "fc.weight" in key or "classifier.weight" in key:
            num_classes = state_dict[key].shape[0]
            break

    # Build model using factory and then load checkpoint weights
    from psorad.config import ModelConfig
    from psorad.models.factory import build_model

    model_cfg = ModelConfig(backbone=backbone, pretrained_path=None, freeze_backbone=False, num_classes=num_classes)
    model = build_model(model_cfg, num_classes=num_classes)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.to(device)
    model.eval()
    return model


def _load_sample_from_manifest(
    manifest_csv: str,
    sample_index: int,
    image_size: int,
) -> tuple[np.ndarray, int, Path, str]:
    manifest = pd.read_csv(manifest_csv)
    if sample_index < 0 or sample_index >= len(manifest):
        raise IndexError(f"sample_index 越界: {sample_index}, 数据总量: {len(manifest)}")

    row = manifest.iloc[sample_index]
    image_path = Path(str(row["file_path"]))
    label = int(row["class_idx"])
    class_name = str(row.get("class_name", str(label)))

    with Image.open(image_path) as img:
        image = center_crop_resize(img, image_size=image_size)
        x = np.asarray(image, dtype=np.float32) / 255.0

    return x, label, image_path, class_name


def _load_sample_for_attack_split(
    manifest_csv: str,
    sample_index: int,
    image_size: int,
    attack_split: str,
    val_ratio: float,
    split_seed: int,
) -> tuple[np.ndarray, int, Path, str, int, int]:
    split_name = attack_split.lower().strip()
    if split_name not in {"all", "train", "val"}:
        raise ValueError("attack_split 仅支持 all/train/val")

    if split_name == "all":
        x, label, image_path, class_name = _load_sample_from_manifest(
            manifest_csv=manifest_csv,
            sample_index=sample_index,
            image_size=image_size,
        )
        total = len(pd.read_csv(manifest_csv))
        return x, label, image_path, class_name, total, sample_index

    manifest = pd.read_csv(manifest_csv)
    if "split" in manifest.columns:
        manifest_with_idx = manifest.reset_index().rename(columns={"index": "global_index"})
        subset = manifest_with_idx[manifest_with_idx["split"].astype(str).str.lower() == split_name].reset_index(drop=True)
    else:
        total = len(manifest)
        val_len = max(int(total * val_ratio), 1)
        train_len = total - val_len
        if train_len <= 0:
            raise ValueError("训练集为空，请增大数据量或减小 val_ratio")

        generator = torch.Generator().manual_seed(split_seed)
        indices = torch.randperm(total, generator=generator).tolist()
        train_indices = indices[:train_len]
        val_indices = indices[train_len:]
        selected_indices = train_indices if split_name == "train" else val_indices

        manifest_with_idx = manifest.reset_index().rename(columns={"index": "global_index"})
        subset = manifest_with_idx.iloc[selected_indices].reset_index(drop=True)

    if subset.empty:
        raise ValueError(f"在 manifest 中未找到 split={split_name} 的样本")

    if sample_index < 0 or sample_index >= len(subset):
        raise IndexError(f"sample_index 越界: {sample_index}, split={split_name} 样本总量: {len(subset)}")

    row = subset.iloc[sample_index]
    global_index = int(row["global_index"])
    image_path = Path(str(row["file_path"]))
    label = int(row["class_idx"])
    class_name = str(row.get("class_name", str(label)))

    with Image.open(image_path) as img:
        image = center_crop_resize(img, image_size=image_size)
        x = np.asarray(image, dtype=np.float32) / 255.0

    return x, label, image_path, class_name, len(subset), global_index


class AttackRunLogger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, message: str) -> None:
        print(message)
        self.lines.append(message)


def _format_topk_probs(probs: np.ndarray, k: int = 5) -> str:
    if probs.size == 0:
        return "[]"
    topk = np.argsort(-probs)[: min(k, probs.size)]
    parts = [f"class_{int(idx)}={float(probs[idx]):.4f}" for idx in topk]
    return ", ".join(parts)


def run_samoo_attack(
    backbone: str,
    checkpoint_path: str,
    datadir: str = "psoriasis_normal",
    manifest_csv: str = "dataset/processed_data/psoriasis_normal/class_manifest.csv",
    sample_index: int = 0,
    attack_split: str = "val",
    val_ratio: float = 0.2,
    split_seed: int = 42,
    image_size: int = 224,
    save_path: str | None = None,
    export_dir: str | None = None,
    keep_raw_npy: bool = True,
    eps: int | None = None,
    iterations: int | None = None,
    pc: float | None = None,
    pm: float | None = None,
    pm_end: float | None = None,
    pop_size: int | None = None,
    query_budget: int | None = None,
    zero_probability: float | None = None,
    include_dist: bool = False,
    max_dist: float = 1e9,
    p_size: float | None = None,
    tournament_size: int | None = None,
    seed: int = 42,
) -> Path:
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_checkpoint(backbone=backbone, checkpoint_path=checkpoint_path, device=device)
    adapter = BinaryModelAdapter(model=model, device=device, backbone=backbone)

    x, y_true, image_path, class_name, split_total, sample_index_global = _load_sample_for_attack_split(
        manifest_csv=manifest_csv,
        sample_index=sample_index,
        image_size=image_size,
        attack_split=attack_split,
        val_ratio=val_ratio,
        split_seed=split_seed,
    )

    eps_user_provided = eps is not None
    iterations_user_provided = iterations is not None
    pop_size_user_provided = pop_size is not None
    query_budget_user_provided = query_budget is not None

    probs_before = adapter.predict_proba(x)
    pred_before = int(np.argmax(probs_before))

    resolved_hparams = _resolve_attack_hparams(
        height=int(x.shape[0]),
        width=int(x.shape[1]),
        eps=eps,
        iterations=iterations,
        pc=pc,
        pm=pm,
        pm_end=pm_end,
        pop_size=pop_size,
        query_budget=query_budget,
        zero_probability=zero_probability,
        p_size=p_size,
        tournament_size=tournament_size,
    )
    resolved_hparams, boost_applied, boost_reason = _apply_confidence_boost(
        hparams=resolved_hparams,
        probs_before=probs_before,
        true_label=y_true,
        image_height=int(x.shape[0]),
        image_width=int(x.shape[1]),
        eps_user_provided=eps_user_provided,
        iterations_user_provided=iterations_user_provided,
        pop_size_user_provided=pop_size_user_provided,
        query_budget_user_provided=query_budget_user_provided,
    )
    resolved_eps = int(resolved_hparams["eps"])
    eps_is_adaptive = bool(resolved_hparams["eps_is_adaptive"])
    loss = UnTargeted(model=adapter, true=y_true, to_pytorch_input=True)

    logger = AttackRunLogger()

    model_name = Path(checkpoint_path).stem
    default_run_dir = Path("output") / "attack" / _safe_path_token(backbone) / f"{_safe_path_token(datadir)}-{_safe_path_token(model_name)}-{sample_index}"

    if save_path is None and export_dir is None:
        save_file = default_run_dir / "samoo_result.npy"
        export_path = default_run_dir
    else:
        save_file = Path(save_path) if save_path is not None else default_run_dir / "samoo_result.npy"
        export_path = Path(export_dir) if export_dir is not None else default_run_dir

    save_file.parent.mkdir(parents=True, exist_ok=True)
    export_path.mkdir(parents=True, exist_ok=True)

    logger.log("=" * 80)
    logger.log("[Attack] 启动 SAMOO 攻击")
    logger.log(f"[Attack] 数据清单: {manifest_csv}")
    logger.log(f"[Attack] 攻击子集: split={attack_split}, split_total={split_total}, val_ratio={val_ratio}, split_seed={split_seed}")
    logger.log(f"[Attack] 数据集目录(datadir): {datadir}")
    logger.log(f"[Attack] 采样索引: subset_index={sample_index}, global_index={sample_index_global}")
    logger.log(f"[Attack] 原图路径: {image_path}")
    logger.log(f"[Attack] 原始标签: class_idx={y_true}, class_name={class_name}")
    logger.log(f"[Attack] 攻击模型: backbone={backbone}, checkpoint={checkpoint_path}")
    logger.log(
        "[Attack] 超参数: "
        f"eps={resolved_eps}{' (adaptive)' if eps_is_adaptive else ''}, "
        f"eps_ratio={resolved_eps / float(x.shape[0] * x.shape[1]):.4%}, "
        f"iterations={resolved_hparams['iterations']}, pop_size={resolved_hparams['pop_size']}, "
        f"pc={resolved_hparams['pc']}, pm={resolved_hparams['pm']}, pm_end={resolved_hparams['pm_end']}, "
        f"query_budget={resolved_hparams['query_budget']}, "
        f"zero_probability={resolved_hparams['zero_probability']}, p_size={resolved_hparams['p_size']}, "
        f"tournament_size={resolved_hparams['tournament_size']}, "
        f"include_dist={include_dist}, max_dist={max_dist}"
    )
    logger.log(f"[Attack] 分辨率预设档位: {resolved_hparams['resolution_profile']}")
    logger.log(f"[Attack] 置信度增强: applied={boost_applied}, reason={boost_reason}")

    params = AttackParams(
        x=x,
        eps=resolved_eps,
        iterations=int(resolved_hparams["iterations"]),
        pc=float(resolved_hparams["pc"]),
        pm=float(resolved_hparams["pm"]),
        pm_end=float(resolved_hparams["pm_end"]) if resolved_hparams["pm_end"] is not None else None,
        pop_size=int(resolved_hparams["pop_size"]),
        query_budget=int(resolved_hparams["query_budget"]) if resolved_hparams["query_budget"] is not None else None,
        zero_probability=float(resolved_hparams["zero_probability"]),
        include_dist=include_dist,
        max_dist=max_dist,
        p_size=float(resolved_hparams["p_size"]),
        tournament_size=int(resolved_hparams["tournament_size"]),
        save_directory=str(save_file),
    )

    def _progress_callback(event: dict[str, Any]) -> None:
        phase = str(event.get("phase", "unknown"))
        if phase == "init_population_start":
            logger.log(f"[Process] 初始化种群: image={event.get('height')}x{event.get('width')}, eps={event.get('eps')}, pop_size={event.get('pop_size')}, zero_prob={event.get('zero_probability')}, p_size={event.get('p_size')}")
        elif phase == "init_population_done":
            logger.log(f"[Process] 初始种群评估完成: population_size={event.get('population_size')}")
        elif phase == "attack_start":
            logger.log(
                "[Process] 进入进化循环: "
                f"iterations={event.get('iterations')}, pc={event.get('pc')}, pm={event.get('pm')}, pm_end={event.get('pm_end')}, "
                f"tournament={event.get('tournament_size')}, include_dist={event.get('include_dist')}, "
                f"max_dist={event.get('max_dist')}, initial_queries={event.get('query_count')}, query_budget={event.get('query_budget')}"
            )
        elif phase == "iteration":
            logger.log(f"[Process] 迭代进度: iter={event.get('iteration')}/{event.get('total_iterations')}, queries={event.get('query_count')}, feasible={event.get('feasible_count')}, best_loss={event.get('best_loss'):.6f}")
        elif phase == "generation_operators":
            logger.log(f"[Process] 进化算子: iter={event.get('iteration')}, parents_pairs={event.get('parents_pairs')}, children={event.get('children')}, pm_current={event.get('pm_current')}, post_queries={event.get('post_query_count')}")
        elif phase == "query_budget_reached":
            logger.log(f"[Process] 达到查询预算，提前停止: iter={event.get('iteration')}, queries={event.get('query_count')}, budget={event.get('query_budget')}")
        elif phase == "early_success":
            logger.log(f"[Process] 提前命中可行对抗解: iter={event.get('iteration')}, feasible={event.get('feasible_count')}, queries={event.get('query_count')}")
        elif phase == "attack_end":
            logger.log(f"[Process] 进化结束: success={event.get('success')}, queries={event.get('query_count')}, query_budget={event.get('query_budget')}, best_loss={event.get('best_loss'):.6f}")

    logger.log(f"[Model] 原图预测: pred_before=class_{pred_before}, conf={float(probs_before[pred_before]):.6f}, true_class_conf={float(probs_before[y_true]):.6f}")
    logger.log(f"[Model] 原图 Top-K 置信度: {_format_topk_probs(probs_before, k=5)}")

    attacker = Attack(params, progress_callback=_progress_callback)
    attacker.attack(loss)

    payload = np.load(save_file, allow_pickle=True).item()
    selected_idx, selected_adv, _ = _pick_best_candidate(payload, y_true)
    pred_after = int(loss.get_label(selected_adv))
    logit_before = adapter.predict_binary_logit(x)
    logit_after = adapter.predict_binary_logit(selected_adv)
    probs_after = adapter.predict_proba(selected_adv)

    perturb = selected_adv - x
    modified_pixel_count = int(np.sum(np.any(np.abs(perturb) > 1e-8, axis=2)))
    modified_channel_count = int(np.sum(np.abs(perturb) > 1e-8))
    queries = int(payload.get("queries", -1))
    success = bool(pred_after != y_true)

    logger.log("[Result] 攻击完成")
    logger.log(f"[Result] 结果: success={success}, queries={queries}, selected_front0_index={selected_idx}, modified_pixels={modified_pixel_count}, modified_channels={modified_channel_count}")
    logger.log(f"[Model] 对抗图预测: pred_after=class_{pred_after}, conf={float(probs_after[pred_after]):.6f}, true_class_conf={float(probs_after[y_true]):.6f}")
    logger.log(f"[Model] 对抗图 Top-K 置信度: {_format_topk_probs(probs_after, k=5)}")
    logger.log(f"[Model] 二分类读数(兼容字段): logit_before={logit_before:.6f}, logit_after={logit_after:.6f}")
    logger.log(f"[Output] 结果目录: {export_path}")
    logger.log("=" * 80)

    _export_attack_artifacts(
        export_dir=export_path,
        backbone=backbone,
        checkpoint_path=checkpoint_path,
        sample_index_subset=sample_index,
        sample_index_global=sample_index_global,
        attack_split=attack_split,
        raw_npy_path=save_file,
        payload=payload,
        original_img=x,
        true_label=y_true,
        pred_before=pred_before,
        pred_after=pred_after,
        logit_before=logit_before,
        logit_after=logit_after,
        selected_idx=selected_idx,
        selected_adv=selected_adv,
        modified_pixel_count=modified_pixel_count,
        modified_channel_count=modified_channel_count,
        prob_before_vec=probs_before,
        prob_after_vec=probs_after,
        detailed_log_lines=logger.lines,
    )

    if not keep_raw_npy and save_file.exists():
        save_file.unlink()

    return export_path
