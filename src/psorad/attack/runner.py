from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch import nn

from psorad.attack.losses import UnTargeted
from psorad.attack.samoo_core.attack import Attack, AttackParams
from psorad.models.classifier import build_resnet50_classifier
from psorad.utils.image import center_crop_resize


def _safe_path_token(value: str) -> str:
    token = value.strip().replace(" ", "_")
    token = token.replace("/", "_").replace("\\", "_")
    return token or "unknown"


def _resolve_eps(eps: int | None, height: int, width: int) -> tuple[int, bool]:
    total_pixels = int(height * width)
    if eps is None:
        adaptive = max(128, int(total_pixels * 0.02))
        return min(adaptive, total_pixels), True
    return min(int(eps), total_pixels), False


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
    return (clipped * 255.0).round().astype(np.uint8)


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
    sample_index: int,
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
        "sample_index": sample_index,
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
                f"sample_index: {sample_index}",
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

    if backbone == "resnet50":
        model = build_resnet50_classifier(num_classes=num_classes)
    elif backbone == "siglip":
        from psorad.models.classifier import SiglipClassifier

        model = SiglipClassifier(pretrained_dir_or_id="model/pretrained_model/siglip", num_classes=num_classes, freeze_backbone=False)
    else:
        raise ValueError("backbone 仅支持 resnet50 或 siglip")

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
    image_size: int = 224,
    save_path: str | None = None,
    export_dir: str | None = None,
    keep_raw_npy: bool = True,
    eps: int | None = None,
    iterations: int = 400,
    pc: float = 0.3,
    pm: float = 0.6,
    pop_size: int = 12,
    zero_probability: float = 0.2,
    include_dist: bool = False,
    max_dist: float = 1e9,
    p_size: float = 0.25,
    tournament_size: int = 2,
    seed: int = 42,
) -> Path:
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_checkpoint(backbone=backbone, checkpoint_path=checkpoint_path, device=device)
    adapter = BinaryModelAdapter(model=model, device=device, backbone=backbone)

    x, y_true, image_path, class_name = _load_sample_from_manifest(
        manifest_csv=manifest_csv,
        sample_index=sample_index,
        image_size=image_size,
    )
    resolved_eps, eps_is_adaptive = _resolve_eps(eps=eps, height=int(x.shape[0]), width=int(x.shape[1]))
    loss = UnTargeted(model=adapter, true=y_true, to_pytorch_input=True)

    logger = AttackRunLogger()

    model_name = Path(checkpoint_path).stem
    default_run_dir = (
        Path("output")
        / "attack"
        / _safe_path_token(backbone)
        / f"{_safe_path_token(datadir)}-{_safe_path_token(model_name)}-{sample_index}"
    )

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
    logger.log(f"[Attack] 数据集目录(datadir): {datadir}")
    logger.log(f"[Attack] 采样索引: {sample_index}")
    logger.log(f"[Attack] 原图路径: {image_path}")
    logger.log(f"[Attack] 原始标签: class_idx={y_true}, class_name={class_name}")
    logger.log(f"[Attack] 攻击模型: backbone={backbone}, checkpoint={checkpoint_path}")
    logger.log(
        "[Attack] 超参数: "
        f"eps={resolved_eps}{' (adaptive)' if eps_is_adaptive else ''}, "
        f"eps_ratio={resolved_eps / float(x.shape[0] * x.shape[1]):.4%}, "
        f"iterations={iterations}, pop_size={pop_size}, pc={pc}, pm={pm}, "
        f"zero_probability={zero_probability}, p_size={p_size}, tournament_size={tournament_size}, "
        f"include_dist={include_dist}, max_dist={max_dist}"
    )

    params = AttackParams(
        x=x,
        eps=resolved_eps,
        iterations=iterations,
        pc=pc,
        pm=pm,
        pop_size=pop_size,
        zero_probability=zero_probability,
        include_dist=include_dist,
        max_dist=max_dist,
        p_size=p_size,
        tournament_size=tournament_size,
        save_directory=str(save_file),
    )

    def _progress_callback(event: dict[str, Any]) -> None:
        phase = str(event.get("phase", "unknown"))
        if phase == "init_population_start":
            logger.log(
                "[Process] 初始化种群: "
                f"image={event.get('height')}x{event.get('width')}, eps={event.get('eps')}, "
                f"pop_size={event.get('pop_size')}, zero_prob={event.get('zero_probability')}, p_size={event.get('p_size')}"
            )
        elif phase == "init_population_done":
            logger.log(f"[Process] 初始种群评估完成: population_size={event.get('population_size')}")
        elif phase == "attack_start":
            logger.log(
                "[Process] 进入进化循环: "
                f"iterations={event.get('iterations')}, pc={event.get('pc')}, pm={event.get('pm')}, "
                f"tournament={event.get('tournament_size')}, include_dist={event.get('include_dist')}, "
                f"max_dist={event.get('max_dist')}, initial_queries={event.get('query_count')}"
            )
        elif phase == "iteration":
            logger.log(
                "[Process] 迭代进度: "
                f"iter={event.get('iteration')}/{event.get('total_iterations')}, "
                f"queries={event.get('query_count')}, feasible={event.get('feasible_count')}, "
                f"best_loss={event.get('best_loss'):.6f}"
            )
        elif phase == "generation_operators":
            logger.log(
                "[Process] 进化算子: "
                f"iter={event.get('iteration')}, parents_pairs={event.get('parents_pairs')}, "
                f"children={event.get('children')}, post_queries={event.get('post_query_count')}"
            )
        elif phase == "early_success":
            logger.log(
                "[Process] 提前命中可行对抗解: "
                f"iter={event.get('iteration')}, feasible={event.get('feasible_count')}, queries={event.get('query_count')}"
            )
        elif phase == "attack_end":
            logger.log(
                "[Process] 进化结束: "
                f"success={event.get('success')}, queries={event.get('query_count')}, best_loss={event.get('best_loss'):.6f}"
            )

    probs_before = adapter.predict_proba(x)
    pred_before = int(np.argmax(probs_before))
    logger.log(
        "[Model] 原图预测: "
        f"pred_before=class_{pred_before}, conf={float(probs_before[pred_before]):.6f}, "
        f"true_class_conf={float(probs_before[y_true]):.6f}"
    )
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
    logger.log(
        f"[Result] 结果: success={success}, queries={queries}, selected_front0_index={selected_idx}, "
        f"modified_pixels={modified_pixel_count}, modified_channels={modified_channel_count}"
    )
    logger.log(
        "[Model] 对抗图预测: "
        f"pred_after=class_{pred_after}, conf={float(probs_after[pred_after]):.6f}, "
        f"true_class_conf={float(probs_after[y_true]):.6f}"
    )
    logger.log(f"[Model] 对抗图 Top-K 置信度: {_format_topk_probs(probs_after, k=5)}")
    logger.log(
        f"[Model] 二分类读数(兼容字段): logit_before={logit_before:.6f}, logit_after={logit_after:.6f}"
    )
    logger.log(f"[Output] 结果目录: {export_path}")
    logger.log("=" * 80)

    _export_attack_artifacts(
        export_dir=export_path,
        backbone=backbone,
        checkpoint_path=checkpoint_path,
        sample_index=sample_index,
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
