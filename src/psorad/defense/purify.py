from __future__ import annotations

import numpy as np


def local_median(img: np.ndarray, window: int = 3) -> np.ndarray:
    # 逐通道局部中值滤波
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    h, w = img.shape[:2]
    result = np.empty_like(img)
    for c in range(3):
        channel = padded[:, :, c]
        views = np.lib.stride_tricks.sliding_window_view(channel, (window, window))
        result[:, :, c] = np.median(views, axis=(-2, -1))
    return result


def region_aware_mask(
    img: np.ndarray,
    *,
    window: int = 3,
    threshold: float = 0.08,
    dilation: int = 1,
) -> np.ndarray:
    # 二值掩膜：偏离局部中值的像素
    median = local_median(img, window=window)
    diff = np.max(np.abs(img - median), axis=2)  # 通道间最大偏差
    mask: np.ndarray = diff > threshold

    if dilation > 0 and mask.any():
        kernel = 2 * dilation + 1
        padded = np.pad(mask.astype(np.float32), dilation, mode="edge")
        dilated = np.lib.stride_tricks.sliding_window_view(padded, (kernel, kernel)).max(axis=(-2, -1)) > 0
        mask = dilated

    return mask


def fft_lowpass(img: np.ndarray, cutoff: float = 0.25) -> np.ndarray:
    # FFT低通滤波
    h, w = img.shape[:2]
    result = np.empty_like(img)
    cy, cx = h // 2, w // 2
    max_r = min(cy, cx)
    r_cut = int(max_r * cutoff)

    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    low_mask = r <= r_cut

    for c in range(3):
        fft = np.fft.fft2(img[:, :, c].astype(np.float64))
        fft_shift = np.fft.fftshift(fft)
        fft_shift[~low_mask] = 0.0
        recon = np.fft.ifft2(np.fft.ifftshift(fft_shift))
        result[:, :, c] = np.clip(np.real(recon), 0.0, 1.0).astype(np.float32)

    return result


def purify(
    img: np.ndarray,
    *,
    window: int = 3,
    threshold: float = 0.08,
    dilation: int = 1,
    low_freq_cutoff: float = 0.25,
    low_freq: str = "fft",
) -> np.ndarray:
    # 区域感知输入净化
    mask = region_aware_mask(img, window=window, threshold=threshold, dilation=dilation)

    if not mask.any():
        return img.copy()

    if low_freq == "fft":
        estimate = fft_lowpass(img, cutoff=low_freq_cutoff)
    elif low_freq == "median":
        estimate = local_median(img, window=window)
    else:
        raise ValueError(f"unsupported low_freq method: {low_freq}")

    out = img.copy()
    for c in range(3):
        channel_out = out[:, :, c]
        channel_out[mask] = estimate[:, :, c][mask]
    return out
