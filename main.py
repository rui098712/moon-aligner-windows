#!/usr/bin/env python3
"""批量配准、统一裁切并堆栈月亮照片。

程序不会修改原图。它会：
1. 自动检测每张照片中的月面；
2. 选择较清晰的照片作为参考帧；
3. 使用 ECC 纠正位移与小幅旋转；
4. 排除配准失败或月面缺边的照片；
5. 输出尺寸完全一致的 16 位/8 位 TIFF；
6. 可将配准后的照片进行平均或中位数堆栈。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
APP_VERSION = "1.5.1"
STACK_LABELS = {
    "不堆栈（只配准和裁切）": "none",
    "平均堆栈（推荐，降噪更好）": "mean",
    "中位数堆栈（排除偶发干扰）": "median",
    "多点局部堆栈（实验，改善大气扰动）": "local",
}
STACK_LABELS_REVERSE = {value: key for key, value in STACK_LABELS.items()}


def resource_path(relative: str) -> Path:
    """返回源码运行或PyInstaller打包后的资源文件路径。"""
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return bundle_root / relative


@dataclass
class ScanResult:
    path: Path
    width: int
    height: int
    center: tuple[float, float] | None
    radius: float | None
    sharpness: float
    scale: float
    error: str = ""


@dataclass
class AlignmentResult:
    scan: ScanResult
    warp: np.ndarray | None
    correlation: float | None
    status: str
    message: str = ""


def list_images(folder: Path) -> list[Path]:
    """按文件名自然顺序返回支持的图像。"""
    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES),
        key=lambda p: p.name.lower(),
    )


def read_image(path: Path) -> np.ndarray:
    """读取图像；使用 imdecode 以兼容包含中文的路径。"""
    raw = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("OpenCV 无法读取该文件")
    return image


def write_tiff(path: Path, image: np.ndarray) -> None:
    """写入 TIFF；使用 imencode 以兼容包含中文的路径。"""
    ok, encoded = cv2.imencode(".tif", image, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
    if not ok:
        raise ValueError("OpenCV 无法编码 TIFF")
    encoded.tofile(path)


def write_png(path: Path, image: np.ndarray) -> None:
    """写入 PNG；使用 imencode 兼容包含中文的路径。"""
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise ValueError("OpenCV 无法编码 PNG")
    encoded.tofile(path)


def to_gray_float(image: np.ndarray) -> np.ndarray:
    """把任意常见位深的图像转换为 0~1 的单通道 float32。"""
    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = gray.astype(np.float32)
    finite = gray[np.isfinite(gray)]
    if finite.size == 0:
        raise ValueError("图像中没有有效像素")

    low = float(np.percentile(finite, 0.2))
    high = float(np.percentile(finite, 99.8))
    if high <= low:
        high = float(finite.max())
        low = float(finite.min())
    if high <= low:
        raise ValueError("图像没有可用于配准的明暗变化")

    return np.clip((gray - low) / (high - low), 0.0, 1.0)


def resize_for_registration(gray: np.ndarray, max_size: int) -> tuple[np.ndarray, float]:
    height, width = gray.shape
    scale = min(1.0, max_size / max(height, width))
    if scale == 1.0:
        return gray, scale
    resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return resized, scale


def detect_moon(gray: np.ndarray) -> tuple[tuple[float, float], float, np.ndarray]:
    """检测最可能的月面轮廓，返回圆心、半径和月面掩膜。"""
    image_u8 = np.round(gray * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(image_u8, (7, 7), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_size = max(3, int(round(min(gray.shape) / 300)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = gray.shape[0] * gray.shape[1]
    candidates: list[tuple[float, np.ndarray, tuple[float, float], float]] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < image_area * 0.0002 or area > image_area * 0.85:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if radius < 4:
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = 4.0 * math.pi * area / (perimeter * perimeter) if perimeter else 0.0
        fill_ratio = area / (math.pi * radius * radius)
        # 面积优先，同时允许月牙等填充率较低的目标。
        score = area * (0.35 + min(1.0, circularity)) * (0.35 + min(1.0, fill_ratio))
        candidates.append((score, contour, (float(cx), float(cy)), float(radius)))

    if not candidates:
        raise ValueError("未检测到月面；请确认月亮明显亮于背景且没有严重过曝")

    _, contour, center, radius = max(candidates, key=lambda item: item[0])
    mask = np.zeros_like(image_u8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    return center, radius, mask


def measure_sharpness(gray: np.ndarray, center: tuple[float, float], radius: float) -> float:
    """用月面内部拉普拉斯方差估计清晰度。"""
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (round(center[0]), round(center[1])), max(2, round(radius * 0.9)), 255, -1)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    pixels = laplacian[mask > 0]
    return float(np.var(pixels)) if pixels.size else 0.0


def scan_image(path: Path, max_size: int) -> ScanResult:
    try:
        image = read_image(path)
        height, width = image.shape[:2]
        gray, scale = resize_for_registration(to_gray_float(image), max_size)
        center, radius, _ = detect_moon(gray)
        sharpness = measure_sharpness(gray, center, radius)
        return ScanResult(path, width, height, center, radius, sharpness, scale)
    except Exception as exc:  # 单张坏图不应终止整个批处理
        return ScanResult(path, 0, 0, None, None, 0.0, 1.0, str(exc))


def registration_image(path: Path, max_size: int) -> tuple[np.ndarray, float]:
    gray, scale = resize_for_registration(to_gray_float(read_image(path)), max_size)
    # 轻微模糊可以降低噪点对 ECC 的干扰，同时保留环形山结构。
    return cv2.GaussianBlur(gray, (5, 5), 0), scale


def align_one(
    scan: ScanResult,
    reference_gray: np.ndarray,
    reference_scan: ScanResult,
    max_size: int,
    min_correlation: float,
) -> AlignmentResult:
    if scan.path == reference_scan.path:
        return AlignmentResult(scan, np.eye(2, 3, dtype=np.float32), 1.0, "accepted", "参考帧")
    if scan.error or scan.center is None or scan.radius is None:
        return AlignmentResult(scan, None, None, "skipped", scan.error or "未检测到月面")
    if (scan.width, scan.height) != (reference_scan.width, reference_scan.height):
        return AlignmentResult(scan, None, None, "skipped", "图像尺寸与参考帧不同")

    try:
        moving_gray, scale = registration_image(scan.path, max_size)
        if moving_gray.shape != reference_gray.shape:
            return AlignmentResult(scan, None, None, "skipped", "缩略图尺寸与参考帧不同")

        assert reference_scan.center is not None
        initial = np.array(
            [
                [1.0, 0.0, scan.center[0] - reference_scan.center[0]],
                [0.0, 1.0, scan.center[1] - reference_scan.center[1]],
            ],
            dtype=np.float32,
        )

        mask = np.zeros(reference_gray.shape, dtype=np.uint8)
        assert reference_scan.radius is not None
        cv2.circle(
            mask,
            (round(reference_scan.center[0]), round(reference_scan.center[1])),
            max(8, round(reference_scan.radius * 1.12)),
            255,
            -1,
        )
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 180, 1e-6)
        correlation, warp = cv2.findTransformECC(
            reference_gray,
            moving_gray,
            initial,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            mask,
            5,
        )
        correlation = float(correlation)
        if correlation < min_correlation:
            return AlignmentResult(
                scan,
                None,
                correlation,
                "skipped",
                f"配准相关度过低（{correlation:.3f}）",
            )

        # ECC 在缩略图坐标中工作；旋转不变，位移换算回原图坐标。
        warp_full = warp.astype(np.float32).copy()
        warp_full[:, 2] /= scale
        return AlignmentResult(scan, warp_full, correlation, "accepted")
    except cv2.error as exc:
        message = str(exc).splitlines()[0] if str(exc) else "ECC 配准失败"
        return AlignmentResult(scan, None, None, "skipped", message)
    except Exception as exc:
        return AlignmentResult(scan, None, None, "skipped", str(exc))


def warp_valid_mask(width: int, height: int, warp: np.ndarray) -> np.ndarray:
    source = np.full((height, width), 255, dtype=np.uint8)
    return cv2.warpAffine(
        source,
        warp,
        (width, height),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def square_bounds(
    center: tuple[float, float], half_size: int, width: int, height: int
) -> tuple[int, int, int, int]:
    cx, cy = center
    x1 = max(0, int(math.floor(cx - half_size)))
    y1 = max(0, int(math.floor(cy - half_size)))
    x2 = min(width, int(math.ceil(cx + half_size)))
    y2 = min(height, int(math.ceil(cy + half_size)))
    return x1, y1, x2, y2


def region_is_valid(mask: np.ndarray, bounds: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = bounds
    region = mask[y1:y2, x1:x2]
    return region.size > 0 and bool(np.all(region == 255))


def find_safe_crop(
    common_mask: np.ndarray,
    center: tuple[float, float],
    radius: float,
    padding: float,
) -> tuple[int, int, int, int]:
    height, width = common_mask.shape
    minimum_half = max(2, math.ceil(radius * 1.02))
    desired_half = max(minimum_half, math.ceil(radius * (1.0 + padding)))

    minimum = square_bounds(center, minimum_half, width, height)
    if not region_is_valid(common_mask, minimum):
        raise RuntimeError("共同有效区域无法完整容纳月面；请移除月亮出框的照片")

    low, high = minimum_half, desired_half
    while low < high:
        middle = (low + high + 1) // 2
        if region_is_valid(common_mask, square_bounds(center, middle, width, height)):
            low = middle
        else:
            high = middle - 1
    return square_bounds(center, low, width, height)


def save_report(
    output: Path,
    results: Iterable[AlignmentResult],
    stacked_paths: set[Path] | None = None,
) -> None:
    stacked_paths = stacked_paths or set()
    with (output / "alignment_report.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "文件",
                "状态",
                "用于堆栈",
                "相关度",
                "清晰度",
                "水平位移(px)",
                "垂直位移(px)",
                "旋转(度)",
                "说明",
            ]
        )
        for result in results:
            warp = result.warp
            angle = math.degrees(math.atan2(float(warp[1, 0]), float(warp[0, 0]))) if warp is not None else ""
            writer.writerow(
                [
                    result.scan.path.name,
                    result.status,
                    "是" if result.scan.path in stacked_paths else "否",
                    "" if result.correlation is None else f"{result.correlation:.6f}",
                    f"{result.scan.sharpness:.8f}",
                    "" if warp is None else f"{warp[0, 2]:.3f}",
                    "" if warp is None else f"{warp[1, 2]:.3f}",
                    "" if warp is None else f"{angle:.4f}",
                    result.message,
                ]
            )


def save_processing_report(
    output: Path,
    input_folder: Path,
    reference: ScanResult,
    total_frames: int,
    accepted_frames: int,
    crop: tuple[int, int, int, int],
    stack_method: str,
    stack_percent: int,
    ap_size: int,
    save_aligned: bool,
) -> None:
    x1, y1, x2, y2 = crop
    report = {
        "application": "Moon Aligner",
        "version": APP_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_folder": str(input_folder),
        "reference_frame": reference.path.name,
        "frames": {
            "total": total_frames,
            "accepted": accepted_frames,
            "skipped": total_frames - accepted_frames,
        },
        "crop": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
        "stack": {
            "method": stack_method,
            "selected_percent": stack_percent,
            "alignment_point_size": ap_size,
            "saved_aligned_frames": save_aligned,
        },
    }
    with (output / "processing_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def cast_stack_result(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """把浮点堆栈结果安全地转换回原照片的位深。"""
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        return np.clip(np.rint(values), limits.min, limits.max).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        return values.astype(dtype)
    raise ValueError(f"不支持堆栈这种像素类型：{dtype}")


def gray_unit(image: np.ndarray) -> np.ndarray:
    """转换为用于局部匹配的 float32 灰度图，并保留帧间亮度关系。"""
    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)
    if np.issubdtype(image.dtype, np.integer):
        maximum = float(np.iinfo(image.dtype).max)
        if maximum > 0:
            gray /= maximum
    return gray


def quadratic_peak_offset(left: float, center: float, right: float) -> float:
    """用相关峰相邻三个采样点估计亚像素偏移。"""
    denominator = left - 2.0 * center + right
    if abs(denominator) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denominator, -0.75, 0.75))


def local_quality(gray: np.ndarray) -> float:
    """用 Tenengrad 梯度能量评估一个局部块的清晰度。"""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    energy = float(np.mean(gx * gx + gy * gy))
    brightness = float(np.mean(np.abs(gray)))
    return energy / (brightness * brightness + 1e-8)


def mean_reference(frames: np.ndarray, working_memory_mb: int) -> np.ndarray:
    """分块计算稳定的平均参考帧。"""
    frame_count, height, width = frames.shape[:3]
    channels = 1 if frames.ndim == 3 else frames.shape[3]
    bytes_per_row = max(1, frame_count * width * channels * frames.dtype.itemsize)
    chunk_rows = max(1, min(height, working_memory_mb * 1024 * 1024 // bytes_per_row))
    reference = np.empty(frames.shape[1:], dtype=np.float32)
    for y1 in range(0, height, chunk_rows):
        y2 = min(height, y1 + chunk_rows)
        reference[y1:y2] = np.mean(frames[:, y1:y2], axis=0, dtype=np.float64)
    return reference


def alignment_points(
    reference_gray: np.ndarray,
    ap_size: int,
    search_radius: int,
    maximum_points: int = 500,
) -> tuple[list[tuple[int, int]], int]:
    """在有纹理的月面区域自动生成重叠的局部对齐点。"""
    center, radius, _ = detect_moon(reference_gray)
    height, width = reference_gray.shape
    if ap_size <= 0:
        # 月面直径约分成 10~12 块，同时限制极端大小。
        ap_size = int(np.clip(round(radius / 5.5), 48, 256))
    ap_size = max(32, min(ap_size, min(height, width) - 2 * search_radius - 2))
    if ap_size % 2:
        ap_size += 1

    half = ap_size // 2
    margin = half + search_radius + 1
    base_step = max(16, ap_size // 2)
    moon_area = math.pi * (radius * 0.92) ** 2
    point_limited_step = math.ceil(math.sqrt(moon_area / maximum_points))
    step = max(base_step, point_limited_step)

    candidates: list[tuple[int, int, float]] = []
    start_x = max(margin, round(center[0] - radius * 0.9))
    end_x = min(width - margin, round(center[0] + radius * 0.9))
    start_y = max(margin, round(center[1] - radius * 0.9))
    end_y = min(height - margin, round(center[1] + radius * 0.9))
    for cy in range(start_y, end_y + 1, step):
        for cx in range(start_x, end_x + 1, step):
            if math.hypot(cx - center[0], cy - center[1]) > radius * 0.9:
                continue
            patch = reference_gray[cy - half : cy + half, cx - half : cx + half]
            if patch.shape != (ap_size, ap_size):
                continue
            candidates.append((cx, cy, local_quality(patch)))

    if not candidates:
        raise RuntimeError("月面太小，无法建立局部对齐点；请使用普通平均堆栈")
    qualities = np.array([item[2] for item in candidates], dtype=np.float64)
    threshold = float(np.percentile(qualities, 12)) if len(candidates) >= 8 else -math.inf
    points = [(cx, cy) for cx, cy, quality in candidates if quality >= threshold]
    if len(points) < 4:
        points = [(cx, cy) for cx, cy, _ in candidates]
    return points, ap_size


def match_local_patch(
    frame: np.ndarray,
    reference_patch: np.ndarray,
    cx: int,
    cy: int,
    ap_size: int,
    search_radius: int,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """模板匹配一个对齐点，返回亚像素对齐后的彩色块、灰度块和相关度。"""
    half = ap_size // 2
    x1 = cx - half
    y1 = cy - half
    search = frame[
        y1 - search_radius : y1 + ap_size + search_radius,
        x1 - search_radius : x1 + ap_size + search_radius,
    ]
    expected = ap_size + 2 * search_radius
    if search.shape[0] != expected or search.shape[1] != expected:
        return None

    search_float = search.astype(np.float32)
    search_gray = gray_unit(search)
    response = cv2.matchTemplate(search_gray, reference_patch, cv2.TM_CCOEFF_NORMED)
    _, peak, _, location = cv2.minMaxLoc(response)
    px, py = location
    sub_x = 0.0
    sub_y = 0.0
    if 0 < px < response.shape[1] - 1:
        sub_x = quadratic_peak_offset(response[py, px - 1], response[py, px], response[py, px + 1])
    if 0 < py < response.shape[0] - 1:
        sub_y = quadratic_peak_offset(response[py - 1, px], response[py, px], response[py + 1, px])

    offset_x = (px + sub_x) - search_radius
    offset_y = (py + sub_y) - search_radius
    sample_center = (
        search_radius + (ap_size - 1) / 2.0 + offset_x,
        search_radius + (ap_size - 1) / 2.0 + offset_y,
    )
    aligned = cv2.getRectSubPix(search_float, (ap_size, ap_size), sample_center)
    aligned_gray = cv2.getRectSubPix(search_gray, (ap_size, ap_size), sample_center)
    return aligned, aligned_gray, float(peak)


def save_alignment_points_preview(
    reference: np.ndarray,
    points: list[tuple[int, int]],
    ap_size: int,
    target: Path,
    maximum_size: int = 1800,
) -> None:
    """保存带对齐点网格的预览图，便于检查自动布局是否合理。"""
    finite = reference[np.isfinite(reference)]
    if finite.size == 0:
        return
    low = float(np.percentile(finite, 0.5))
    high = float(np.percentile(finite, 99.7))
    if high <= low:
        return
    preview = np.clip((reference.astype(np.float32) - low) / (high - low), 0.0, 1.0)
    preview = np.round(preview * 255).astype(np.uint8)
    if preview.ndim == 2:
        preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
    elif preview.shape[2] == 4:
        preview = cv2.cvtColor(preview, cv2.COLOR_BGRA2BGR)

    height, width = preview.shape[:2]
    scale = min(1.0, maximum_size / max(height, width))
    if scale < 1.0:
        preview = cv2.resize(preview, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    half = ap_size / 2.0
    line_width = max(1, round(1.5 * scale))
    marker_radius = max(2, round(3 * scale))
    for cx, cy in points:
        top_left = (round((cx - half) * scale), round((cy - half) * scale))
        bottom_right = (round((cx + half) * scale), round((cy + half) * scale))
        center = (round(cx * scale), round(cy * scale))
        cv2.rectangle(preview, top_left, bottom_right, (60, 220, 60), line_width)
        cv2.circle(preview, center, marker_radius, (30, 60, 255), -1)
    write_png(target, preview)


def local_stack_frames(
    frames: np.ndarray,
    target: Path,
    stack_percent: int,
    ap_size: int,
    log: Callable[[str], None],
    working_memory_mb: int,
    temp_folder: Path,
) -> None:
    """多点局部对齐、局部择优并以重叠窗口融合为一张图像。"""
    frame_count, height, width = frames.shape[:3]
    reference = mean_reference(frames, working_memory_mb)
    reference_gray = gray_unit(cast_stack_result(reference, frames.dtype))
    search_radius = max(4, min(20, (ap_size if ap_size > 0 else 96) // 6))
    points, actual_ap_size = alignment_points(reference_gray, ap_size, search_radius)
    if ap_size <= 0:
        search_radius = max(4, min(20, actual_ap_size // 6))
        points, actual_ap_size = alignment_points(reference_gray, actual_ap_size, search_radius)
    log(
        f"多点局部堆栈：{len(points)} 个对齐点，块大小 {actual_ap_size} 像素，"
        f"每个点选择最清晰的 {stack_percent}% 帧"
    )
    preview_target = target.with_name("local_alignment_points.png")
    save_alignment_points_preview(reference, points, actual_ap_size, preview_target)
    log(f"对齐点预览：{preview_target.name}")

    accumulator = np.memmap(
        temp_folder / "local_accumulator.dat",
        mode="w+",
        dtype=np.float32,
        shape=frames.shape[1:],
    )
    weight_sum = np.memmap(
        temp_folder / "local_weights.dat",
        mode="w+",
        dtype=np.float32,
        shape=(height, width),
    )
    accumulator[:] = 0
    weight_sum[:] = 0
    window = cv2.createHanningWindow((actual_ap_size, actual_ap_size), cv2.CV_32F)
    # 避免窗口边缘严格为零导致细小接缝。
    window = np.maximum(window, 0.02)
    half = actual_ap_size // 2
    keep_count = max(2, min(frame_count, math.ceil(frame_count * stack_percent / 100)))

    for point_index, (cx, cy) in enumerate(points, 1):
        reference_patch = reference_gray[cy - half : cy + half, cx - half : cx + half]
        candidates: list[tuple[float, float, np.ndarray]] = []
        for frame_index in range(frame_count):
            matched = match_local_patch(
                frames[frame_index],
                reference_patch,
                cx,
                cy,
                actual_ap_size,
                search_radius,
            )
            if matched is None:
                continue
            patch, patch_gray, correlation = matched
            if correlation < 0.12:
                continue
            candidates.append((local_quality(patch_gray), correlation, patch))

        if len(candidates) < 2:
            continue
        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = candidates[: min(keep_count, len(candidates))]
        median_quality = float(np.median([item[0] for item in selected])) + 1e-12
        y1, y2 = cy - half, cy + half
        x1, x2 = cx - half, cx + half
        for quality, correlation, patch in selected:
            quality_weight = float(np.clip(math.sqrt(quality / median_quality), 0.5, 2.0))
            match_weight = float(np.clip((correlation - 0.1) / 0.9, 0.25, 1.0))
            weights = window * quality_weight * match_weight
            if patch.ndim == 3:
                accumulator[y1:y2, x1:x2] += patch * weights[:, :, None]
            else:
                accumulator[y1:y2, x1:x2] += patch * weights
            weight_sum[y1:y2, x1:x2] += weights

        if point_index == 1 or point_index % 25 == 0 or point_index == len(points):
            log(f"  局部对齐点 {point_index}/{len(points)}")

    result = np.empty(frames.shape[1:], dtype=frames.dtype)
    rows_per_chunk = max(1, min(height, 256))
    for y1 in range(0, height, rows_per_chunk):
        y2 = min(height, y1 + rows_per_chunk)
        weights = weight_sum[y1:y2]
        if reference.ndim == 3:
            locally_aligned = accumulator[y1:y2] / np.maximum(weights[:, :, None], 1e-8)
            values = np.where(weights[:, :, None] > 1e-6, locally_aligned, reference[y1:y2])
        else:
            locally_aligned = accumulator[y1:y2] / np.maximum(weights, 1e-8)
            values = np.where(weights > 1e-6, locally_aligned, reference[y1:y2])
        result[y1:y2] = cast_stack_result(values, frames.dtype)

    write_tiff(target, result)
    log(f"多点局部堆栈结果：{target.name}")


def stack_frames(
    stack_file: Path,
    target: Path,
    method: str,
    stack_percent: int,
    ap_size: int,
    log: Callable[[str], None],
    working_memory_mb: int = 192,
) -> None:
    """分块堆栈磁盘映射中的照片，避免一次把全部原图装入内存。"""
    frames = np.load(stack_file, mmap_mode="r")
    if frames.ndim not in (3, 4) or frames.shape[0] < 2:
        raise ValueError("堆栈至少需要两张尺寸和通道数相同的照片")

    frame_count, height, width = frames.shape[:3]
    if method == "local":
        local_stack_frames(
            frames,
            target,
            stack_percent,
            ap_size,
            log,
            working_memory_mb,
            stack_file.parent,
        )
        return

    channels = 1 if frames.ndim == 3 else frames.shape[3]
    bytes_per_row = max(1, frame_count * width * channels * frames.dtype.itemsize)
    # 中位数需要同时查看同一区块的所有帧；把临时工作内存限制在约 192 MB。
    chunk_rows = max(1, min(height, working_memory_mb * 1024 * 1024 // bytes_per_row))
    result = np.empty(frames.shape[1:], dtype=frames.dtype)

    method_name = "平均" if method == "mean" else "中位数"
    log(f"正在进行{method_name}堆栈（{frame_count} 张，分块高度 {chunk_rows} 像素）……")
    for y1 in range(0, height, chunk_rows):
        y2 = min(height, y1 + chunk_rows)
        block = frames[:, y1:y2]
        if method == "mean":
            values = np.mean(block, axis=0, dtype=np.float64)
        elif method == "median":
            values = np.median(block, axis=0)
        else:
            raise ValueError(f"未知的堆栈方式：{method}")
        result[y1:y2] = cast_stack_result(values, frames.dtype)

    write_tiff(target, result)
    log(f"堆栈结果：{target.name}")


def process_folder(
    input_folder: Path,
    output_folder: Path,
    max_size: int,
    min_correlation: float,
    padding: float,
    reference_name: str | None,
    stack_method: str = "none",
    stack_percent: int = 100,
    ap_size: int = 0,
    save_aligned: bool = True,
    log: Callable[[str], None] = print,
) -> None:
    if stack_method not in {"none", "mean", "median", "local"}:
        raise ValueError("堆栈方式必须是 none、mean、median 或 local")
    if not 1 <= stack_percent <= 100:
        raise ValueError("堆栈保留比例必须在 1 到 100 之间")
    if ap_size != 0 and not 32 <= ap_size <= 512:
        raise ValueError("局部对齐块大小必须为 0（自动）或 32 到 512")
    if stack_method == "none" and not save_aligned:
        raise ValueError("不堆栈时必须保留裁切后的单张照片")

    paths = list_images(input_folder)
    if len(paths) < 2:
        raise RuntimeError("输入文件夹中至少需要两张 JPG、PNG 或 TIFF 照片")

    log(f"找到 {len(paths)} 张照片，正在检测月面和评估清晰度……")
    scans: list[ScanResult] = []
    for index, path in enumerate(paths, 1):
        scan = scan_image(path, max_size)
        scans.append(scan)
        state = "失败" if scan.error else "完成"
        log(f"  [{index:>3}/{len(paths)}] {state}: {path.name}")

    valid_scans = [scan for scan in scans if not scan.error]
    if len(valid_scans) < 2:
        details = "；".join(f"{s.path.name}: {s.error}" for s in scans if s.error)
        raise RuntimeError(f"可用照片少于两张。{details}")

    if reference_name:
        matches = [scan for scan in valid_scans if scan.path.name == reference_name]
        if not matches:
            raise RuntimeError(f"找不到指定参考帧：{reference_name}")
        reference = matches[0]
    else:
        reference = max(valid_scans, key=lambda scan: scan.sharpness)

    log(f"参考帧：{reference.path.name}（清晰度 {reference.sharpness:.6f}）")
    reference_gray, _ = registration_image(reference.path, max_size)

    log("正在纠正位移和旋转……")
    results: list[AlignmentResult] = []
    for index, scan in enumerate(scans, 1):
        result = align_one(scan, reference_gray, reference, max_size, min_correlation)
        results.append(result)
        detail = result.message or (
            f"相关度 {result.correlation:.3f}" if result.correlation is not None else result.status
        )
        log(f"  [{index:>3}/{len(scans)}] {result.status}: {scan.path.name} — {detail}")

    accepted = [result for result in results if result.status == "accepted" and result.warp is not None]
    if len(accepted) < 2:
        raise RuntimeError("成功配准的照片少于两张；可尝试降低 --min-correlation")

    assert reference.center is not None and reference.radius is not None
    center_full = (reference.center[0] / reference.scale, reference.center[1] / reference.scale)
    radius_full = reference.radius / reference.scale
    width, height = reference.width, reference.height
    required_bounds = square_bounds(center_full, math.ceil(radius_full * 1.02), width, height)

    # 排除变换后无法完整覆盖月面的帧，避免最终裁切掉月球边缘。
    common_mask = np.full((height, width), 255, dtype=np.uint8)
    coverage_checked: list[AlignmentResult] = []
    for result in accepted:
        assert result.warp is not None
        valid_mask = warp_valid_mask(width, height, result.warp)
        if not region_is_valid(valid_mask, required_bounds):
            result.status = "skipped"
            result.message = "配准后无法完整覆盖月面（原图中的月亮可能已经出框）"
            continue
        common_mask = cv2.bitwise_and(common_mask, valid_mask)
        coverage_checked.append(result)

    accepted = coverage_checked
    if len(accepted) < 2:
        raise RuntimeError("没有足够照片能够完整覆盖月面")

    crop = find_safe_crop(common_mask, center_full, radius_full, padding)
    x1, y1, x2, y2 = crop
    output_folder.mkdir(parents=True, exist_ok=True)
    log(f"统一裁切范围：{x2 - x1} × {y2 - y1} 像素")
    if save_aligned:
        log("正在输出对齐后的 TIFF……")
    else:
        log("正在准备堆栈帧（不保留单张 TIFF）……")

    stack_results: list[AlignmentResult] = []
    if stack_method == "local":
        # 多点模式会在每个局部独立择优，因此需要缓存全部已接受帧。
        stack_results = list(accepted)
        log(f"局部堆栈将在每个对齐点独立选择最清晰的 {stack_percent}% 照片")
    elif stack_method != "none":
        stack_count = max(2, math.ceil(len(accepted) * stack_percent / 100))
        stack_count = min(len(accepted), stack_count)
        stack_results = sorted(
            accepted, key=lambda item: item.scan.sharpness, reverse=True
        )[:stack_count]
        log(f"堆栈将使用清晰度最高的 {stack_count}/{len(accepted)} 张照片（{stack_percent}%）")
    stack_paths = {result.scan.path for result in stack_results}

    with tempfile.TemporaryDirectory(prefix="moon-stack-") as temp_dir:
        stack_store: np.memmap | None = None
        stack_file = Path(temp_dir) / "aligned_frames.npy"
        stack_index = 0
        expected_shape: tuple[int, ...] | None = None
        expected_dtype: np.dtype | None = None

        for index, result in enumerate(accepted, 1):
            assert result.warp is not None
            image = read_image(result.scan.path)
            aligned = cv2.warpAffine(
                image,
                result.warp,
                (width, height),
                flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            cropped = aligned[y1:y2, x1:x2]

            if save_aligned:
                target = output_folder / f"{index:04d}_{result.scan.path.stem}_aligned.tif"
                write_tiff(target, cropped)
                log(f"  [{index:>3}/{len(accepted)}] {target.name}")
            else:
                log(f"  [{index:>3}/{len(accepted)}] 已准备：{result.scan.path.name}")

            if result.scan.path in stack_paths:
                if stack_store is None:
                    expected_shape = cropped.shape
                    expected_dtype = cropped.dtype
                    stack_store = np.lib.format.open_memmap(
                        stack_file,
                        mode="w+",
                        dtype=cropped.dtype,
                        shape=(len(stack_results), *cropped.shape),
                    )
                if cropped.shape != expected_shape or cropped.dtype != expected_dtype:
                    raise RuntimeError(
                        "照片的颜色通道或位深不一致，无法可靠堆栈；请先统一导出格式"
                    )
                stack_store[stack_index] = cropped
                stack_index += 1

        if stack_store is not None:
            stack_store.flush()
            del stack_store
            stack_target = output_folder / f"moon_stack_{stack_method}.tif"
            stack_frames(
                stack_file,
                stack_target,
                stack_method,
                stack_percent,
                ap_size,
                log,
            )

    save_report(output_folder, results, stack_paths)
    save_processing_report(
        output_folder,
        input_folder,
        reference,
        len(results),
        len(accepted),
        crop,
        stack_method,
        stack_percent,
        ap_size,
        save_aligned,
    )
    log("")
    output_description = "配准帧和堆栈图" if save_aligned and stack_method != "none" else (
        "配准帧" if save_aligned else "堆栈图"
    )
    log(
        f"完成：已生成{output_description}；使用 {len(accepted)} 张，"
        f"跳过 {len(results) - len(accepted)} 张。"
    )
    log(f"输出目录：{output_folder.resolve()}")
    if stack_method == "none":
        log("这些 TIFF 可以直接导入 PlanetarySystemStacker、AutoStakkert 或 Siril 继续堆栈。")


class MoonAlignerApp:
    """Windows友好的主界面；耗时处理在后台线程运行。"""

    def __init__(self, root: object, args: argparse.Namespace) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.args = args
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.running = False
        self.last_output: Path | None = None
        self.output_was_selected = bool(args.output)

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar(value=str(args.output) if args.output else "")
        self.max_size_var = tk.StringVar(value=str(args.max_size))
        self.correlation_var = tk.StringVar(value=str(args.min_correlation))
        self.padding_var = tk.StringVar(value=str(args.padding))
        self.stack_var = tk.StringVar(value=STACK_LABELS_REVERSE[args.stack])
        self.stack_percent_var = tk.StringVar(value=str(args.stack_percent))
        self.ap_size_var = tk.StringVar(value=str(args.ap_size))
        self.save_aligned_var = tk.BooleanVar(value=args.save_aligned)
        self.status_var = tk.StringVar(value="请选择照片文件夹和输出文件夹")

        self.root.title("月亮裁切与堆栈")
        self.root.geometry("840x720")
        self.root.minsize(720, 620)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"))
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10))
        style.configure("Status.TLabel", font=("Segoe UI", 10, "bold"))

        outer = ttk.Frame(self.root, padding=20)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="月亮照片自动配准、裁切与堆栈", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text="选择照片目录和处理方式，确认后点击“开始处理”。原始照片不会被修改。",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(4, 18))

        paths = ttk.LabelFrame(outer, text="文件夹", padding=12)
        paths.pack(fill="x")
        paths.columnconfigure(1, weight=1)

        ttk.Label(paths, text="输入文件夹：").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=6)
        self.input_entry = ttk.Entry(paths, textvariable=self.input_var)
        self.input_entry.grid(row=0, column=1, sticky="ew", pady=6)
        self.input_button = ttk.Button(paths, text="选择…", width=10, command=self.choose_input)
        self.input_button.grid(row=0, column=2, padx=(8, 0), pady=6)

        ttk.Label(paths, text="输出文件夹：").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=6)
        self.output_entry = ttk.Entry(paths, textvariable=self.output_var)
        self.output_entry.grid(row=1, column=1, sticky="ew", pady=6)
        self.output_button = ttk.Button(paths, text="选择…", width=10, command=self.choose_output)
        self.output_button.grid(row=1, column=2, padx=(8, 0), pady=6)

        stack_options = ttk.LabelFrame(outer, text="输出与堆栈", padding=10)
        stack_options.pack(fill="x", pady=(12, 0))
        ttk.Label(stack_options, text="处理方式：").grid(row=0, column=0, sticky="w")
        self.stack_box = ttk.Combobox(
            stack_options,
            state="readonly",
            width=29,
            textvariable=self.stack_var,
            values=list(STACK_LABELS),
        )
        self.stack_box.grid(row=0, column=1, sticky="w", padx=(6, 22))
        self.stack_box.bind("<<ComboboxSelected>>", self.on_stack_mode_changed)
        ttk.Label(stack_options, text="使用最清晰的：").grid(row=0, column=2, sticky="w")
        self.stack_percent_box = ttk.Spinbox(
            stack_options,
            from_=1,
            to=100,
            increment=5,
            width=6,
            textvariable=self.stack_percent_var,
        )
        self.stack_percent_box.grid(row=0, column=3, sticky="w", padx=(6, 2))
        ttk.Label(stack_options, text="%").grid(row=0, column=4, sticky="w")
        self.save_aligned_check = ttk.Checkbutton(
            stack_options,
            text="同时保留每张配准裁切后的 TIFF",
            variable=self.save_aligned_var,
        )
        self.save_aligned_check.grid(row=1, column=0, columnspan=5, sticky="w", pady=(8, 0))

        advanced = ttk.LabelFrame(outer, text="高级设置（一般保持默认即可）", padding=10)
        advanced.pack(fill="x", pady=(12, 0))
        ttk.Label(advanced, text="配准精度：").grid(row=0, column=0, sticky="w")
        self.max_size_box = ttk.Spinbox(
            advanced, from_=400, to=4000, increment=100, width=8, textvariable=self.max_size_var
        )
        self.max_size_box.grid(row=0, column=1, padx=(6, 22))
        ttk.Label(advanced, text="最低相关度：").grid(row=0, column=2, sticky="w")
        self.correlation_box = ttk.Spinbox(
            advanced, from_=0.05, to=1.0, increment=0.05, width=8, textvariable=self.correlation_var
        )
        self.correlation_box.grid(row=0, column=3, padx=(6, 22))
        ttk.Label(advanced, text="边缘留白：").grid(row=0, column=4, sticky="w")
        self.padding_box = ttk.Spinbox(
            advanced, from_=0.0, to=2.0, increment=0.05, width=8, textvariable=self.padding_var
        )
        self.padding_box.grid(row=0, column=5, padx=(6, 0))
        ttk.Label(advanced, text="局部块大小：").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.ap_size_box = ttk.Spinbox(
            advanced, from_=0, to=512, increment=32, width=8, textvariable=self.ap_size_var
        )
        self.ap_size_box.grid(row=1, column=1, padx=(6, 22), pady=(8, 0))
        ttk.Label(advanced, text="0 表示自动；仅用于多点局部堆栈").grid(
            row=1, column=2, columnspan=4, sticky="w", pady=(8, 0)
        )

        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=(14, 8))
        self.start_button = ttk.Button(actions, text="开始处理", command=self.start_processing)
        self.start_button.pack(side="left")
        self.open_button = ttk.Button(
            actions, text="打开输出文件夹", command=self.open_output_folder, state="disabled"
        )
        self.open_button.pack(side="left", padx=(8, 0))
        self.progress = ttk.Progressbar(actions, mode="indeterminate", length=180)
        self.progress.pack(side="right")

        ttk.Label(outer, textvariable=self.status_var, style="Status.TLabel").pack(anchor="w", pady=(0, 6))
        log_frame = ttk.Frame(outer)
        log_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(
            log_frame,
            wrap="word",
            state="disabled",
            font=("Consolas", 10),
            background="#f6f6f6",
            relief="solid",
            borderwidth=1,
        )
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.on_stack_mode_changed()
        self.root.after(100, self.poll_events)

    def on_stack_mode_changed(self, _event: object | None = None) -> None:
        if self.running:
            return
        method = STACK_LABELS[self.stack_var.get()]
        self.stack_percent_box.configure(state="disabled" if method == "none" else "normal")
        self.ap_size_box.configure(state="normal" if method == "local" else "disabled")

    def choose_input(self) -> None:
        from tkinter import filedialog

        selected = filedialog.askdirectory(title="选择包含月亮照片的文件夹", parent=self.root)
        if not selected:
            return
        self.input_var.set(selected)
        if not self.output_was_selected:
            self.output_var.set(str(Path(selected) / "moon_aligned"))

    def choose_output(self) -> None:
        from tkinter import filedialog

        selected = filedialog.askdirectory(title="选择裁切结果的保存文件夹", parent=self.root)
        if selected:
            self.output_var.set(selected)
            self.output_was_selected = True

    def append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def set_running(self, running: bool) -> None:
        self.running = running
        state = "disabled" if running else "normal"
        for widget in (
            self.input_entry,
            self.output_entry,
            self.input_button,
            self.output_button,
            self.start_button,
            self.max_size_box,
            self.correlation_box,
            self.padding_box,
            self.stack_percent_box,
            self.ap_size_box,
            self.save_aligned_check,
        ):
            widget.configure(state=state)
        self.stack_box.configure(state="disabled" if running else "readonly")
        if not running:
            self.on_stack_mode_changed()
        if running:
            self.open_button.configure(state="disabled")
            self.progress.start(12)
        else:
            self.progress.stop()

    def start_processing(self) -> None:
        from tkinter import messagebox

        input_text = self.input_var.get().strip()
        output_text = self.output_var.get().strip()
        if not input_text:
            messagebox.showwarning("缺少输入文件夹", "请先选择包含月亮照片的文件夹。", parent=self.root)
            return
        if not output_text:
            output_text = str(Path(input_text) / "moon_aligned")
            self.output_var.set(output_text)

        input_folder = Path(input_text).expanduser().resolve()
        output_folder = Path(output_text).expanduser().resolve()
        if not input_folder.is_dir():
            messagebox.showerror("输入错误", f"输入文件夹不存在：\n{input_folder}", parent=self.root)
            return

        try:
            max_size = int(self.max_size_var.get())
            min_correlation = float(self.correlation_var.get())
            padding = float(self.padding_var.get())
            stack_method = STACK_LABELS[self.stack_var.get()]
            stack_percent = int(self.stack_percent_var.get())
            ap_size = int(self.ap_size_var.get())
            save_aligned = bool(self.save_aligned_var.get())
            if max_size < 400:
                raise ValueError("配准精度不能小于400")
            if not 0.0 <= min_correlation <= 1.0:
                raise ValueError("最低相关度必须在0到1之间")
            if not 0.0 <= padding <= 2.0:
                raise ValueError("边缘留白必须在0到2之间")
            if not 1 <= stack_percent <= 100:
                raise ValueError("堆栈保留比例必须在1到100之间")
            if ap_size != 0 and not 32 <= ap_size <= 512:
                raise ValueError("局部块大小必须为0（自动）或32到512")
            if stack_method == "none" and not save_aligned:
                raise ValueError("选择“不堆栈”时，必须保留裁切后的单张照片")
        except ValueError as exc:
            messagebox.showerror("设置错误", str(exc), parent=self.root)
            return

        self.clear_log()
        self.append_log(f"输入目录：{input_folder}")
        self.append_log(f"输出目录：{output_folder}\n")
        self.status_var.set("正在处理，请保持程序打开……")
        self.set_running(True)
        self.last_output = output_folder

        def worker() -> None:
            try:
                process_folder(
                    input_folder,
                    output_folder,
                    max_size,
                    min_correlation,
                    padding,
                    None,
                    stack_method=stack_method,
                    stack_percent=stack_percent,
                    ap_size=ap_size,
                    save_aligned=save_aligned,
                    log=lambda message: self.events.put(("log", message)),
                )
                self.events.put(("done", output_folder))
            except Exception as exc:
                self.events.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True, name="moon-aligner-worker").start()

    def poll_events(self) -> None:
        from tkinter import messagebox

        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "log":
                    self.append_log(str(payload))
                elif event == "done":
                    self.set_running(False)
                    self.open_button.configure(state="normal")
                    self.status_var.set("处理完成")
                    messagebox.showinfo(
                        "处理完成", f"照片已经处理完成。\n\n输出目录：\n{payload}", parent=self.root
                    )
                elif event == "error":
                    self.set_running(False)
                    self.status_var.set("处理失败")
                    self.append_log(f"\n错误：{payload}")
                    messagebox.showerror("处理失败", str(payload), parent=self.root)
        except queue.Empty:
            pass
        self.root.after(100, self.poll_events)

    def open_output_folder(self) -> None:
        if self.last_output is None or not self.last_output.exists():
            return
        if sys.platform == "win32":
            os.startfile(self.last_output)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(self.last_output)])
        else:
            subprocess.Popen(["xdg-open", str(self.last_output)])

    def on_close(self) -> None:
        if self.running:
            from tkinter import messagebox

            if not messagebox.askyesno(
                "仍在处理中", "照片尚未处理完成，确定要退出吗？", parent=self.root
            ):
                return
        self.root.destroy()


def run_gui(args: argparse.Namespace) -> None:
    try:
        import tkinter as tk
    except ImportError as exc:
        raise RuntimeError("当前Python不包含tkinter，无法启动图形界面") from exc

    root = tk.Tk()
    try:
        icon_image = tk.PhotoImage(file=str(resource_path("assets/app_icon.png")))
        root.iconphoto(True, icon_image)
        # 保留Python引用，避免Tk图片对象被垃圾回收。
        root._moon_aligner_icon = icon_image  # type: ignore[attr-defined]
        if sys.platform == "win32":
            root.iconbitmap(default=str(resource_path("assets/app_icon.ico")))
    except (tk.TclError, OSError):
        # 图标加载失败不应阻止主程序启动。
        pass
    MoonAlignerApp(root, args)
    root.mainloop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="自动配准、统一裁切并堆栈月亮照片，不修改原图。"
    )
    parser.add_argument("input", nargs="?", type=Path, help="包含照片的文件夹；省略时弹出选择窗口")
    parser.add_argument("-o", "--output", type=Path, help="输出文件夹，默认是输入目录下的 moon_aligned")
    parser.add_argument(
        "--max-size",
        type=int,
        default=1600,
        help="配准缩略图的最长边，越大越精确但越慢（默认：1600）",
    )
    parser.add_argument(
        "--min-correlation",
        type=float,
        default=0.45,
        help="最低 ECC 相关度，低于此值的照片会被跳过（默认：0.45）",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.18,
        help="月面四周留白占月球半径的比例（默认：0.18）",
    )
    parser.add_argument("--reference", help="手动指定输入目录中的参考帧文件名")
    parser.add_argument(
        "--stack",
        choices=("none", "mean", "median", "local"),
        default="mean",
        help="堆栈方式：不堆栈、平均、中位数或多点局部堆栈（默认：mean）",
    )
    parser.add_argument(
        "--stack-percent",
        type=int,
        default=100,
        help="按清晰度排序后用于堆栈的照片比例（默认：100）",
    )
    parser.add_argument(
        "--ap-size",
        type=int,
        default=0,
        help="多点局部堆栈的块大小；0表示自动，或设置32到512（默认：0）",
    )
    parser.add_argument(
        "--no-save-aligned",
        action="store_false",
        dest="save_aligned",
        help="只保存堆栈结果，不保留每张配准裁切后的 TIFF",
    )
    parser.set_defaults(save_aligned=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.input is None:
        try:
            run_gui(args)
            return 0
        except Exception as exc:
            print(f"\n错误：{exc}", file=sys.stderr)
            return 1

    try:
        input_folder = args.input.expanduser().resolve()
        output_folder = args.output or input_folder / "moon_aligned"

        if not input_folder.is_dir():
            raise RuntimeError(f"输入目录不存在：{input_folder}")
        if args.max_size < 400:
            raise RuntimeError("--max-size 建议不小于 400")
        if not 0.0 <= args.min_correlation <= 1.0:
            raise RuntimeError("--min-correlation 必须在 0 到 1 之间")
        if not 0.0 <= args.padding <= 2.0:
            raise RuntimeError("--padding 必须在 0 到 2 之间")
        if not 1 <= args.stack_percent <= 100:
            raise RuntimeError("--stack-percent 必须在 1 到 100 之间")
        if args.ap_size != 0 and not 32 <= args.ap_size <= 512:
            raise RuntimeError("--ap-size 必须为 0（自动）或 32 到 512")
        if args.stack == "none" and not args.save_aligned:
            raise RuntimeError("--stack none 不能与 --no-save-aligned 同时使用")

        process_folder(
            input_folder,
            Path(output_folder).expanduser().resolve(),
            args.max_size,
            args.min_correlation,
            args.padding,
            args.reference,
            stack_method=args.stack,
            stack_percent=args.stack_percent,
            ap_size=args.ap_size,
            save_aligned=args.save_aligned,
            log=print,
        )
        return 0
    except KeyboardInterrupt:
        print("\n已取消。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\n错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

