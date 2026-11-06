#!/usr/bin/env python3
"""Compare saved geometry and SIFT layouts using the same Stage 4 renderer.

Both methods are rendered directly from measured RGBA fragments onto the
reference pixel grid. Only a single global frame normalization is added;
individual poses/homographies are never optimized against the reference.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import sys

import cv2
import numpy as np
import scipy
import shapely
import skimage

try:
    from . import assemble as assembly
except ImportError:
    import assemble as assembly

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIFT = ROOT / "stage3_feature_matching/output/assignments.json"
DEFAULT_LAYOUT = ROOT / "stage1_capture_calibration/output/cut_pdf/jigsaw_layout.json"
DEFAULT_OUTPUT = ROOT / "stage4_assembly/output/method_comparison"


def normalize_fragments(fragments, rectangle, size, angle):
    """Map the continuous outer frame to the common grid in one matrix."""
    rotation, rotated = assembly.board_rotation(angle, rectangle)
    scale = np.diag([size[0] / rotated[0], size[1] / rotated[1], 1.0])
    matrix = scale @ rotation
    return [replace(f, h=matrix @ f.h,
                    target=assembly.transform(matrix, f.target))
            for f in fragments], matrix


def render_method(fragments, reference, config):
    """Return images, alpha coverage and separate observed/target geometry."""
    size = (reference.shape[1], reference.shape[0])
    accumulators = [np.zeros(reference.shape, np.float32) for _ in range(2)]
    weights = np.zeros(reference.shape[:2], np.float32)
    counts = np.zeros(reference.shape[:2], np.uint16)
    observed = []
    for f in fragments:
        corners = assembly.finite_array(
            f.metadata.get("image_corners_px"), f.target.shape,
            f"{f.id}.image_corners_px")
        observed.append(replace(f, target=assembly.transform(f.h, corners)))
        color, alpha, weight = assembly.warp_fragment(f, size, config)
        corrected, _ = assembly.color_correct(color, reference, alpha, config)
        counts += (alpha > 1e-3).astype(np.uint16)
        weights += weight
        accumulators[0] += color * weight[:, :, None]
        accumulators[1] += corrected * weight[:, :, None]
    raw, corrected = [assembly.finish(a, weights) for a in accumulators]
    covered = counts > 0
    metrics = {
        "raw": assembly.image_metrics(raw, reference, covered, config["evaluate_ssim"]),
        "corrected": assembly.image_metrics(
            corrected, reference, covered, config["evaluate_ssim"]),
        "raster": {
            "covered_fraction": float(covered.mean()),
            "uncovered_pixels": int((~covered).sum()),
            "overlap_pixels": int((counts > 1).sum()),
        },
        "warped_measured_geometry": assembly.geometry_report(observed, size),
        "target_geometry": assembly.geometry_report(fragments, size),
    }
    return raw, corrected, covered, metrics


def panel_row(images, titles):
    panels = []
    for image, title in zip(images, titles):
        factor = min(1.0, 640 / image.shape[1])
        small = cv2.resize(image, (round(image.shape[1] * factor),
                                  round(image.shape[0] * factor)))
        panel = cv2.copyMakeBorder(small, 36, 0, 0, 0, cv2.BORDER_CONSTANT,
                                   value=(255, 255, 255))
        cv2.putText(panel, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (0, 0, 0), 1, cv2.LINE_AA)
        panels.append(panel)
    return np.hstack(panels)


def summary_markdown(report):
    rows = [
        "# 几何与 SIFT 重建对比", "",
        "同一组实拍 RGBA 碎片、同一参考图和相同融合/校色参数。",
        "统一外框后直接渲染到参考图像素网格，不重新优化单片位姿。",
        "几何方案的整体朝向由参考图辅助判断；SIFT 在匹配时使用原图模板和切割布局。",
        "两种方法的信息条件及变换自由度不同，本表不是无参考条件下的算法排名。", "",
        "| 指标 | 几何方案 | SIFT 方案 |", "| --- | ---: | ---: |",
    ]
    g, s = report["methods"]["geometry"], report["methods"]["sift"]
    rows += [f"| 真实纹理栅格覆盖率 | {g['raster']['covered_fraction']:.6%} | {s['raster']['covered_fraction']:.6%} |"]
    rows += [f"| 变换后实测轮廓综合误差 | {g['warped_measured_geometry']['score']:.6%} | {s['warped_measured_geometry']['score']:.6%} |"]
    for stage, label in (("raw", "未校色"), ("corrected", "校色后")):
        for key, name, suffix in (("psnr_db", "全画布 RGB PSNR", " dB"),
                                  ("ssim_gray", "全画布灰度 SSIM", "")):
            values = [f"{m[stage][key]:.5f}{suffix}" if m[stage][key] is not None
                      else "不可用或完全相同" for m in (g, s)]
            rows.append(f"| {label}{name} | {values[0]} | {values[1]} |")
    rows += ["", "参考图大小：" + str(report["evaluation_size_px"]), "",
             "## 指标解释", "",
             "覆盖率由变换后的 Alpha 支持区域计算，不能用 RGB 非零与否代替。",
             "warped_measured_geometry 来自实测碎片角点经实际变换后的几何；",
             "target_geometry 是渲染裁剪目标。SIFT 的后者来自已知源多边形，接近 100% 不代表实测匹配精度。",
             "PSNR 包含黑色缺口；covered_psnr_db 另列于 JSON。SSIM 是灰度相似性，不是碎片正确率。",
             "校色是有参考图的后处理；未校色指标应同时保留。", "",
             "## 复核", ""]
    for name, method in report["methods"].items():
        rows.append(f"- {name}：" + ("；".join(method["warnings"]) or "无复核项"))
    rows += ["", "原生尺度的两份人工摆放计划保存在 geometry/ 与 sift/ 中。",
             "SIFT 的单应可包含尺度、剪切和透视变形；其参考图像素坐标不可直接当作实物毫米摆放坐标。",
             "这份对比使用已有 Stage 3 结果，不测量求解时间，也不声明一一分配就是经真值验证的正确率。", ""]
    return "\n".join(rows)


def compare(pieces=assembly.DEFAULT_PIECES, geometry=assembly.DEFAULT_PLACEMENTS,
            sift=DEFAULT_SIFT, layout=DEFAULT_LAYOUT,
            reference=assembly.DEFAULT_REFERENCE, output=DEFAULT_OUTPUT,
            write_native=True):
    paths = {k: Path(v) for k, v in dict(pieces=pieces, geometry=geometry,
                                      sift=sift, layout=layout, reference=reference).items()}
    output = Path(output)
    config = assembly.resolve_config(None)
    ref = assembly.imread(paths["reference"])
    size = (ref.shape[1], ref.shape[0])
    assembly.require(size[0] * size[1] <= config["max_canvas_pixels"],
                     "reference exceeds max_canvas_pixels")
    prepared = {}
    input_paths = {p.resolve() for p in paths.values()}
    for name, mode in (("geometry", "geometry_placements"), ("sift", "reference_assignments")):
        manifest, upstream, fragments, rect, actual_mode, warnings = assembly.load_inputs(
            paths["pieces"], paths[name], paths["layout"] if name == "sift" else None, config)
        assembly.require(actual_mode == mode, f"{name}: expected {mode}")
        if name == "sift":
            assembly.require(tuple(rect) == size, "SIFT layout and reference sizes differ")
            angle, orientation = 0, {"method": "reference_assignments"}
        else:
            angle, orientation = assembly.choose_orientation(fragments, rect, ref, config, "auto")
            if orientation.get("ambiguous"):
                warnings.append("Reference does not confidently resolve geometry orientation.")
        orientation = {**orientation, "chosen_clockwise_deg": angle}
        normalized, matrix = normalize_fragments(fragments, rect, size, angle)
        # Validate measured-corner availability before writing any outputs.
        for f in normalized:
            assembly.finite_array(f.metadata.get("image_corners_px"), f.target.shape,
                                  f"{f.id}.image_corners_px")
            for key in ("rgba_path", "mask_path"):
                if f.metadata.get(key):
                    input_paths.add((paths["pieces"].parent / f.metadata[key]).resolve())
        prepared[name] = (normalized, rect, actual_mode, warnings, orientation, matrix, upstream)
    assembly.require(not any(p.is_relative_to(output.resolve()) for p in input_paths),
                     "output directory must not contain input files")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "1.0", "algorithm_version": "stage4-compare-1.0.0",
        "evaluation_size_px": list(size),
        "normalization": "one global outer-frame mapping; no per-piece refinement",
        "coverage_definition": "union of warped alpha > 0.001, clipped to target polygons",
        "config": config,
        "input_sha256": {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "opencv": cv2.__version__, "shapely": shapely.__version__,
                        "scipy": scipy.__version__, "scikit_image": skimage.__version__},
        "methods": {},
    }
    images = {}
    for name, (fragments, rect, mode, warnings, orientation, matrix, upstream) in prepared.items():
        raw, corrected, covered, metrics = render_method(fragments, ref, config)
        method = {
            "input_mode": mode, "piece_count": len(fragments),
            "native_rectangle_px": list(rect), "native_canvas_size_px": [int(np.ceil(v)) for v in rect],
            "H_native_to_evaluation": matrix.tolist(), "orientation": orientation,
            "reference_used_for_matching": name == "sift",
            "target_geometry_source": "known_cut_layout" if name == "sift" else "estimated_layout",
            "upstream_status": upstream.get("status"),
            "matching_review": upstream.get("review", []), "warnings": warnings,
            **metrics,
        }
        if write_native:
            native = assembly.assemble(paths["pieces"], paths[name], output / name,
                layout_path=paths["layout"] if name == "sift" else None,
                reference_path=paths["reference"], config={"write_steps": False})
            method["native_status"] = native["status"]
        report["methods"][name] = method
        images[name] = (raw, corrected)
        for suffix, img in (("raw", raw), ("corrected", corrected),
                            ("coverage", covered.astype(np.uint8) * 255)):
            assembly.imwrite(output / f"{name}_{suffix}.png", img)
    for index, name in enumerate(("raw", "corrected")):
        row = panel_row([ref, images["geometry"][index], images["sift"][index]],
                        ["Reference", f"Geometry ({name})", f"SIFT ({name})"])
        assembly.imwrite(output / f"comparison_{name}.png", row)
    assembly.save_json(output / "comparison_report.json", report)
    (output / "comparison.md").write_text(summary_markdown(report), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("pieces", assembly.DEFAULT_PIECES),
                          ("geometry", assembly.DEFAULT_PLACEMENTS), ("sift", DEFAULT_SIFT),
                          ("layout", DEFAULT_LAYOUT), ("reference", assembly.DEFAULT_REFERENCE),
                          ("output", DEFAULT_OUTPUT)):
        parser.add_argument("--" + name, type=Path, default=default)
    args = parser.parse_args()
    try:
        report = compare(**vars(args))
    except (ValueError, KeyError, OSError, cv2.error) as exc:
        print(f"comparison failed: {exc}", file=sys.stderr)
        return 2
    print(summary_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
