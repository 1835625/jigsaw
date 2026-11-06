#!/usr/bin/env python3
"""Map photographed pieces back to the reference image with SIFT.

Scan pieces (S01..S16) are matched to source templates (P01..P16) cut from
the original reference.  Lowe ratio test + RANSAC homography score each pair,
Hungarian assignment forces a 1-to-1 mapping, then each piece is warped onto
the reference canvas.

Does not use the geometric quad solver.  Homographies map piece PNG pixels
onto reference.png coordinates so Stage 4 can consume assignments.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Point, Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))
from jigsaw_common.io import imread, imwrite, read_json, write_json  # noqa: E402

VERSION = "stage3-sift-0.1.0"
RATIO = 0.75
RANSAC_PX = 4.0
MIN_INLIERS = 8
BORDER_ERODE_PX = 5
SCALE_RANGE = (0.25, 2.0)
TEMPLATE_PAD = 2


def homography_points(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    ones = np.ones((len(pts), 1))
    mapped = np.hstack((pts, ones)) @ h.T
    denom = mapped[:, 2:3]
    if np.any(np.abs(denom) < 1e-12):
        raise ValueError("homography hits infinity")
    return mapped[:, :2] / denom


def affine_scale(h: np.ndarray) -> float:
    sx = float(math.hypot(h[0, 0], h[1, 0]))
    sy = float(math.hypot(h[0, 1], h[1, 1]))
    return 0.5 * (sx + sy)


def signed_area(points: np.ndarray) -> float:
    p = np.asarray(points, dtype=np.float64)
    return float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1)))


def preserves_orientation(h: np.ndarray, points: np.ndarray) -> bool:
    mapped = homography_points(h, points)
    return signed_area(points) * signed_area(mapped) > 0


def prepare_gray(rgba: np.ndarray, erode: int) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    mask = (rgba[:, :, 3] > 0).astype(np.uint8) * 255
    if erode > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode * 2 + 1, erode * 2 + 1))
        mask = cv2.erode(mask, k)
    return gray, mask


def extract_features(sift: cv2.SIFT, rgba: np.ndarray) -> tuple[list, np.ndarray | None]:
    gray, mask = prepare_gray(rgba, BORDER_ERODE_PX)
    if int(np.count_nonzero(mask)) < 64:
        return [], None
    keypoints, descriptors = sift.detectAndCompute(gray, mask)
    if descriptors is None or len(keypoints) < 4:
        return [], None
    return list(keypoints), descriptors


def match_pair(
    sift: cv2.SIFT,
    matcher: cv2.BFMatcher,
    scan: dict,
    source: dict,
) -> dict:
    empty = {
        "good": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "median_reprojection_error_px": None,
        "score": 0.0,
        "iou": 0.0,
        "scale": None,
        "H_piece_to_template": None,
        "H_piece_to_source": None,
        "usable": False,
        "reason": "insufficient_features",
    }
    if scan["descriptors"] is None or source["descriptors"] is None:
        return empty
    knn = matcher.knnMatch(scan["descriptors"], source["descriptors"], k=2)
    good = [a for pair in knn if len(pair) == 2 for a, b in (pair,) if a.distance < RATIO * b.distance]
    empty["good"] = len(good)
    if len(good) < MIN_INLIERS:
        empty["reason"] = "few_ratio_matches"
        return empty
    src_pts = np.float32([scan["keypoints"][m.queryIdx].pt for m in good])
    dst_pts = np.float32([source["keypoints"][m.trainIdx].pt for m in good])
    h_tmpl, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_PX)
    if h_tmpl is None or inlier_mask is None:
        empty["reason"] = "homography_failed"
        return empty
    inlier_mask = inlier_mask.ravel().astype(bool)
    inliers = int(inlier_mask.sum())
    if inliers < MIN_INLIERS:
        empty["inliers"] = inliers
        empty["inlier_ratio"] = inliers / len(good)
        empty["reason"] = "few_inliers"
        return empty
    h_ref = source["to_reference"] @ h_tmpl
    try:
        mapped = homography_points(h_ref, scan["corners"])
        if not preserves_orientation(h_ref, scan["corners"]):
            empty["inliers"] = inliers
            empty["reason"] = "reflected_or_degenerate"
            return empty
    except ValueError:
        empty["inliers"] = inliers
        empty["reason"] = "homography_infinity"
        return empty
    scale = affine_scale(h_ref)
    if not (SCALE_RANGE[0] <= scale <= SCALE_RANGE[1]):
        empty["inliers"] = inliers
        empty["scale"] = scale
        empty["reason"] = "unreasonable_scale"
        return empty
    piece_poly = Polygon(mapped)
    source_poly = source["polygon"]
    if not piece_poly.is_valid or piece_poly.area <= 0:
        empty["inliers"] = inliers
        empty["reason"] = "invalid_mapped_polygon"
        return empty
    inter = piece_poly.intersection(source_poly).area
    union = piece_poly.union(source_poly).area
    iou = float(inter / union) if union > 1e-9 else 0.0
    centroid = homography_points(h_ref, np.array([scan["center"]]))[0]
    inside = source_poly.buffer(8.0).contains(Point(centroid))
    reproj = np.linalg.norm(
        homography_points(h_tmpl, src_pts[inlier_mask]) - dst_pts[inlier_mask],
        axis=1,
    )
    median_err = float(np.median(reproj))
    inlier_ratio = inliers / len(good)
    score = (inliers * inlier_ratio * max(iou, 0.02)) / (1.0 + median_err)
    if not inside:
        score *= 0.15
    usable = inliers >= MIN_INLIERS and iou >= 0.15 and inside
    return {
        "good": len(good),
        "inliers": inliers,
        "inlier_ratio": float(inlier_ratio),
        "median_reprojection_error_px": median_err,
        "score": float(score),
        "iou": iou,
        "scale": float(scale),
        "H_piece_to_template": h_tmpl.tolist(),
        "H_piece_to_source": (h_ref / h_ref[2, 2]).tolist(),
        "usable": usable,
        "reason": None if usable else "weak_geometry",
        "matches": good,
        "inlier_mask": inlier_mask,
    }


def template_origin(polygon: Polygon, width: int, height: int) -> tuple[int, int]:
    min_x, min_y, max_x, max_y = polygon.bounds
    left = max(0, math.floor(min_x) - TEMPLATE_PAD)
    top = max(0, math.floor(min_y) - TEMPLATE_PAD)
    right = min(width, math.ceil(max_x) + TEMPLATE_PAD)
    bottom = min(height, math.ceil(max_y) + TEMPLATE_PAD)
    return left, top, right, bottom


def load_scan_pieces(manifest_path: Path, sift: cv2.SIFT) -> list[dict]:
    manifest = read_json(manifest_path)
    root = manifest_path.parent
    pieces = []
    for record in manifest["pieces"]:
        rgba = imread(root / record["rgba_path"], cv2.IMREAD_UNCHANGED)
        if record.get("mask_path"):
            mask = imread(root / record["mask_path"], cv2.IMREAD_GRAYSCALE)
            rgba = rgba.copy()
            rgba[:, :, 3] = np.minimum(rgba[:, :, 3], mask)
        corners = np.asarray(record["image_corners_px"], dtype=np.float64)
        keypoints, descriptors = extract_features(sift, rgba)
        pieces.append(
            {
                "id": record["piece_id"],
                "rgba": rgba,
                "corners": corners,
                "center": corners.mean(axis=0),
                "keypoints": keypoints,
                "descriptors": descriptors,
            }
        )
    return pieces


def load_sources(layout_path: Path, source_dir: Path, sift: cv2.SIFT) -> list[dict]:
    layout = read_json(layout_path)
    width, height = layout["source_size_px"]
    sources = []
    for record in layout["pieces"]:
        polygon = Polygon(record["polygon_px"])
        left, top, right, bottom = template_origin(polygon, width, height)
        rgba = imread(source_dir / f"{record['piece_id']}_rgba.png", cv2.IMREAD_UNCHANGED)
        keypoints, descriptors = extract_features(sift, rgba)
        to_reference = np.array(
            [[1.0, 0.0, float(left)], [0.0, 1.0, float(top)], [0.0, 0.0, 1.0]]
        )
        sources.append(
            {
                "id": record["piece_id"],
                "rgba": rgba,
                "polygon": polygon,
                "polygon_px": np.asarray(record["polygon_px"], dtype=np.float64),
                "to_reference": to_reference,
                "keypoints": keypoints,
                "descriptors": descriptors,
            }
        )
    return sources, (int(width), int(height)), layout


def second_best_margin(scores: np.ndarray, row: int, col: int) -> float:
    order = np.argsort(-scores[row])
    best = float(scores[row, col])
    others = [float(scores[row, j]) for j in order if j != col]
    second = others[0] if others else 0.0
    if best <= 1e-12:
        return 0.0
    return float((best - second) / best)


def draw_cost_matrix(scores: np.ndarray, scans: list[dict], sources: list[dict], assigned: dict) -> np.ndarray:
    n = len(scans)
    cell = 48
    header = 72
    img = np.full((header + n * cell, header + n * cell, 3), 245, np.uint8)
    finite = scores[np.isfinite(scores)]
    peak = float(finite.max()) if len(finite) else 1.0
    peak = max(peak, 1e-6)
    for i, scan in enumerate(scans):
        for j, source in enumerate(sources):
            x0, y0 = header + j * cell, header + i * cell
            t = float(np.clip(scores[i, j] / peak, 0, 1))
            color = (int(40 + 20 * t), int(40 + 160 * t), int(40 + 40 * t))
            if assigned.get(scan["id"]) == source["id"]:
                color = (40, 180, 40)
            cv2.rectangle(img, (x0 + 1, y0 + 1), (x0 + cell - 2, y0 + cell - 2), color, -1)
            label = f"{scores[i, j]:.1f}" if scores[i, j] > 0 else ""
            if label:
                cv2.putText(img, label, (x0 + 4, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, scan["id"], (8, header + i * cell + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    for j, source in enumerate(sources):
        cv2.putText(img, source["id"], (header + j * cell + 6, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(img, "score  scan\\source  green=Hungarian", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return img


def draw_matches_image(scan: dict, source: dict, pair: dict) -> np.ndarray:
    matches = pair.get("matches") or []
    mask = pair.get("inlier_mask")
    if not matches:
        canvas = np.full((220, 480, 3), 32, np.uint8)
        cv2.putText(canvas, f"{scan['id']} x {source['id']}: no matches", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        return canvas
    inlier_matches = [m for m, keep in zip(matches, mask) if keep] if mask is not None else matches[:40]
    vis = cv2.drawMatches(
        scan["rgba"][:, :, :3],
        scan["keypoints"],
        source["rgba"][:, :, :3],
        source["keypoints"],
        inlier_matches[:80],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return vis


def reconstruct(scans: list[dict], sources: list[dict], assignment: list[tuple[int, int]], pairs: list[list[dict]], size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    width, height = size
    color = np.zeros((height, width, 3), dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)
    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    for i, j in assignment:
        pair = pairs[i][j]
        h = np.asarray(pair["H_piece_to_source"], dtype=np.float64)
        bgr = scans[i]["rgba"][:, :, :3].astype(np.float32)
        alpha = scans[i]["rgba"][:, :, 3].astype(np.float32) / 255.0
        warped = cv2.warpPerspective(bgr, h, (width, height), flags=cv2.INTER_LINEAR)
        warped_a = cv2.warpPerspective(alpha, h, (width, height), flags=cv2.INTER_LINEAR)
        poly = np.round(sources[j]["polygon_px"]).astype(np.int32)
        clip = np.zeros((height, width), dtype=np.float32)
        cv2.fillPoly(clip, [poly], 1.0)
        a = warped_a * clip
        color += warped * a[:, :, None]
        weight += a
        vis = np.clip(warped, 0, 255).astype(np.uint8)
        overlay[a > 0.05] = vis[a > 0.05]
        cv2.polylines(overlay, [poly], True, (0, 220, 0), 1, cv2.LINE_AA)
        cx, cy = sources[j]["polygon"].centroid.coords[0]
        cv2.putText(overlay, f"{scans[i]['id']}>{sources[j]['id']}", (int(cx) - 40, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    assembled = np.zeros((height, width, 3), dtype=np.uint8)
    nz = weight > 1e-6
    assembled[nz] = np.clip(color[nz] / weight[nz, None], 0, 255).astype(np.uint8)
    return assembled, overlay


def quality_vs_reference(assembled: np.ndarray, reference: np.ndarray) -> dict:
    mask = np.any(assembled > 0, axis=2)
    if not np.any(mask):
        return {"coverage": 0.0, "psnr_covered": None, "ssim_gray": None}
    coverage = float(mask.mean())
    diff = assembled.astype(np.float32) - reference.astype(np.float32)
    mse = float(np.mean(diff[mask] ** 2))
    psnr = None if mse <= 1e-12 else float(10.0 * math.log10((255.0 ** 2) / mse))
    gray_a = cv2.cvtColor(assembled, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    try:
        from skimage.metrics import structural_similarity
        ssim = float(structural_similarity(gray_a, gray_r))
    except Exception:
        ssim = None
    return {"coverage": coverage, "psnr_covered": psnr, "ssim_gray": ssim, "mse_covered": mse}


def side_by_side(reference: np.ndarray, assembled: np.ndarray) -> np.ndarray:
    residual = cv2.absdiff(reference, assembled)
    residual[~np.any(assembled > 0, axis=2)] = 0
    gap = np.full((reference.shape[0], 16, 3), 30, np.uint8)
    row = np.hstack((reference, gap, assembled, gap, residual))
    header = np.full((36, row.shape[1], 3), 20, np.uint8)
    cv2.putText(header, "reference", (20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(header, "SIFT warp", (reference.shape[1] + 36, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(header, "|diff|", (2 * reference.shape[1] + 52, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1, cv2.LINE_AA)
    return np.vstack((header, row))


def run(args: argparse.Namespace) -> dict:
    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02, edgeThreshold=10)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    scans = load_scan_pieces(args.pieces, sift)
    sources, size, _layout = load_sources(args.layout, args.source_dir, sift)
    print(f"loaded {len(scans)} scan pieces, {len(sources)} source templates, canvas {size[0]}x{size[1]}")
    for scan in scans:
        n = 0 if scan["descriptors"] is None else len(scan["descriptors"])
        print(f"  {scan['id']}: {n} SIFT keypoints")

    pairs = [[match_pair(sift, matcher, scan, source) for source in sources] for scan in scans]
    scores = np.array([[pair["score"] for pair in row] for row in pairs], dtype=np.float64)
    cost = np.where(scores > 0, -scores, 1e6)
    rows, cols = linear_sum_assignment(cost)
    assignment = list(zip(rows.tolist(), cols.tolist()))
    assigned = {scans[i]["id"]: sources[j]["id"] for i, j in assignment}

    output_dir: Path = args.output
    matches_dir = output_dir / "matches"
    matches_dir.mkdir(parents=True, exist_ok=True)

    assignments = []
    review = []
    print("\nHungarian assignment:")
    for i, j in assignment:
        pair = pairs[i][j]
        margin = second_best_margin(scores, i, j)
        confidence = float(min(1.0, 0.35 * pair["inlier_ratio"] + 0.45 * pair["iou"] + 0.20 * margin))
        status = "accepted" if pair["usable"] and confidence >= 0.45 and pair["inliers"] >= 12 else "review"
        record = {
            "scan_piece_id": scans[i]["id"],
            "source_piece_id": sources[j]["id"],
            "rotation_deg": float(math.degrees(math.atan2(
                np.asarray(pair["H_piece_to_source"])[1, 0],
                np.asarray(pair["H_piece_to_source"])[0, 0],
            ))) if pair["H_piece_to_source"] else None,
            "homography_piece_to_source": pair["H_piece_to_source"],
            "confidence": confidence,
            "inliers": pair["inliers"],
            "inlier_ratio": pair["inlier_ratio"],
            "median_reprojection_error_px": pair["median_reprojection_error_px"],
            "iou": pair["iou"],
            "scale": pair["scale"],
            "second_best_margin": margin,
            "score": pair["score"],
            "status": status if pair["H_piece_to_source"] is not None else "failed",
            "reason": pair["reason"],
        }
        print(
            f"  {record['scan_piece_id']} -> {record['source_piece_id']}  "
            f"inliers={record['inliers']:3d}  iou={record['iou']:.2f}  "
            f"conf={confidence:.2f}  {record['status']}"
        )
        if record["status"] != "accepted":
            review.append(record["scan_piece_id"])
        assignments.append(record)
        vis = draw_matches_image(scans[i], sources[j], pair)
        imwrite(matches_dir / f"{scans[i]['id']}_{sources[j]['id']}.png", vis)

    failed = [a for a in assignments if a["homography_piece_to_source"] is None]
    if failed:
        raise RuntimeError(f"no usable homography for {[a['scan_piece_id'] for a in failed]}")

    reference = imread(args.reference)
    assembled, overlay = reconstruct(scans, sources, assignment, pairs, size)
    comparison = side_by_side(reference, assembled)
    metrics = quality_vs_reference(assembled, reference)
    print("\nreconstruction vs reference.png:")
    print(json.dumps(metrics, indent=2))

    result = {
        "schema_version": "1.0",
        "algorithm": "SIFT+RANSAC+Hungarian",
        "algorithm_version": VERSION,
        "ratio_test": RATIO,
        "ransac_reprojection_px": RANSAC_PX,
        "assignments": assignments,
        "unassigned": [],
        "review": review,
        "quality": metrics,
        "score_matrix": {
            "scan_ids": [s["id"] for s in scans],
            "source_ids": [s["id"] for s in sources],
            "scores": scores.tolist(),
        },
    }
    write_json(output_dir / "assignments.json", result)
    write_json(output_dir / "report.json", {"quality": metrics, "review": review, "assignments": [
        {k: a[k] for k in ("scan_piece_id", "source_piece_id", "inliers", "iou", "confidence", "status")}
        for a in assignments
    ]})
    imwrite(output_dir / "cost_matrix.png", draw_cost_matrix(scores, scans, sources, assigned))
    imwrite(output_dir / "reconstructed.png", assembled)
    imwrite(output_dir / "overlay.png", overlay)
    imwrite(output_dir / "comparison.png", comparison)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SIFT-map Stage 2 pieces back onto the Stage 1 reference image.")
    parser.add_argument("--pieces", type=Path, default=ROOT / "stage2_piece_extraction/output/pieces.json")
    parser.add_argument("--layout", type=Path, default=ROOT / "stage1_capture_calibration/output/cut_pdf/jigsaw_layout.json")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "stage1_capture_calibration/output/reference/source_pieces")
    parser.add_argument("--reference", type=Path, default=ROOT / "stage1_capture_calibration/output/reference/reference.png")
    parser.add_argument("--output", type=Path, default=ROOT / "stage3_feature_matching/output")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
