"""
Coordinate matching utilities (simplified)
Used to detect containment relationships (tooth box in quadrant box,
disease box in tooth box, etc.).
"""

import logging
from typing import List, Dict, Tuple, Any  # noqa: F401 List used in _filter_and_dedup_teeth_by_fdi

logger = logging.getLogger(__name__)


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute IoU (Intersection over Union) of two boxes.

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        IoU value in [0, 1].
    """
    if len(box1) == 4 and box1[2] < box1[0]:  # [x, y, w, h] layout
        box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]

    if len(box2) == 4 and box2[2] < box2[0]:
        box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]

    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def generate_fdi_notation(quadrant_name: str, tooth_number: str) -> str:
    """
    Build the full FDI notation.

    Args:
        quadrant_name: quadrant label (Upperright / Upperleft / Lowerleft / Lowerright or Q1-Q4)
        tooth_number: tooth position (1-8)

    Returns:
        Full FDI notation (e.g. "11", "48").
    """
    quadrant_mapping = {
        "Upperright": "1",
        "Upperleft": "2",
        "Lowerleft": "3",
        "Lowerright": "4",
        "Q1": "1",
        "Q2": "2",
        "Q3": "3",
        "Q4": "4"
    }
    
    quadrant_digit = quadrant_mapping.get(quadrant_name, "0")
    return f"{quadrant_digit}{tooth_number}"


def _intersection_area(box1: List[float], box2: List[float]) -> float:
    """Compute the intersection area of two bboxes."""
    if not box1 or not box2 or len(box1) != 4 or len(box2) != 4:
        return 0.0
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def _box_mask_intersection_area(bbox: List[float], mask_contour: List[List[float]]) -> float:
    """Compute the intersection area of a bbox (rectangle) and a mask (polygon)."""
    if not bbox or len(bbox) != 4 or not mask_contour or len(mask_contour) < 3:
        return 0.0
    try:
        from shapely.geometry import Polygon, box as shapely_box
        rect = shapely_box(bbox[0], bbox[1], bbox[2], bbox[3])
        poly = Polygon([(float(p[0]), float(p[1])) for p in mask_contour])
        if not poly.is_valid:
            poly = poly.buffer(0)
        return float(rect.intersection(poly).area)
    except Exception as e:
        logger.debug(f"box-mask intersection failed: {e}")
        return 0.0


def assign_teeth_to_quadrants(
    teeth: Dict[str, Dict],
    quadrants: Dict[str, Dict],
    iou_threshold: float = 0.0,  # compatibility parameter
) -> Dict[str, str]:
    """
    Assign teeth to quadrants using a combined bbox + mask intersection score.
    - If the quadrant has a mask_contour, score = bbox_intersection + mask_intersection.
    - Otherwise use bbox_intersection only.
    Highest-scoring quadrant wins; teeth with no intersection go to "other_quad".
    """
    assignments = {}
    for tooth_id, tooth_data in teeth.items():
        tooth_box = tooth_data.get("box")
        if not tooth_box:
            continue
        best_quadrant = None
        best_score = 0.0
        for quadrant_id, quadrant_data in quadrants.items():
            quadrant_box = quadrant_data.get("box")
            if not quadrant_box:
                continue
            bbox_area = _intersection_area(tooth_box, quadrant_box)
            mask_area = _box_mask_intersection_area(tooth_box, quadrant_data.get("mask_contour"))
            score = bbox_area + mask_area
            if score > best_score:
                best_score = score
                best_quadrant = quadrant_id
        assignments[tooth_id] = best_quadrant if best_quadrant else "other_quad"
    return assignments


def _box_overlap_score(a: List[float], b: List[float]) -> float:
    """Overlap score = max(IoU, intersection / min(area_a, area_b)).
    The second term handles the case where a small box is fully enclosed
    in a larger one, which yields an artificially low IoU."""
    inter = _intersection_area(a, b)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0.0
    min_area = min(area_a, area_b)
    iomin = inter / min_area if min_area > 0 else 0.0
    return max(iou, iomin)


def _dedup_teeth_by_iou(teeth: Dict[str, Dict], iou_threshold: float = 0.9) -> Dict[str, Dict]:
    """
    All-pairs deduplication: when max(IoU, intersection / smaller-box area) > threshold,
    the pair is treated as the same tooth and the one with the smaller ``number`` (position 1-8) is kept.
    Called before FDI assignment so that duplicate detections do not affect downstream matching.
    """
    def _num(t: Dict) -> int:
        try:
            return int(t.get("number", 99))
        except (TypeError, ValueError):
            return 99

    items = [(tid, t) for tid, t in teeth.items() if t.get("box") and len(t["box"]) == 4]
    items.sort(key=lambda x: (_num(x[1]), str(x[0])))
    removed = set()
    for i in range(len(items)):
        if items[i][0] in removed:
            continue
        for j in range(i + 1, len(items)):
            if items[j][0] in removed:
                continue
            if _box_overlap_score(items[i][1]["box"], items[j][1]["box"]) > iou_threshold:
                removed.add(items[j][0])
    return {tid: t for tid, t in teeth.items() if tid not in removed}


def _filter_and_dedup_teeth_by_fdi(teeth: Dict[str, Dict]) -> Dict[str, Dict]:
    """Keep only valid FDI (1x-4x); all-pairs dedup: when max(IoU, intersection / smaller-box) > 0.9, keep the smaller FDI."""
    valid_prefix = {"1", "2", "3", "4"}
    valid_suffix = {str(i) for i in range(1, 9)}
    items: List[Dict] = []
    for tid, t in list(teeth.items()):
        fdi = str(t.get("fdi", ""))
        if len(fdi) == 2 and fdi[0] in valid_prefix and fdi[1] in valid_suffix and t.get("box"):
            items.append({"id": tid, "fdi": int(fdi), "box": t["box"]})
    items.sort(key=lambda x: x["fdi"])
    removed = set()
    for i in range(len(items)):
        if items[i]["id"] in removed:
            continue
        for j in range(i + 1, len(items)):
            if items[j]["id"] in removed:
                continue
            if _box_overlap_score(items[i]["box"], items[j]["box"]) > 0.9:
                removed.add(items[j]["id"])
    kept = {x["id"] for x in items if x["id"] not in removed}
    return {tid: t for tid, t in teeth.items() if tid in kept}


def build_fdi_teeth_like_refactor(
    quadrants: Dict[str, Dict],
    teeth: Dict[str, Dict],
) -> Dict[str, Dict]:
    """
    Compute FDI for every tooth, keep only valid FDI after deduplication, and
    return {fdi: {number, box, confidence}}.

    Pipeline:
    1. All-pairs dedup: max(IoU, intersection/smaller-box) > 0.9 means the same
       tooth; keep the one with the smaller position number.
    2. Assign teeth to quadrants using combined bbox + mask intersection.
    3. Generate FDI notation and drop invalid FDIs.
    """
    teeth = _dedup_teeth_by_iou(teeth, iou_threshold=0.9)
    tooth_to_quad = assign_teeth_to_quadrants(teeth, quadrants, iou_threshold=0.0)
    for tid, tooth in teeth.items():
        qid = tooth_to_quad.get(tid)
        if qid and qid != "other_quad":
            name = quadrants.get(qid, {}).get("name", qid)
            tooth["fdi"] = generate_fdi_notation(name, str(tooth.get("number", "1")))
        else:
            tooth["fdi"] = f"0{tooth.get('number', '0')}"
    filtered = _filter_and_dedup_teeth_by_fdi(teeth)
    teeth_by_fdi: Dict[str, Dict] = {}
    for _, tooth in filtered.items():
        fdi = str(tooth.get("fdi"))
        teeth_by_fdi[fdi] = {
            "number": fdi,
            "box": tooth.get("box") or [],
            "confidence": float(tooth.get("confidence", 0.0) or 0.0),
        }
    return teeth_by_fdi


def _box_center(box: List[float]) -> tuple:
    """Return the center point of a bbox."""
    if len(box) != 4:
        return (0.0, 0.0)
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _center_distance(box1: List[float], box2: List[float]) -> float:
    """Euclidean distance between the centers of two bboxes."""
    c1 = _box_center(box1)
    c2 = _box_center(box2)
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


# Disease classes that cannot be matched to a tooth (the tooth itself is gone
# or the class is not a tooth). Matching is skipped for these classes.
UNMATCHABLE_DISEASE_CLASSES = {
    "root piece",         # residual root (TVEM 11disease class 8)
    "missing teeth",      # missing tooth (TVEM 11disease class 5) — not a tooth target
    "mandibular canal",   # mandibular canal (TVEM 11disease class 4) — anatomy
    "maxillary sinus",    # maxillary sinus (TVEM 11disease class 10) — anatomy
}


def match_diseases_to_teeth(
    diseases: List[Dict],
    teeth: Dict[str, Dict],
    iou_threshold: float = 0.3
) -> Dict[str, List[Dict]]:
    """
    Match diseases to teeth using a two-strategy fallback.

    Strategies:
    1. IoU >= iou_threshold (works well for large targets such as Impacted).
    2. Nearest center-to-center distance (works for small targets such as
       Filling / Crown where IoU is unstable).
    Combined rule: prefer IoU match; fall back to nearest center distance.

    Note: some disease classes (e.g. Residual Root, Residual Crown) have no
    corresponding tooth and are filtered out instead of matched.

    Args:
        diseases: list like [{"box": [...], "class_name": "Filling", ...}]
        teeth: dict like {tooth_id: {"box": [...], "fdi": "11", ...}}
        iou_threshold: IoU threshold

    Returns:
        tooth -> disease mapping {tooth_id: [disease1, disease2, ...], "other_tooth": [...]}.
    """
    assignments = {tooth_id: [] for tooth_id in teeth}
    assignments["other_tooth"] = []

    for disease in diseases:
        disease_box = disease.get("box")
        if not disease_box:
            continue

        # Skip disease classes that can never be matched to a tooth
        class_name = (disease.get("class_name") or disease.get("class") or "").strip().lower()
        if class_name in UNMATCHABLE_DISEASE_CLASSES:
            continue

        best_tooth_iou = None
        best_iou = 0.0
        best_tooth_dist = None
        best_dist = float("inf")

        for tooth_id, tooth_data in teeth.items():
            tooth_box = tooth_data.get("box")
            if not tooth_box:
                continue

            # Strategy 1: IoU
            iou = calculate_iou(disease_box, tooth_box)
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_tooth_iou = tooth_id

            # Strategy 2: center distance
            dist = _center_distance(disease_box, tooth_box)
            if dist < best_dist:
                best_dist = dist
                best_tooth_dist = tooth_id

        if best_tooth_iou:
            assignments[best_tooth_iou].append(disease)
        elif best_tooth_dist:
            assignments[best_tooth_dist].append(disease)
        else:
            assignments["other_tooth"].append(disease)

    return assignments


def merge_detection_results(
    quadrants: Dict[str, Dict],
    teeth: Dict[str, Dict],
    diseases: List[Dict],
    iou_threshold: float = 0.3
) -> Dict[str, Any]:
    """
    Merge all detection results and build a hierarchical structure.

    Args:
        quadrants: quadrant detection result
        teeth: tooth detection result
        diseases: disease detection result
        iou_threshold: IoU threshold

    Returns:
        Merged structured data.
    """
    # 1. Assign teeth to quadrants
    tooth_to_quadrant = assign_teeth_to_quadrants(teeth, quadrants, iou_threshold)

    # 2. Generate full FDI notation
    for tooth_id, tooth_data in teeth.items():
        quadrant_id = tooth_to_quadrant.get(tooth_id)

        if quadrant_id and quadrant_id != "other_quad":
            quadrant_name = quadrants[quadrant_id].get("name", "Q1")
            tooth_number = tooth_data.get("number", "1")
            tooth_data["fdi"] = generate_fdi_notation(quadrant_name, tooth_number)
        else:
            tooth_data["fdi"] = f"0{tooth_data.get('number', '0')}"

    # 3. Assign diseases to teeth
    disease_to_tooth = match_diseases_to_teeth(diseases, teeth, iou_threshold)

    # 4. Build structured output
    result = {
        "quadrants": {},
        "teeth": {},
        "other_quad": {"teeth": []},
        "other_tooth": {"diseases": []}
    }

    # Populate quadrants
    for quadrant_id, quadrant_data in quadrants.items():
        quadrant_name = quadrant_data.get("name", f"Q{quadrant_id}")
        result["quadrants"][quadrant_name] = {
            "box": quadrant_data.get("box"),
            "confidence": quadrant_data.get("confidence"),
            "teeth": []
        }

    # Populate teeth and diseases
    for tooth_id, tooth_data in teeth.items():
        quadrant_id = tooth_to_quadrant.get(tooth_id)
        fdi = tooth_data.get("fdi", "00")

        rounded_box = [round(x, 1) for x in tooth_data.get("box", [])]

        tooth_info = {
            "fdi": fdi,
            "box": rounded_box,
            "confidence": tooth_data.get("confidence"),
            "diseases": disease_to_tooth.get(tooth_id, [])
        }

        result["teeth"][fdi] = tooth_info

        if quadrant_id and quadrant_id != "other_quad":
            quadrant_name = quadrants[quadrant_id].get("name", "Q1")
            result["quadrants"][quadrant_name]["teeth"].append(fdi)
        else:
            result["other_quad"]["teeth"].append(fdi)

    # Unmatched diseases
    result["other_tooth"]["diseases"] = disease_to_tooth.get("other_tooth", [])

    logger.info(f"Detection merge complete: {len(result['quadrants'])} quadrants, "
                f"{len(result['teeth'])} teeth")

    return result


# ==================== Shapely contour distance (extraction risk: tooth vs. maxillary sinus / mandibular canal) ====================

def _contour_to_polygon(contour: List) -> Any:
    """
    Convert a mask_contour ([[x, y], ...]) into a Shapely Polygon.
    Auto-closes the ring if the first/last points differ.
    """
    try:
        from shapely.geometry import Polygon
    except ImportError:
        return None
    if not contour or len(contour) < 3:
        return None
    coords = []
    for pt in contour:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            coords.append((float(pt[0]), float(pt[1])))
    if len(coords) < 3:
        return None
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    try:
        return Polygon(coords)
    except Exception:
        return None


def contour_min_distance_pixels(contour_a: List, contour_b: List) -> float:
    """
    Minimum pixel distance between two mask_contours.
    Returns infinity if either side cannot be turned into a Polygon.
    """
    poly_a = _contour_to_polygon(contour_a)
    poly_b = _contour_to_polygon(contour_b)
    if poly_a is None or poly_b is None or not poly_a.is_valid or not poly_b.is_valid:
        return float("inf")
    try:
        return float(poly_a.distance(poly_b))
    except Exception:
        return float("inf")
