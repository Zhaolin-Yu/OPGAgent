"""
坐标匹配工具（简化版）
用于检测包含关系（牙齿 box 在象限 box 内，疾病 box 在牙齿 box 内等）
"""

import logging
from typing import List, Dict, Tuple, Any  # noqa: F401 List used in _filter_and_dedup_teeth_by_fdi

logger = logging.getLogger(__name__)


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    计算两个框的 IoU (Intersection over Union)
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        IoU 值 (0-1)
    """
    # 统一转换为 [x1, y1, x2, y2] 格式
    if len(box1) == 4 and box1[2] < box1[0]:  # 如果是 [x, y, w, h] 格式
        box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
    
    if len(box2) == 4 and box2[2] < box2[0]:
        box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]
    
    # 计算交集区域
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # 如果没有交集
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    # 计算交集面积
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # 计算各自面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算并集面积
    union_area = box1_area + box2_area - inter_area
    
    # 计算 IoU
    iou = inter_area / union_area if union_area > 0 else 0.0
    
    return iou


def generate_fdi_notation(quadrant_name: str, tooth_number: str) -> str:
    """
    生成完整的 FDI 编号
    
    Args:
        quadrant_name: 象限名称 (Upperright, Upperleft, Lowerleft, Lowerright 或 Q1-Q4)
        tooth_number: 牙齿编号 (1-8)
        
    Returns:
        完整 FDI 编号 (如 "11", "48")
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


def assign_teeth_to_quadrants(
    teeth: Dict[str, Dict],
    quadrants: Dict[str, Dict],
    iou_threshold: float = 0.0
) -> Dict[str, str]:
    """
    将牙齿分配到象限（与 Agent_refactor 一致：取 IoU 最大的象限，不设阈值）。
    iou_threshold 保留用于兼容，0 表示始终分配最佳象限。
    """
    assignments = {}
    for tooth_id, tooth_data in teeth.items():
        tooth_box = tooth_data.get("box")
        if not tooth_box:
            continue
        best_quadrant = None
        best_iou = 0.0
        for quadrant_id, quadrant_data in quadrants.items():
            quadrant_box = quadrant_data.get("box")
            if not quadrant_box:
                continue
            iou = calculate_iou(tooth_box, quadrant_box)
            if iou > best_iou:
                best_iou = iou
                best_quadrant = quadrant_id
        assignments[tooth_id] = best_quadrant if best_quadrant else "other_quad"
    return assignments


def _filter_and_dedup_teeth_by_fdi(teeth: Dict[str, Dict]) -> Dict[str, Dict]:
    """与 Agent_refactor 一致：只保留有效 FDI（1x–4x），并按 FDI 去重（IoU>0.9 保留编号更小）。"""
    valid_prefix = {"1", "2", "3", "4"}
    valid_suffix = {str(i) for i in range(1, 9)}
    items: List[Dict] = []
    for tid, t in list(teeth.items()):
        fdi = str(t.get("fdi", ""))
        if len(fdi) == 2 and fdi[0] in valid_prefix and fdi[1] in valid_suffix:
            items.append({"id": tid, "fdi": int(fdi), "box": t.get("box")})
    items.sort(key=lambda x: x["fdi"])
    i = 0
    while i < len(items) - 1:
        cur, nxt = items[i], items[i + 1]
        if not cur.get("box") or not nxt.get("box"):
            i += 1
            continue
        if calculate_iou(cur["box"], nxt["box"]) > 0.9:
            items.pop(i + 1)
        else:
            i += 1
    kept = {x["id"] for x in items}
    return {tid: t for tid, t in teeth.items() if tid in kept}


def build_fdi_teeth_like_refactor(
    quadrants: Dict[str, Dict],
    teeth: Dict[str, Dict],
) -> Dict[str, Dict]:
    """
    与 Agent_refactor build_fdi_teeth 一致：为每颗牙算 FDI，过滤仅保留有效 FDI 并去重，
    返回 {fdi: {number, box, confidence}}。
    """
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
    """计算 bbox 中心点"""
    if len(box) != 4:
        return (0.0, 0.0)
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _center_distance(box1: List[float], box2: List[float]) -> float:
    """计算两个 bbox 中心点的欧氏距离"""
    c1 = _box_center(box1)
    c2 = _box_center(box2)
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


# 无法匹配到牙齿的疾病类型（因为牙齿本身已不存在，只剩残留部分，或非牙齿目标）
# 这些疾病会被过滤，不参与匹配
UNMATCHABLE_DISEASE_CLASSES = {
    "root piece",         # 残根（TVEM 11disease class 8）
    "missing teeth",      # 缺失牙（TVEM 11disease class 5）- 非牙齿目标
    "mandibular canal",   # 下颌管（TVEM 11disease class 4）- 解剖结构
    "maxillary sinus",    # 上颌窦（TVEM 11disease class 10）- 解剖结构
}


def match_diseases_to_teeth(
    diseases: List[Dict],
    teeth: Dict[str, Dict],
    iou_threshold: float = 0.3
) -> Dict[str, List[Dict]]:
    """
    将疾病匹配到牙齿（双重保险策略）。
    
    匹配策略：
    1. IoU >= iou_threshold（适用于大目标如 Impacted）
    2. 疾病框中心到牙齿框中心距离最近（适用于 Filling、Crown 等小目标）
    综合：优先 IoU 匹配，若 IoU 不满足则使用中心距离最近。
    
    注意：部分疾病类型（如 Residual Root、Residual Crown）因无对应牙齿，会被过滤跳过。
    
    Args:
        diseases: 疾病列表 [{"box": [...], "class_name": "Filling", ...}]
        teeth: 牙齿字典 {tooth_id: {"box": [...], "fdi": "11", ...}}
        iou_threshold: IoU 阈值
        
    Returns:
        牙齿到疾病的映射 {tooth_id: [disease1, disease2, ...], "other_tooth": [...]}
    """
    assignments = {tooth_id: [] for tooth_id in teeth}
    assignments["other_tooth"] = []
    
    for disease in diseases:
        disease_box = disease.get("box")
        if not disease_box:
            continue
        
        # 过滤无法匹配牙齿的疾病类型
        class_name = (disease.get("class_name") or disease.get("class") or "").strip().lower()
        if class_name in UNMATCHABLE_DISEASE_CLASSES:
            # 这些疾病不参与匹配，直接跳过（不放入 other_tooth）
            continue
        
        best_tooth_iou = None
        best_iou = 0.0
        best_tooth_dist = None
        best_dist = float("inf")
        
        for tooth_id, tooth_data in teeth.items():
            tooth_box = tooth_data.get("box")
            if not tooth_box:
                continue
            
            # 策略 1：IoU
            iou = calculate_iou(disease_box, tooth_box)
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_tooth_iou = tooth_id
            
            # 策略 2：中心距离
            dist = _center_distance(disease_box, tooth_box)
            if dist < best_dist:
                best_dist = dist
                best_tooth_dist = tooth_id
        
        # 综合判断：优先 IoU 匹配，否则使用中心距离最近
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
    合并所有检测结果，建立层级关系
    
    Args:
        quadrants: 象限检测结果
        teeth: 牙齿检测结果
        diseases: 疾病检测结果
        iou_threshold: IoU 阈值
        
    Returns:
        合并后的结构化数据
    """
    # 1. 分配牙齿到象限
    tooth_to_quadrant = assign_teeth_to_quadrants(teeth, quadrants, iou_threshold)
    
    # 2. 生成完整 FDI 编号
    for tooth_id, tooth_data in teeth.items():
        quadrant_id = tooth_to_quadrant.get(tooth_id)
        
        if quadrant_id and quadrant_id != "other_quad":
            quadrant_name = quadrants[quadrant_id].get("name", "Q1")
            tooth_number = tooth_data.get("number", "1")
            tooth_data["fdi"] = generate_fdi_notation(quadrant_name, tooth_number)
        else:
            tooth_data["fdi"] = f"0{tooth_data.get('number', '0')}"
    
    # 3. 分配疾病到牙齿
    disease_to_tooth = match_diseases_to_teeth(diseases, teeth, iou_threshold)
    
    # 4. 构建结构化数据
    result = {
        "quadrants": {},
        "teeth": {},
        "other_quad": {"teeth": []},
        "other_tooth": {"diseases": []}
    }
    
    # 添加象限数据
    for quadrant_id, quadrant_data in quadrants.items():
        quadrant_name = quadrant_data.get("name", f"Q{quadrant_id}")
        result["quadrants"][quadrant_name] = {
            "box": quadrant_data.get("box"),
            "confidence": quadrant_data.get("confidence"),
            "teeth": []
        }
    
    # 添加牙齿和疾病数据
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
    
    # 添加未匹配的疾病
    result["other_tooth"]["diseases"] = disease_to_tooth.get("other_tooth", [])
    
    logger.info(f"✓ 检测结果合并完成: {len(result['quadrants'])} 个象限, "
                f"{len(result['teeth'])} 颗牙齿")
    
    return result


# ==================== Shapely 轮廓距离（用于拔牙风险：牙齿与上颌窦/下颌管是否接近） ====================

def _contour_to_polygon(contour: List) -> Any:
    """
    将 mask_contour（[[x,y], ...]）转为 Shapely Polygon。
    若首尾不闭合则自动闭合。
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
    两段 mask_contour 之间的最小距离（像素）。
    若任一侧无法成 Polygon 则返回无穷大。
    """
    poly_a = _contour_to_polygon(contour_a)
    poly_b = _contour_to_polygon(contour_b)
    if poly_a is None or poly_b is None or not poly_a.is_valid or not poly_b.is_valid:
        return float("inf")
    try:
        return float(poly_a.distance(poly_b))
    except Exception:
        return float("inf")


def contours_near(contour_a: List, contour_b: List, max_pixels: float = 10.0) -> bool:
    """判断两段轮廓是否重合或接近（最小距离 < max_pixels 像素）。"""
    return contour_min_distance_pixels(contour_a, contour_b) < max_pixels
