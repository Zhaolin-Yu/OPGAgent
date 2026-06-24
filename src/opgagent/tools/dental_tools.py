"""
LangChain 工具定义
将牙科工具注册表转换为 LangChain Tool 格式
"""

import json
import logging
import os
import requests
import base64
import contextvars
import concurrent.futures
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path

# 当前 run 的真实图像路径（由 agent.run() 设置，工具优先使用此路径而非 LLM 传入的路径）
current_image_path_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "current_image_path", default=None
)

from langchain_core.tools import StructuredTool
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from .coordinate_utils import (
    merge_detection_results,
    match_diseases_to_teeth,
    build_fdi_teeth_like_refactor,
    contour_min_distance_pixels,
    contours_near,
    calculate_iou,
)

logger = logging.getLogger(__name__)

# 象限名与 FDI 前缀映射（Q1=1x, Q2=2x, Q3=3x, Q4=4x）
QUADRANT_TO_FDI_PREFIX = {"Upperright": "1", "Upperleft": "2", "Lowerleft": "3", "Lowerright": "4", "Q1": "1", "Q2": "2", "Q3": "3", "Q4": "4"}

# Quadrant name -> English description (for bone loss region description)
_QUADRANT_TO_EN = {
    "Upperright": "upper right",
    "Upperleft": "upper left",
    "Lowerleft": "lower left",
    "Lowerright": "lower right",
}


def _bone_loss_quadrants_to_description(quadrants: List[str]) -> str:
    """
    根据涉及的象限列表生成简洁英文描述。
    例如：lower left + lower right -> "lower region"; 仅 lower left -> "lower left quadrant"。
    """
    if not quadrants:
        return "none"
    qset = set(quadrants)
    en_list = sorted([_QUADRANT_TO_EN.get(q, q) for q in qset if _QUADRANT_TO_EN.get(q)])
    if not en_list:
        return ", ".join(sorted(qset))
    # 聚合：upper region = upper left + upper right, lower region = lower left + lower right
    # left side = upper left + lower left, right side = upper right + lower right
    upper = {"upper left", "upper right"}
    lower = {"lower left", "lower right"}
    left = {"upper left", "lower left"}
    right = {"upper right", "lower right"}
    en_set = set(en_list)
    # 单个象限时返回 "X quadrant"
    if len(en_set) == 1:
        return f"{list(en_set)[0]} quadrant"
    if en_set == lower:
        return "lower region"
    if en_set == upper:
        return "upper region"
    if en_set == left:
        return "left side"
    if en_set == right:
        return "right side"
    return ", ".join(sorted(en_list))


# 与 Agent_refactor preprocess._QUADRANT_MAPPING 一致，保证 detections 与 reference 一致
_QUADRANT_MAPPING = {
    "lower left": "Lowerright",
    "lowerleft": "Lowerright",
    "upper left": "Upperright",
    "upperleft": "Upperright",
    "upper right": "Lowerleft",
    "upperright": "Lowerleft",
    "lower right": "Upperleft",
    "lowerright": "Upperleft",
    "class_0": "Upperleft",
    "q1": "Upperright",
    "q2": "Upperleft",
    "q3": "Lowerleft",
    "q4": "Lowerright",
}


def _map_quadrant_name(name: str) -> str:
    """与 Agent_refactor 一致"""
    key = (name or "").strip().lower()
    if key in _QUADRANT_MAPPING:
        return _QUADRANT_MAPPING[key]
    key2 = key.replace(" ", "")
    for k, v in _QUADRANT_MAPPING.items():
        if k.replace(" ", "") == key2:
            return v
    if name in {"Upperright", "Upperleft", "Lowerleft", "Lowerright"}:
        return name
    return name


def _normalize_quadrants_for_merge(quadrants: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    将象限检测 API 响应（含 detections 列表）转换为 merge 期望的格式。
    使用与 Agent_refactor 一致的象限名称映射，保证 teeth_fdi 与 reference 一致。
    """
    if not isinstance(quadrants, dict):
        return {}
    if "detections" in quadrants:
        out = {}
        for idx, det in enumerate(quadrants.get("detections", [])):
            if not isinstance(det, dict):
                continue
            raw_name = det.get("class_name") or det.get("class") or f"quadrant_{idx}"
            name = _map_quadrant_name(raw_name)
            if name not in {"Upperright", "Upperleft", "Lowerleft", "Lowerright"}:
                continue
            bbox = det.get("bbox") or det.get("box") or []
            if isinstance(bbox, dict):
                bbox = [bbox.get("x1", 0), bbox.get("y1", 0), bbox.get("x2", 0), bbox.get("y2", 0)]
            out[name] = {"name": name, "box": bbox, "confidence": det.get("confidence", 0.0)}
        return out
    return quadrants


def _normalize_teeth_for_merge(teeth: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    将牙齿检测 API 响应（含 detections 列表）转换为 merge_detection_results 期望的格式。
    Agent 传入的往往是 tooth_enumeration 的原始 JSON，需先转换。
    """
    if not isinstance(teeth, dict):
        return {}
    if "detections" in teeth:
        out = {}
        for idx, det in enumerate(teeth.get("detections", [])):
            if not isinstance(det, dict):
                continue
            tooth_id = f"t{idx}"
            class_name = det.get("class_name") or det.get("class") or str(idx + 1)
            tooth_number = str(class_name)
            bbox = det.get("bbox") or det.get("box") or []
            if isinstance(bbox, dict):
                bbox = [bbox.get("x1", 0), bbox.get("y1", 0), bbox.get("x2", 0), bbox.get("y2", 0)]
            out[tooth_id] = {"number": tooth_number, "box": bbox, "confidence": det.get("confidence", 0.0)}
        return out
    return teeth


# ==================== 工具输入 Schema ====================

class QuadrantDetectionInput(BaseModel):
    """象限检测输入"""
    image_path: str = Field(description="Path to OPG image")
    confidence_threshold: float = Field(default=0.5, description="Confidence threshold (mapped to confidence)")


class ToothEnumerationInput(BaseModel):
    """牙齿枚举输入"""
    image_path: str = Field(description="Path to OPG image")


class DiseaseDetectionTVEMInput(BaseModel):
    """TVEM 疾病检测输入"""
    image_path: str = Field(description="Path to OPG image")
    confidence: float = Field(default=0.5, description="Confidence threshold")
    return_vis: bool = Field(default=False, description="Return visualization")


class BoneLossDetectionInput(BaseModel):
    """骨吸收检测输入（与 Agent_refactor 一致，默认 0.5）"""
    image_path: str = Field(description="Path to OPG image")
    confidence: float = Field(default=0.5, description="Confidence threshold")
    return_vis: bool = Field(default=False, description="Return visualization")


class AnatomyDetectionInput(BaseModel):
    """解剖结构检测输入"""
    image_path: str = Field(description="Path to OPG image")
    confidence: float = Field(default=0.5, description="Confidence threshold")
    return_vis: bool = Field(default=False, description="Return visualization")


class SegmentObjectInput(BaseModel):
    """对象分割输入"""
    image_path: str = Field(description="Path to image")
    boxes: List[List[float]] = Field(description="List of bbox [[x1,y1,x2,y2], ...]")


class DentalExpertAnalysisInput(BaseModel):
    """牙科专家分析输入"""
    image_path: str = Field(description="Path to image")
    analysis_type: str = Field(description="Analysis type: overall/quadrant/tooth/custom")
    target_id: Optional[str] = Field(default=None, description="Target ID (quadrant Q1-Q4 or FDI e.g. 48)")
    fdi: Optional[str] = Field(default=None, description="FDI tooth number (e.g. 48), same as target_id")
    quadrant: Optional[str] = Field(default=None, description="Quadrant name (Q1-Q4), same as target_id")
    custom_prompt: Optional[str] = Field(default=None, description="Custom prompt when analysis_type=custom")
    detected_findings: Optional[List[str]] = Field(default=None, description="Pre-detected findings list")
    focus_areas: Optional[List[str]] = Field(default=None, description="Focus: caries/periapical/periodontal/impaction/restoration/anatomy/bone_loss/root_canal")


class RunAllDetectionsInput(BaseModel):
    """一键运行完整检测管道输入（与 reference detections 顺序一致）"""
    image_path: str = Field(description="Path to OPG image")


class GetToothByFDIInput(BaseModel):
    """按 FDI 获取单颗牙信息"""
    image_path: str = Field(description="Path to OPG image")
    fdi: str = Field(description="FDI tooth number, e.g. 11, 18, 48")
    detections_json: Optional[str] = Field(default=None, description="Optional JSON from run_all_detections; omit to run detection pipeline")


class GetQuadrantInput(BaseModel):
    """按象限名获取一个或多个象限信息"""
    image_path: str = Field(description="Path to OPG image")
    quadrant_names: str = Field(description="One or more quadrants, comma-separated. E.g. 'Q1', 'Q1,Q2', 'Q1,Q2,Q3,Q4' or 'Upperright,Lowerleft'")
    detections_json: Optional[str] = Field(default=None, description="Optional JSON from run_all_detections")


class GetToothMaskInput(BaseModel):
    """获取某颗牙的 mask（MedSAM 分割）"""
    image_path: str = Field(description="Path to OPG image")
    fdi: str = Field(description="FDI tooth number")
    detections_json: Optional[str] = Field(default=None, description="Optional JSON from run_all_detections")


class GetStatusOnToothInput(BaseModel):
    """获取某颗牙上的状态列表（TVEM 匹配结果，不含 YOLO Caries/Deep Caries）"""
    image_path: str = Field(description="Path to OPG image")
    fdi: str = Field(description="FDI tooth number")
    detections_json: Optional[str] = Field(default=None, description="Optional JSON from run_all_detections")


class ExtractionRiskNearAnatomyInput(BaseModel):
    """拔牙风险：该牙与上颌窦/下颌管是否接近（Shapely 检查 mask 轮廓距离）"""
    image_path: str = Field(description="Path to OPG image")
    fdi: str = Field(description="FDI tooth number (18/28: maxillary sinus, 38/48: mandibular canal)")
    proximity_pixels: float = Field(default=10.0, description="Proximity threshold in pixels")
    detections_json: Optional[str] = Field(default=None, description="Optional JSON from run_all_detections")


class GetQuadrantTeethInput(BaseModel):
    """获取某象限内所有牙齿的 FDI 及信息"""
    image_path: str = Field(description="Path to OPG image")
    quadrant_name: str = Field(description="Quadrant: Upperright/Upperleft/Lowerleft/Lowerright or Q1-Q4")
    detections_json: Optional[str] = Field(default=None, description="Optional JSON from run_all_detections")


class ListTeethWithStatusInput(BaseModel):
    """列出具有某类状态的所有牙位（FDI）- 不含 YOLO Caries/Deep Caries"""
    image_path: str = Field(description="Path to OPG image")
    status_class: Optional[str] = Field(default=None, description="Status class to filter (e.g. Filling, Crown, impacted tooth). If empty or 'all', returns all detected statuses grouped by class.")
    detections_json: Optional[str] = Field(default=None, description="Optional JSON from run_all_detections")


class GetBoneLossDescriptionInput(BaseModel):
    """骨吸收区域描述：根据涉及象限/牙齿生成「下方、左下、右下」等描述"""
    image_path: str = Field(description="Path to OPG image")
    detections_json: Optional[str] = Field(default=None, description="Optional JSON from run_all_detections")
    iou_threshold: float = Field(default=0.1, description="IoU threshold for bone-loss box vs quadrant/tooth overlap")


class GetAnnotatedImageInput(BaseModel):
    """生成带标注的图像：裁剪区域或带 bbox 的整张 OPG"""
    image_path: str = Field(description="Path to OPG image")
    target_type: str = Field(description="Target type: 'tooth' or 'quadrant'")
    target_id: str = Field(description="Target ID: FDI number (e.g. 11, 48) for tooth, or quadrant name (Q1-Q4 / Upperright/Upperleft/Lowerleft/Lowerright) for quadrant")
    output_mode: str = Field(default="bbox_overlay", description="Output mode: 'crop' (cropped region only) or 'bbox_overlay' (full OPG with bbox drawn)")
    detections_json: Optional[str] = Field(default=None, description="Optional JSON from run_all_detections")


class ResolveFindingDisagreementInput(BaseModel):
    """解决疾病位置/分类不一致的问题"""
    image_path: str = Field(description="Path to OPG image")
    finding_type: str = Field(description="Type of finding, e.g. 'implant', 'bone_loss', 'periapical_lesion', 'filling', 'crown', 'rct'")
    disagreement_type: str = Field(description="Type of disagreement: 'position' (FDI location varies) or 'classification' (severity/type varies)")
    vlm_opinions: str = Field(description="JSON array of VLM opinions, each with {source, opinion, position_or_classification}. E.g. [{\"source\": \"DentalGPT\", \"opinion\": \"implant present\", \"position_or_classification\": \"lower left 36\"}, ...]")
    gold_standard_info: str = Field(description="JSON with gold standard info: {teeth_fdi: [...], quadrants: {...}, not_detected: [...]}") 
    confirmed_findings: Optional[str] = Field(default=None, description="Optional JSON of already confirmed findings for context")


class CalculateFDIInput(BaseModel):
    """计算 FDI 输入"""
    quadrants: Dict = Field(description="Quadrant detection result")
    teeth: Dict = Field(description="Tooth detection result")


class MatchDiseaseToToothInput(BaseModel):
    """疾病匹配输入"""
    diseases: List[Dict] = Field(description="Disease detection result")
    teeth: Dict = Field(description="Teeth with FDI (teeth_fdi)")
    iou_threshold: float = Field(default=0.3, description="IoU threshold")


class LLMZooAnalysisInput(BaseModel):
    """LLM Zoo 分析输入"""
    image_path: str = Field(description="Path to image")
    task_type: str = Field(
        default="analysis",
        description="Task: analysis/verification/cross_check/second_opinion/summarize"
    )
    target_fdi: Optional[str] = Field(default=None, description="Target FDI tooth number")
    custom_prompt: Optional[str] = Field(default=None, description="Custom prompt")
    detection_findings: Optional[Any] = Field(default=None, description="Detection findings (dict or str)")
    dentist_analysis: Optional[Any] = Field(default=None, description="Dentist model analysis (dict or str)")
    focus: Optional[str] = Field(default=None, description="Focus description")
    analysis_level: str = Field(
        default="overall",
        description="Analysis level: 'overall' (full OPG, max 2048 tokens), 'quadrant' (max 1024 tokens), 'tooth' (max 256 tokens)"
    )


class ConvertToStructuredInput(BaseModel):
    """Convert natural language report to structured JSON"""
    natural_report: str = Field(description="Natural language diagnostic report to convert")
    schema_path: Optional[str] = Field(default=None, description="Optional path to Schema&Enum_standard.md file")


# ==================== 工具实现 ====================

class DentalToolkit:
    """牙科工具集"""

    def __init__(self, tools_config: Dict[str, Any]):
        """
        初始化工具集

        Args:
            tools_config: 工具配置（从 tools_config.yaml 加载）
        """
        self.config = tools_config
        # 运行期缓存：同一张图在一次 Agent run 内，多次高级工具调用时避免重复请求基础工具
        # key: resolved_image_path, value: { "quadrants": {...}, "teeth": {...}, "teeth_fdi": {...}, ... }
        self._runtime_cache: Dict[str, Dict[str, Any]] = {}
        # 轮询计数器：为每个启用负载均衡的服务维护独立计数器
        self._round_robin_counters: Dict[str, int] = {}
        logger.info("✓ DentalToolkit 初始化")

    def _call_remote_service(
        self,
        service_name: str,
        endpoint: str,
        data: Dict[str, Any],
        timeout: int = 120,
        file_field: str = "file"
    ) -> Dict[str, Any]:
        """调用远程服务"""
        # 优先从 tools.*.service.base_url 读取（tools_config.yaml 的标准结构）
        # 兼容直接 service_name 在顶层的情况
        service_config = self.config.get("tools", {}).get(service_name, {}) or self.config.get(service_name, {})
        service_info = service_config.get("service", {})
        base_url = service_info.get("base_url")
        if not base_url:
            logger.warning(f"服务 {service_name} 未配置 base_url，请检查 tools_config.yaml")
            return {"error": f"服务 {service_name} 未配置，请在 tools_config.yaml 中添加配置"}
        
        # 处理多副本负载均衡（轮询策略）
        if "load_balancing" in service_info and service_info["load_balancing"].get("enabled"):
            endpoints = service_info["load_balancing"].get("endpoints", [base_url])
            if endpoints:
                # 轮询：每次调用递增计数器，取模选择下一个端点
                counter = self._round_robin_counters.get(service_name, 0)
                base_url = endpoints[counter % len(endpoints)]
                self._round_robin_counters[service_name] = counter + 1
        
        url = f"{base_url}{endpoint}"
        
        try:
            # 根据数据类型选择请求方式
            if "image_path" in data:
                image_path = data.pop("image_path")
                # 优先使用 context 注入的真实路径（agent.run 设置），避免 LLM 传入错误路径
                resolved = current_image_path_ctx.get()
                if resolved and os.path.isfile(resolved):
                    image_path = resolved
                if not os.path.isfile(image_path):
                    raise FileNotFoundError(f"图像文件不存在: {image_path}")
                with open(image_path, "rb") as f:
                    files = {file_field: f}
                    form_data = {k: str(v) for k, v in data.items()}
                    response = requests.post(url, files=files, data=form_data, timeout=timeout)
            else:
                response = requests.post(url, json=data, timeout=timeout)
            
            response.raise_for_status()
            return response.json()
            
        except requests.RequestException as e:
            logger.error(f"调用 {service_name} 失败: {e}")
            return {"error": str(e)}
    
    # ==================== 检测类工具 ====================
    
    def _resolve_image_path(self, image_path: str, config: RunnableConfig = None) -> str:  # noqa: ARG002
        """从 config 或 context 解析真实图像路径，避免 LLM 传入错误路径"""
        if config:
            path = config.get("configurable", {}).get("current_image_path")
            if path and os.path.isfile(path):
                return path
        resolved = current_image_path_ctx.get()
        if resolved and os.path.isfile(resolved):
            return resolved
        return image_path

    def _resize_image_short_edge(self, image_path: str, short_edge: int = 768) -> bytes:
        """
        将图像缩放到短边为 short_edge 像素，返回 PNG 字节流。
        用于 VLM 工具调用前预处理图像（统一尺寸，减少传输量）。
        """
        from PIL import Image
        import io
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        if min(w, h) <= short_edge:
            # 图像已经足够小，不缩放
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        # 按短边缩放
        if w < h:
            new_w = short_edge
            new_h = int(h * short_edge / w)
        else:
            new_h = short_edge
            new_w = int(w * short_edge / h)
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        img_resized.save(buf, format="PNG")
        return buf.getvalue()

    def _resize_pil_image_short_edge(self, pil_img, short_edge: int = 768):
        """
        将 PIL.Image 对象缩放到短边为 short_edge 像素，返回缩放后的 PIL.Image。
        用于 get_annotated_image 等需要在内存中处理图像的场景。
        """
        from PIL import Image
        w, h = pil_img.size
        if min(w, h) <= short_edge:
            return pil_img
        if w < h:
            new_w = short_edge
            new_h = int(h * short_edge / w)
        else:
            new_h = short_edge
            new_w = int(w * short_edge / h)
        return pil_img.resize((new_w, new_h), Image.LANCZOS)

    def _cache_key(self, image_path: str, config: RunnableConfig = None) -> str:
        """生成缓存 key（使用解析后的绝对路径），避免同图不同相对路径导致重复计算"""
        resolved = self._resolve_image_path(image_path, config)
        try:
            return str(Path(resolved).resolve())
        except Exception:
            return str(resolved)

    def _cache_get(self, image_path: str, key: str, config: RunnableConfig = None) -> Optional[Any]:
        cache = self._runtime_cache.get(self._cache_key(image_path, config), {})
        return cache.get(key)

    def _cache_set(self, image_path: str, key: str, value: Any, config: RunnableConfig = None) -> Any:
        ck = self._cache_key(image_path, config)
        if ck not in self._runtime_cache:
            self._runtime_cache[ck] = {}
        self._runtime_cache[ck][key] = value
        return value

    def _parse_detections_json(self, detections_json: Optional[str]) -> Optional[Dict[str, Any]]:
        """解析外部传入的 detections_json（用于复现/对照）；失败则返回 None"""
        if not detections_json or not detections_json.strip():
            return None
        try:
            parsed = json.loads(detections_json)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    def _ensure_quadrants(self, image_path: str, detections_json: Optional[str], config: RunnableConfig = None) -> Dict[str, Any]:
        """确保 quadrants 可用（优先 detections_json，其次缓存，其次调用 quadrant_detection）"""
        det = self._parse_detections_json(detections_json)
        if det and isinstance(det.get("quadrants"), dict):
            return self._cache_set(image_path, "quadrants", det["quadrants"], config)
        cached = self._cache_get(image_path, "quadrants", config)
        if isinstance(cached, dict):
            return cached
        data = json.loads(self.quadrant_detection(image_path, confidence_threshold=0.5, config=config))
        return self._cache_set(image_path, "quadrants", data, config)

    def _ensure_teeth(self, image_path: str, detections_json: Optional[str], config: RunnableConfig = None) -> Dict[str, Any]:
        """确保 teeth 可用（优先 detections_json，其次缓存，其次调用 tooth_enumeration）"""
        det = self._parse_detections_json(detections_json)
        if det and isinstance(det.get("teeth"), dict):
            return self._cache_set(image_path, "teeth", det["teeth"], config)
        cached = self._cache_get(image_path, "teeth", config)
        if isinstance(cached, dict):
            return cached
        data = json.loads(self.tooth_enumeration(image_path, config=config))
        return self._cache_set(image_path, "teeth", data, config)

    def _ensure_teeth_fdi(self, image_path: str, detections_json: Optional[str], config: RunnableConfig = None) -> Dict[str, Any]:
        """确保 teeth_fdi 可用（优先 detections_json，其次缓存，其次按 preprocess 顺序计算）"""
        det = self._parse_detections_json(detections_json)
        if det and isinstance(det.get("teeth_fdi"), dict) and "error" not in det.get("teeth_fdi", {}):
            return self._cache_set(image_path, "teeth_fdi", det["teeth_fdi"], config)
        cached = self._cache_get(image_path, "teeth_fdi", config)
        if isinstance(cached, dict) and "error" not in cached:
            return cached
        quadrants_raw = self._ensure_quadrants(image_path, detections_json, config)
        teeth_raw = self._ensure_teeth(image_path, detections_json, config)
        # 若检测服务不可用，返回 error
        if isinstance(quadrants_raw, dict) and "error" in quadrants_raw:
            return self._cache_set(image_path, "teeth_fdi", {"error": f"quadrants: {quadrants_raw.get('error')}"}, config)
        if isinstance(teeth_raw, dict) and "error" in teeth_raw:
            return self._cache_set(image_path, "teeth_fdi", {"error": f"teeth: {teeth_raw.get('error')}"}, config)
        q_norm = _normalize_quadrants_for_merge(quadrants_raw)
        t_norm = _normalize_teeth_for_merge(teeth_raw)
        teeth_fdi = build_fdi_teeth_like_refactor(quadrants=q_norm, teeth=t_norm)
        return self._cache_set(image_path, "teeth_fdi", teeth_fdi, config)

    def _ensure_tvem_disease(self, image_path: str, detections_json: Optional[str], config: RunnableConfig = None) -> Dict[str, Any]:
        """确保 tvem_disease 可用"""
        det = self._parse_detections_json(detections_json)
        if det and isinstance(det.get("tvem_disease"), dict):
            return self._cache_set(image_path, "tvem_disease", det["tvem_disease"], config)
        cached = self._cache_get(image_path, "tvem_disease", config)
        if isinstance(cached, dict):
            return cached
        data = json.loads(self.disease_detection_tvem(image_path, confidence=0.5, return_vis=False, config=config))
        return self._cache_set(image_path, "tvem_disease", data, config)

    def _ensure_tvem_matched(self, image_path: str, detections_json: Optional[str], config: RunnableConfig = None) -> Dict[str, Any]:
        """确保 tvem_matched 可用（疾病匹配需 box 字段，若只有 bbox 则补齐）"""
        det = self._parse_detections_json(detections_json)
        if det and isinstance(det.get("tvem_matched"), dict) and "error" not in det.get("tvem_matched", {}):
            return self._cache_set(image_path, "tvem_matched", det["tvem_matched"], config)
        cached = self._cache_get(image_path, "tvem_matched", config)
        if isinstance(cached, dict) and "error" not in cached:
            return cached
        teeth_fdi = self._ensure_teeth_fdi(image_path, detections_json, config)
        tvem = self._ensure_tvem_disease(image_path, detections_json, config)
        dets: List[Dict[str, Any]] = []
        for d in (tvem.get("detections") or []):
            dd = dict(d)
            if "box" not in dd and "bbox" in dd:
                dd["box"] = dd["bbox"]
            dets.append(dd)
        matched = json.loads(self.match_disease_to_tooth(dets, teeth_fdi, iou_threshold=0.3))
        return self._cache_set(image_path, "tvem_matched", matched, config)

    def _ensure_anatomy(self, image_path: str, detections_json: Optional[str], config: RunnableConfig = None) -> Dict[str, Any]:
        """确保 anatomy 可用"""
        det = self._parse_detections_json(detections_json)
        if det and isinstance(det.get("anatomy"), dict):
            return self._cache_set(image_path, "anatomy", det["anatomy"], config)
        cached = self._cache_get(image_path, "anatomy", config)
        if isinstance(cached, dict):
            return cached
        data = json.loads(self.anatomy_detection(image_path, confidence=0.5, return_vis=False, config=config))
        return self._cache_set(image_path, "anatomy", data, config)

    def _ensure_bone_loss(self, image_path: str, detections_json: Optional[str], config: RunnableConfig = None) -> Dict[str, Any]:
        """确保 bone_loss 可用"""
        det = self._parse_detections_json(detections_json)
        if det and isinstance(det.get("bone_loss"), dict):
            return self._cache_set(image_path, "bone_loss", det["bone_loss"], config)
        cached = self._cache_get(image_path, "bone_loss", config)
        if isinstance(cached, dict):
            return cached
        data = json.loads(self.bone_loss_detection(image_path, confidence=0.5, return_vis=False, config=config))
        return self._cache_set(image_path, "bone_loss", data, config)

    def _parse_detections_json(self, detections_json: Optional[str]) -> Optional[Dict[str, Any]]:
        """解析外部传入的 detections_json（用于复现/对照）；失败则返回 None"""
        if not detections_json or not detections_json.strip():
            return None
        try:
            parsed = json.loads(detections_json)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    def _ensure_quadrants(self, image_path: str, detections_json: Optional[str], config: RunnableConfig = None) -> Dict[str, Any]:
        """确保 quadrants 可用（优先 detections_json，其次缓存，其次调用 quadrant_detection）"""
        det = self._parse_detections_json(detections_json)
        if det and isinstance(det.get("quadrants"), dict):
            return self._cache_set(image_path, "quadrants", det["quadrants"], config)
        cached = self._cache_get(image_path, "quadrants", config)
        if isinstance(cached, dict):
            return cached
        data = json.loads(self.quadrant_detection(image_path, confidence_threshold=0.5, config=config))
        return self._cache_set(image_path, "quadrants", data, config)

    def _ensure_teeth(self, image_path: str, detections_json: Optional[str], config: RunnableConfig = None) -> Dict[str, Any]:
        """确保 teeth 可用（优先 detections_json，其次缓存，其次调用 tooth_enumeration）"""
        det = self._parse_detections_json(detections_json)
        if det and isinstance(det.get("teeth"), dict):
            return self._cache_set(image_path, "teeth", det["teeth"], config)
        cached = self._cache_get(image_path, "teeth", config)
        if isinstance(cached, dict):
            return cached
        data = json.loads(self.tooth_enumeration(image_path, config=config))
        return self._cache_set(image_path, "teeth", data, config)

    def _ensure_teeth_fdi(self, image_path: str, detections_json: Optional[str], config: RunnableConfig = None) -> Dict[str, Any]:
        """确保 teeth_fdi 可用（优先 detections_json，其次缓存，其次按 preprocess 顺序计算）"""
        det = self._parse_detections_json(detections_json)
        if det and isinstance(det.get("teeth_fdi"), dict) and "error" not in det.get("teeth_fdi", {}):
            return self._cache_set(image_path, "teeth_fdi", det["teeth_fdi"], config)
        cached = self._cache_get(image_path, "teeth_fdi", config)
        if isinstance(cached, dict) and "error" not in cached:
            return cached
        quadrants_raw = self._ensure_quadrants(image_path, detections_json, config)
        teeth_raw = self._ensure_teeth(image_path, detections_json, config)
        # 若检测服务不可用，返回 error
        if isinstance(quadrants_raw, dict) and "error" in quadrants_raw:
            return self._cache_set(image_path, "teeth_fdi", {"error": f"quadrants: {quadrants_raw.get('error')}"}, config)
        if isinstance(teeth_raw, dict) and "error" in teeth_raw:
            return self._cache_set(image_path, "teeth_fdi", {"error": f"teeth: {teeth_raw.get('error')}"}, config)
        q_norm = _normalize_quadrants_for_merge(quadrants_raw)
        t_norm = _normalize_teeth_for_merge(teeth_raw)
        teeth_fdi = build_fdi_teeth_like_refactor(quadrants=q_norm, teeth=t_norm)
        return self._cache_set(image_path, "teeth_fdi", teeth_fdi, config)

    def _ensure_tvem_disease(self, image_path: str, detections_json: Optional[str], config: RunnableConfig = None) -> Dict[str, Any]:
        """确保 tvem_disease 可用"""
        det = self._parse_detections_json(detections_json)
        if det and isinstance(det.get("tvem_disease"), dict):
            return self._cache_set(image_path, "tvem_disease", det["tvem_disease"], config)
        cached = self._cache_get(image_path, "tvem_disease", config)
        if isinstance(cached, dict):
            return cached
        data = json.loads(self.disease_detection_tvem(image_path, confidence=0.5, return_vis=False, config=config))
        return self._cache_set(image_path, "tvem_disease", data, config)

    def _ensure_tvem_matched(self, image_path: str, detections_json: Optional[str], config: RunnableConfig = None) -> Dict[str, Any]:
        """确保 tvem_matched 可用（疾病匹配需 box 字段，若只有 bbox 则补齐）"""
        det = self._parse_detections_json(detections_json)
        if det and isinstance(det.get("tvem_matched"), dict) and "error" not in det.get("tvem_matched", {}):
            return self._cache_set(image_path, "tvem_matched", det["tvem_matched"], config)
        cached = self._cache_get(image_path, "tvem_matched", config)
        if isinstance(cached, dict) and "error" not in cached:
            return cached
        teeth_fdi = self._ensure_teeth_fdi(image_path, detections_json, config)
        tvem = self._ensure_tvem_disease(image_path, detections_json, config)
        dets = []
        for d in (tvem.get("detections") or []):
            dd = dict(d)
            if "box" not in dd and "bbox" in dd:
                dd["box"] = dd["bbox"]
            dets.append(dd)
        matched = json.loads(self.match_disease_to_tooth(dets, teeth_fdi, iou_threshold=0.3))
        return self._cache_set(image_path, "tvem_matched", matched, config)

    def _ensure_anatomy(self, image_path: str, detections_json: Optional[str], config: RunnableConfig = None) -> Dict[str, Any]:
        """确保 anatomy 可用"""
        det = self._parse_detections_json(detections_json)
        if det and isinstance(det.get("anatomy"), dict):
            return self._cache_set(image_path, "anatomy", det["anatomy"], config)
        cached = self._cache_get(image_path, "anatomy", config)
        if isinstance(cached, dict):
            return cached
        data = json.loads(self.anatomy_detection(image_path, confidence=0.5, return_vis=False, config=config))
        return self._cache_set(image_path, "anatomy", data, config)

    def _ensure_bone_loss(self, image_path: str, detections_json: Optional[str], config: RunnableConfig = None) -> Dict[str, Any]:
        """确保 bone_loss 可用"""
        det = self._parse_detections_json(detections_json)
        if det and isinstance(det.get("bone_loss"), dict):
            return self._cache_set(image_path, "bone_loss", det["bone_loss"], config)
        cached = self._cache_get(image_path, "bone_loss", config)
        if isinstance(cached, dict):
            return cached
        data = json.loads(self.bone_loss_detection(image_path, confidence=0.5, return_vis=False, config=config))
        return self._cache_set(image_path, "bone_loss", data, config)

    def quadrant_detection(self, image_path: str, confidence_threshold: float = 0.5, config: RunnableConfig = None) -> str:
        """检测 OPG 图像中的 4 个象限（Q1-Q4）"""
        image_path = self._resolve_image_path(image_path, config)
        # Agent_refactor TVEMClient 期望参数名为 confidence（不是 confidence_threshold）
        result = self._call_remote_service(
            "tvem",
            "/detect/4quadrants",
            {"image_path": image_path, "confidence": confidence_threshold, "return_vis": False}
        )
        return json.dumps(result, ensure_ascii=False)
    
    def tooth_enumeration(self, image_path: str, config: RunnableConfig = None) -> str:
        """检测 OPG 图像中每颗牙齿的位置和编号（1-8）"""
        image_path = self._resolve_image_path(image_path, config)
        result = self._call_remote_service(
            "yolo_enumeration",
            "/detect",
            {"image_path": image_path}
        )
        return json.dumps(result, ensure_ascii=False)
    
    def disease_detection_tvem(
        self,
        image_path: str,
        confidence: float = 0.5,
        return_vis: bool = False,
        config: RunnableConfig = None,
    ) -> str:
        """使用 TVEM 检测 11 类牙科疾病"""
        image_path = self._resolve_image_path(image_path, config)
        result = self._call_remote_service(
            "tvem",
            "/detect/11diseases",
            {
                "image_path": image_path,
                "confidence": confidence,
                "return_vis": return_vis
            }
        )
        # 修正 class_id -> class_name 映射（TVEM 服务使用 1-based category file，但模型输出 0-based class_id）
        # TVEM 11disease 的 COCO_CATEGORIES（1-based id），转换为 0-based class_id：
        # 0=Caries, 1=Crown, 2=Filling, 3=Implant, 4=Mandibular Canal, 5=Missing teeth,
        # 6=Periapical lesion, 7=Root Canal Treatment, 8=Root Piece, 9=impacted tooth, 10=maxillary sinus
        TVEM_11DISEASES_CLASS_MAP = {
            0: "Caries",
            1: "Crown",
            2: "Filling",
            3: "Implant",
            4: "Mandibular Canal",
            5: "Missing teeth",
            6: "Periapical lesion",
            7: "Root Canal Treatment",
            8: "Root Piece",
            9: "impacted tooth",
            10: "maxillary sinus",
        }
        if isinstance(result, dict) and "detections" in result:
            for d in result["detections"]:
                class_id = d.get("class_id")
                if class_id is not None and class_id in TVEM_11DISEASES_CLASS_MAP:
                    d["class_name"] = TVEM_11DISEASES_CLASS_MAP[class_id]
        return json.dumps(result, ensure_ascii=False)
    
    def bone_loss_detection(
        self,
        image_path: str,
        confidence: float = 0.5,
        return_vis: bool = False,
        config: RunnableConfig = None,
    ) -> str:
        """检测牙周病引起的骨吸收区域"""
        image_path = self._resolve_image_path(image_path, config)
        result = self._call_remote_service(
            "tvem",
            "/detect/bone_loss",
            {
                "image_path": image_path,
                "confidence": confidence,
                "return_vis": return_vis
            }
        )
        return json.dumps(result, ensure_ascii=False)
    
    def anatomy_detection(
        self,
        image_path: str,
        confidence: float = 0.5,
        return_vis: bool = False,
        config: RunnableConfig = None,
    ) -> str:
        """检测重要的解剖结构（下颌管、上颌窦）"""
        image_path = self._resolve_image_path(image_path, config)
        result = self._call_remote_service(
            "tvem",
            "/detect/mandibular_maxillary",
            {
                "image_path": image_path,
                "confidence": confidence,
                "return_vis": return_vis
            }
        )
        return json.dumps(result, ensure_ascii=False)
    
    def segment_object(self, image_path: str, boxes: List[List[float]], config: RunnableConfig = None) -> str:
        """使用 MedSAM 对指定 bbox 内的目标进行精确分割
        
        注意：Agent_refactor MedSAMClient 只处理单个 bbox，这里处理第一个 box
        如需处理多个 boxes，需要多次调用此工具
        """
        image_path = self._resolve_image_path(image_path, config)
        if not boxes or len(boxes) == 0:
            return json.dumps({"error": "boxes cannot be empty"}, ensure_ascii=False)
        
        # Agent_refactor MedSAMClient 期望单个 bbox，格式为逗号分隔的字符串 "x1,y1,x2,y2"
        first_box = boxes[0]
        if len(first_box) != 4:
            return json.dumps({"error": f"bbox format error, expected 4 values [x1,y1,x2,y2], got {len(first_box)} values"}, ensure_ascii=False)
        
        bbox_str = ",".join(map(str, first_box))
        result = self._call_remote_service(
            "medsam",
            "/segment",
            {"image_path": image_path, "bbox": bbox_str}
        )
        return json.dumps(result, ensure_ascii=False)
    
    # ==================== 分析类工具 ====================
    
    def _build_dentist_prompt(
        self,
        analysis_type: str,
        target_id: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        detected_findings: Optional[List[str]] = None,
        focus_areas: Optional[List[str]] = None
    ) -> str:
        """
        Build Dentist prompt (unbiased, no tool detection results included).
        
        Important: detected_findings param is kept for logging but NOT included in prompt to avoid biasing the model.
        All prompts are in English. OPG contains only permanent teeth.
        """
        # Reminder about permanent teeth only
        PERMANENT_TEETH_REMINDER = "\n\nIMPORTANT: This OPG contains ONLY permanent teeth (no deciduous/primary teeth)."
        
        if analysis_type == "custom" and custom_prompt:
            return custom_prompt + PERMANENT_TEETH_REMINDER
        
        # OPG 方向说明
        OPG_ORIENTATION = """

OPG ORIENTATION (CRITICAL):
- Image RIGHT side = Patient's LEFT side
- Image LEFT side = Patient's RIGHT side
- FDI Q1 (upper right) & Q4 (lower right) = Patient's RIGHT = IMAGE LEFT
- FDI Q2 (upper left) & Q3 (lower left) = Patient's LEFT = IMAGE RIGHT
- ALL descriptions MUST use PATIENT orientation (not image orientation)!"""

        # 简洁回答指令
        CONCISE_INSTRUCTION = """

IMPORTANT INSTRUCTIONS:
- Your response will be sent to an AI Agent for synthesis with other VLM opinions.
- Be CONCISE. Only report CONFIRMED findings with HIGH CONFIDENCE.
- Do NOT report uncertain or ambiguous findings.
- If unsure, do NOT mention the finding at all.
- Keep response brief and factual."""

        # Base prompts (unbiased, English)
        # NOTE: DentalGPT/OralGPT 只推荐分析整张 OPG (analysis_type="overall")
        if analysis_type == "overall":
            prompt = (
                "You are a professional dentist and radiologist. Please analyze this OPG panoramic radiograph.\n\n"
                "Independently analyze the image and identify any abnormal findings.\n\n"
                "Pay special attention to periodontal assessment:\n"
                "- Evaluate alveolar bone loss severity: mild, moderate, or severe\n"
                "- Mild: alveolar crest 1-2mm from CEJ, or bone loss < 1/3 root length\n"
                "- Moderate: alveolar crest 3-4mm from CEJ, or bone loss 1/3-2/3 root length\n"
                "- Severe: alveolar crest > 4mm from CEJ, or bone loss > 2/3 root length\n"
                "- If no significant bone loss, state explicitly\n\n"
                "Output: Key abnormal findings organized by quadrant/tooth."
            )
        elif analysis_type == "quadrant":
            quadrant_map = {
                "Upperright": "upper right quadrant (FDI 11-18)",
                "Upperleft": "upper left quadrant (FDI 21-28)",
                "Lowerleft": "lower left quadrant (FDI 31-38)",
                "Lowerright": "lower right quadrant (FDI 41-48)"
            }
            quadrant_name = quadrant_map.get(target_id or "Upperright", target_id or "Q1")
            prompt = (
                f"You are a professional dentist. Please independently analyze the {quadrant_name} in this OPG image.\n\n"
                "Independently analyze the image and identify any abnormal findings.\n\n"
                "Output: Main radiographic findings and possible diagnoses for this quadrant."
            )
        elif analysis_type == "tooth":
            prompt = (
                f"You are a professional dentist. Please independently analyze this OPG image with bbox annotation (tooth FDI={target_id or '11'}).\n\n"
                "Independently analyze the image and identify any abnormal findings.\n\n"
                "Carefully examine: crown, root, periapical region, periodontal ligament space within the marked box.\n"
                "Output: Main radiographic findings and possible diagnosis.\n\n"
                "Important: At the beginning of your response, clearly state one of:\n"
                "1. [NO ABNORMALITY]: if no obvious abnormality found\n"
                "2. [ABNORMALITY SUPPORTS DISEASE]: if abnormality found supporting the diagnosis\n"
                "3. [ABNORMALITY DOES NOT SUPPORT DISEASE]: if abnormality found but not supporting the diagnosis"
            )
        else:
            prompt = "Independently analyze this OPG panoramic radiograph and identify any abnormal findings."
        
        # Add OPG orientation, permanent teeth reminder, and concise instruction
        prompt += OPG_ORIENTATION + PERMANENT_TEETH_REMINDER + CONCISE_INSTRUCTION
        
        # Note: detected_findings NOT included in prompt to maintain unbiased analysis
        # focus_areas can be added as areas of interest (not implying presence)
        if focus_areas:
            focus_terms = [f for f in focus_areas if f in {"caries", "periapical", "periodontal", "impaction", "restoration", "anatomy", "bone_loss", "root_canal"}]
            if focus_terms:
                prompt += f"\n\nFocus areas: {', '.join(focus_terms)} related findings."
        
        return prompt
    
    def dental_expert_analysis(
        self,
        image_path: str,
        analysis_type: str,
        target_id: Optional[str] = None,
        fdi: Optional[str] = None,
        quadrant: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        detected_findings: Optional[List[str]] = None,
        focus_areas: Optional[List[str]] = None,
        config: RunnableConfig = None,
    ) -> str:
        """使用 DentalGPT 进行图像分析。图像缩放到短边 768。"""
        import tempfile
        image_path = self._resolve_image_path(image_path, config)
        actual_target_id = target_id or fdi or quadrant
        
        prompt = self._build_dentist_prompt(
            analysis_type=analysis_type,
            target_id=actual_target_id,
            custom_prompt=custom_prompt,
            detected_findings=detected_findings,
            focus_areas=focus_areas
        )
        
        # 缩放图像到短边 768，写入临时文件
        resized_bytes = self._resize_image_short_edge(image_path, short_edge=768)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(resized_bytes)
            tmp_path = tmp.name
        try:
            result = self._call_remote_service(
                "dental_gpt",
                "/analyze",
                {
                    "image_path": tmp_path,
                    "question": prompt
                },
                timeout=100,
                file_field="image"
            )
        finally:
            os.unlink(tmp_path)  # 清理临时文件
        # 只返回 answer 部分，不包含 question 以节省 token
        if isinstance(result, dict):
            answer = result.get("answer", "")
            return json.dumps({"analysis": answer, "model": "DentalGPT"}, ensure_ascii=False)
        return json.dumps(result, ensure_ascii=False)
    
    def oral_expert_analysis(
        self,
        image_path: str,
        analysis_type: str,
        target_id: Optional[str] = None,
        fdi: Optional[str] = None,
        quadrant: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        detected_findings: Optional[List[str]] = None,
        focus_areas: Optional[List[str]] = None,
        config: RunnableConfig = None,
    ) -> str:
        """使用 OralGPT 进行图像分析。图像缩放到短边 768。"""
        import tempfile
        image_path = self._resolve_image_path(image_path, config)
        actual_target_id = target_id or fdi or quadrant
        
        prompt = self._build_dentist_prompt(
            analysis_type=analysis_type,
            target_id=actual_target_id,
            custom_prompt=custom_prompt,
            detected_findings=detected_findings,
            focus_areas=focus_areas
        )
        
        # 缩放图像到短边 768，写入临时文件
        resized_bytes = self._resize_image_short_edge(image_path, short_edge=768)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(resized_bytes)
            tmp_path = tmp.name
        try:
            result = self._call_remote_service(
                "oral_gpt",
                "/analyze",
                {
                    "image_path": tmp_path,
                    "prompt": prompt
                },
                timeout=100,
                file_field="file"
            )
        finally:
            os.unlink(tmp_path)  # 清理临时文件
        # 只返回 analysis 部分，不包含 prompt 以节省 token
        if isinstance(result, dict):
            analysis = result.get("analysis", result.get("answer", ""))
            return json.dumps({"analysis": analysis, "model": "OralGPT"}, ensure_ascii=False)
        return json.dumps(result, ensure_ascii=False)
    
    def llm_zoo_analysis(
        self,
        image_path: str,
        task_type: str = "analysis",
        target_fdi: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        detection_findings: Optional[Any] = None,
        dentist_analysis: Optional[Any] = None,
        focus: Optional[str] = None,
        config: RunnableConfig = None,
    ) -> str:
        """
        使用 LLM Zoo 同时调用 OpenAI 与 Google（保留供内部/调试）。对外请用 llm_zoo_openai / llm_zoo_google。
        """
        image_path = self._resolve_image_path(image_path, config)
        prompt = custom_prompt if custom_prompt else self._build_llm_zoo_prompt(task_type, target_fdi, custom_prompt, focus)
        result = self._call_llm_zoo_apis(image_path, prompt)
        return json.dumps(result, ensure_ascii=False)

    # 简洁回答要求，用于所有 VLM prompt
    # OPG 方向说明（标准放射学方向约定）
    _OPG_ORIENTATION_INSTRUCTION = """

OPG ORIENTATION (CRITICAL):
- Image RIGHT side = Patient's LEFT side
- Image LEFT side = Patient's RIGHT side
- FDI Q1 (upper right) & Q4 (lower right) = Patient's RIGHT = IMAGE LEFT
- FDI Q2 (upper left) & Q3 (lower left) = Patient's LEFT = IMAGE RIGHT
- ALL descriptions MUST use PATIENT orientation (not image orientation)!"""

    _CONCISE_ANSWER_INSTRUCTION = """

IMPORTANT INSTRUCTIONS:
- Your response will be sent to an AI Agent for synthesis with other VLM opinions.
- Be CONCISE. Only report CONFIRMED findings with HIGH CONFIDENCE.
- Do NOT report uncertain or ambiguous findings.
- If unsure, do NOT mention the finding at all.
- Keep response brief and factual."""

    def _build_llm_zoo_prompt(
        self,
        task_type: str,
        target_fdi: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        focus: Optional[str] = None,
    ) -> str:
        """Build LLM Zoo prompt (English only). OPG contains only permanent teeth."""
        # Reminder about permanent teeth only
        PERMANENT_TEETH_REMINDER = "\n\nIMPORTANT: This OPG contains ONLY permanent teeth (no deciduous/primary teeth)."
        
        if custom_prompt:
            return custom_prompt + self._OPG_ORIENTATION_INSTRUCTION + PERMANENT_TEETH_REMINDER + self._CONCISE_ANSWER_INSTRUCTION
        if task_type == "analysis":
            return (
                "You are a dental radiology expert. Provide a whole-OPG assessment.\n\n"
                "Requirements:\n"
                "1. Base your analysis solely on the image itself, without influence from any external information.\n"
                "2. Identify any abnormal findings.\n\n"
                "Pay special attention to periodontal assessment:\n"
                "- Evaluate alveolar bone loss severity, classified as: mild, moderate, or severe\n"
                "- Mild: alveolar crest 1-2mm apical to CEJ, or bone loss < 1/3 of root length\n"
                "- Moderate: alveolar crest 3-4mm apical to CEJ, or bone loss 1/3-2/3 of root length\n"
                "- Severe: alveolar crest >4mm apical to CEJ, or bone loss > 2/3 of root length\n"
                "- If no significant bone loss is observed, explicitly state so\n\n"
                "Output: key abnormalities (grouped by quadrant/tooth where possible)."
                + self._OPG_ORIENTATION_INSTRUCTION
                + PERMANENT_TEETH_REMINDER
                + self._CONCISE_ANSWER_INSTRUCTION
            )
        if task_type == "verification":
            prompt = (
                "You are a dental radiology expert. Independently analyze this OPG image.\n\n"
                "Requirements:\n"
                "1. Base your analysis solely on the image itself, without influence from any external information.\n"
                "2. Identify any abnormal findings.\n\n"
            )
            if target_fdi:
                prompt += f"Focus on tooth FDI={target_fdi}.\n\n"
            if focus:
                prompt += f"Pay attention to findings related to: {focus}.\n\n"
            prompt += "Output: key radiographic findings, most likely diagnosis."
            return prompt + self._OPG_ORIENTATION_INSTRUCTION + PERMANENT_TEETH_REMINDER + self._CONCISE_ANSWER_INSTRUCTION
        if task_type == "cross_check":
            prompt = (
                "You are a dental radiology expert. Independently analyze this OPG image.\n\n"
                "Requirements:\n"
                "1. Base your analysis solely on the image itself, without influence from any external information.\n"
                "2. Identify any abnormal findings.\n\n"
            )
            if target_fdi:
                prompt += f"Focus on tooth FDI={target_fdi}.\n\n"
            if focus:
                prompt += f"Pay attention to findings related to: {focus}.\n\n"
            prompt += "Output: key radiographic findings, most likely diagnosis."
            return prompt + self._OPG_ORIENTATION_INSTRUCTION + PERMANENT_TEETH_REMINDER + self._CONCISE_ANSWER_INSTRUCTION
        if task_type == "second_opinion":
            prompt = (
                "You are a dental radiology expert. Provide an independent second opinion on this OPG image.\n\n"
                "Requirements:\n"
                "1. Base your analysis solely on the image itself, without influence from any external information.\n"
                "2. Identify any abnormal findings.\n\n"
            )
            if target_fdi:
                prompt += f"Focus on tooth FDI={target_fdi}.\n\n"
            prompt += "Output: key radiographic findings, most likely diagnosis."
            return prompt + self._OPG_ORIENTATION_INSTRUCTION + PERMANENT_TEETH_REMINDER + self._CONCISE_ANSWER_INSTRUCTION
        return "You are a dental radiology expert. Independently analyze this OPG image and identify any abnormal findings." + self._OPG_ORIENTATION_INSTRUCTION + PERMANENT_TEETH_REMINDER + self._CONCISE_ANSWER_INSTRUCTION

    def _call_llm_zoo_openai(self, image_path: str, prompt: str, analysis_level: str = "overall") -> Dict[str, Any]:
        """
        仅调用 OpenAI GPT-5.2，返回与 _call_llm_zoo_apis 中 openai 键相同结构。
        图像缩放到短边 768。温度 0.3。
        
        analysis_level 控制 max_tokens（整体为原值 4 倍）:
        - "overall": 8192 (整张 OPG)
        - "quadrant": 4096 (象限分析)
        - "tooth": 1024 (单颗牙齿分析)
        """
        resolved = current_image_path_ctx.get()
        if resolved and os.path.isfile(resolved):
            image_path = resolved
        # 根据分析级别设置 max_tokens（4 倍）
        max_tokens_map = {"overall": 8192, "quadrant": 4096, "tooth": 1024}
        max_tokens = max_tokens_map.get(analysis_level, 8192)
        # 有有效图像则发「图像 + 文本」；否则纯文本（如 convert_to_structured 的报告→JSON 任务）
        has_image = bool(image_path) and os.path.isfile(image_path)
        if has_image:
            resized_bytes = self._resize_image_short_edge(image_path, short_edge=768)
            image_data = base64.standard_b64encode(resized_bytes).decode("utf-8")
            content = [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
                {"type": "text", "text": prompt},
            ]
        else:
            content = [{"type": "text", "text": prompt}]
        try:
            import openai
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-5.2",
                messages=[{"role": "user", "content": content}],
                max_completion_tokens=max_tokens,
                temperature=0.3  # GPT 温度 0.3
            )
            return {"model": "gpt-5.2", "response": response.choices[0].message.content, "status": "success", "max_tokens": max_tokens}
        except Exception as e:
            logger.error(f"OpenAI API 调用失败: {e}")
            return {"model": "gpt-5.2", "error": str(e), "status": "error"}

    def _call_llm_zoo_google(self, image_path: str, prompt: str, analysis_level: str = "overall") -> Dict[str, Any]:
        """
        仅调用 Google Gemini（Gemini Developer API），优先 GEMINI_API_KEY。
        模型：gemini-3-flash-preview。温度 0.3。图像缩放到短边 768。
        analysis_level 控制 max_output_tokens（整体为原值 4 倍）:
        - "overall": 8192, "quadrant": 4096, "tooth": 1024, "binary": 64
        """
        resolved = current_image_path_ctx.get()
        if resolved and os.path.isfile(resolved):
            image_path = resolved
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.error("未设置 GEMINI_API_KEY 或 GOOGLE_API_KEY")
            return {"model": "gemini-3-flash-preview", "error": "GEMINI_API_KEY or GOOGLE_API_KEY not set", "status": "error"}
        # 根据分析级别设置 max_output_tokens（4 倍）
        max_tokens_map = {"overall": 8192, "quadrant": 4096, "tooth": 1024, "binary": 64}
        max_tokens = max_tokens_map.get(analysis_level, 8192)
        # 缩放图像到短边 768
        resized_bytes = self._resize_image_short_edge(image_path, short_edge=768)
        image_data = base64.standard_b64encode(resized_bytes).decode("utf-8")
        mime_type = "image/png"  # 缩放后统一输出 PNG
        data_url = f"data:{mime_type};base64,{image_data}"
        from langchain_core.messages import HumanMessage
        msg = HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ]
        )
        model_name = "gemini-3-flash-preview"
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            model = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.3,  # Gemini 温度 0.3
                max_output_tokens=max_tokens,
            )
            response = model.invoke([msg])
            raw_content = response.content if hasattr(response, "content") else str(response)
            # 提取纯文本，移除 extras/signature（减少 context 膨胀）
            if isinstance(raw_content, list):
                text_parts = []
                for item in raw_content:
                    if isinstance(item, dict):
                        if "text" in item:
                            text_parts.append(item["text"])
                        # 忽略 extras/signature
                    elif isinstance(item, str):
                        text_parts.append(item)
                text = "\n".join(text_parts)
            else:
                text = raw_content
            return {"model": model_name, "response": text, "status": "success", "max_tokens": max_tokens}
        except Exception as e:
            logger.error(f"Gemini API 调用失败: {e}")
            return {"model": model_name, "error": str(e), "status": "error"}

    def llm_zoo_openai(
        self,
        image_path: str,
        task_type: str = "analysis",
        target_fdi: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        detection_findings: Optional[Any] = None,
        dentist_analysis: Optional[Any] = None,
        focus: Optional[str] = None,
        analysis_level: str = "overall",
        config: RunnableConfig = None,
    ) -> str:
        """
        LLM Zoo 仅调用 OpenAI GPT-5.2。温度 0.3。
        analysis_level 控制 max_tokens: overall=2048, quadrant=1024, tooth=256.
        """
        image_path = self._resolve_image_path(image_path, config)
        prompt = custom_prompt if custom_prompt else self._build_llm_zoo_prompt(task_type, target_fdi, custom_prompt, focus)
        result = self._call_llm_zoo_openai(image_path, prompt, analysis_level)
        return json.dumps(result, ensure_ascii=False)

    def llm_zoo_google(
        self,
        image_path: str,
        task_type: str = "analysis",
        target_fdi: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        detection_findings: Optional[Any] = None,
        dentist_analysis: Optional[Any] = None,
        focus: Optional[str] = None,
        analysis_level: str = "overall",
        config: RunnableConfig = None,
    ) -> str:
        """
        LLM Zoo 仅调用 Google Gemini。温度 0.3。
        analysis_level 控制 max_output_tokens: overall=8192, quadrant=4096, tooth=1024（均为 4 倍）。
        """
        image_path = self._resolve_image_path(image_path, config)
        prompt = custom_prompt if custom_prompt else self._build_llm_zoo_prompt(task_type, target_fdi, custom_prompt, focus)
        result = self._call_llm_zoo_google(image_path, prompt, analysis_level)
        return json.dumps(result, ensure_ascii=False)
    
    def _call_llm_zoo_apis(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """直接调用 OpenAI GPT-5.2 和 Google Gemini，返回 openai + google + summary（保留供内部/调试）"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            openai_future = executor.submit(self._call_llm_zoo_openai, image_path, prompt)
            google_future = executor.submit(self._call_llm_zoo_google, image_path, prompt)
            results = {"openai": openai_future.result(), "google": google_future.result()}
        google_model = results.get("google", {}).get("model", "gemini-3-flash-preview")
        results["summary"] = {
            "models_called": 2,
            "successful": sum(1 for r in [results["openai"], results["google"]] if r.get("status") == "success"),
            "prompt_used": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "models": {"openai": "gpt-5.2", "google": google_model}
        }
        return results
    
    # ==================== 辅助工具 ====================
    
    def calculate_fdi(self, quadrants: Dict, teeth: Dict) -> str:
        """计算 FDI 牙齿编号（与 Agent_refactor 一致：best IoU 分配、仅保留有效 FDI 并去重）。"""
        quadrants = _normalize_quadrants_for_merge(quadrants)
        teeth = _normalize_teeth_for_merge(teeth)
        teeth_by_fdi = build_fdi_teeth_like_refactor(quadrants=quadrants, teeth=teeth)
        return json.dumps(teeth_by_fdi, ensure_ascii=False)
    
    def match_disease_to_tooth(
        self,
        diseases: List[Dict],
        teeth: Dict,
        iou_threshold: float = 0.3
    ) -> str:
        """将疾病检测结果匹配到具体牙齿（本地执行）"""
        matched = match_diseases_to_teeth(diseases, teeth, iou_threshold)
        return json.dumps(matched, ensure_ascii=False)

    def _get_detections_or_run(
        self,
        image_path: str,
        detections_json: Optional[str],
        config: RunnableConfig = None,
    ) -> Dict[str, Any]:
        """
        若提供 detections_json 则解析返回，否则执行 run_all_detections 并返回解析后的 dict。
        供高级组合工具复用，避免重复跑完整检测管道。
        """
        if detections_json and detections_json.strip():
            try:
                return json.loads(detections_json)
            except json.JSONDecodeError as e:
                logger.warning(f"detections_json 解析失败，将重新运行检测: {e}")
        raw = self.run_all_detections(image_path, config)
        return json.loads(raw)

    def run_all_detections(self, image_path: str, config: RunnableConfig = None) -> str:
        """
        一键运行完整检测管道（并行优化版本）。
        阶段 1（并行）：quadrant_detection + tooth_enumeration
        阶段 2（顺序）：calculate_fdi（依赖阶段 1）
        阶段 3（并行）：disease_detection_tvem + bone_loss_detection + anatomy_detection
        阶段 4（顺序）：match_disease_to_tooth（依赖阶段 2、3）
        返回包含所有 detections 的单一 JSON，便于 Agent 一次获取全部结果。
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        image_path = self._resolve_image_path(image_path, config)
        out: Dict[str, Any] = {}

        # 阶段 1：并行执行 quadrant_detection 和 tooth_enumeration
        def run_quadrant():
            try:
                return "quadrants", json.loads(self.quadrant_detection(image_path, confidence_threshold=0.5, config=config))
            except Exception as e:
                return "quadrants", {"error": str(e)}

        def run_teeth():
            try:
                return "teeth", json.loads(self.tooth_enumeration(image_path, config=config))
            except Exception as e:
                return "teeth", {"error": str(e)}

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(run_quadrant), executor.submit(run_teeth)]
            for future in as_completed(futures):
                key, val = future.result()
                out[key] = val

        # 阶段 2：calculate_fdi（依赖阶段 1）
        try:
            q_norm = _normalize_quadrants_for_merge(out.get("quadrants") or {})
            t_norm = _normalize_teeth_for_merge(out.get("teeth") or {})
            out["teeth_fdi"] = build_fdi_teeth_like_refactor(quadrants=q_norm, teeth=t_norm)
        except Exception as e:
            out["teeth_fdi"] = {"error": str(e)}

        # 阶段 3：并行执行 tvem_disease、bone_loss、anatomy（TVEM 为唯一疾病检测器）
        tasks = []
        def run_tvem():
            try:
                return "tvem_disease", json.loads(self.disease_detection_tvem(image_path, confidence=0.5, return_vis=False, config=config))
            except Exception as e:
                return "tvem_disease", {"error": str(e)}
        tasks.append(("tvem_disease", run_tvem))
        def run_bone_loss():
            try:
                return "bone_loss", json.loads(self.bone_loss_detection(image_path, confidence=0.5, return_vis=False, config=config))
            except Exception as e:
                return "bone_loss", {"error": str(e)}
        tasks.append(("bone_loss", run_bone_loss))
        def run_anatomy():
            try:
                return "anatomy", json.loads(self.anatomy_detection(image_path, confidence=0.5, return_vis=False, config=config))
            except Exception as e:
                return "anatomy", {"error": str(e)}
        tasks.append(("anatomy", run_anatomy))
        if tasks:
            with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
                futures = [executor.submit(fn) for _, fn in tasks]
                for future in as_completed(futures):
                    key, val = future.result()
                    out[key] = val

        # 阶段 4：match_disease_to_tooth（依赖阶段 2、3）；仅 TVEM 匹配到牙位
        # TVEM: 8th tooth (18/28/38/48) with Root Piece -> reported_as impacted tooth
        teeth_fdi = out.get("teeth_fdi")
        if isinstance(teeth_fdi, dict) and "error" not in teeth_fdi:
            # TVEM matched (with 8th tooth Root Piece -> impacted tooth normalization)
            tvem_src = out.get("tvem_disease")
            if isinstance(tvem_src, dict) and "detections" in tvem_src:
                try:
                    # 将 bbox 转换为 box（match_diseases_to_teeth 使用 box）
                    tvem_dets = []
                    for d in tvem_src["detections"]:
                        dd = dict(d)
                        if "box" not in dd and "bbox" in dd:
                            dd["box"] = dd["bbox"]
                        tvem_dets.append(dd)
                    raw_m = self.match_disease_to_tooth(tvem_dets, teeth_fdi, iou_threshold=0.3)
                    tvem_matched = json.loads(raw_m)
                    # Normalize 8th tooth Root Piece -> impacted tooth
                    for fdi, diseases in tvem_matched.items():
                        if fdi in ("18", "28", "38", "48") and isinstance(diseases, list):
                            for d in diseases:
                                cls = (d.get("class_name") or d.get("class") or "").lower()
                                if "root piece" in cls:
                                    d["reported_as"] = "impacted tooth"
                    out["tvem_matched"] = tvem_matched
                except Exception:
                    out["tvem_matched"] = {"error": "match failed"}
        return json.dumps(out, ensure_ascii=False)

    # ==================== 高级组合工具 ====================
    # 高级工具会按需组合基础工具（并缓存）以完成任务；其结果与 run_all_detections 的同名字段保持一致，便于对照与测试。
    # 若传入 detections_json，则直接复用以保证可复现。
    # - get_tooth_by_fdi: 返回 teeth_fdi[fdi]（number, box, confidence）；未找到时返回 error + available_fdi
    # - get_quadrant: 返回一个或多个象限的 quadrant、box、teeth_fdi 列表（支持逗号分隔多象限）
    # - get_tooth_mask: 内部用 teeth_fdi[fdi].box 调 segment_object，返回 segment 的 JSON（mask_contour）
    # - get_status_on_tooth: 返回 tvem_matched[fdi]，8号牙 Root Piece -> impacted tooth
    # - extraction_risk_near_anatomy: 用 teeth_fdi[fdi] + anatomy + segment 轮廓距离，返回 risk_near
    # - get_quadrant_teeth: 同 get_quadrant 的牙齿子集，返回 quadrant + teeth: [{fdi, box, confidence}]
    # - list_teeth_with_status: 遍历 tvem_matched 按 status_class 模糊匹配，返回 fdi_list
    # - get_bone_loss_description: 用 bone_loss.detections + quadrants + teeth_fdi 做 IoU，返回英文 description、quadrants_involved、teeth_involved

    def get_tooth_by_fdi(
        self,
        image_path: str,
        fdi: str,
        detections_json: Optional[str] = None,
        config: RunnableConfig = None,
    ) -> str:
        """
        按 FDI 获取单颗牙信息（box、confidence 等）。
        若提供 detections_json（run_all_detections 的 JSON），则直接从中取；否则按需调用基础工具并缓存。
        """
        image_path = self._resolve_image_path(image_path, config)
        fdi = str(fdi).strip()
        if len(fdi) == 1:
            fdi = "0" + fdi
        teeth_fdi = self._ensure_teeth_fdi(image_path, detections_json, config)
        if not isinstance(teeth_fdi, dict) or "error" in teeth_fdi:
            return json.dumps({"error": "teeth_fdi unavailable", "detail": teeth_fdi}, ensure_ascii=False)
        if fdi not in teeth_fdi:
            return json.dumps({
                "error": f"Tooth FDI={fdi} not found",
                "available_fdi": list(teeth_fdi.keys()),
            }, ensure_ascii=False)
        return json.dumps(teeth_fdi[fdi], ensure_ascii=False)

    def get_quadrant(
        self,
        image_path: str,
        quadrant_names: str,
        detections_json: Optional[str] = None,
        config: RunnableConfig = None,
    ) -> str:
        """
        按象限名获取一个或多个象限信息（box、该象限内牙齿 FDI 列表）。
        象限名支持 Upperright/Upperleft/Lowerleft/Lowerright 或 Q1/Q2/Q3/Q4，逗号分隔。
        """
        image_path = self._resolve_image_path(image_path, config)
        # Parse comma-separated quadrant names
        raw_names = [n.strip() for n in str(quadrant_names).split(",") if n.strip()]
        if not raw_names:
            return json.dumps({"error": "quadrant_names cannot be empty", "supported": "Q1-Q4 or Upperright/Upperleft/Lowerleft/Lowerright, comma-separated"}, ensure_ascii=False)
        
        valid_quadrants = {"Upperright", "Upperleft", "Lowerleft", "Lowerright"}
        qnames = []
        invalid_names = []
        for rn in raw_names:
            qn = _map_quadrant_name(rn)
            if qn in valid_quadrants:
                if qn not in qnames:  # deduplicate
                    qnames.append(qn)
            else:
                invalid_names.append(rn)
        
        if invalid_names:
            return json.dumps({"error": f"Invalid quadrant name(s): {invalid_names}", "supported": "Q1-Q4 or Upperright/Upperleft/Lowerleft/Lowerright"}, ensure_ascii=False)
        
        quadrants_raw = self._ensure_quadrants(image_path, detections_json, config)
        quadrants_norm = _normalize_quadrants_for_merge(quadrants_raw)
        teeth_fdi = self._ensure_teeth_fdi(image_path, detections_json, config)
        if not isinstance(teeth_fdi, dict):
            teeth_fdi = {}
        
        q_to_digit = {"Upperright": "1", "Upperleft": "2", "Lowerleft": "3", "Lowerright": "4"}
        
        results = []
        for qname in qnames:
            digit = q_to_digit.get(qname, "")
            teeth_in_q = [fdi for fdi in teeth_fdi if str(fdi).startswith(digit)]
            quadrant_box = (quadrants_norm.get(qname) or {}).get("box")
            results.append({"quadrant": qname, "box": quadrant_box, "teeth_fdi": sorted(teeth_in_q)})
        
        # Return single object if only one quadrant, else return list
        if len(results) == 1:
            return json.dumps(results[0], ensure_ascii=False)
        return json.dumps({"quadrants": results}, ensure_ascii=False)

    def get_tooth_mask(
        self,
        image_path: str,
        fdi: str,
        detections_json: Optional[str] = None,
        config: RunnableConfig = None,
    ) -> str:
        """
        获取某颗牙的 mask（MedSAM 分割）。先按 FDI 取 box，再调用 segment_object。
        """
        image_path = self._resolve_image_path(image_path, config)
        fdi = str(fdi).strip()
        if len(fdi) == 1:
            fdi = "0" + fdi
        teeth_fdi = self._ensure_teeth_fdi(image_path, detections_json, config)
        if not isinstance(teeth_fdi, dict) or fdi not in teeth_fdi:
            return json.dumps({"error": f"Tooth FDI={fdi} not found", "available_fdi": list(teeth_fdi.keys()) if isinstance(teeth_fdi, dict) else []}, ensure_ascii=False)
        box = teeth_fdi[fdi].get("box")
        if not box or len(box) != 4:
            return json.dumps({"error": f"FDI={fdi} has no valid box"}, ensure_ascii=False)
        return self.segment_object(image_path, [box], config)

    def get_status_on_tooth(
        self,
        image_path: str,
        fdi: str,
        detections_json: Optional[str] = None,
        config: RunnableConfig = None,
    ) -> str:
        """
        获取某颗牙上的状态列表（TVEM 匹配结果）。
        8 号牙（FDI 18/28/38/48）若 TVEM 为 Root Piece 则自动改为 impacted tooth。
        """
        image_path = self._resolve_image_path(image_path, config)
        fdi = str(fdi).strip()
        if len(fdi) == 1:
            fdi = "0" + fdi
        tvem_matched = self._ensure_tvem_matched(image_path, detections_json, config)
        if isinstance(tvem_matched, dict) and "error" in tvem_matched:
            tvem_matched = {}

        # TVEM matched results (TVEM is the sole disease detector)
        raw_statuses = [dict(d) for d in tvem_matched.get(fdi, [])]
        
        # Normalize: 8th tooth (18/28/38/48) with Root Piece -> impacted tooth
        # Clean output: only keep bbox (1 decimal) and bbox_normalized (4 decimals)
        statuses = []
        for d in raw_statuses:
            cls = (d.get("class_name") or d.get("class") or "").strip()
            cls_lower = cls.lower()
            reported = cls
            if fdi in ("18", "28", "38", "48"):
                if "root piece" in cls_lower:
                    reported = "impacted tooth"
            
            # 构建精简的输出，只保留必要字段
            clean_status = {
                "class_id": d.get("class_id"),
                "class_name": d.get("class_name"),
                "confidence": d.get("confidence"),
                "reported_as": reported,
            }
            # bbox 保留 1 位小数
            bbox = d.get("bbox") or d.get("box")
            if bbox and isinstance(bbox, (list, tuple)):
                clean_status["bbox"] = [round(v, 1) for v in bbox]
            # bbox_normalized 保留 4 位小数（如果存在）
            bbox_norm = d.get("bbox_normalized")
            if bbox_norm and isinstance(bbox_norm, (list, tuple)):
                clean_status["bbox_normalized"] = [round(v, 4) for v in bbox_norm]
            # 不包含 mask_contour, box 等冗余字段
            statuses.append(clean_status)
        return json.dumps({"fdi": fdi, "statuses": statuses}, ensure_ascii=False)

    def extraction_risk_near_anatomy(
        self,
        image_path: str,
        fdi: str,
        proximity_pixels: float = 10.0,
        detections_json: Optional[str] = None,
        config: RunnableConfig = None,
    ) -> str:
        """
        拔牙风险：该牙与上颌窦/下颌管是否接近（Shapely 检查 mask 轮廓距离）。
        上颌牙（FDI 1x、2x）检查上颌窦；下颌牙（FDI 3x、4x）检查下颌管。
        """
        image_path = self._resolve_image_path(image_path, config)
        fdi = str(fdi).strip()
        if len(fdi) == 1:
            fdi = "0" + fdi
        teeth_fdi = self._ensure_teeth_fdi(image_path, detections_json, config)
        if not isinstance(teeth_fdi, dict) or fdi not in teeth_fdi:
            return json.dumps({"error": f"Tooth FDI={fdi} not found", "risk": None}, ensure_ascii=False)
        # 获取牙齿 mask 轮廓
        box = teeth_fdi[fdi].get("box")
        if not box or len(box) != 4:
            return json.dumps({"error": f"FDI={fdi} has no valid box", "risk": None}, ensure_ascii=False)
        seg_result = self.segment_object(image_path, [box], config)
        try:
            seg_data = json.loads(seg_result)
        except json.JSONDecodeError:
            return json.dumps({"error": "Tooth segmentation result parsing failed", "risk": None}, ensure_ascii=False)
        if seg_data.get("error") or not seg_data.get("mask_contour"):
            return json.dumps({"error": seg_data.get("error", "No mask_contour"), "risk": None}, ensure_ascii=False)
        tooth_contour = seg_data["mask_contour"]
        anatomy = self._ensure_anatomy(image_path, detections_json, config)
        if not isinstance(anatomy, dict) or "error" in anatomy:
            return json.dumps({"fdi": fdi, "risk": None, "message": "Anatomy detection unavailable"}, ensure_ascii=False)
        detections_anatomy = anatomy.get("detections") or []
        # class_id 0 = 下颌管, 1 = 上颌窦
        digit = fdi[0] if fdi else ""
        if digit in ("1", "2"):
            target_class_id = 1  # 上颌窦
            anatomy_name = "maxillary sinus"
            risk_description = "oroantral communication risk"
        elif digit in ("3", "4"):
            target_class_id = 0  # 下颌管
            anatomy_name = "mandibular canal (inferior alveolar nerve)"
            risk_description = "nerve damage / paresthesia risk"
        else:
            return json.dumps({"fdi": fdi, "risk": None, "message": "Only FDI 1x, 2x, 3x, 4x supported"}, ensure_ascii=False)
        
        # 计算最小距离
        min_distance = float('inf')
        for d in detections_anatomy:
            if d.get("class_id") != target_class_id:
                continue
            contour = d.get("mask_contour") or []
            if not contour:
                continue
            dist = contour_min_distance_pixels(tooth_contour, contour)
            if dist < min_distance:
                min_distance = dist
        
        # 判断风险等级
        if min_distance == float('inf'):
            return json.dumps({
                "fdi": fdi,
                "anatomy": anatomy_name,
                "risk_level": "unknown",
                "message": f"No {anatomy_name} detected in image"
            }, ensure_ascii=False)
        
        # 风险分级
        if min_distance <= 10:
            risk_level = "HIGH"
        elif min_distance <= 20:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"
        
        return json.dumps({
            "fdi": fdi,
            "anatomy": anatomy_name,
            "distance_pixels": round(min_distance, 1),
            "risk_level": risk_level,
            "risk_near": min_distance < proximity_pixels,
            "risk_description": risk_description if risk_level in ("HIGH", "MODERATE") else None,
            "recommendation": f"Exercise caution during extraction - {risk_description}" if risk_level == "HIGH" else None
        }, ensure_ascii=False)

    def get_quadrant_teeth(
        self,
        image_path: str,
        quadrant_name: str,
        detections_json: Optional[str] = None,
        config: RunnableConfig = None,
    ) -> str:
        """
        获取某象限内所有牙齿的 FDI 及信息列表。
        """
        image_path = self._resolve_image_path(image_path, config)
        qname = _map_quadrant_name(str(quadrant_name).strip())
        if qname not in {"Upperright", "Upperleft", "Lowerleft", "Lowerright"}:
            return json.dumps({"error": f"Invalid quadrant name: {quadrant_name}"}, ensure_ascii=False)
        teeth_fdi = self._ensure_teeth_fdi(image_path, detections_json, config)
        if not isinstance(teeth_fdi, dict):
            teeth_fdi = {}
        q_to_digit = {"Upperright": "1", "Upperleft": "2", "Lowerleft": "3", "Lowerright": "4"}
        digit = q_to_digit.get(qname, "")
        teeth_in_q = [fdi for fdi in teeth_fdi if str(fdi).startswith(digit)]
        list_info = []
        for fdi in sorted(teeth_in_q):
            tooth_info = teeth_fdi[fdi]
            box = tooth_info.get("box")
            # 保留 1 位小数
            if box and isinstance(box, (list, tuple)):
                box = [round(v, 1) for v in box]
            list_info.append({"fdi": fdi, "box": box, "confidence": tooth_info.get("confidence")})
        return json.dumps({"quadrant": qname, "teeth": list_info}, ensure_ascii=False)

    def list_teeth_with_status(
        self,
        image_path: str,
        status_class: Optional[str] = None,
        detections_json: Optional[str] = None,
        config: RunnableConfig = None,
    ) -> str:
        """
        列出具有某类状态的所有牙位（FDI），或返回所有检测到的状态汇总。
        
        若 status_class 为空或 "all"：返回所有检测到的状态 {status: [fdi_list]}
        若 status_class 指定：返回该状态对应的 fdi_list（模糊匹配）

        状态来源为 TVEM 匹配结果。
        """
        image_path = self._resolve_image_path(image_path, config)
        tvem_matched = self._ensure_tvem_matched(image_path, detections_json, config)
        if isinstance(tvem_matched, dict) and "error" in tvem_matched:
            tvem_matched = {}

        # 判断是否返回所有状态
        return_all = not status_class or str(status_class).strip().lower() in ("", "all")

        if return_all:
            # 返回所有检测到的状态: {status_class: [fdi_list]}
            status_to_fdi: Dict[str, set] = {}

            # TVEM: all statuses
            for fdi, diseases in tvem_matched.items():
                if fdi == "other_tooth" or not isinstance(diseases, list):
                    continue
                for d in diseases:
                    cls = (d.get("class_name") or d.get("class") or "").strip()
                    if cls:
                        if cls not in status_to_fdi:
                            status_to_fdi[cls] = set()
                        status_to_fdi[cls].add(fdi)
            
            # 转换为 sorted list
            result = {status: sorted(fdi_set) for status, fdi_set in status_to_fdi.items()}
            return json.dumps({"all_statuses": result, "status_count": len(result)}, ensure_ascii=False)
        
        else:
            # 返回指定状态的 fdi_list
            status_class = str(status_class).strip().lower()
            fdi_set = set()

            # TVEM: all statuses
            for fdi, diseases in tvem_matched.items():
                if fdi == "other_tooth" or not isinstance(diseases, list):
                    continue
                for d in diseases:
                    cls = (d.get("class_name") or d.get("class") or "").lower()
                    if status_class in cls or cls in status_class:
                        fdi_set.add(fdi)
                        break
            
            return json.dumps({"status_class": status_class, "fdi_list": sorted(fdi_set)}, ensure_ascii=False)

    def get_bone_loss_description(
        self,
        image_path: str,
        detections_json: Optional[str] = None,
        iou_threshold: float = 0.1,
        config: RunnableConfig = None,
    ) -> str:
        """
        根据骨吸收（bone loss）检测结果，将涉及范围描述为 "lower region", "lower left quadrant" 等。
        通过骨吸收框与象限/牙齿的 IoU 判定涉及哪些象限和牙位，再聚合成英文描述。
        单个象限时返回 "X quadrant"，多个象限时返回聚合描述如 "lower region"。
        """
        image_path = self._resolve_image_path(image_path, config)
        bone_loss = self._ensure_bone_loss(image_path, detections_json, config)
        if not isinstance(bone_loss, dict) or "error" in bone_loss:
            return json.dumps({
                "description": "none",
                "quadrants_involved": [],
                "teeth_involved": [],
                "message": "bone_loss detection unavailable",
            }, ensure_ascii=False)
        detections_bl = bone_loss.get("detections") or []
        quadrants_raw = self._ensure_quadrants(image_path, detections_json, config)
        quadrants_norm = _normalize_quadrants_for_merge(quadrants_raw)
        teeth_fdi = self._ensure_teeth_fdi(image_path, detections_json, config)
        if not isinstance(teeth_fdi, dict):
            teeth_fdi = {}

        quadrants_involved: set = set()
        teeth_involved: set = set()

        for bl in detections_bl:
            box = bl.get("bbox") or bl.get("box")
            if not box or len(box) != 4:
                continue
            # 匹配到最佳象限（IoU 最大）
            best_q = None
            best_iou = 0.0
            for qid, qdata in quadrants_norm.items():
                qbox = qdata.get("box")
                if not qbox or len(qbox) != 4:
                    continue
                iou = calculate_iou(box, qbox)
                if iou > best_iou:
                    best_iou = iou
                    best_q = qid
            if best_q and best_iou >= iou_threshold:
                quadrants_involved.add(best_q)
            # 匹配到重叠牙齿（IoU >= iou_threshold）
            for fdi, tdata in teeth_fdi.items():
                if not isinstance(tdata, dict):
                    continue
                tbox = tdata.get("box")
                if not tbox or len(tbox) != 4:
                    continue
                if calculate_iou(box, tbox) >= iou_threshold:
                    teeth_involved.add(fdi)

        description = _bone_loss_quadrants_to_description(list(quadrants_involved))
        return json.dumps({
            "description": description,
            "quadrants_involved": sorted(quadrants_involved),
            "teeth_involved": sorted(teeth_involved),
            "total_bone_loss_regions": len(detections_bl),
        }, ensure_ascii=False)

    def get_annotated_image(
        self,
        image_path: str,
        target_type: str,
        target_id: str,
        output_mode: str = "bbox_overlay",
        detections_json: Optional[str] = None,
        config: RunnableConfig = None,
    ) -> str:
        """
        生成带标注的图像：裁剪区域（crop）或带 bbox 的整张 OPG（bbox_overlay）。
        - crop：只返回目标区域的裁剪图（base64）
        - bbox_overlay：返回整张 OPG 并在目标区域画红色 bbox（base64）
        DentalGPT/OralGPT 偏好 bbox_overlay；GPT/Gemini 两者均可。
        """
        from PIL import Image, ImageDraw
        image_path = self._resolve_image_path(image_path, config)
        target_type = str(target_type).strip().lower()
        target_id = str(target_id).strip()
        output_mode = str(output_mode).strip().lower()
        if output_mode not in ("crop", "bbox_overlay"):
            return json.dumps({"error": f"Invalid output_mode: {output_mode}, should be 'crop' or 'bbox_overlay'"}, ensure_ascii=False)
        # 获取目标 box
        box = None
        if target_type == "tooth":
            fdi = target_id if len(target_id) == 2 else "0" + target_id
            teeth_fdi = self._ensure_teeth_fdi(image_path, detections_json, config)
            if not isinstance(teeth_fdi, dict) or "error" in teeth_fdi:
                return json.dumps({"error": f"Tooth detection unavailable: {teeth_fdi.get('error') if isinstance(teeth_fdi, dict) else 'unknown'}"}, ensure_ascii=False)
            if fdi not in teeth_fdi:
                return json.dumps({"error": f"Tooth FDI={fdi} not found", "available_fdi": list(teeth_fdi.keys())}, ensure_ascii=False)
            box = teeth_fdi[fdi].get("box")
        elif target_type == "quadrant":
            qname = _map_quadrant_name(target_id)
            if qname not in {"Upperright", "Upperleft", "Lowerleft", "Lowerright"}:
                return json.dumps({"error": f"Invalid quadrant name: {target_id}", "supported": "Q1-Q4 or Upperright/Upperleft/Lowerleft/Lowerright"}, ensure_ascii=False)
            quadrants_raw = self._ensure_quadrants(image_path, detections_json, config)
            if not isinstance(quadrants_raw, dict) or "error" in quadrants_raw:
                return json.dumps({"error": f"Quadrant detection unavailable: {quadrants_raw.get('error') if isinstance(quadrants_raw, dict) else 'unknown'}"}, ensure_ascii=False)
            quadrants_norm = _normalize_quadrants_for_merge(quadrants_raw)
            box = (quadrants_norm.get(qname) or {}).get("box")
        else:
            return json.dumps({"error": f"Invalid target_type: {target_type}, should be 'tooth' or 'quadrant'"}, ensure_ascii=False)
        if not box or len(box) != 4:
            return json.dumps({"error": f"No valid box found for {target_type}={target_id}"}, ensure_ascii=False)
        # 加载图像
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            return json.dumps({"error": f"Cannot load image: {e}"}, ensure_ascii=False)
        x1, y1, x2, y2 = [int(v) for v in box]
        # 生成输出（缩放到短边 768 便于 VLM 使用）
        # NOTE: 返回临时文件路径而非 base64，避免消息历史膨胀
        import tempfile
        if output_mode == "crop":
            cropped = img.crop((x1, y1, x2, y2))
            cropped_resized = self._resize_pil_image_short_edge(cropped, short_edge=768)
            # 保存到临时文件
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            cropped_resized.save(tmp.name, format="PNG")
            return json.dumps({
                "target_type": target_type,
                "target_id": target_id,
                "output_mode": "crop",
                "box": [x1, y1, x2, y2],
                "image_path": tmp.name,
                "note": "Image saved to temp file. Use this path for VLM tools.",
            }, ensure_ascii=False)
        else:  # bbox_overlay
            draw = ImageDraw.Draw(img)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
            img_resized = self._resize_pil_image_short_edge(img, short_edge=768)
            # 保存到临时文件
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img_resized.save(tmp.name, format="PNG")
            return json.dumps({
                "target_type": target_type,
                "target_id": target_id,
                "output_mode": "bbox_overlay",
                "box": [x1, y1, x2, y2],
                "image_path": tmp.name,
                "note": "Image saved to temp file. Use this path for VLM tools.",
            }, ensure_ascii=False)

    def convert_to_structured(
        self,
        natural_report: str,
        schema_path: Optional[str] = None,
        config: RunnableConfig = None,
    ) -> str:
        """
        将自然语言诊断报告转换为结构化 JSON。
        Agent 决定何时调用此工具（通常在完成分析并生成报告后）。
        
        Args:
            natural_report: 自然语言诊断报告
            schema_path: Schema&Enum_standard.md 文件路径（可选）
            
        Returns:
            JSON 格式的结构化报告，或错误信息
        """
        import re
        
        # 读取 schema 标准
        if schema_path and os.path.exists(schema_path):
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_content = f.read()[:4000]  # 截断
        else:
            schema_content = self._get_builtin_schema()
        
        conversion_prompt = f"""You are a medical data structuring specialist. Convert the following OPG diagnostic report into a structured JSON following the Schema&Enum_standard.

## Schema Standard (Key Rules):
{schema_content}

## Natural Language Report to Convert:
{natural_report}

## Conversion Rules:
1. Use snake_case for all enum values
2. Use FDI notation as string keys (e.g., "38", not 38)
3. OMIT normal findings (sparse representation)
4. Include only findings explicitly mentioned in the report
5. List ALL not-detected teeth in "not_detected_fdi" (no special treatment for wisdom teeth)

## Output Format:
Return ONLY valid JSON, no explanations. Example structure:
```json
{{
  "dentition_summary": {{
    "total_teeth_detected": 28,
    "not_detected_fdi": ["18", "28", "38", "46", "48"]
  }},
  "teeth": {{
    "47": {{
      "restoration": "filling"
    }}
  }}
}}
```

Now convert the report to structured JSON:"""

        try:
            # 调用 GPT-5.2 进行转换
            result = self._call_llm_zoo_openai("", conversion_prompt, analysis_level="overall")
            if result.get("status") == "error":
                return json.dumps({"error": result.get("error", "conversion failed"), "status": "error"}, ensure_ascii=False)
            
            response_text = result.get("response", "")
            
            # 提取 JSON
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_text.strip()
            
            structured_report = json.loads(json_str)
            return json.dumps({
                "structured_report": structured_report,
                "conversion_status": "success"
            }, ensure_ascii=False)
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败: {e}")
            return json.dumps({
                "structured_report": None,
                "conversion_status": "json_parse_error",
                "error": str(e)
            }, ensure_ascii=False)
        except Exception as e:
            logger.error(f"结构化转换失败: {e}")
            return json.dumps({
                "structured_report": None,
                "conversion_status": "error",
                "error": str(e)
            }, ensure_ascii=False)

    def resolve_finding_disagreement(
        self,
        image_path: str,
        finding_type: str,
        disagreement_type: str,
        vlm_opinions: str,
        gold_standard_info: str,
        confirmed_findings: Optional[str] = None,
        config: RunnableConfig = None,
    ) -> str:
        """
        通过 subagent 解决疾病位置/分类不一致的问题。
        
        当多个 VLM 对某个疾病的位置或分类存在分歧时，此工具会：
        1. 收集所有相关意见的上下文
        2. 结合金标准信息（牙齿列表、象限等）
        3. 使用 subagent (GPT-5.2) 进行分析判断
        4. 返回确定的位置 (FDI) 或分类
        
        Args:
            image_path: OPG 图像路径
            finding_type: 疾病类型 (implant, bone_loss, periapical_lesion, filling, crown, rct 等)
            disagreement_type: 分歧类型 ('position' 或 'classification')
            vlm_opinions: JSON 格式的 VLM 意见数组
            gold_standard_info: JSON 格式的金标准信息
            confirmed_findings: JSON 格式的已确认发现（可选）
            
        Returns:
            JSON 格式的解析结果，包含确定的位置或分类
        """
        import re
        
        # 解析输入
        try:
            opinions = json.loads(vlm_opinions) if isinstance(vlm_opinions, str) else vlm_opinions
            gold_info = json.loads(gold_standard_info) if isinstance(gold_standard_info, str) else gold_standard_info
            confirmed = json.loads(confirmed_findings) if confirmed_findings and isinstance(confirmed_findings, str) else (confirmed_findings or {})
        except json.JSONDecodeError as e:
            return json.dumps({
                "error": f"JSON parse error: {e}",
                "status": "error"
            }, ensure_ascii=False)
        
        # 构建 subagent prompt
        if disagreement_type == "position":
            task_description = f"""You are a dental radiology expert resolving a **position disagreement** for a finding of type: **{finding_type}**.

Multiple VLMs have confirmed this finding EXISTS, but they DISAGREE on its exact POSITION/FDI location.

## Your Task:
1. Analyze each VLM's position claim
2. Consider the gold standard tooth list (which teeth actually exist on this OPG)
3. Use OPG orientation: Image RIGHT = Patient LEFT, Image LEFT = Patient RIGHT
4. Determine the most likely FDI position based on:
   - Majority agreement on region (e.g., "lower left" vs "lower right")
   - Anatomical plausibility (is the claimed tooth present?)
   - Consistency with gold standard teeth list

## GOLD STANDARD Information

**IMPORTANT**: The following tool outputs are GOLD STANDARD:
- **Total tooth count** from detection tools is gold standard
- **FDI numbering** is nearly gold standard; at most 1-tooth offset error in missing tooth cases

## VLM Opinions (with position claims):
"""
            for i, op in enumerate(opinions, 1):
                source = op.get("source", f"VLM_{i}")
                opinion = op.get("opinion", "")
                pos = op.get("position_or_classification", "unspecified")
                task_description += f"\n**{source}**: {opinion} | Position: {pos}"
            
            task_description += f"""

## Gold Standard Information:
- **Detected teeth (FDI)**: {gold_info.get('teeth_fdi', [])}
- **Not detected (could be missing)**: {gold_info.get('not_detected', [])}
- **Quadrants**: {gold_info.get('quadrants', {})}

## Already Confirmed Findings:
{json.dumps(confirmed, indent=2) if confirmed else "None"}

## Output Format (JSON only):
```json
{{
  "resolved_fdi": "XX",  // Specific FDI number, or null if cannot determine
  "resolved_region": "lower left posterior",  // General region description
  "confidence": "high|medium|low",
  "reasoning": "Brief explanation of how position was determined"
}}
```

Analyze and provide the most likely FDI position:"""

        elif disagreement_type == "classification":
            task_description = f"""You are a dental radiology expert resolving a **classification/severity disagreement** for a finding of type: **{finding_type}**.

Multiple VLMs agree this finding EXISTS at a similar location, but they DISAGREE on its CLASSIFICATION or SEVERITY.

## Your Task:
1. Analyze each VLM's classification claim
2. Consider the range of classifications reported
3. Apply the conservative classification rule:
   - If severity varies (mild/moderate/severe): use the mildest or most general
   - If type varies but related: use the more general term
4. Determine the most appropriate classification

## VLM Opinions (with classification claims):
"""
            for i, op in enumerate(opinions, 1):
                source = op.get("source", f"VLM_{i}")
                opinion = op.get("opinion", "")
                cls = op.get("position_or_classification", "unspecified")
                task_description += f"\n**{source}**: {opinion} | Classification: {cls}"
            
            task_description += f"""

## Gold Standard Information:
{json.dumps(gold_info, indent=2)}

## Already Confirmed Findings:
{json.dumps(confirmed, indent=2) if confirmed else "None"}

## Output Format (JSON only):
```json
{{
  "resolved_classification": "mild bone loss",  // Conservative/general classification
  "alternatives_considered": ["mild", "moderate", "severe"],
  "confidence": "high|medium|low",
  "reasoning": "Brief explanation of classification choice"
}}
```

Analyze and provide the most appropriate classification:"""

        else:
            return json.dumps({
                "error": f"Unknown disagreement_type: {disagreement_type}. Use 'position' or 'classification'.",
                "status": "error"
            }, ensure_ascii=False)
        
        # 调用 subagent (GPT-5.2)
        try:
            result = self._call_llm_zoo_openai("", task_description, analysis_level="quadrant")  # 使用 quadrant 级别 (1024 tokens)
            
            if result.get("status") == "error":
                return json.dumps({
                    "error": result.get("error", "subagent call failed"),
                    "status": "error"
                }, ensure_ascii=False)
            
            response_text = result.get("response", "")
            
            # 提取 JSON
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 尝试直接解析
                json_str = response_text.strip()
            
            resolved = json.loads(json_str)
            
            # 添加元数据
            resolved["finding_type"] = finding_type
            resolved["disagreement_type"] = disagreement_type
            resolved["vlm_count"] = len(opinions)
            resolved["status"] = "resolved"
            
            logger.info(f"✓ 解决 {finding_type} 的 {disagreement_type} 分歧: {resolved}")
            return json.dumps(resolved, ensure_ascii=False)
            
        except json.JSONDecodeError as e:
            logger.error(f"Subagent 返回的 JSON 解析失败: {e}")
            return json.dumps({
                "error": f"JSON parse error in subagent response: {e}",
                "raw_response": response_text[:500] if 'response_text' in dir() else "N/A",
                "status": "parse_error"
            }, ensure_ascii=False)
        except Exception as e:
            logger.error(f"resolve_finding_disagreement 失败: {e}")
            return json.dumps({
                "error": str(e),
                "status": "error"
            }, ensure_ascii=False)

    def _get_builtin_schema(self) -> str:
        '''返回内置的简化 Schema 标准'''
        return """
### Allowed Root Keys (OPG-only)
- dentition_summary: Always include
- teeth: Per-tooth findings keyed by FDI string
- periodontium: Periodontal findings (if abnormal)
- tmj: TMJ findings (if abnormal)
- sinuses: Sinus findings (if abnormal)
- jaws: Jaw findings (if abnormal)

### dentition_summary Fields
- total_teeth_detected: integer (GOLD STANDARD)
- not_detected_fdi: [FDI strings] - ALL teeth not detected (no special wisdom tooth treatment)

### teeth.{FDI} Fields (snake_case enums)
- status: present | missing | unerupted | impacted | residual_root | implant | root_filled | supernumerary
- restoration: filling | crown | implant | root_canal_treatment
- winters_class: vertical | angled | horizontal
- icdas_code: early | moderate | advanced
- pai_score: normal | mild_change | moderate_change | severe_change
- relationship: approximates_iac | approximates_sinus

### periodontium Fields
- severity: mild | moderate | severe
- bone_loss_pattern: horizontal | vertical
- region: generalized | localized

### Sparse Representation
- OMIT any field/object if normal
- OMIT teeth.{FDI} if no abnormal findings
"""


def create_dental_tools(
    tools_config: Dict[str, Any],
    analysis_only: bool = False,
    return_toolkit: bool = False,
) -> Union[List[StructuredTool], Tuple[List[StructuredTool], "DentalToolkit"]]:
    """
    Create LangChain tool list and optionally return the toolkit instance.

    Args:
        tools_config: Tool configuration dict
        analysis_only: If True, only create VLM analysis tools (default False)
        return_toolkit: If True, return (tools, toolkit) tuple for preloading cache

    Returns:
        LangChain Tool list, or (tools, toolkit) tuple if return_toolkit=True
    """
    toolkit = DentalToolkit(tools_config)

    tools = []

    # NOTE: run_all_detections is NOT exposed as a tool.
    # Instead, Agent should call toolkit.run_all_detections() at startup to preload cache.
    # Only high-level tools are exposed; base tools are kept for internal use.
    if not analysis_only:
        detection_tools_list = [
            StructuredTool.from_function(
                func=toolkit.get_tooth_by_fdi,
                name="get_tooth_by_fdi",
                description="""Get single-tooth info by FDI number.

**GOLD STANDARD**: FDI numbering is nearly gold standard; at most 1-tooth offset error in missing tooth cases.

Input: image_path (required), fdi (e.g. 11, 18, 48), optional detections_json.
Returns: JSON { number, box, confidence } for that tooth. On error: { error, available_fdi } (available_fdi lists valid FDI on this image).
Use when: you need one tooth's box/confidence (e.g. before get_tooth_mask or extraction_risk_near_anatomy).""",
                args_schema=GetToothByFDIInput,
            ),
            StructuredTool.from_function(
                func=toolkit.get_quadrant,
                name="get_quadrant",
                description="""Get one or more quadrant info by quadrant name(s).

**GOLD STANDARD**: Total tooth count and per-quadrant tooth count are gold standard.

Input: image_path (required), quadrant_names (comma-separated, e.g. 'Q1', 'Q1,Q2', 'Q1,Q2,Q3,Q4'), optional detections_json.
Returns:
- Single quadrant: JSON { quadrant, box, teeth_fdi: [FDI strings] }
- Multiple quadrants: JSON { quadrants: [ {quadrant, box, teeth_fdi}, ... ] }
On invalid name: { error, supported }.
Use when: you need quadrant bbox and the list of FDI in one or more quadrants. For full tooth info use get_quadrant_teeth.""",
                args_schema=GetQuadrantInput,
            ),
            StructuredTool.from_function(
                func=toolkit.get_tooth_mask,
                name="get_tooth_mask",
                description="""Get tooth mask (MedSAM segment) by FDI. Uses tooth box then segment_object.

Input: image_path (required), fdi, optional detections_json.
Returns: JSON { mask_contour: [[x,y],...], success } on success; { error } or { error, available_fdi } on failure.
Use when: you need tooth contour (e.g. for distance to anatomy); extraction_risk_near_anatomy calls this internally.""",
                args_schema=GetToothMaskInput,
            ),
            StructuredTool.from_function(
                func=toolkit.get_status_on_tooth,
                name="get_status_on_tooth",
                description="""Get statuses on a tooth (TVEM matched + YOLO filtered) by FDI. YOLO Caries/Deep Caries are excluded; only Impacted and Periapical Lesion from YOLO are kept. FDI 18/28/38/48 with Root Piece auto-reported as impacted tooth.

TVEM classes: Caries, Crown, Filling, Implant, Mandibular Canal, Missing teeth, Periapical lesion, Root Canal Treatment, Root Piece, impacted tooth, maxillary sinus.

Input: image_path (required), fdi, optional detections_json.
Returns: JSON { fdi, statuses: [ { class_name, confidence, bbox, reported_as } ] }. Empty statuses list if none. Use reported_as for display (impacted tooth/original class).
Use when: you need which statuses/abnormalities are on a specific tooth.""",
                args_schema=GetStatusOnToothInput,
            ),
        ]
        detection_tools_list.append(StructuredTool.from_function(
                func=toolkit.extraction_risk_near_anatomy,
                name="extraction_risk_near_anatomy",
                description="""Extraction risk assessment: checks if tooth root is close to critical anatomy (Shapely contour distance).

**IMPORTANT for impacted teeth**: Always call this tool when an impacted tooth is detected to assess extraction risk.

**Anatomy checked by quadrant**:
- Upper teeth (FDI quadrants 1 and 2): **maxillary sinus** - risk of oroantral communication
- Lower teeth (FDI quadrants 3 and 4): **mandibular canal** (inferior alveolar nerve) - risk of nerve damage/paresthesia

**Input**: image_path, fdi (e.g. 18, 38, 48), proximity_pixels (default 10), optional detections_json.

**Returns**: JSON with fdi, anatomy, distance_pixels, risk_level, risk_near.
- risk_level: high if distance <= 10px, moderate (10-20px), low if > 20px
- risk_near: true if distance < proximity_pixels

**Use when**: 
- Impacted tooth detected (18/28/38/48)
- Planning extraction and need to assess nerve/sinus proximity
- Evaluating surgical difficulty""",
                args_schema=ExtractionRiskNearAnatomyInput,
            ))
        detection_tools_list.extend([
            StructuredTool.from_function(
                func=toolkit.get_quadrant_teeth,
                name="get_quadrant_teeth",
                description="""Get all teeth in a quadrant with full info (box, confidence).

**GOLD STANDARD**: Total tooth count and per-quadrant tooth count are gold standard. FDI is nearly gold standard (max 1-tooth offset in missing tooth cases).

Input: image_path (required), quadrant_name (Q1-Q4 or Upperright/Upperleft/Lowerleft/Lowerright), optional detections_json.
Returns: JSON { quadrant, teeth: [ { fdi, box, confidence } ] }. On invalid name: { error }.
Use when: you need box/confidence for every tooth in a quadrant (use get_quadrant if only FDI list is needed).""",
                args_schema=GetQuadrantTeethInput,
            ),
            StructuredTool.from_function(
                func=toolkit.list_teeth_with_status,
                name="list_teeth_with_status",
                description="""List FDI of teeth with detected statuses. YOLO Caries/Deep Caries are excluded.

**Two modes:**
1. **All statuses** (default): If status_class is empty or "all", returns ALL detected statuses grouped by class.
   - Returns: { all_statuses: { "Filling": ["15","17",...], "Crown": ["36"] }, status_count: N }
2. **Filter by class**: If status_class specified, returns FDI list for that class (fuzzy match).
   - Returns: { status_class, fdi_list: [FDI strings] }

TVEM detectable classes: Crown, Filling, Implant, Periapical lesion, Root Canal Treatment, impacted tooth.
YOLO detectable classes (after filtering): Impacted, Periapical Lesion.

Input: image_path (required), status_class (optional, default=all), detections_json (optional).
Use when: you want to see ALL detected abnormalities at once, or filter for specific status.""",
                args_schema=ListTeethWithStatusInput,
            ),
        ])
        detection_tools_list.append(StructuredTool.from_function(
                func=toolkit.get_bone_loss_description,
                name="get_bone_loss_description",
                description="""Describe bone-loss regions by involved quadrants/teeth (English: "lower region", "lower left quadrant", "upper region", "left side", "right side", etc.).

Input: image_path (required), optional detections_json, optional iou_threshold (default 0.1).
Returns: JSON { description, quadrants_involved: [names], teeth_involved: [FDI], total_bone_loss_regions }. If no bone loss: description "none", empty lists.
Use when: you need a textual summary of where bone loss is (e.g. for report).""",
                args_schema=GetBoneLossDescriptionInput,
            ))
        detection_tools_list.append(StructuredTool.from_function(
            func=toolkit.get_annotated_image,
            name="get_annotated_image",
            description="""Generate annotated image: cropped region or full OPG with bbox drawn.

Input: image_path (required), target_type ('tooth' or 'quadrant'), target_id (FDI e.g. 11/48 or quadrant Q1-Q4), output_mode ('crop' or 'bbox_overlay', default bbox_overlay), optional detections_json.
Returns: JSON { target_type, target_id, output_mode, box, image_path }. image_path is a temp file path to the generated PNG.
- output_mode='crop': cropped region only (smaller image focused on target)
- output_mode='bbox_overlay': full OPG with red bbox drawn around target (preserves context)

**Preference for VLM tools**:
- DentalGPT / OralGPT: prefer 'bbox_overlay' (full OPG with bbox, better context)
- GPT-5.2 / Gemini: both 'crop' and 'bbox_overlay' work well

Use when: you need to pass a localized/annotated image to VLM tools for focused analysis. Pass the returned image_path to VLM tools.""",
            args_schema=GetAnnotatedImageInput,
        ))
        tools.extend(detection_tools_list)
    
    # VLM 分析工具：DentalGPT + OralGPT + GPT-5.2 + Gemini + resolve
    resolve_desc = "When at least 3 of 5 sources agree a finding EXISTS"
    vlm_tools = [
        StructuredTool.from_function(
            func=toolkit.dental_expert_analysis,
            name="dental_expert_analysis",
            description="""DentalGPT multimodal image analysis. **RECOMMENDED: Use analysis_type='overall' for full OPG analysis only.**

Input: image_path (required), analysis_type (overall RECOMMENDED, quadrant/tooth/custom for specific needs), fdi (for tooth), quadrant (for quadrant), custom_prompt (for custom), detected_findings (list), focus_areas (caries/periapical/periodontal/impaction/restoration/anatomy/bone_loss/root_canal).
Returns: JSON from service (e.g. answer/analysis text); may include error key on failure.
Use when: you need expert-level image interpretation on the whole OPG.

**IMPORTANT**: DentalGPT works best on full OPG images. For tooth/quadrant-level analysis, prefer GPT or Gemini tools instead.""",
            args_schema=DentalExpertAnalysisInput,
        ),
        StructuredTool.from_function(
            func=toolkit.oral_expert_analysis,
            name="oral_expert_analysis",
            description="""OralGPT multimodal analysis. **RECOMMENDED: Use analysis_type='overall' for full OPG analysis only.**

Input: image_path (required), analysis_type (overall RECOMMENDED, quadrant/tooth/custom for specific needs), fdi, quadrant, custom_prompt, detected_findings, focus_areas.
Returns: JSON from service (e.g. answer/analysis); may include error on failure.
Use when: cross-validate DentalGPT or get second opinion on the whole OPG.

**IMPORTANT**: OralGPT works best on full OPG images. For tooth/quadrant-level analysis, prefer GPT or Gemini tools instead.""",
            args_schema=DentalExpertAnalysisInput,
        ),
    ]
    vlm_tools.extend([
            StructuredTool.from_function(
                func=toolkit.llm_zoo_openai,
                name="llm_zoo_openai",
                description="""LLM Zoo: OpenAI GPT-5.2 (temperature 0.3, high precision).

Input: image_path (required), task_type (analysis/verification/cross_check/second_opinion/summarize), target_fdi, custom_prompt, detection_findings, dentist_analysis, focus, analysis_level.
- analysis_level controls max_tokens: 'overall'=2048, 'quadrant'=1024, 'tooth'=256.
Returns: JSON { model, response, status, max_tokens }.
Use when: you need GPT-5.2 analysis or verification; pair with llm_zoo_google for cross-model check.

**Image preference**: Both 'crop' and 'bbox_overlay' from get_annotated_image work well with GPT-5.2. GPT excels at focused tooth/quadrant analysis.""",
                args_schema=LLMZooAnalysisInput,
            ),
            StructuredTool.from_function(
                func=toolkit.llm_zoo_google,
                name="llm_zoo_google",
                description="""LLM Zoo: Google Gemini 3 Flash Preview (temperature 0.3, max_output_tokens 4x: overall=8192, quadrant=4096, tooth=1024).

Input: image_path (required), task_type (analysis/verification/cross_check/second_opinion/summarize), target_fdi, custom_prompt, detection_findings, dentist_analysis, focus, analysis_level.
- analysis_level controls max_tokens: 'overall'=2048, 'quadrant'=1024, 'tooth'=256.
Returns: JSON { model, response, status, max_tokens }.
Use when: you need Gemini analysis or verification; pair with llm_zoo_openai for cross-model check.

**Image preference**: Both 'crop' and 'bbox_overlay' from get_annotated_image work well with Gemini. Gemini excels at focused tooth/quadrant analysis.""",
                args_schema=LLMZooAnalysisInput,
            ),
        ])
    vlm_tools.append(
        StructuredTool.from_function(
            func=toolkit.resolve_finding_disagreement,
            name="resolve_finding_disagreement",
            description=f"""Resolve position or classification disagreement for a confirmed finding using a specialized subagent.

**When to use**: {resolve_desc}, but DISAGREE on its exact POSITION (FDI) or CLASSIFICATION (severity/type).

**Input**:
- image_path: OPG image path
- finding_type: e.g. implant, bone_loss, periapical_lesion, filling
- disagreement_type: position (FDI location varies) or classification (severity/type varies)
- vlm_opinions: JSON array of opinions
- gold_standard_info: JSON with teeth_fdi, not_detected, quadrants
- confirmed_findings: Optional JSON of already confirmed findings

**Returns** (position): resolved_fdi, resolved_region, confidence, reasoning.
**Returns** (classification): resolved_classification, alternatives_considered, confidence, reasoning.

**Use when**: You have a confirmed finding ({resolve_desc}) but need to resolve position or classification disagreement to produce a specific FDI number or classification.""",
            args_schema=ResolveFindingDisagreementInput,
        ),
    )
    tools.extend(vlm_tools)
    logger.info("✓ 已添加 VLM 工具 (DentalGPT, OralGPT, GPT-5.2, Gemini, resolve_finding_disagreement)")
    
    logger.info(f"✓ Created {len(tools)} LangChain tools (analysis_only={analysis_only})")
    if return_toolkit:
        return tools, toolkit
    return tools
