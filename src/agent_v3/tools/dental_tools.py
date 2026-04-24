"""
LangChain tool definitions.
Converts the dental tool registry into LangChain Tool format.
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

# The actual image path for the current run (set by agent.run()); tools prefer this path over whatever the LLM passes.
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
    calculate_iou,
)

logger = logging.getLogger(__name__)

# Quadrant name to FDI prefix mapping (Q1=1x, Q2=2x, Q3=3x, Q4=4x)
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
    Generate a concise English description based on the involved quadrants.
    Example: lower left + lower right -> "lower region"; lower left only -> "lower left quadrant".
    """
    if not quadrants:
        return "none"
    qset = set(quadrants)
    en_list = sorted([_QUADRANT_TO_EN.get(q, q) for q in qset if _QUADRANT_TO_EN.get(q)])
    if not en_list:
        return ", ".join(sorted(qset))
    # Aggregate: upper region = upper left + upper right, lower region = lower left + lower right
    # left side = upper left + lower left, right side = upper right + lower right
    upper = {"upper left", "upper right"}
    lower = {"lower left", "lower right"}
    left = {"upper left", "lower left"}
    right = {"upper right", "lower right"}
    en_set = set(en_list)
    # Return "X quadrant" for a single quadrant
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


# TVEM quadrant labels are already in patient orientation (verified by bbox spatial analysis)
_QUADRANT_MAPPING = {
    "upper right": "Upperright",
    "upperright": "Upperright",
    "upper left": "Upperleft",
    "upperleft": "Upperleft",
    "lower left": "Lowerleft",
    "lowerleft": "Lowerleft",
    "lower right": "Lowerright",
    "lowerright": "Lowerright",
    "q1": "Upperright",
    "q2": "Upperleft",
    "q3": "Lowerleft",
    "q4": "Lowerright",
}


def _map_quadrant_name(name: str) -> str:
    """Consistent with Agent_refactor."""
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
    Convert the quadrant-detection API response (with a detections list) into the format expected by merge.
    Uses the same quadrant name mapping as Agent_refactor so teeth_fdi matches the reference.
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
            out[name] = {
                "name": name,
                "box": bbox,
                "confidence": det.get("confidence", 0.0),
                "mask_contour": det.get("mask_contour"),
            }
        return out
    return quadrants


def _normalize_teeth_for_merge(teeth: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Convert the tooth-detection API response (with a detections list) into the format expected by merge_detection_results.
    The agent typically passes the raw tooth_enumeration JSON, which needs conversion first.
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


# ==================== Tool Input Schemas ====================

class QuadrantDetectionInput(BaseModel):
    """Quadrant detection input"""
    image_path: str = Field(description="Path to OPG image")
    confidence_threshold: float = Field(default=0.5, description="Confidence threshold (mapped to confidence)")


class ToothEnumerationInput(BaseModel):
    """Tooth enumeration input"""
    image_path: str = Field(description="Path to OPG image")


class DiseaseDetectionInput(BaseModel):
    """Disease detection input (YOLO)"""
    image_path: str = Field(description="Path to OPG image")
    conf: float = Field(default=0.25, description="Confidence threshold")
    return_visualization: bool = Field(default=False, description="Return visualization")


class DiseaseDetectionTVEMInput(BaseModel):
    """TVEM disease detection input"""
    image_path: str = Field(description="Path to OPG image")
    confidence: float = Field(default=0.5, description="Confidence threshold")
    return_vis: bool = Field(default=False, description="Return visualization")


class BoneLossDetectionInput(BaseModel):
    """Bone loss detection input (matches Agent_refactor; default 0.5)"""
    image_path: str = Field(description="Path to OPG image")
    confidence: float = Field(default=0.5, description="Confidence threshold")
    return_vis: bool = Field(default=False, description="Return visualization")


class AnatomyDetectionInput(BaseModel):
    """Anatomy detection input"""
    image_path: str = Field(description="Path to OPG image")
    confidence: float = Field(default=0.5, description="Confidence threshold")
    return_vis: bool = Field(default=False, description="Return visualization")


class SegmentObjectInput(BaseModel):
    """Object segmentation input"""
    image_path: str = Field(description="Path to image")
    boxes: List[List[float]] = Field(description="List of bbox [[x1,y1,x2,y2], ...]")


class DentalExpertAnalysisInput(BaseModel):
    """Dental-expert analysis input"""
    image_path: str = Field(description="Path to image")
    analysis_type: str = Field(description="Analysis type: overall/quadrant/tooth/custom")
    target_id: Optional[str] = Field(default=None, description="Target ID (quadrant Q1-Q4 or FDI e.g. 48)")
    fdi: Optional[str] = Field(default=None, description="FDI tooth number (e.g. 48), same as target_id")
    quadrant: Optional[str] = Field(default=None, description="Quadrant name (Q1-Q4), same as target_id")
    custom_prompt: Optional[str] = Field(default=None, description="Custom prompt when analysis_type=custom")
    detected_findings: Optional[List[str]] = Field(default=None, description="Pre-detected findings list")
    focus_areas: Optional[List[str]] = Field(default=None, description="Focus: caries/periapical/periodontal/impaction/restoration/anatomy/bone_loss/root_canal")


class RunAllDetectionsInput(BaseModel):
    """One-click input for running the full detection pipeline (order matches reference detections)"""
    image_path: str = Field(description="Path to OPG image")


class GetToothByFDIInput(BaseModel):
    """Get info for a single tooth by FDI"""
    image_path: str = Field(description="Path to OPG image")
    fdi: str = Field(description="FDI tooth number, e.g. 11, 18, 48")
    detections_json: Optional[str] = Field(default=None, description="Optional JSON from run_all_detections; omit to run detection pipeline")


class GetQuadrantInput(BaseModel):
    """Get info for one or more quadrants by name"""
    image_path: str = Field(description="Path to OPG image")
    quadrant_names: str = Field(description="One or more quadrants, comma-separated. E.g. 'Q1', 'Q1,Q2', 'Q1,Q2,Q3,Q4' or 'Upperright,Lowerleft'")
    detections_json: Optional[str] = Field(default=None, description="Optional JSON from run_all_detections")


class GetToothMaskInput(BaseModel):
    """Get the mask of a tooth (MedSAM segmentation)"""
    image_path: str = Field(description="Path to OPG image")
    fdi: str = Field(description="FDI tooth number")
    detections_json: Optional[str] = Field(default=None, description="Optional JSON from run_all_detections")


class GetStatusOnToothInput(BaseModel):
    """Get the list of statuses on a tooth (TVEM matched results, excluding YOLO Caries/Deep Caries)"""
    image_path: str = Field(description="Path to OPG image")
    fdi: str = Field(description="FDI tooth number")
    detections_json: Optional[str] = Field(default=None, description="Optional JSON from run_all_detections")


class ExtractionRiskNearAnatomyInput(BaseModel):
    """Extraction risk: whether the tooth is close to the maxillary sinus / mandibular canal (Shapely checks mask contour distance)"""
    image_path: str = Field(description="Path to OPG image")
    fdi: str = Field(description="FDI tooth number (18/28: maxillary sinus, 38/48: mandibular canal)")
    proximity_pixels: float = Field(default=10.0, description="Proximity threshold in pixels")
    detections_json: Optional[str] = Field(default=None, description="Optional JSON from run_all_detections")


class GetQuadrantTeethInput(BaseModel):
    """Get FDI and info for all teeth in a quadrant"""
    image_path: str = Field(description="Path to OPG image")
    quadrant_name: str = Field(description="Quadrant: Upperright/Upperleft/Lowerleft/Lowerright or Q1-Q4")
    detections_json: Optional[str] = Field(default=None, description="Optional JSON from run_all_detections")


class ListTeethWithStatusInput(BaseModel):
    """List all tooth positions (FDI) with a given status class; excludes YOLO Caries/Deep Caries"""
    image_path: str = Field(description="Path to OPG image")
    status_class: Optional[str] = Field(default=None, description="Status class to filter (e.g. Filling, Crown, impacted tooth). If empty or 'all', returns all detected statuses grouped by class.")
    detections_json: Optional[str] = Field(default=None, description="Optional JSON from run_all_detections")


class GetBoneLossDescriptionInput(BaseModel):
    """Bone loss region description: generates descriptions like 'lower region', 'lower left', 'lower right' from the involved quadrants/teeth"""
    image_path: str = Field(description="Path to OPG image")
    detections_json: Optional[str] = Field(default=None, description="Optional JSON from run_all_detections")
    iou_threshold: float = Field(default=0.1, description="IoU threshold for bone-loss box vs quadrant/tooth overlap")


class GetAnnotatedImageInput(BaseModel):
    """Generate an annotated image: cropped region or full OPG with bbox overlay"""
    image_path: str = Field(description="Path to OPG image")
    target_type: str = Field(description="Target type: 'tooth' or 'quadrant'")
    target_id: str = Field(description="Target ID: FDI number (e.g. 11, 48) for tooth, or quadrant name (Q1-Q4 / Upperright/Upperleft/Lowerleft/Lowerright) for quadrant")
    output_mode: str = Field(default="bbox_overlay", description="Output mode: 'crop' (cropped region only) or 'bbox_overlay' (full OPG with bbox drawn)")
    detections_json: Optional[str] = Field(default=None, description="Optional JSON from run_all_detections")


class RAGSimilarCasesInput(BaseModel):
    """RAG similar-case retrieval input"""
    image_path: str = Field(description="Path to tooth crop image (from get_annotated_image with output_mode='crop')")
    top_k: int = Field(default=5, description="Number of similar cases to retrieve")
    fdi_filter: Optional[str] = Field(default=None, description="Optional FDI number to restrict search (e.g. '18')")


class RegionAnalysisInput(BaseModel):
    """Combined region-analysis input: (target_type, target_id) + optional prompt knobs.
    Internally handles get_annotated_image + VLM call in one step."""
    target_type: str = Field(description="'tooth', 'quadrant', or 'overall' (whole OPG)")
    target_id: Optional[str] = Field(default=None, description="FDI (e.g. 36) for tooth, Q1-Q4 / Upperright etc for quadrant. Omit when target_type='overall'.")
    custom_prompt: Optional[str] = Field(default=None, description="Optional custom prompt for the VLM")
    focus_areas: Optional[List[str]] = Field(default=None, description="Optional focus list: caries/periapical/periodontal/impaction/restoration/anatomy/bone_loss/root_canal")
    detected_findings: Optional[List[str]] = Field(default=None, description="Optional pre-detected findings to inform the VLM")


class RegionRagInput(BaseModel):
    """Combined region-based RAG retrieval: fetches tooth crop internally."""
    target_type: str = Field(default="tooth", description="'tooth' or 'quadrant' — RAG typically uses tooth crops")
    target_id: str = Field(description="FDI for tooth (e.g. 36) or Q1-Q4 for quadrant")
    top_k: int = Field(default=5, description="Number of similar cases to retrieve")


class ResolveFindingDisagreementInput(BaseModel):
    """Resolve disagreement on disease position/classification"""
    image_path: str = Field(description="Path to OPG image")
    finding_type: str = Field(description="Type of finding, e.g. 'implant', 'bone_loss', 'periapical_lesion', 'filling', 'crown', 'rct'")
    disagreement_type: str = Field(description="Type of disagreement: 'position' (FDI location varies) or 'classification' (severity/type varies)")
    vlm_opinions: str = Field(description="JSON array of VLM opinions, each with {source, opinion, position_or_classification}. E.g. [{\"source\": \"DentalGPT\", \"opinion\": \"implant present\", \"position_or_classification\": \"lower left 36\"}, ...]")
    gold_standard_info: str = Field(description="JSON with gold standard info: {teeth_fdi: [...], quadrants: {...}, not_detected: [...]}") 
    confirmed_findings: Optional[str] = Field(default=None, description="Optional JSON of already confirmed findings for context")


class CalculateFDIInput(BaseModel):
    """FDI calculation input"""
    quadrants: Dict = Field(description="Quadrant detection result")
    teeth: Dict = Field(description="Tooth detection result")


class MatchDiseaseToToothInput(BaseModel):
    """Disease-to-tooth matching input"""
    diseases: List[Dict] = Field(description="Disease detection result")
    teeth: Dict = Field(description="Teeth with FDI (teeth_fdi)")
    iou_threshold: float = Field(default=0.3, description="IoU threshold")


class LLMZooAnalysisInput(BaseModel):
    """LLM Zoo analysis input"""
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


# ==================== Tool Implementations ====================

class ServicePool:
    """Manages concurrency control and queue scheduling for multi-replica service endpoints.
    Each endpoint has a semaphore controlling concurrency; requests prefer idle endpoints and queue when all are busy."""

    def __init__(self, endpoints: List[str], max_concurrent: int = 1):
        import threading
        self.endpoints = list(endpoints)
        self.max_concurrent = max_concurrent
        self._semaphores = {ep: threading.Semaphore(max_concurrent) for ep in self.endpoints}
        self._lock = threading.Lock()
        self._idx = 0  # round-robin start

    def acquire(self, timeout: float = 300) -> Optional[str]:
        """Acquire an available endpoint (try all non-blocking first; if all busy, block and wait)."""
        import time
        n = len(self.endpoints)
        with self._lock:
            start = self._idx
            self._idx = (self._idx + 1) % n

        # Fast attempt: non-blocking round-robin
        for i in range(n):
            ep = self.endpoints[(start + i) % n]
            if self._semaphores[ep].acquire(blocking=False):
                return ep

        # All busy: block and wait for any endpoint to become free
        deadline = time.monotonic() + timeout
        for i in range(n):
            ep = self.endpoints[(start + i) % n]
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if self._semaphores[ep].acquire(blocking=True, timeout=remaining):
                return ep
        return None

    def release(self, endpoint: str):
        if endpoint in self._semaphores:
            self._semaphores[endpoint].release()


class DentalToolkit:
    """Dental toolkit"""

    # Ablation: detection sources that can be skipped (keys match phase 3 of run_all_detections)
    ABLATION_SKIP_KEYS = ("yolo_disease", "tvem_disease", "bone_loss", "anatomy")

    def __init__(self, tools_config: Dict[str, Any], skip_detection_sources: Optional[List[str]] = None):
        """
        Initialize the toolkit.

        Args:
            tools_config: tool configuration (loaded from tools_config.yaml)
            skip_detection_sources: detection sources to skip for ablation, e.g. ["yolo_disease", "tvem_disease", "bone_loss", "anatomy"]
        """
        self.config = tools_config
        self._skip_detection_sources = list(skip_detection_sources or [])
        # Runtime cache: avoids repeated basic-tool requests during multiple high-level tool calls on the same image in one agent run.
        # key: resolved_image_path, value: { "quadrants": {...}, "teeth": {...}, "teeth_fdi": {...}, ... }
        self._runtime_cache: Dict[str, Dict[str, Any]] = {}
        # Round-robin counters (used only for services without load_balancing)
        self._round_robin_counters: Dict[str, int] = {}
        # Service endpoint pools (with queueing and concurrency control)
        self._service_pools: Dict[str, ServicePool] = {}
        self._init_service_pools()
        if self._skip_detection_sources:
            logger.info("DentalToolkit initialized (ablation: skipping detection sources %s)", self._skip_detection_sources)
        else:
            logger.info("DentalToolkit initialized")

    def _init_service_pools(self):
        """Create a ServicePool for each service that has load_balancing configured."""
        for tool_name, tool_cfg in self.config.get("tools", {}).items():
            service_info = tool_cfg.get("service", {})
            lb = service_info.get("load_balancing", {})
            if lb.get("enabled") and lb.get("endpoints"):
                endpoints = lb["endpoints"]
                max_concurrent = lb.get("max_concurrent_per_endpoint", 1)
                queue_timeout = lb.get("queue_timeout", 300)
                pool = ServicePool(endpoints, max_concurrent)
                pool.queue_timeout = queue_timeout
                self._service_pools[tool_name] = pool
                logger.info("✓ ServicePool[%s]: %d endpoints, max_concurrent=%d, queue_timeout=%ds",
                            tool_name, len(endpoints), max_concurrent, queue_timeout)
    
    def _call_remote_service(
        self,
        service_name: str,
        endpoint: str,
        data: Dict[str, Any],
        timeout: int = 120,
        file_field: str = "file"
    ) -> Dict[str, Any]:
        """Call a remote service."""
        # Prefer reading from tools.*.service.base_url (standard structure in tools_config.yaml);
        # fall back to service_name at the top level for compatibility.
        service_config = self.config.get("tools", {}).get(service_name, {}) or self.config.get(service_name, {})
        service_info = service_config.get("service", {})
        base_url = service_info.get("base_url")
        if not base_url:
            logger.warning(f"Service {service_name} has no base_url configured; check tools_config.yaml")
            return {"error": f"Service {service_name} is not configured; please add it to tools_config.yaml"}

        # Use ServicePool (with queueing and concurrency control) or fall back to simple round-robin.
        pool = self._service_pools.get(service_name)
        acquired_ep = None
        if pool:
            acquired_ep = pool.acquire(timeout=getattr(pool, 'queue_timeout', 300))
            if acquired_ep is None:
                logger.error(f"All endpoints for service {service_name} are busy and queue timed out")
                return {"error": f"Service {service_name} is busy; please retry later"}
            base_url = acquired_ep
        elif "load_balancing" in service_info and service_info["load_balancing"].get("enabled"):
            # Simple round-robin compatibility when no ServicePool is configured
            endpoints = service_info["load_balancing"].get("endpoints", [base_url])
            if endpoints:
                counter = self._round_robin_counters.get(service_name, 0)
                base_url = endpoints[counter % len(endpoints)]
                self._round_robin_counters[service_name] = counter + 1

        url = f"{base_url}{endpoint}"

        try:
            # Choose request form based on data type
            if "image_path" in data:
                image_path = data.pop("image_path")
                # If the supplied path does not exist, fall back to the context-injected path
                if not os.path.isfile(image_path):
                    resolved = current_image_path_ctx.get()
                    if resolved and os.path.isfile(resolved):
                        image_path = resolved
                if not os.path.isfile(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                with open(image_path, "rb") as f:
                    files = {file_field: f}
                    form_data = {k: str(v) for k, v in data.items()}
                    response = requests.post(url, files=files, data=form_data, timeout=timeout)
            else:
                response = requests.post(url, json=data, timeout=timeout)

            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            logger.error(f"Call to {service_name} failed: {e}")
            return {"error": str(e)}
        finally:
            if pool and acquired_ep:
                pool.release(acquired_ep)

    # ==================== Detection Tools ====================
    
    def _resolve_image_path(self, image_path: str, config: RunnableConfig = None) -> str:  # noqa: ARG002
        """Resolve an image path: use the given path if it exists; otherwise fall back to the context-injected path."""
        if os.path.isfile(image_path):
            return image_path
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
        Resize the image so its shorter edge equals short_edge pixels and return PNG bytes.
        Used to preprocess images before VLM tool calls (normalize size, reduce transfer volume).
        """
        from PIL import Image
        import io
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        if min(w, h) <= short_edge:
            # Image is already small enough; skip resizing
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        # Resize by shorter edge
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
        Resize a PIL.Image so its shorter edge equals short_edge pixels and return the resized PIL.Image.
        Used in scenarios like get_annotated_image that need to process images in memory.
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
        """Generate a cache key (using the resolved absolute path) to avoid recomputation when the same image is referred to by different relative paths."""
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
        """Parse an externally provided detections_json (used for reproduction/comparison); return None on failure."""
        if not detections_json or not detections_json.strip():
            return None
        try:
            parsed = json.loads(detections_json)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    def _ensure_quadrants(self, image_path: str, detections_json: Optional[str], config: RunnableConfig = None) -> Dict[str, Any]:
        """Ensure quadrants are available (prefer detections_json, then cache, then call quadrant_detection)."""
        det = self._parse_detections_json(detections_json)
        if det and isinstance(det.get("quadrants"), dict):
            return self._cache_set(image_path, "quadrants", det["quadrants"], config)
        cached = self._cache_get(image_path, "quadrants", config)
        if isinstance(cached, dict):
            return cached
        data = json.loads(self.quadrant_detection(image_path, confidence_threshold=0.5, config=config))
        return self._cache_set(image_path, "quadrants", data, config)

    def _ensure_teeth(self, image_path: str, detections_json: Optional[str], config: RunnableConfig = None) -> Dict[str, Any]:
        """Ensure teeth are available (prefer detections_json, then cache, then call tooth_enumeration)."""
        det = self._parse_detections_json(detections_json)
        if det and isinstance(det.get("teeth"), dict):
            return self._cache_set(image_path, "teeth", det["teeth"], config)
        cached = self._cache_get(image_path, "teeth", config)
        if isinstance(cached, dict):
            return cached
        data = json.loads(self.tooth_enumeration(image_path, config=config))
        return self._cache_set(image_path, "teeth", data, config)

    def _ensure_teeth_fdi(self, image_path: str, detections_json: Optional[str], config: RunnableConfig = None) -> Dict[str, Any]:
        """Ensure teeth_fdi is available (prefer detections_json, then cache, then compute in preprocess order)."""
        det = self._parse_detections_json(detections_json)
        if det and isinstance(det.get("teeth_fdi"), dict) and "error" not in det.get("teeth_fdi", {}):
            return self._cache_set(image_path, "teeth_fdi", det["teeth_fdi"], config)
        cached = self._cache_get(image_path, "teeth_fdi", config)
        if isinstance(cached, dict) and "error" not in cached:
            return cached
        quadrants_raw = self._ensure_quadrants(image_path, detections_json, config)
        teeth_raw = self._ensure_teeth(image_path, detections_json, config)
        # If the detection service is unavailable, return an error
        if isinstance(quadrants_raw, dict) and "error" in quadrants_raw:
            return self._cache_set(image_path, "teeth_fdi", {"error": f"quadrants: {quadrants_raw.get('error')}"}, config)
        if isinstance(teeth_raw, dict) and "error" in teeth_raw:
            return self._cache_set(image_path, "teeth_fdi", {"error": f"teeth: {teeth_raw.get('error')}"}, config)
        q_norm = _normalize_quadrants_for_merge(quadrants_raw)
        t_norm = _normalize_teeth_for_merge(teeth_raw)
        teeth_fdi = build_fdi_teeth_like_refactor(quadrants=q_norm, teeth=t_norm)
        return self._cache_set(image_path, "teeth_fdi", teeth_fdi, config)

    def _ensure_yolo_disease(self, image_path: str, detections_json: Optional[str], config: RunnableConfig = None) -> Dict[str, Any]:
        """Ensure yolo_disease is available."""
        det = self._parse_detections_json(detections_json)
        if det and isinstance(det.get("yolo_disease"), dict):
            return self._cache_set(image_path, "yolo_disease", det["yolo_disease"], config)
        cached = self._cache_get(image_path, "yolo_disease", config)
        if isinstance(cached, dict):
            return cached
        data = json.loads(self.disease_detection_yolo(image_path, conf=0.25, return_visualization=False, config=config))
        return self._cache_set(image_path, "yolo_disease", data, config)

    def _ensure_tvem_disease(self, image_path: str, detections_json: Optional[str], config: RunnableConfig = None) -> Dict[str, Any]:
        """Ensure tvem_disease is available."""
        det = self._parse_detections_json(detections_json)
        if det and isinstance(det.get("tvem_disease"), dict):
            return self._cache_set(image_path, "tvem_disease", det["tvem_disease"], config)
        cached = self._cache_get(image_path, "tvem_disease", config)
        if isinstance(cached, dict):
            return cached
        data = json.loads(self.disease_detection_tvem(image_path, confidence=0.5, return_vis=False, config=config))
        return self._cache_set(image_path, "tvem_disease", data, config)

    def _ensure_yolo_matched(self, image_path: str, detections_json: Optional[str], config: RunnableConfig = None) -> Dict[str, Any]:
        """Ensure yolo_matched is available (disease matching needs the 'box' field; derive it from 'bbox' if absent)."""
        det = self._parse_detections_json(detections_json)
        if det and isinstance(det.get("yolo_matched"), dict) and "error" not in det.get("yolo_matched", {}):
            return self._cache_set(image_path, "yolo_matched", det["yolo_matched"], config)
        cached = self._cache_get(image_path, "yolo_matched", config)
        if isinstance(cached, dict) and "error" not in cached:
            return cached
        teeth_fdi = self._ensure_teeth_fdi(image_path, detections_json, config)
        yolo = self._ensure_yolo_disease(image_path, detections_json, config)
        dets: List[Dict[str, Any]] = []
        for d in (yolo.get("detections") or []):
            dd = dict(d)
            if "box" not in dd and "bbox" in dd:
                dd["box"] = dd["bbox"]
            dets.append(dd)
        matched = json.loads(self.match_disease_to_tooth(dets, teeth_fdi, iou_threshold=0.3))
        return self._cache_set(image_path, "yolo_matched", matched, config)

    def _ensure_tvem_matched(self, image_path: str, detections_json: Optional[str], config: RunnableConfig = None) -> Dict[str, Any]:
        """Ensure tvem_matched is available (disease matching needs the 'box' field; derive it from 'bbox' if absent)."""
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
        """Ensure anatomy is available."""
        det = self._parse_detections_json(detections_json)
        if det and isinstance(det.get("anatomy"), dict):
            return self._cache_set(image_path, "anatomy", det["anatomy"], config)
        cached = self._cache_get(image_path, "anatomy", config)
        if isinstance(cached, dict):
            return cached
        data = json.loads(self.anatomy_detection(image_path, confidence=0.5, return_vis=False, config=config))
        return self._cache_set(image_path, "anatomy", data, config)

    def _ensure_bone_loss(self, image_path: str, detections_json: Optional[str], config: RunnableConfig = None) -> Dict[str, Any]:
        """Ensure bone_loss is available."""
        det = self._parse_detections_json(detections_json)
        if det and isinstance(det.get("bone_loss"), dict):
            return self._cache_set(image_path, "bone_loss", det["bone_loss"], config)
        cached = self._cache_get(image_path, "bone_loss", config)
        if isinstance(cached, dict):
            return cached
        data = json.loads(self.bone_loss_detection(image_path, confidence=0.5, return_vis=False, config=config))
        return self._cache_set(image_path, "bone_loss", data, config)

    def _parse_detections_json(self, detections_json: Optional[str]) -> Optional[Dict[str, Any]]:
        """Parse an externally provided detections_json (used for reproduction/comparison); return None on failure."""
        if not detections_json or not detections_json.strip():
            return None
        try:
            parsed = json.loads(detections_json)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    def _ensure_quadrants(self, image_path: str, detections_json: Optional[str], config: RunnableConfig = None) -> Dict[str, Any]:
        """Ensure quadrants are available (prefer detections_json, then cache, then call quadrant_detection)."""
        det = self._parse_detections_json(detections_json)
        if det and isinstance(det.get("quadrants"), dict):
            return self._cache_set(image_path, "quadrants", det["quadrants"], config)
        cached = self._cache_get(image_path, "quadrants", config)
        if isinstance(cached, dict):
            return cached
        data = json.loads(self.quadrant_detection(image_path, confidence_threshold=0.5, config=config))
        return self._cache_set(image_path, "quadrants", data, config)

    def _ensure_teeth(self, image_path: str, detections_json: Optional[str], config: RunnableConfig = None) -> Dict[str, Any]:
        """Ensure teeth are available (prefer detections_json, then cache, then call tooth_enumeration)."""
        det = self._parse_detections_json(detections_json)
        if det and isinstance(det.get("teeth"), dict):
            return self._cache_set(image_path, "teeth", det["teeth"], config)
        cached = self._cache_get(image_path, "teeth", config)
        if isinstance(cached, dict):
            return cached
        data = json.loads(self.tooth_enumeration(image_path, config=config))
        return self._cache_set(image_path, "teeth", data, config)

    def _ensure_teeth_fdi(self, image_path: str, detections_json: Optional[str], config: RunnableConfig = None) -> Dict[str, Any]:
        """Ensure teeth_fdi is available (prefer detections_json, then cache, then compute in preprocess order)."""
        det = self._parse_detections_json(detections_json)
        if det and isinstance(det.get("teeth_fdi"), dict) and "error" not in det.get("teeth_fdi", {}):
            return self._cache_set(image_path, "teeth_fdi", det["teeth_fdi"], config)
        cached = self._cache_get(image_path, "teeth_fdi", config)
        if isinstance(cached, dict) and "error" not in cached:
            return cached
        quadrants_raw = self._ensure_quadrants(image_path, detections_json, config)
        teeth_raw = self._ensure_teeth(image_path, detections_json, config)
        # If the detection service is unavailable, return an error
        if isinstance(quadrants_raw, dict) and "error" in quadrants_raw:
            return self._cache_set(image_path, "teeth_fdi", {"error": f"quadrants: {quadrants_raw.get('error')}"}, config)
        if isinstance(teeth_raw, dict) and "error" in teeth_raw:
            return self._cache_set(image_path, "teeth_fdi", {"error": f"teeth: {teeth_raw.get('error')}"}, config)
        q_norm = _normalize_quadrants_for_merge(quadrants_raw)
        t_norm = _normalize_teeth_for_merge(teeth_raw)
        teeth_fdi = build_fdi_teeth_like_refactor(quadrants=q_norm, teeth=t_norm)
        return self._cache_set(image_path, "teeth_fdi", teeth_fdi, config)

    def _ensure_yolo_disease(self, image_path: str, detections_json: Optional[str], config: RunnableConfig = None) -> Dict[str, Any]:
        """Ensure yolo_disease is available."""
        det = self._parse_detections_json(detections_json)
        if det and isinstance(det.get("yolo_disease"), dict):
            return self._cache_set(image_path, "yolo_disease", det["yolo_disease"], config)
        cached = self._cache_get(image_path, "yolo_disease", config)
        if isinstance(cached, dict):
            return cached
        data = json.loads(self.disease_detection_yolo(image_path, conf=0.25, return_visualization=False, config=config))
        return self._cache_set(image_path, "yolo_disease", data, config)

    def _ensure_tvem_disease(self, image_path: str, detections_json: Optional[str], config: RunnableConfig = None) -> Dict[str, Any]:
        """Ensure tvem_disease is available."""
        det = self._parse_detections_json(detections_json)
        if det and isinstance(det.get("tvem_disease"), dict):
            return self._cache_set(image_path, "tvem_disease", det["tvem_disease"], config)
        cached = self._cache_get(image_path, "tvem_disease", config)
        if isinstance(cached, dict):
            return cached
        data = json.loads(self.disease_detection_tvem(image_path, confidence=0.5, return_vis=False, config=config))
        return self._cache_set(image_path, "tvem_disease", data, config)

    def _ensure_yolo_matched(self, image_path: str, detections_json: Optional[str], config: RunnableConfig = None) -> Dict[str, Any]:
        """Ensure yolo_matched is available (disease matching needs the 'box' field; derive it from 'bbox' if absent)."""
        det = self._parse_detections_json(detections_json)
        if det and isinstance(det.get("yolo_matched"), dict) and "error" not in det.get("yolo_matched", {}):
            return self._cache_set(image_path, "yolo_matched", det["yolo_matched"], config)
        cached = self._cache_get(image_path, "yolo_matched", config)
        if isinstance(cached, dict) and "error" not in cached:
            return cached
        teeth_fdi = self._ensure_teeth_fdi(image_path, detections_json, config)
        yolo = self._ensure_yolo_disease(image_path, detections_json, config)
        dets = []
        for d in (yolo.get("detections") or []):
            dd = dict(d)
            if "box" not in dd and "bbox" in dd:
                dd["box"] = dd["bbox"]
            dets.append(dd)
        matched = json.loads(self.match_disease_to_tooth(dets, teeth_fdi, iou_threshold=0.3))
        return self._cache_set(image_path, "yolo_matched", matched, config)

    def _ensure_tvem_matched(self, image_path: str, detections_json: Optional[str], config: RunnableConfig = None) -> Dict[str, Any]:
        """Ensure tvem_matched is available (disease matching needs the 'box' field; derive it from 'bbox' if absent)."""
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
        """Ensure anatomy is available."""
        det = self._parse_detections_json(detections_json)
        if det and isinstance(det.get("anatomy"), dict):
            return self._cache_set(image_path, "anatomy", det["anatomy"], config)
        cached = self._cache_get(image_path, "anatomy", config)
        if isinstance(cached, dict):
            return cached
        data = json.loads(self.anatomy_detection(image_path, confidence=0.5, return_vis=False, config=config))
        return self._cache_set(image_path, "anatomy", data, config)

    def _ensure_bone_loss(self, image_path: str, detections_json: Optional[str], config: RunnableConfig = None) -> Dict[str, Any]:
        """Ensure bone_loss is available."""
        det = self._parse_detections_json(detections_json)
        if det and isinstance(det.get("bone_loss"), dict):
            return self._cache_set(image_path, "bone_loss", det["bone_loss"], config)
        cached = self._cache_get(image_path, "bone_loss", config)
        if isinstance(cached, dict):
            return cached
        data = json.loads(self.bone_loss_detection(image_path, confidence=0.5, return_vis=False, config=config))
        return self._cache_set(image_path, "bone_loss", data, config)

    def quadrant_detection(self, image_path: str, confidence_threshold: float = 0.5, config: RunnableConfig = None) -> str:
        """Detect the 4 quadrants (Q1-Q4) in an OPG image."""
        image_path = self._resolve_image_path(image_path, config)
        # Agent_refactor TVEMClient expects parameter name "confidence" (not "confidence_threshold")
        result = self._call_remote_service(
            "tvem",
            "/detect/4quadrants",
            {"image_path": image_path, "confidence": confidence_threshold, "return_vis": False}
        )
        return json.dumps(result, ensure_ascii=False)
    
    def tooth_enumeration(self, image_path: str, config: RunnableConfig = None) -> str:
        """Detect the position and number (1-8) of each tooth in an OPG image."""
        image_path = self._resolve_image_path(image_path, config)
        result = self._call_remote_service(
            "yolo_enumeration",
            "/detect",
            {"image_path": image_path}
        )
        return json.dumps(result, ensure_ascii=False)
    
    def disease_detection_yolo(
        self,
        image_path: str,
        conf: float = 0.25,
        return_visualization: bool = False,
        config: RunnableConfig = None,
    ) -> str:
        """Use YOLO to detect 4 categories of common dental diseases."""
        image_path = self._resolve_image_path(image_path, config)
        # Agent_refactor YoloDiseaseClient expects return_visualization as a lowercase string "true"/"false"
        result = self._call_remote_service(
            "yolo_disease",
            "/predict",
            {
                "image_path": image_path,
                "conf": conf,
                "return_visualization": str(return_visualization).lower()
            }
        )
        return json.dumps(result, ensure_ascii=False)
    
    def disease_detection_tvem(
        self,
        image_path: str,
        confidence: float = 0.5,
        return_vis: bool = False,
        config: RunnableConfig = None,
    ) -> str:
        """Use TVEM to detect 11 categories of dental diseases."""
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
        # Fix class_id -> class_name mapping (the TVEM service uses a 1-based category file, but the model outputs 0-based class_id)
        # TVEM 11disease COCO_CATEGORIES (1-based id), converted to 0-based class_id:
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
        """Detect regions of periodontal bone loss."""
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
        """Detect important anatomical structures (mandibular canal, maxillary sinus)."""
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
        """Use MedSAM to precisely segment the target inside the given bbox.

        Note: Agent_refactor MedSAMClient handles only a single bbox, so this uses the first box.
        To segment multiple boxes, call this tool multiple times.
        """
        image_path = self._resolve_image_path(image_path, config)
        if not boxes or len(boxes) == 0:
            return json.dumps({"error": "boxes cannot be empty"}, ensure_ascii=False)

        # Agent_refactor MedSAMClient expects a single bbox as a comma-separated string "x1,y1,x2,y2"
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
    
    # ==================== Analysis Tools ====================
    
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
        
        # OPG orientation note
        OPG_ORIENTATION = """

OPG ORIENTATION (CRITICAL):
- Image RIGHT side = Patient's LEFT side
- Image LEFT side = Patient's RIGHT side
- FDI Q1 (upper right) & Q4 (lower right) = Patient's RIGHT = IMAGE LEFT
- FDI Q2 (upper left) & Q3 (lower left) = Patient's LEFT = IMAGE RIGHT
- ALL descriptions MUST use PATIENT orientation (not image orientation)!"""

        # Concise-answer instruction
        CONCISE_INSTRUCTION = """

IMPORTANT INSTRUCTIONS:
- Your response will be sent to an AI Agent for synthesis with other VLM opinions.
- Be CONCISE. Only report CONFIRMED findings with HIGH CONFIDENCE.
- Do NOT report uncertain or ambiguous findings.
- If unsure, do NOT mention the finding at all.
- Keep response brief and factual."""

        # Base prompts (unbiased, English)
        # NOTE: DentalGPT/OralGPT are recommended only for analyzing the full OPG (analysis_type="overall")
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
        """Run image analysis with DentalGPT. The image is resized so its shorter edge is 768 pixels."""
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
        
        # Resize image to shorter-edge 768 and write to a temp file
        resized_bytes = self._resize_image_short_edge(image_path, short_edge=768)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(resized_bytes)
            tmp_path = tmp.name
        try:
            cfg_timeout = self.config.get("tools", {}).get("dental_gpt", {}).get("timeout", 300)
            result = self._call_remote_service(
                "dental_gpt",
                "/analyze",
                {
                    "image_path": tmp_path,
                    "question": prompt
                },
                timeout=cfg_timeout,
                file_field="image"
            )
        finally:
            os.unlink(tmp_path)  # clean up temp file
        # Return only the answer to save tokens (drop the question)
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
        """Run image analysis with OralGPT. The image is resized so its shorter edge is 768 pixels."""
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
        
        # Resize image to shorter-edge 768 and write to a temp file
        resized_bytes = self._resize_image_short_edge(image_path, short_edge=768)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(resized_bytes)
            tmp_path = tmp.name
        try:
            cfg_timeout = self.config.get("tools", {}).get("oral_gpt", {}).get("timeout", 300)
            result = self._call_remote_service(
                "oral_gpt",
                "/analyze",
                {
                    "image_path": tmp_path,
                    "prompt": prompt
                },
                timeout=cfg_timeout,
                file_field="file"
            )
        finally:
            os.unlink(tmp_path)  # clean up temp file
        # Return only the analysis to save tokens (drop the prompt)
        if isinstance(result, dict):
            analysis = result.get("analysis", result.get("answer", ""))
            return json.dumps({"analysis": analysis, "model": "OralGPT"}, ensure_ascii=False)
        return json.dumps(result, ensure_ascii=False)
    
    # Concise-answer requirement used in all VLM prompts
    # OPG orientation note (standard radiographic orientation convention)
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
        Call only OpenAI GPT-5.4; return the same structure as the "openai" key in _call_llm_zoo_apis.
        The image is resized to shorter-edge 768. Temperature 0.3.

        analysis_level controls max_tokens (4x the baseline):
        - "overall": 8192 (full OPG)
        - "quadrant": 4096 (quadrant analysis)
        - "tooth": 1024 (single-tooth analysis)
        """
        # If the given path does not exist, fall back to the context-injected path
        if not os.path.isfile(image_path):
            resolved = current_image_path_ctx.get()
            if resolved and os.path.isfile(resolved):
                image_path = resolved
        # Set max_tokens based on analysis level (4x)
        max_tokens_map = {"overall": 8192, "quadrant": 4096, "tooth": 1024}
        max_tokens = max_tokens_map.get(analysis_level, 8192)
        # Resize image to shorter-edge 768
        resized_bytes = self._resize_image_short_edge(image_path, short_edge=768)
        image_data = base64.standard_b64encode(resized_bytes).decode("utf-8")
        mime_type = "image/png"  # after resizing we always output PNG
        try:
            import openai
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-5.4",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}},
                        {"type": "text", "text": prompt}
                    ]
                }],
                max_completion_tokens=max_tokens,
                temperature=0.3  # GPT temperature 0.3
            )
            return {"model": "gpt-5.4", "response": response.choices[0].message.content, "status": "success", "max_tokens": max_tokens}
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return {"model": "gpt-5.4", "error": str(e), "status": "error"}

    def call_gpt4o_mini_binary(
        self,
        image_path: str,
        prompt: str,
        config: Optional[Any] = None,
        max_tokens: int = 32,
    ) -> Dict[str, Any]:
        """
        Use gpt-4o-mini for very short yes/no classification (e.g. VQA question-type detection).
        Low max_tokens; only yes/no-style answers are expected. Returns a structure compatible with _call_llm_zoo_openai.
        """
        image_path = self._resolve_image_path(image_path, config)
        resized_bytes = self._resize_image_short_edge(image_path, short_edge=768)
        image_data = base64.standard_b64encode(resized_bytes).decode("utf-8")
        mime_type = "image/png"
        try:
            import openai
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                max_completion_tokens=max_tokens,
                temperature=0.0,
            )
            text = (response.choices[0].message.content or "").strip()
            return {"model": "gpt-4o-mini", "response": text, "status": "success", "max_tokens": max_tokens}
        except Exception as e:
            logger.error("gpt-4o-mini binary call failed: %s", e)
            return {"model": "gpt-4o-mini", "error": str(e), "status": "error"}

    def _call_llm_zoo_google(self, image_path: str, prompt: str, analysis_level: str = "overall") -> Dict[str, Any]:
        """
        Call only Google Gemini (Gemini Developer API); GEMINI_API_KEY is preferred.
        Model: gemini-3-flash-preview. Temperature 0.3. Image resized to shorter-edge 768.
        analysis_level controls max_output_tokens (4x the baseline):
        - "overall": 8192, "quadrant": 4096, "tooth": 1024, "binary": 64
        """
        # If the given path does not exist, fall back to the context-injected path
        if not os.path.isfile(image_path):
            resolved = current_image_path_ctx.get()
            if resolved and os.path.isfile(resolved):
                image_path = resolved
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY or GOOGLE_API_KEY is not set")
            return {"model": "gemini-3-flash-preview", "error": "GEMINI_API_KEY or GOOGLE_API_KEY not set", "status": "error"}
        # Set max_output_tokens based on analysis level (4x)
        max_tokens_map = {"overall": 8192, "quadrant": 4096, "tooth": 1024, "binary": 64}
        max_tokens = max_tokens_map.get(analysis_level, 8192)
        # Resize image to shorter-edge 768
        resized_bytes = self._resize_image_short_edge(image_path, short_edge=768)
        image_data = base64.standard_b64encode(resized_bytes).decode("utf-8")
        mime_type = "image/png"  # after resizing we always output PNG
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
            # max_retries=2 keeps Gemini 503 storms bounded — default is 6 which
            # can stretch a single call to 2+ minutes and, combined with
            # langgraph concurrent tool batching, has led to apparent deadlocks.
            model = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.3,  # Gemini temperature 0.3
                max_output_tokens=max_tokens,
                max_retries=2,
                timeout=60,
            )
            response = model.invoke([msg])
            raw_content = response.content if hasattr(response, "content") else str(response)
            # Extract plain text and drop extras/signature (reduces context bloat)
            if isinstance(raw_content, list):
                text_parts = []
                for item in raw_content:
                    if isinstance(item, dict):
                        if "text" in item:
                            text_parts.append(item["text"])
                        # ignore extras/signature
                    elif isinstance(item, str):
                        text_parts.append(item)
                text = "\n".join(text_parts)
            else:
                text = raw_content
            return {"model": model_name, "response": text, "status": "success", "max_tokens": max_tokens}
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
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
        LLM Zoo: call only OpenAI GPT-5.4. Temperature 0.3.
        analysis_level controls max_tokens: overall=2048, quadrant=1024, tooth=256.
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
        LLM Zoo: call only Google Gemini. Temperature 0.3.
        analysis_level controls max_output_tokens: overall=8192, quadrant=4096, tooth=1024 (all 4x).
        """
        image_path = self._resolve_image_path(image_path, config)
        prompt = custom_prompt if custom_prompt else self._build_llm_zoo_prompt(task_type, target_fdi, custom_prompt, focus)
        result = self._call_llm_zoo_google(image_path, prompt, analysis_level)
        return json.dumps(result, ensure_ascii=False)
    
    def _call_llm_zoo_anthropic(self, image_path: str, prompt: str, analysis_level: str = "overall") -> Dict[str, Any]:
        """
        Call Anthropic Claude Opus 4.6 for image analysis (via LangChain ChatAnthropic).
        The image is resized to shorter-edge 768.
        analysis_level controls max_tokens: overall=8192, quadrant=4096, tooth=1024.
        """
        if not os.path.isfile(image_path):
            resolved = current_image_path_ctx.get()
            if resolved and os.path.isfile(resolved):
                image_path = resolved
        max_tokens_map = {"overall": 8192, "quadrant": 4096, "tooth": 1024}
        max_tokens = max_tokens_map.get(analysis_level, 8192)
        resized_bytes = self._resize_image_short_edge(image_path, short_edge=768)
        image_data = base64.standard_b64encode(resized_bytes).decode("utf-8")
        data_url = f"data:image/png;base64,{image_data}"
        model_name = "claude-opus-4-6"
        from langchain_core.messages import HumanMessage
        msg = HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ]
        )
        try:
            from langchain_anthropic import ChatAnthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                return {"model": model_name, "error": "ANTHROPIC_API_KEY not set", "status": "error"}
            model = ChatAnthropic(
                model=model_name,
                anthropic_api_key=api_key,
                max_tokens=max_tokens,
            )
            response = model.invoke([msg])
            raw_content = response.content if hasattr(response, "content") else str(response)
            if isinstance(raw_content, list):
                text = "\n".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in raw_content if isinstance(item, (dict, str))
                )
            else:
                text = raw_content
            return {"model": model_name, "response": text, "status": "success", "max_tokens": max_tokens}
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            return {"model": model_name, "error": str(e), "status": "error"}

    def llm_zoo_anthropic(
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
        LLM Zoo: call Anthropic Claude Opus 4.6.
        analysis_level controls max_tokens: overall=8192, quadrant=4096, tooth=1024.
        """
        image_path = self._resolve_image_path(image_path, config)
        prompt = custom_prompt if custom_prompt else self._build_llm_zoo_prompt(task_type, target_fdi, custom_prompt, focus)
        result = self._call_llm_zoo_anthropic(image_path, prompt, analysis_level)
        return json.dumps(result, ensure_ascii=False)

    def _call_llm_zoo_apis(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """Directly call OpenAI GPT-5.4 and Google Gemini; return openai + google + summary (kept for internal/debug use)."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            openai_future = executor.submit(self._call_llm_zoo_openai, image_path, prompt)
            google_future = executor.submit(self._call_llm_zoo_google, image_path, prompt)
            results = {"openai": openai_future.result(), "google": google_future.result()}
        google_model = results.get("google", {}).get("model", "gemini-3-flash-preview")
        results["summary"] = {
            "models_called": 2,
            "successful": sum(1 for r in [results["openai"], results["google"]] if r.get("status") == "success"),
            "prompt_used": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "models": {"openai": "gpt-5.4", "google": google_model}
        }
        return results
    
    # ==================== Helper Tools ====================

    def calculate_fdi(self, quadrants: Dict, teeth: Dict) -> str:
        """Compute FDI tooth numbers (same as Agent_refactor: best-IoU assignment, keep only valid FDI, deduplicated)."""
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
        """Match disease detection results to specific teeth (executed locally)."""
        matched = match_diseases_to_teeth(diseases, teeth, iou_threshold)
        return json.dumps(matched, ensure_ascii=False)

    def _get_detections_or_run(
        self,
        image_path: str,
        detections_json: Optional[str],
        config: RunnableConfig = None,
    ) -> Dict[str, Any]:
        """
        If detections_json is provided, parse and return it; otherwise run run_all_detections and return the parsed dict.
        Reused by high-level composite tools to avoid running the full detection pipeline repeatedly.
        """
        if detections_json and detections_json.strip():
            try:
                return json.loads(detections_json)
            except json.JSONDecodeError as e:
                logger.warning(f"detections_json parse failed; will re-run detection: {e}")
        raw = self.run_all_detections(image_path, config)
        return json.loads(raw)

    def run_all_detections(self, image_path: str, config: RunnableConfig = None) -> str:
        """
        Run the full detection pipeline in one shot (parallel-optimized version).
        Phase 1 (parallel): quadrant_detection + tooth_enumeration
        Phase 2 (sequential): calculate_fdi (depends on phase 1)
        Phase 3 (parallel): disease_detection_yolo + disease_detection_tvem + bone_loss_detection + anatomy_detection
        Phase 4 (sequential): match_disease_to_tooth (depends on phases 2 and 3)
        Returns a single JSON containing all detections so the agent can fetch everything at once.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        image_path = self._resolve_image_path(image_path, config)
        out: Dict[str, Any] = {}

        # Phase 1: run quadrant_detection and tooth_enumeration in parallel
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

        # Phase 2: calculate_fdi (depends on phase 1)
        try:
            q_norm = _normalize_quadrants_for_merge(out.get("quadrants") or {})
            t_norm = _normalize_teeth_for_merge(out.get("teeth") or {})
            out["teeth_fdi"] = build_fdi_teeth_like_refactor(quadrants=q_norm, teeth=t_norm)
        except Exception as e:
            out["teeth_fdi"] = {"error": str(e)}

        # Phase 3: run yolo_disease, tvem_disease, bone_loss, anatomy in parallel (some may be skipped for ablation)
        skip = set(self._skip_detection_sources or [])
        tasks = []
        if "yolo_disease" not in skip:
            def run_yolo():
                try:
                    return "yolo_disease", json.loads(self.disease_detection_yolo(image_path, conf=0.25, return_visualization=False, config=config))
                except Exception as e:
                    return "yolo_disease", {"error": str(e)}
            tasks.append(("yolo_disease", run_yolo))
        else:
            out["yolo_disease"] = {}
        if "tvem_disease" not in skip:
            def run_tvem():
                try:
                    return "tvem_disease", json.loads(self.disease_detection_tvem(image_path, confidence=0.5, return_vis=False, config=config))
                except Exception as e:
                    return "tvem_disease", {"error": str(e)}
            tasks.append(("tvem_disease", run_tvem))
        else:
            out["tvem_disease"] = {}
        if "bone_loss" not in skip:
            def run_bone_loss():
                try:
                    return "bone_loss", json.loads(self.bone_loss_detection(image_path, confidence=0.5, return_vis=False, config=config))
                except Exception as e:
                    return "bone_loss", {"error": str(e)}
            tasks.append(("bone_loss", run_bone_loss))
        else:
            out["bone_loss"] = {}
        if "anatomy" not in skip:
            def run_anatomy():
                try:
                    return "anatomy", json.loads(self.anatomy_detection(image_path, confidence=0.5, return_vis=False, config=config))
                except Exception as e:
                    return "anatomy", {"error": str(e)}
            tasks.append(("anatomy", run_anatomy))
        else:
            out["anatomy"] = {}
        if tasks:
            with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
                futures = [executor.submit(fn) for _, fn in tasks]
                for future in as_completed(futures):
                    key, val = future.result()
                    out[key] = val

        # Phase 4: match_disease_to_tooth (depends on phases 2 and 3)
        # YOLO: filter out Caries/Deep Caries before matching, keep only Impacted and Periapical Lesion
        # TVEM: 8th tooth (18/28/38/48) with Root Piece -> reported_as impacted tooth
        teeth_fdi = out.get("teeth_fdi")
        if isinstance(teeth_fdi, dict) and "error" not in teeth_fdi:
            # YOLO matched (filtered)
            yolo_src = out.get("yolo_disease")
            if isinstance(yolo_src, dict) and "detections" in yolo_src:
                try:
                    # Filter out Caries and Deep Caries
                    filtered_dets = [
                        d for d in yolo_src["detections"]
                        if (d.get("class_name") or d.get("class") or "").lower() not in ("caries", "deep caries", "deep_caries")
                    ]
                    raw_m = self.match_disease_to_tooth(filtered_dets, teeth_fdi, iou_threshold=0.3)
                    out["yolo_matched"] = json.loads(raw_m)
                except Exception:
                    out["yolo_matched"] = {"error": "match failed"}
            
            # TVEM matched (with 8th tooth Root Piece -> impacted tooth normalization)
            tvem_src = out.get("tvem_disease")
            if isinstance(tvem_src, dict) and "detections" in tvem_src:
                try:
                    # Convert bbox to box (match_diseases_to_teeth uses 'box')
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

    # ==================== High-level Composite Tools ====================
    # High-level tools combine basic tools on demand (with caching) to complete tasks; their results
    # match the same-named fields in run_all_detections for cross-checking and testing.
    # If detections_json is provided, it is reused directly to ensure reproducibility.
    # - get_tooth_by_fdi: returns teeth_fdi[fdi] (number, box, confidence); returns error + available_fdi if not found
    # - get_quadrant: returns quadrant, box, and teeth_fdi list for one or more quadrants (comma-separated supported)
    # - get_tooth_mask: internally calls segment_object on teeth_fdi[fdi].box and returns the segment JSON (mask_contour)
    # - get_status_on_tooth: returns yolo_matched[fdi] (Caries/Deep Caries filtered out) + tvem_matched[fdi]; 8th-tooth Root Piece -> impacted tooth
    # - extraction_risk_near_anatomy: uses teeth_fdi[fdi] + anatomy + segment contour distance; returns risk_near
    # - get_quadrant_teeth: a tooth-level subset of get_quadrant; returns quadrant + teeth: [{fdi, box, confidence}]
    # - list_teeth_with_status: scans yolo_matched (Caries filtered) / tvem_matched by fuzzy match on status_class and returns fdi_list
    # - get_bone_loss_description: uses bone_loss.detections + quadrants + teeth_fdi with IoU; returns English description, quadrants_involved, teeth_involved

    def get_tooth_by_fdi(
        self,
        image_path: str,
        fdi: str,
        detections_json: Optional[str] = None,
        config: RunnableConfig = None,
    ) -> str:
        """
        Get info for a single tooth by FDI (box, confidence, etc.).
        If detections_json (JSON from run_all_detections) is provided, read from it directly; otherwise call basic tools on demand and cache.
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
        Get info for one or more quadrants by name (box, FDI list of teeth in that quadrant).
        Quadrant names may be Upperright/Upperleft/Lowerleft/Lowerright or Q1/Q2/Q3/Q4, comma-separated.
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
        Get the mask of a tooth (MedSAM segmentation). First fetch the box by FDI, then call segment_object.
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
        Get the list of statuses on a tooth (TVEM matched results, with YOLO Caries/Deep Caries filtered out).
        For the 8th teeth (FDI 18/28/38/48), a TVEM "Root Piece" status is automatically reported as "impacted tooth".
        """
        image_path = self._resolve_image_path(image_path, config)
        fdi = str(fdi).strip()
        if len(fdi) == 1:
            fdi = "0" + fdi
        yolo_matched = self._ensure_yolo_matched(image_path, detections_json, config)
        tvem_matched = self._ensure_tvem_matched(image_path, detections_json, config)
        if isinstance(yolo_matched, dict) and "error" in yolo_matched:
            yolo_matched = {}
        if isinstance(tvem_matched, dict) and "error" in tvem_matched:
            tvem_matched = {}
        
        # Merge YOLO (filtered) + TVEM results
        raw_statuses = []
        # YOLO: filter out Caries and Deep Caries, keep only Impacted and Periapical Lesion
        for d in yolo_matched.get(fdi, []):
            cls = (d.get("class_name") or d.get("class") or "").strip().lower()
            if cls in ("caries", "deep caries", "deep_caries"):
                continue  # skip caries
            raw_statuses.append(dict(d))
        # TVEM: all results included
        raw_statuses.extend([dict(d) for d in tvem_matched.get(fdi, [])])
        
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
            
            # Build a compact output keeping only essential fields
            clean_status = {
                "class_id": d.get("class_id"),
                "class_name": d.get("class_name"),
                "confidence": d.get("confidence"),
                "reported_as": reported,
            }
            # Round bbox to 1 decimal
            bbox = d.get("bbox") or d.get("box")
            if bbox and isinstance(bbox, (list, tuple)):
                clean_status["bbox"] = [round(v, 1) for v in bbox]
            # Round bbox_normalized to 4 decimals (if present)
            bbox_norm = d.get("bbox_normalized")
            if bbox_norm and isinstance(bbox_norm, (list, tuple)):
                clean_status["bbox_normalized"] = [round(v, 4) for v in bbox_norm]
            # Redundant fields like mask_contour and box are intentionally omitted
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
        Extraction risk: whether the tooth is close to the maxillary sinus / mandibular canal (Shapely checks mask contour distance).
        Upper teeth (FDI 1x, 2x) are checked against the maxillary sinus; lower teeth (FDI 3x, 4x) against the mandibular canal.
        """
        image_path = self._resolve_image_path(image_path, config)
        fdi = str(fdi).strip()
        if len(fdi) == 1:
            fdi = "0" + fdi
        teeth_fdi = self._ensure_teeth_fdi(image_path, detections_json, config)
        if not isinstance(teeth_fdi, dict) or fdi not in teeth_fdi:
            return json.dumps({"error": f"Tooth FDI={fdi} not found", "risk": None}, ensure_ascii=False)
        # Get the tooth mask contour
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
        # class_id 0 = mandibular canal, 1 = maxillary sinus
        digit = fdi[0] if fdi else ""
        if digit in ("1", "2"):
            target_class_id = 1  # maxillary sinus
            anatomy_name = "maxillary sinus"
            risk_description = "oroantral communication risk"
        elif digit in ("3", "4"):
            target_class_id = 0  # mandibular canal
            anatomy_name = "mandibular canal (inferior alveolar nerve)"
            risk_description = "nerve damage / paresthesia risk"
        else:
            return json.dumps({"fdi": fdi, "risk": None, "message": "Only FDI 1x, 2x, 3x, 4x supported"}, ensure_ascii=False)
        
        # Compute minimum distance
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
        
        # Determine risk level
        if min_distance == float('inf'):
            return json.dumps({
                "fdi": fdi,
                "anatomy": anatomy_name,
                "risk_level": "unknown",
                "message": f"No {anatomy_name} detected in image"
            }, ensure_ascii=False)
        
        # Risk tiering
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
        Get the FDI and info list of all teeth in a quadrant.
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
            # Round to 1 decimal
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
        List all tooth positions (FDI) with a given status class, or return a summary of all detected statuses.

        If status_class is empty or "all": return all detected statuses as {status: [fdi_list]}.
        If status_class is specified: return the fdi_list corresponding to that status (fuzzy match).

        YOLO Caries/Deep Caries are filtered out; only Impacted and Periapical Lesion are kept.
        """
        image_path = self._resolve_image_path(image_path, config)
        yolo_matched = self._ensure_yolo_matched(image_path, detections_json, config)
        tvem_matched = self._ensure_tvem_matched(image_path, detections_json, config)
        if isinstance(yolo_matched, dict) and "error" in yolo_matched:
            yolo_matched = {}
        if isinstance(tvem_matched, dict) and "error" in tvem_matched:
            tvem_matched = {}
        
        # Decide whether to return all statuses
        return_all = not status_class or str(status_class).strip().lower() in ("", "all")

        if return_all:
            # Return all detected statuses: {status_class: [fdi_list]}
            status_to_fdi: Dict[str, set] = {}
            
            # YOLO: filter out Caries/Deep Caries
            for fdi, diseases in yolo_matched.items():
                if fdi == "other_tooth" or not isinstance(diseases, list):
                    continue
                for d in diseases:
                    cls = (d.get("class_name") or d.get("class") or "").strip()
                    cls_lower = cls.lower()
                    if cls_lower in ("caries", "deep caries", "deep_caries"):
                        continue  # skip caries
                    if cls:
                        if cls not in status_to_fdi:
                            status_to_fdi[cls] = set()
                        status_to_fdi[cls].add(fdi)
            
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
            
            # Convert to a sorted list
            result = {status: sorted(fdi_set) for status, fdi_set in status_to_fdi.items()}
            return json.dumps({"all_statuses": result, "status_count": len(result)}, ensure_ascii=False)
        
        else:
            # Return the fdi_list for the specified status
            status_class = str(status_class).strip().lower()
            fdi_set = set()
            
            # YOLO: filter out Caries/Deep Caries
            for fdi, diseases in yolo_matched.items():
                if fdi == "other_tooth" or not isinstance(diseases, list):
                    continue
                for d in diseases:
                    cls = (d.get("class_name") or d.get("class") or "").lower()
                    if cls in ("caries", "deep caries", "deep_caries"):
                        continue  # skip caries
                    if status_class in cls or cls in status_class:
                        fdi_set.add(fdi)
                        break
            
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
        Based on bone-loss detections, describe the affected area as "lower region", "lower left quadrant", etc.
        Uses IoU between bone-loss boxes and quadrant/tooth boxes to determine which quadrants and tooth positions are involved, then aggregates into an English description.
        Returns "X quadrant" for a single quadrant, or an aggregated description such as "lower region" for multiple quadrants.
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
            # Match to best quadrant (highest IoU)
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
            # Match overlapping teeth (IoU >= iou_threshold)
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

    def rag_similar_cases(
        self,
        image_path: str,
        top_k: int = 5,
        fdi_filter: Optional[str] = None,
        config: RunnableConfig = None,
    ) -> str:
        """Search similar dental cases via RAG (MedImageInsights embedding)."""
        image_path = self._resolve_image_path(image_path, config)
        data = {"image_path": image_path, "top_k": str(top_k)}
        if fdi_filter:
            data["fdi_filter"] = fdi_filter
        # Exclude current patient to prevent self-retrieval (data leakage)
        ctx_path = current_image_path_ctx.get()
        if ctx_path:
            patient_id = Path(ctx_path).parent.name
            if len(patient_id) > 8 and "-" in patient_id:  # UUID-like
                data["exclude_patient_id"] = patient_id
        result = self._call_remote_service(
            "dental_rag", "/search", data, timeout=60, file_field="file",
        )
        if isinstance(result, dict) and "error" in result:
            return json.dumps(result, ensure_ascii=False)
        results = result.get("results", [])
        return json.dumps({
            "num_results": len(results),
            "similar_cases": results,
        }, ensure_ascii=False)

    # ============ Region-based combined tools (image-prep + VLM in one call) ============
    # These wrap get_annotated_image + a specific VLM so the Agent only emits one tool call.
    # Used by the local profile (bbox_overlay for DentalGPT/OralGPT, crop for GPT/RAG).

    def _resolve_region_image(
        self,
        target_type: str,
        target_id: Optional[str],
        output_mode: str,
        config: RunnableConfig = None,
    ) -> Dict[str, Any]:
        """Return {image_path, box, analysis_type} for a region; raises via return dict on error.
        target_type='overall' returns the raw OPG path (no annotation)."""
        base_image = self._resolve_image_path("", config) or current_image_path_ctx.get()
        if not base_image or not os.path.isfile(base_image):
            return {"error": "No OPG image path in context; open a case first."}
        tt = (target_type or "").strip().lower()
        if tt in ("overall", "full", "whole"):
            return {"image_path": base_image, "analysis_type": "overall"}
        if tt not in ("tooth", "quadrant"):
            return {"error": f"Invalid target_type {target_type!r}, expected tooth/quadrant/overall"}
        if not target_id:
            return {"error": f"target_id required for target_type={tt}"}
        raw = self.get_annotated_image(
            image_path=base_image, target_type=tt, target_id=str(target_id),
            output_mode=output_mode, config=config,
        )
        info = json.loads(raw)
        if "error" in info:
            return info
        return {"image_path": info["image_path"], "analysis_type": tt, "box": info.get("box")}

    def region_analyze_dentalgpt(
        self,
        target_type: str,
        target_id: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        focus_areas: Optional[List[str]] = None,
        detected_findings: Optional[List[str]] = None,
        config: RunnableConfig = None,
    ) -> str:
        """DentalGPT analysis. Always uses OPG-with-bbox (bbox_overlay) image."""
        prep = self._resolve_region_image(target_type, target_id, "bbox_overlay", config)
        if "error" in prep:
            return json.dumps(prep, ensure_ascii=False)
        return self.dental_expert_analysis(
            image_path=prep["image_path"],
            analysis_type=prep["analysis_type"],
            target_id=target_id,
            custom_prompt=custom_prompt,
            detected_findings=detected_findings,
            focus_areas=focus_areas,
            config=config,
        )

    def region_analyze_oralgpt(
        self,
        target_type: str,
        target_id: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        focus_areas: Optional[List[str]] = None,
        detected_findings: Optional[List[str]] = None,
        config: RunnableConfig = None,
    ) -> str:
        """OralGPT analysis. Always uses OPG-with-bbox (bbox_overlay) image."""
        prep = self._resolve_region_image(target_type, target_id, "bbox_overlay", config)
        if "error" in prep:
            return json.dumps(prep, ensure_ascii=False)
        return self.oral_expert_analysis(
            image_path=prep["image_path"],
            analysis_type=prep["analysis_type"],
            target_id=target_id,
            custom_prompt=custom_prompt,
            detected_findings=detected_findings,
            focus_areas=focus_areas,
            config=config,
        )

    def region_analyze_gpt(
        self,
        target_type: str,
        target_id: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        focus_areas: Optional[List[str]] = None,
        detected_findings: Optional[List[str]] = None,
        config: RunnableConfig = None,
    ) -> str:
        """GPT-5.4 (llm_zoo_openai) analysis. Always uses tooth/quadrant crop image."""
        prep = self._resolve_region_image(target_type, target_id, "crop", config)
        if "error" in prep:
            return json.dumps(prep, ensure_ascii=False)
        level_map = {"overall": "overall", "quadrant": "quadrant", "tooth": "tooth"}
        return self.llm_zoo_openai(
            image_path=prep["image_path"],
            task_type="analysis",
            target_fdi=(target_id if prep["analysis_type"] == "tooth" else None),
            custom_prompt=custom_prompt,
            focus=(",".join(focus_areas) if focus_areas else None),
            analysis_level=level_map.get(prep["analysis_type"], "tooth"),
            config=config,
        )

    def region_analyze_gemini(
        self,
        target_type: str,
        target_id: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        focus_areas: Optional[List[str]] = None,
        detected_findings: Optional[List[str]] = None,
        config: RunnableConfig = None,
    ) -> str:
        """Gemini 3 Flash (llm_zoo_google) analysis. Always uses tooth/quadrant crop image."""
        prep = self._resolve_region_image(target_type, target_id, "crop", config)
        if "error" in prep:
            return json.dumps(prep, ensure_ascii=False)
        level_map = {"overall": "overall", "quadrant": "quadrant", "tooth": "tooth"}
        return self.llm_zoo_google(
            image_path=prep["image_path"],
            task_type="analysis",
            target_fdi=(target_id if prep["analysis_type"] == "tooth" else None),
            custom_prompt=custom_prompt,
            focus=(",".join(focus_areas) if focus_areas else None),
            analysis_level=level_map.get(prep["analysis_type"], "tooth"),
            config=config,
        )

    def region_analyze_claude(
        self,
        target_type: str,
        target_id: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        focus_areas: Optional[List[str]] = None,
        detected_findings: Optional[List[str]] = None,
        config: RunnableConfig = None,
    ) -> str:
        """Claude Opus 4.6 (llm_zoo_anthropic) analysis. Always uses tooth/quadrant crop image."""
        prep = self._resolve_region_image(target_type, target_id, "crop", config)
        if "error" in prep:
            return json.dumps(prep, ensure_ascii=False)
        level_map = {"overall": "overall", "quadrant": "quadrant", "tooth": "tooth"}
        return self.llm_zoo_anthropic(
            image_path=prep["image_path"],
            task_type="analysis",
            target_fdi=(target_id if prep["analysis_type"] == "tooth" else None),
            custom_prompt=custom_prompt,
            focus=(",".join(focus_areas) if focus_areas else None),
            analysis_level=level_map.get(prep["analysis_type"], "tooth"),
            config=config,
        )

    def region_rag_search(
        self,
        target_type: str = "tooth",
        target_id: str = "",
        top_k: int = 5,
        config: RunnableConfig = None,
    ) -> str:
        """RAG similar-case retrieval. Always uses tooth crop image."""
        prep = self._resolve_region_image(target_type, target_id, "crop", config)
        if "error" in prep:
            return json.dumps(prep, ensure_ascii=False)
        return self.rag_similar_cases(
            image_path=prep["image_path"],
            top_k=top_k,
            fdi_filter=(target_id if (target_type or "").lower() == "tooth" else None),
            config=config,
        )

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
        Generate an annotated image: cropped region ("crop") or full OPG with bbox drawn ("bbox_overlay").
        - crop: return only the cropped target region (base64)
        - bbox_overlay: return the full OPG with a red bbox drawn over the target region (base64)
        DentalGPT/OralGPT prefer bbox_overlay; GPT/Gemini accept either.
        """
        from PIL import Image, ImageDraw
        image_path = self._resolve_image_path(image_path, config)
        target_type = str(target_type).strip().lower()
        target_id = str(target_id).strip()
        output_mode = str(output_mode).strip().lower()
        if output_mode not in ("crop", "bbox_overlay"):
            return json.dumps({"error": f"Invalid output_mode: {output_mode}, should be 'crop' or 'bbox_overlay'"}, ensure_ascii=False)
        # Get target box
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
        # Load image
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            return json.dumps({"error": f"Cannot load image: {e}"}, ensure_ascii=False)
        x1, y1, x2, y2 = [int(v) for v in box]
        # Generate output (resize to shorter-edge 768 for VLM use)
        # NOTE: return a temp file path rather than base64 to avoid bloating the message history
        import tempfile
        if output_mode == "crop":
            # Expand box by 20px for better context
            w, h = img.size
            pad = 20
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(w, x2 + pad)
            cy2 = min(h, y2 + pad)
            cropped = img.crop((cx1, cy1, cx2, cy2))
            cropped_resized = self._resize_pil_image_short_edge(cropped, short_edge=768)
            # Save to a temp file
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
            # Save to a temp file
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

    @staticmethod
    def _clean_structured_report(report: dict) -> dict:
        """Post-process structured report: remove empty objects, 'present'/'normal' noise, etc."""
        # Values to strip (sparse representation: omit normal/default)
        NOISE_VALUES = {"present", "normal", "none", "no", "nil", "n/a", ""}

        def _clean(obj):
            if isinstance(obj, dict):
                cleaned = {}
                for k, v in obj.items():
                    v = _clean(v)
                    # Drop noise string values
                    if isinstance(v, str) and v.lower().strip() in NOISE_VALUES:
                        continue
                    # Drop empty containers
                    if v is None or v == {} or v == []:
                        continue
                    # Drop boolean False (e.g. "root_filled": false)
                    if v is False:
                        continue
                    cleaned[k] = v
                return cleaned
            elif isinstance(obj, list):
                return [_clean(item) for item in obj if item is not None and item != {} and item != ""]
            return obj

        report = _clean(report)

        # Remove teeth entries that became empty after cleaning
        if "teeth" in report:
            report["teeth"] = {
                fdi: info for fdi, info in report["teeth"].items()
                if isinstance(info, dict) and info
            }
            if not report["teeth"]:
                del report["teeth"]

        # Remove other top-level keys that became empty
        for key in list(report.keys()):
            if report[key] == {} or report[key] == []:
                del report[key]

        return report

    def convert_to_structured(
        self,
        natural_report: str,
        schema_path: Optional[str] = None,
        config: RunnableConfig = None,
    ) -> str:
        """
        Convert a natural-language diagnostic report into structured JSON.
        The agent decides when to call this tool (typically after analysis and report generation).

        Args:
            natural_report: natural-language diagnostic report
            schema_path: optional path to the Schema&Enum_standard.md file

        Returns:
            Structured report as JSON, or an error message.
        """
        import re

        # Read the schema standard
        if schema_path and os.path.exists(schema_path):
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_content = f.read()[:4000]  # truncated
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
            # Call GPT-5.4 to perform the conversion
            result = self._call_llm_zoo_openai("", conversion_prompt, analysis_level="overall")
            if result.get("status") == "error":
                return json.dumps({"error": result.get("error", "conversion failed"), "status": "error"}, ensure_ascii=False)
            
            response_text = result.get("response", "")
            
            # Extract JSON
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_text.strip()

            structured_report = json.loads(json_str)
            structured_report = self._clean_structured_report(structured_report)
            return json.dumps({
                "structured_report": structured_report,
                "conversion_status": "success"
            }, ensure_ascii=False)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            return json.dumps({
                "structured_report": None,
                "conversion_status": "json_parse_error",
                "error": str(e)
            }, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Structured conversion failed: {e}")
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
        Resolve disease position/classification disagreement via a subagent.

        When multiple VLMs disagree on the position or classification of a finding, this tool will:
        1. Collect context from all relevant opinions
        2. Combine with gold-standard information (tooth list, quadrants, etc.)
        3. Use a subagent (GPT-5.4) to analyze and judge
        4. Return the resolved position (FDI) or classification

        Args:
            image_path: OPG image path
            finding_type: finding type (implant, bone_loss, periapical_lesion, filling, crown, rct, etc.)
            disagreement_type: kind of disagreement ('position' or 'classification')
            vlm_opinions: JSON array of VLM opinions
            gold_standard_info: JSON with gold-standard info
            confirmed_findings: optional JSON of already confirmed findings

        Returns:
            Resolved result as JSON, including the decided position or classification.
        """
        import re

        # Parse input
        try:
            opinions = json.loads(vlm_opinions) if isinstance(vlm_opinions, str) else vlm_opinions
            gold_info = json.loads(gold_standard_info) if isinstance(gold_standard_info, str) else gold_standard_info
            confirmed = json.loads(confirmed_findings) if confirmed_findings and isinstance(confirmed_findings, str) else (confirmed_findings or {})
        except json.JSONDecodeError as e:
            return json.dumps({
                "error": f"JSON parse error: {e}",
                "status": "error"
            }, ensure_ascii=False)
        
        # Build the subagent prompt
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
        
        # Call the subagent (GPT-5.4)
        try:
            result = self._call_llm_zoo_openai("", task_description, analysis_level="quadrant")  # use "quadrant" level (1024 tokens)
            
            if result.get("status") == "error":
                return json.dumps({
                    "error": result.get("error", "subagent call failed"),
                    "status": "error"
                }, ensure_ascii=False)
            
            response_text = result.get("response", "")
            
            # Extract JSON
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Fall back to parsing directly
                json_str = response_text.strip()

            resolved = json.loads(json_str)

            # Add metadata
            resolved["finding_type"] = finding_type
            resolved["disagreement_type"] = disagreement_type
            resolved["vlm_count"] = len(opinions)
            resolved["status"] = "resolved"
            
            logger.info(f"Resolved {disagreement_type} disagreement for {finding_type}: {resolved}")
            return json.dumps(resolved, ensure_ascii=False)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON returned by subagent: {e}")
            return json.dumps({
                "error": f"JSON parse error in subagent response: {e}",
                "raw_response": response_text[:500] if 'response_text' in dir() else "N/A",
                "status": "parse_error"
            }, ensure_ascii=False)
        except Exception as e:
            logger.error(f"resolve_finding_disagreement failed: {e}")
            return json.dumps({
                "error": str(e),
                "status": "error"
            }, ensure_ascii=False)

    def _get_builtin_schema(self) -> str:
        '''Load full Schema&Enum standard from file, fallback to embedded minimal version.'''
        schema_path = Path(__file__).parent.parent / "config" / "schema_enum_standard.md"
        if schema_path.exists():
            return schema_path.read_text(encoding="utf-8")
        return """
### Allowed Root Keys (OPG-only)
dentition_summary, teeth, periodontium, tmj, sinuses, jaws, anatomical_variants

### teeth.{FDI} Fields (snake_case enums)
- status: present|missing|unerupted|erupted|impacted|residual_root|implant|root_filled|supernumerary|exfoliating
- winters_class: vertical|angled|horizontal
- icdas_code: early|moderate|advanced
- pai_score: normal|mild_change|moderate_change|severe_change
- apical_status: scar|active_pathology
- relationship: approximates_iac|approximates_sinus
- root_anomaly: abnormal_morphology|root_rounding
- crown_anomaly: abnormal_morphology|developmental_anomaly
- position_anomaly: transposed|insufficient_space|ectopic_eruption
- caries_location: recurrent|root_surface
- restoration_issue: defective

### periodontium: severity(mild|moderate|severe), bone_loss_pattern(horizontal|vertical), findings([calculus|pericoronitis])
### tmj.{left|right}: morphology(normal|degenerative|developmental)
### sinuses: finding(mucosal_change|opacification|air_fluid_level), severity(mild|moderate|severe)
### jaws: finding(sclerotic_lesion|lucent_lesion|bone_variant|osteopenia|marrow_space_prominence|surgical_hardware)

### Sparse Representation: OMIT normal findings, OMIT teeth.{FDI} if healthy
"""


def create_dental_tools(
    tools_config: Dict[str, Any],
    analysis_only: bool = False,
    return_toolkit: bool = False,
    local_vlm_only: bool = False,
    no_llm_zoo_openai_google: bool = False,
    skip_detection_sources: Optional[List[str]] = None,
    no_detection_tools: bool = False,
    skip_tool_names: Optional[List[str]] = None,
) -> Union[List[StructuredTool], Tuple[List[StructuredTool], "DentalToolkit"]]:
    """
    Create LangChain tool list and optionally return the toolkit instance.
    
    Args:
        tools_config: Tool configuration dict
        analysis_only: If True, only create VLM analysis tools (default False)
        local_vlm_only: If True, use local profile (GPT + Gemini + DentalGPT + OralGPT, 4 VLMs, no Claude, no RAG).
                        If False, use cloud profile (GPT + Gemini + Claude Opus 4.6, 3 VLMs + Tool + RAG, 5 sources).
        no_llm_zoo_openai_google: Ablation 1: do not add llm_zoo_openai / llm_zoo_google (GPT, Gemini)
        skip_detection_sources: Ablation 2: detection sources to skip, e.g. ["yolo_disease","tvem_disease","bone_loss","anatomy"]
        no_detection_tools: Ablation 3: do not add any detection tools (keep only VLM + resolve)
        skip_tool_names: Ablations 2/4: additional tool names to skip, e.g. ["get_bone_loss_description","extraction_risk_near_anatomy"]
        return_toolkit: If True, return (tools, toolkit) tuple for preloading cache
        
    Returns:
        LangChain Tool list, or (tools, toolkit) tuple if return_toolkit=True
    """
    # local_vlm_only no longer disables GPT/Gemini — local profile uses GPT+Gemini+DentalGPT+OralGPT.
    toolkit = DentalToolkit(tools_config, skip_detection_sources=skip_detection_sources)
    skip_tools = set(skip_tool_names or [])
    
    tools = []
    
    # NOTE: run_all_detections is NOT exposed as a tool.
    # Instead, Agent should call toolkit.run_all_detections() at startup to preload cache.
    # Only high-level tools are exposed; base tools are kept for internal use.
    if not analysis_only and not no_detection_tools:
        # Basic detection tools (for ablations 2/4, extraction_risk_near_anatomy and get_bone_loss_description can be dropped via skip_tool_names)
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
        if "extraction_risk_near_anatomy" not in skip_tools:
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
        if "get_bone_loss_description" not in skip_tools:
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
- GPT-5.4 / Gemini: both 'crop' and 'bbox_overlay' work well

Use when: you need to pass a localized/annotated image to VLM tools for focused analysis. Pass the returned image_path to VLM tools.""",
            args_schema=GetAnnotatedImageInput,
        ))
        # RAG similar-case retrieval tool
        rag_config = tools_config.get("tools", {}).get("dental_rag", {})
        if rag_config.get("service", {}).get("base_url") and "rag_similar_cases" not in skip_tools:
            detection_tools_list.append(StructuredTool.from_function(
                func=toolkit.rag_similar_cases,
                name="rag_similar_cases",
                description="""Search similar historical dental cases using RAG (MedImageInsights embedding retrieval).

**ONLY accepts single-tooth crop images.** Do NOT pass quadrant crops, bbox_overlay images, or full OPG images — only tooth-level crops from get_annotated_image(target_type='tooth', output_mode='crop').

Input: image_path (required, MUST be a tooth crop), top_k (default 5), fdi_filter (optional FDI number).
Returns: JSON { num_results, similar_cases: [ { patient_id, fdi_number, report: {status, ...}, similarity } ] }.

**Workflow**: get_annotated_image(target_type='tooth', target_id=FDI, output_mode='crop') → rag_similar_cases(image_path=<returned crop path>)

Use when: you want to find historically similar cases for a specific tooth to support diagnosis.""",
                args_schema=RAGSimilarCasesInput,
            ))
        tools.extend(detection_tools_list)
    
    # VLM analysis tools.
    # Cloud profile: GPT + Gemini + Claude (3 VLMs) + Tool + RAG = 5 sources (≥3/5)
    # Local profile: GPT + DentalGPT + OralGPT (3 VLMs, no Gemini/Claude) + Tool + RAG = 5 sources (≥3/5)
    resolve_desc = "When at least 3 of 5 sources agree a finding EXISTS"
    vlm_tools = []

    if local_vlm_only:
        # Local profile uses 4 COMBINED tools that wrap get_annotated_image + VLM into one call:
        #   - analyze_with_dentalgpt / analyze_with_oralgpt  → bbox_overlay + DentalGPT/OralGPT
        #   - analyze_with_gpt                                → crop + GPT-5.4
        #   - retrieve_similar_cases                          → crop + RAG
        # Agent supplies only (target_type, target_id[, custom_prompt, ...]); no manual image-prep step.
        vlm_tools.extend([
            StructuredTool.from_function(
                func=toolkit.region_analyze_dentalgpt,
                name="analyze_with_dentalgpt",
                description="""DentalGPT (Qwen2.5-VL fine-tuned for dental) — analyses the OPG with a bbox overlay drawn on the target region.
Combines `get_annotated_image(output_mode='bbox_overlay')` + DentalGPT in a single call.

Input: target_type ('tooth' | 'quadrant' | 'overall'), target_id (FDI like '36' for tooth, 'Q1'-'Q4' for quadrant; omit for overall),
optional custom_prompt, focus_areas, detected_findings.
Returns: JSON { analysis, model: 'DentalGPT' }.""",
                args_schema=RegionAnalysisInput,
            ),
            StructuredTool.from_function(
                func=toolkit.region_analyze_oralgpt,
                name="analyze_with_oralgpt",
                description="""OralGPT (Qwen2.5-VL fine-tuned for oral radiology) — analyses the OPG with a bbox overlay drawn on the target region.
Combines `get_annotated_image(output_mode='bbox_overlay')` + OralGPT in a single call.

Input: target_type ('tooth' | 'quadrant' | 'overall'), target_id, optional custom_prompt, focus_areas, detected_findings.
Returns: JSON { analysis, model: 'OralGPT' }.""",
                args_schema=RegionAnalysisInput,
            ),
            StructuredTool.from_function(
                func=toolkit.region_analyze_gpt,
                name="analyze_with_gpt",
                description="""GPT-5.4 (OpenAI) — analyses a CROP of the target region. Combines `get_annotated_image(output_mode='crop')` + GPT in one call.

Input: target_type ('tooth' | 'quadrant' | 'overall'), target_id, optional custom_prompt, focus_areas.
Returns: JSON { model, response, status, max_tokens }.""",
                args_schema=RegionAnalysisInput,
            ),
            StructuredTool.from_function(
                func=toolkit.region_rag_search,
                name="retrieve_similar_cases",
                description="""RAG retrieval over historical dental cases — automatically crops the target tooth and runs similarity search.
Combines `get_annotated_image(output_mode='crop')` + `rag_similar_cases` in one call.

Input: target_type ('tooth' recommended), target_id (FDI, e.g. '36'), top_k (default 5).
Returns: JSON { num_results, similar_cases: [ { patient_id, fdi_number, report, similarity } ] }.""",
                args_schema=RegionRagInput,
            ),
        ])
    else:
        # Cloud profile: GPT + Gemini + Claude — all use combined region tools that
        # wrap get_annotated_image(output_mode='crop') + the specific VLM in one call.
        vlm_tools.extend([
            StructuredTool.from_function(
                func=toolkit.region_analyze_gpt,
                name="analyze_with_gpt",
                description="""GPT-5.4 (OpenAI) — analyses a crop of the target region. Combines `get_annotated_image(output_mode='crop')` + GPT in one call.

Input: target_type ('tooth' | 'quadrant' | 'overall'), target_id (FDI like '36' or 'Q1'-'Q4'; omit for overall),
optional custom_prompt, focus_areas, detected_findings.
Returns: JSON { model, response, status, max_tokens }.""",
                args_schema=RegionAnalysisInput,
            ),
            StructuredTool.from_function(
                func=toolkit.region_analyze_gemini,
                name="analyze_with_gemini",
                description="""Gemini 3 Flash Preview (Google) — analyses a crop of the target region. Combines `get_annotated_image(output_mode='crop')` + Gemini in one call.

Input: target_type ('tooth' | 'quadrant' | 'overall'), target_id, optional custom_prompt, focus_areas, detected_findings.
Returns: JSON { model, response, status, max_tokens }.""",
                args_schema=RegionAnalysisInput,
            ),
            StructuredTool.from_function(
                func=toolkit.region_analyze_claude,
                name="analyze_with_claude",
                description="""Claude Opus 4.6 (Anthropic) — analyses a crop of the target region. Combines `get_annotated_image(output_mode='crop')` + Claude in one call.

Input: target_type ('tooth' | 'quadrant' | 'overall'), target_id, optional custom_prompt, focus_areas, detected_findings.
Returns: JSON { model, response, status, max_tokens }.""",
                args_schema=RegionAnalysisInput,
            ),
            StructuredTool.from_function(
                func=toolkit.region_rag_search,
                name="retrieve_similar_cases",
                description="""RAG retrieval over historical dental cases — automatically crops the target tooth and runs similarity search.
Combines `get_annotated_image(output_mode='crop')` + `rag_similar_cases` in one call.

Input: target_type ('tooth' recommended), target_id (FDI, e.g. '36'), top_k (default 5).
Returns: JSON { num_results, similar_cases }.""",
                args_schema=RegionRagInput,
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
    if local_vlm_only:
        logger.info("Added VLM tools (analyze_with_gpt, analyze_with_dentalgpt, analyze_with_oralgpt, retrieve_similar_cases, resolve_finding_disagreement) — local profile")
    else:
        logger.info("Added VLM tools (analyze_with_gpt, analyze_with_gemini, analyze_with_claude, retrieve_similar_cases, resolve_finding_disagreement) — cloud profile")

    # Structured report conversion tool
    tools.append(StructuredTool.from_function(
        func=toolkit.convert_to_structured,
        name="convert_to_structured",
        description="""Convert the final natural language diagnostic report to structured JSON following the Schema&Enum standard.

**MUST be called as the LAST step** after the natural language report is generated.

Input: natural_report (the full text of your final diagnostic report).
Returns: JSON { structured_report: {...}, conversion_status: "success" }.

The structured report follows the Schema&Enum standard with:
- dentition_summary: total_teeth_detected, not_detected_fdi
- teeth.{FDI}: status, winters_class, icdas_code, pai_score, relationship, etc. (snake_case enums only)
- periodontium: severity, bone_loss_pattern, findings
- tmj, sinuses, jaws: only if abnormal
- Sparse representation: omit normal findings""",
        args_schema=ConvertToStructuredInput,
    ))

    logger.info(f"Created {len(tools)} LangChain tools (analysis_only={analysis_only})")
    if return_toolkit:
        return tools, toolkit
    return tools
