"""
基于 Agent_refactor reference run 的 detections 验收脚本。

目标：
- 对每个在 reference_run 下有 detections/ 的患者，用同一张图调用 Agent_v3 各工具；
- 将输出保存到与 reference 同名的文件（tvem_4quadrants.json, yolo_enumeration.json 等）；
- 验收：每工具正常运行，且输出结构与 reference 一致（含 detections 等关键字段）。

Reference 路径：Agent_refactor/runs/inference_20260128_232845/<patient_id>/detections/
工具与参考文件映射：
  quadrant_detection     -> tvem_4quadrants.json
  tooth_enumeration      -> yolo_enumeration.json
  disease_detection_tvem  -> tvem_11diseases.json
  bone_loss_detection    -> tvem_bone_loss.json
  anatomy_detection      -> tvem_mandibular_maxillary.json
  calculate_fdi          -> teeth_fdi.json（依赖 quadrant + teeth）
  （可选）segment_object  -> 需 bbox，可从 teeth 取第一个
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from opgagent.tools.dental_tools import DentalToolkit, current_image_path_ctx
from opgagent.tools.dental_tools import _normalize_quadrants_for_merge, _normalize_teeth_for_merge
from opgagent.tools.coordinate_utils import build_fdi_teeth_like_refactor
from langchain_core.runnables import RunnableConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 工具名 -> reference detections 文件名（不含路径）
TOOL_TO_REF_FILE: Dict[str, str] = {
    "quadrant_detection": "tvem_4quadrants.json",
    "tooth_enumeration": "yolo_enumeration.json",
    "disease_detection_tvem": "tvem_11diseases.json",
    "bone_loss_detection": "tvem_bone_loss.json",
    "anatomy_detection": "tvem_mandibular_maxillary.json",
}


def load_tools_config() -> Dict[str, Any]:
    """加载工具配置"""
    config_path = Path(__file__).parent.parent / "src" / "opgagent" / "config" / "tools_config.yaml"
    if not config_path.exists():
        logger.warning("工具配置文件不存在: %s，使用空配置", config_path)
        return {}
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def convert_quadrants_to_dict(quadrants_result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """将象限检测 API 响应转为 merge 所需格式"""
    out = {}
    for idx, det in enumerate(quadrants_result.get("detections", [])):
        if not isinstance(det, dict):
            continue
        class_name = det.get("class_name") or det.get("class") or f"quadrant_{idx}"
        name_mapping = {
            "Upper Right": "Upperright", "Upper Left": "Upperleft",
            "Lower Left": "Lowerleft", "Lower Right": "Lowerright",
            "class_0": "Upperleft",
        }
        name = name_mapping.get(class_name, class_name)
        bbox = det.get("bbox") or det.get("box") or []
        out[name] = {"name": name, "box": bbox, "confidence": det.get("confidence", 0.0)}
    return out


def convert_teeth_to_dict(teeth_result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """将牙齿检测 API 响应转为 merge 所需格式"""
    out = {}
    for idx, det in enumerate(teeth_result.get("detections", [])):
        if not isinstance(det, dict):
            continue
        tooth_id = f"t{idx}"
        class_name = det.get("class_name") or det.get("class") or str(idx + 1)
        bbox = det.get("bbox") or det.get("box") or []
        if isinstance(bbox, dict):
            bbox = [bbox.get("x1", 0), bbox.get("y1", 0), bbox.get("x2", 0), bbox.get("y2", 0)]
        out[tooth_id] = {"number": str(class_name), "box": bbox, "confidence": det.get("confidence", 0.0)}
    return out


def structure_ok(got: Dict[str, Any], ref_path: Path) -> Tuple[bool, str]:
    """
    检查工具输出结构是否与 reference 一致（含 detections 或等价键）。
    不要求数值完全一致，只要求可用的结构存在。
    """
    if ref_path.exists():
        try:
            with open(ref_path, "r", encoding="utf-8") as f:
                ref = json.load(f)
        except Exception as e:
            return True, f"reference 读取跳过: {e}"
        # reference 多为 {"detections": [...], "model": ..., ...}
        if "detections" in ref and "detections" not in got:
            return False, "缺少 detections 字段"
        if "detections" in got and not isinstance(got.get("detections"), list):
            return False, "detections 应为 list"
    else:
        if "error" in got and got.get("error"):
            return False, f"工具返回错误: {got.get('error')}"
        if "detections" in got and not isinstance(got.get("detections"), list):
            return False, "detections 应为 list"
    return True, "ok"


def run_tool_and_save(
    toolkit: DentalToolkit,
    patient_id: str,
    image_path: str,
    output_detections_dir: Path,
    config: RunnableConfig,
) -> List[Dict[str, Any]]:
    """
    按 reference 顺序运行各工具，保存到 output_detections_dir，并与 reference 对比。
    返回每步的验收结果列表。
    """
    abs_image_path = str(Path(image_path).resolve())
    token = current_image_path_ctx.set(abs_image_path)
    ref_detections_dir = Path(__file__).parent.parent.parent / "Agent_refactor" / "runs" / "inference_20260128_232845" / patient_id / "detections"
    results_log: List[Dict[str, Any]] = []

    try:
        # 1. quadrant_detection -> tvem_4quadrants.json
        out_file = output_detections_dir / "tvem_4quadrants.json"
        try:
            raw = toolkit.quadrant_detection(abs_image_path, confidence_threshold=0.5, config=config)
            data = json.loads(raw)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            ok, msg = structure_ok(data, ref_detections_dir / "tvem_4quadrants.json")
            n = len(data.get("detections", []))
            results_log.append({"tool": "quadrant_detection", "file": "tvem_4quadrants.json", "status": "ok" if ok else "fail", "message": msg, "detections_count": n})
            if not ok:
                logger.warning("quadrant_detection 验收: %s", msg)
        except Exception as e:
            results_log.append({"tool": "quadrant_detection", "file": "tvem_4quadrants.json", "status": "error", "message": str(e)})
            logger.exception("quadrant_detection 异常")

        # 2. tooth_enumeration -> yolo_enumeration.json
        out_file = output_detections_dir / "yolo_enumeration.json"
        try:
            raw = toolkit.tooth_enumeration(abs_image_path, config=config)
            data = json.loads(raw)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            ok, msg = structure_ok(data, ref_detections_dir / "yolo_enumeration.json")
            n = len(data.get("detections", []))
            results_log.append({"tool": "tooth_enumeration", "file": "yolo_enumeration.json", "status": "ok" if ok else "fail", "message": msg, "detections_count": n})
            if not ok:
                logger.warning("tooth_enumeration 验收: %s", msg)
        except Exception as e:
            results_log.append({"tool": "tooth_enumeration", "file": "yolo_enumeration.json", "status": "error", "message": str(e)})
            logger.exception("tooth_enumeration 异常")

        quadrants_raw: Optional[Dict] = None
        teeth_raw: Optional[Dict] = None
        try:
            with open(output_detections_dir / "tvem_4quadrants.json", "r", encoding="utf-8") as f:
                quadrants_raw = json.load(f)
        except Exception:
            pass
        try:
            with open(output_detections_dir / "yolo_enumeration.json", "r", encoding="utf-8") as f:
                teeth_raw = json.load(f)
        except Exception:
            pass

        # 3. calculate_fdi -> teeth_fdi.json（与 Agent_refactor 一致：build_fdi_teeth_like_refactor）
        out_file = output_detections_dir / "teeth_fdi.json"
        if quadrants_raw and teeth_raw:
            try:
                quadrants_dict = _normalize_quadrants_for_merge(quadrants_raw)
                teeth_dict = _normalize_teeth_for_merge(teeth_raw)
                data = build_fdi_teeth_like_refactor(quadrants=quadrants_dict, teeth=teeth_dict)
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                results_log.append({"tool": "calculate_fdi", "file": "teeth_fdi.json", "status": "ok", "message": "ok", "detections_count": len(data)})
            except Exception as e:
                results_log.append({"tool": "calculate_fdi", "file": "teeth_fdi.json", "status": "error", "message": str(e)})
                logger.exception("calculate_fdi 异常")
        else:
            results_log.append({"tool": "calculate_fdi", "file": "teeth_fdi.json", "status": "skip", "message": "缺少 quadrant 或 teeth"})

        # 5. disease_detection_tvem -> tvem_11diseases.json
        out_file = output_detections_dir / "tvem_11diseases.json"
        try:
            raw = toolkit.disease_detection_tvem(abs_image_path, confidence=0.5, return_vis=False, config=config)
            data = json.loads(raw)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            ok, msg = structure_ok(data, ref_detections_dir / "tvem_11diseases.json")
            n = len(data.get("detections", []))
            results_log.append({"tool": "disease_detection_tvem", "file": "tvem_11diseases.json", "status": "ok" if ok else "fail", "message": msg, "detections_count": n})
        except Exception as e:
            results_log.append({"tool": "disease_detection_tvem", "file": "tvem_11diseases.json", "status": "error", "message": str(e)})
            logger.exception("disease_detection_tvem 异常")

        # 6. bone_loss_detection -> tvem_bone_loss.json
        out_file = output_detections_dir / "tvem_bone_loss.json"
        try:
            raw = toolkit.bone_loss_detection(abs_image_path, confidence=0.5, return_vis=False, config=config)
            data = json.loads(raw)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            ok, msg = structure_ok(data, ref_detections_dir / "tvem_bone_loss.json")
            n = len(data.get("detections", []))
            results_log.append({"tool": "bone_loss_detection", "file": "tvem_bone_loss.json", "status": "ok" if ok else "fail", "message": msg, "detections_count": n})
        except Exception as e:
            results_log.append({"tool": "bone_loss_detection", "file": "tvem_bone_loss.json", "status": "error", "message": str(e)})
            logger.exception("bone_loss_detection 异常")

        # 7. anatomy_detection -> tvem_mandibular_maxillary.json
        out_file = output_detections_dir / "tvem_mandibular_maxillary.json"
        try:
            raw = toolkit.anatomy_detection(abs_image_path, confidence=0.5, return_vis=False, config=config)
            data = json.loads(raw)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            ok, msg = structure_ok(data, ref_detections_dir / "tvem_mandibular_maxillary.json")
            n = len(data.get("detections", []))
            results_log.append({"tool": "anatomy_detection", "file": "tvem_mandibular_maxillary.json", "status": "ok" if ok else "fail", "message": msg, "detections_count": n})
        except Exception as e:
            results_log.append({"tool": "anatomy_detection", "file": "tvem_mandibular_maxillary.json", "status": "error", "message": str(e)})
            logger.exception("anatomy_detection 异常")

        # 8. （可选）segment_object：从 teeth 取第一个 bbox
        if teeth_raw and teeth_raw.get("detections"):
            det0 = teeth_raw["detections"][0]
            bbox = det0.get("bbox") or det0.get("box")
            if isinstance(bbox, dict):
                bbox = [bbox.get("x1", 0), bbox.get("y1", 0), bbox.get("x2", 0), bbox.get("y2", 0)]
            if isinstance(bbox, list) and len(bbox) >= 4:
                try:
                    raw = toolkit.segment_object(abs_image_path, boxes=[bbox], config=config)
                    data = json.loads(raw)
                    out_file = output_detections_dir / "segment_first_tooth.json"
                    with open(out_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    results_log.append({"tool": "segment_object", "file": "segment_first_tooth.json", "status": "ok", "message": "ok"})
                except Exception as e:
                    results_log.append({"tool": "segment_object", "file": "segment_first_tooth.json", "status": "error", "message": str(e)})
            else:
                results_log.append({"tool": "segment_object", "file": "-", "status": "skip", "message": "无有效 bbox"})
        else:
            results_log.append({"tool": "segment_object", "file": "-", "status": "skip", "message": "无 teeth 结果"})

    finally:
        current_image_path_ctx.reset(token)

    return results_log


def main():
    import argparse
    parser = argparse.ArgumentParser(description="基于 reference detections 验收 Agent_v3 工具")
    parser.add_argument("--reference_run_dir", type=str, default=None, help="Reference run 根目录（默认 Agent_refactor/runs/inference_20260128_232845）")
    parser.add_argument("--test_data_dir", type=str, default=None, help="test_data 根目录（默认 workspace test_data）")
    parser.add_argument("--output_dir", type=str, default="runs/tools_validation", help="输出根目录")
    parser.add_argument("--patient", type=str, default=None, help="只跑指定 patient_id，不指定则跑所有有 detections 的 patient")
    args = parser.parse_args()

    # 脚本在 Agent_v3/scripts/，workspace 为 repo 根目录（agent_v2）
    workspace = Path(__file__).resolve().parent.parent.parent
    reference_run_dir = Path(args.reference_run_dir) if args.reference_run_dir else workspace / "Agent_refactor" / "runs" / "inference_20260128_232845"
    test_data_dir = Path(args.test_data_dir) if args.test_data_dir else workspace / "test_data"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 确定要跑的患者：reference 下有 detections/ 且 test_data 下有 image
    patient_ids: List[str] = []
    if args.patient:
        if (reference_run_dir / args.patient / "detections").exists() and (test_data_dir / args.patient).exists():
            patient_ids = [args.patient]
        else:
            logger.error("指定患者 %s 在 reference 或 test_data 中不完整", args.patient)
            sys.exit(1)
    else:
        for d in reference_run_dir.iterdir():
            if d.is_dir() and (d / "detections").exists():
                pid = d.name
                img = test_data_dir / pid / "image_1.png"
                if not img.exists():
                    img = next((test_data_dir / pid).glob("*.png"), None) or next((test_data_dir / pid).glob("*.jpg"), None)
                if img:
                    patient_ids.append(pid)
        patient_ids.sort()

    if not patient_ids:
        logger.error("未找到任何具备 reference detections 且 test_data 图像的患者")
        sys.exit(1)

    logger.info("待验收患者: %s", patient_ids)

    tools_config = load_tools_config()
    toolkit = DentalToolkit(tools_config)
    all_results: Dict[str, List[Dict[str, Any]]] = {}

    for patient_id in patient_ids:
        sample_dir = test_data_dir / patient_id
        image_path = str(sample_dir / "image_1.png")
        if not Path(image_path).exists():
            image_path = str(next(sample_dir.glob("*.png"), None) or next(sample_dir.glob("*.jpg"), None) or "")
        if not image_path or not Path(image_path).exists():
            logger.warning("跳过 %s：无图像", patient_id)
            continue
        out_detections = output_dir / patient_id / "detections"
        out_detections.mkdir(parents=True, exist_ok=True)
        config = RunnableConfig(configurable={"current_image_path": str(Path(image_path).resolve())})
        logger.info("运行患者: %s 图像: %s", patient_id, image_path)
        results_log = run_tool_and_save(toolkit, patient_id, image_path, out_detections, config)
        all_results[patient_id] = results_log
        for r in results_log:
            status = r.get("status", "?")
            msg = r.get("message", "")
            logger.info("  %s %s -> %s  %s", r.get("tool"), r.get("file"), status, msg)

    # 汇总
    summary_file = output_dir / "validation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({"patients": list(all_results.keys()), "results": all_results}, f, ensure_ascii=False, indent=2)
    logger.info("汇总已保存: %s", summary_file)

    failed = sum(1 for per in all_results.values() for r in per if r.get("status") in ("fail", "error"))
    if failed > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
