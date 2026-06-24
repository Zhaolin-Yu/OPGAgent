"""
使用 tools_validation 中已与 reference 一致的 detections 验证 7 个高级组合工具。

流程：
1. 对每个 tools_validation/<patient_id>/detections/ 下的患者，加载各 reference 文件；
2. 组装成 run_all_detections 风格的 JSON（含 yolo_matched/tvem_matched，由 match_disease_to_tooth 计算）；
3. 用 image_path + detections_json 调用 7 个高级工具；
4. 校验输出与 reference 数据一致（get_tooth_by_fdi 与 teeth_fdi[fdi] 一致等）。
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from opgagent.tools.dental_tools import DentalToolkit, current_image_path_ctx
from langchain_core.runnables import RunnableConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 高级工具名（含 bone loss）
HIGH_LEVEL_TOOLS = [
    "get_tooth_by_fdi",
    "get_quadrant",
    "get_tooth_mask",
    "get_diseases_on_tooth",
    "extraction_risk_near_anatomy",
    "get_quadrant_teeth",
    "list_teeth_with_disease",
    "get_bone_loss_description",
]


def load_tools_config() -> Dict[str, Any]:
    """加载工具配置"""
    config_path = Path(__file__).parent.parent / "src" / "opgagent" / "config" / "tools_config.yaml"
    if not config_path.exists():
        return {}
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_box(det: Dict[str, Any]) -> Dict[str, Any]:
    """确保疾病检测项有 box 键（match_disease_to_tooth 使用 box）"""
    if "box" in det:
        return det
    if "bbox" in det:
        return {**det, "box": det["bbox"]}
    return det


def build_run_all_detections_json(
    toolkit: DentalToolkit,
    detections_dir: Path,
) -> Optional[Dict[str, Any]]:
    """
    从 reference detections 目录组装 run_all_detections 风格的 JSON。
    包含 yolo_matched / tvem_matched（由 match_disease_to_tooth 计算）。
    """
    files = {
        "quadrants": "tvem_4quadrants.json",
        "teeth": "yolo_enumeration.json",
        "teeth_fdi": "teeth_fdi.json",
        "tvem_disease": "tvem_11diseases.json",
        "bone_loss": "tvem_bone_loss.json",
        "anatomy": "tvem_mandibular_maxillary.json",
    }
    out: Dict[str, Any] = {}
    for key, fname in files.items():
        path = detections_dir / fname
        if not path.exists():
            logger.warning("缺少 %s", path)
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                out[key] = json.load(f)
        except Exception as e:
            logger.warning("读取 %s 失败: %s", path, e)
            return None

    teeth_fdi = out.get("teeth_fdi")
    if not isinstance(teeth_fdi, dict) or "error" in teeth_fdi:
        return None

    # 计算 yolo_matched / tvem_matched（疾病检测需含 box）
    for disease_key, matched_key in [("tvem_disease", "tvem_matched")]:
        src = out.get(disease_key)
        if isinstance(src, dict) and "detections" in src:
            dets = [ensure_box(d) for d in src["detections"]]
            try:
                raw_m = toolkit.match_disease_to_tooth(dets, teeth_fdi, iou_threshold=0.3)
                out[matched_key] = json.loads(raw_m)
            except Exception as e:
                logger.warning("match_disease_to_tooth %s 失败: %s", matched_key, e)
                out[matched_key] = {}
        else:
            out[matched_key] = {}

    return out


def _sample_fdi(ref_teeth_fdi: Dict[str, Any], prefer: str = "11") -> Optional[str]:
    """从 reference teeth_fdi 中取一个存在的 FDI，优先 prefer（如 11）。"""
    if prefer in ref_teeth_fdi:
        return prefer
    for k in ref_teeth_fdi:
        if k in ("other_tooth",) or not isinstance(k, str) or len(k) != 2:
            continue
        if k[0] in "1234" and k[1] in "12345678":
            return k
    return next(iter(ref_teeth_fdi), None) if ref_teeth_fdi else None


def run_high_level_tools(
    toolkit: DentalToolkit,
    image_path: str,
    detections_json: str,
    config: RunnableConfig,
    patient_id: str,
    ref_teeth_fdi: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    对 7 个高级工具各调用一次，并做基本校验。
    返回每条结果 { tool, status, message, [expected_vs_got] }。
    """
    results: List[Dict[str, Any]] = []
    abs_image_path = str(Path(image_path).resolve())
    sample_fdi = _sample_fdi(ref_teeth_fdi, prefer="11")

    # 1. get_tooth_by_fdi(fdi=sample_fdi) -> 应与 ref teeth_fdi[sample_fdi] 一致
    try:
        if not sample_fdi:
            results.append({"tool": "get_tooth_by_fdi", "status": "skip", "message": "reference 无可用 FDI"})
        else:
            raw = toolkit.get_tooth_by_fdi(abs_image_path, fdi=sample_fdi, detections_json=detections_json, config=config)
            data = json.loads(raw)
            if "error" in data:
                results.append({"tool": "get_tooth_by_fdi", "status": "fail", "message": data.get("error"), "detail": data})
            else:
                ref_t = ref_teeth_fdi.get(sample_fdi)
                ok = "box" in data and "number" in data
                if ok and ref_t and ref_t.get("box") and data.get("box"):
                    ok = len(data["box"]) == 4 and len(ref_t["box"]) == 4
                results.append({"tool": "get_tooth_by_fdi", "status": "ok" if ok else "fail", "message": f"与 reference {sample_fdi} 一致" if ok else "结构或 box 不一致", "expected_keys": list(ref_t.keys()) if ref_t else []})
    except Exception as e:
        results.append({"tool": "get_tooth_by_fdi", "status": "error", "message": str(e)})

    # 2. get_quadrant(quadrant_name="Q1") -> 应有 quadrant, box, teeth_fdi；teeth_fdi 应为 1x
    try:
        raw = toolkit.get_quadrant(abs_image_path, quadrant_name="Q1", detections_json=detections_json, config=config)
        data = json.loads(raw)
        if "error" in data:
            results.append({"tool": "get_quadrant", "status": "fail", "message": data.get("error")})
        else:
            ok = "quadrant" in data and "teeth_fdi" in data
            if ok:
                ok = all(fdi.startswith("1") for fdi in data["teeth_fdi"])
            results.append({"tool": "get_quadrant", "status": "ok" if ok else "fail", "message": "Q1 含 1x 牙位" if ok else "缺少字段或牙位非 1x"})
    except Exception as e:
        results.append({"tool": "get_quadrant", "status": "error", "message": str(e)})

    # 3. get_tooth_mask(fdi=sample_fdi) -> 应返回 mask_contour，无 error
    try:
        if not sample_fdi:
            results.append({"tool": "get_tooth_mask", "status": "skip", "message": "reference 无可用 FDI"})
        else:
            raw = toolkit.get_tooth_mask(abs_image_path, fdi=sample_fdi, detections_json=detections_json, config=config)
            data = json.loads(raw)
            if data.get("error"):
                results.append({"tool": "get_tooth_mask", "status": "fail", "message": data.get("error")})
            elif data.get("mask_contour"):
                results.append({"tool": "get_tooth_mask", "status": "ok", "message": "含 mask_contour"})
            else:
                results.append({"tool": "get_tooth_mask", "status": "fail", "message": "无 mask_contour"})
    except Exception as e:
        results.append({"tool": "get_tooth_mask", "status": "error", "message": str(e)})

    # 4. get_diseases_on_tooth(fdi=sample_fdi) -> 应有 fdi, diseases 列表
    try:
        if not sample_fdi:
            results.append({"tool": "get_diseases_on_tooth", "status": "skip", "message": "reference 无可用 FDI"})
        else:
            raw = toolkit.get_diseases_on_tooth(abs_image_path, fdi=sample_fdi, detections_json=detections_json, config=config)
            data = json.loads(raw)
            if "error" in data and data.get("error"):
                results.append({"tool": "get_diseases_on_tooth", "status": "fail", "message": data.get("error")})
            else:
                ok = "fdi" in data and "diseases" in data and data.get("fdi") == sample_fdi
                results.append({"tool": "get_diseases_on_tooth", "status": "ok" if ok else "fail", "message": f"fdi={sample_fdi}, diseases 数={len(data.get('diseases', []))}"})
    except Exception as e:
        results.append({"tool": "get_diseases_on_tooth", "status": "error", "message": str(e)})

    # 5. extraction_risk_near_anatomy(fdi="38" 或 "48") -> 应有 risk_near 布尔，无 error
    try:
        # 选一个下颌牙（该患者可能无 38/48，选任意存在的下颌牙）
        lower_fdi = next((f for f in ref_teeth_fdi if f.startswith("3") or f.startswith("4")), None)
        if not lower_fdi:
            lower_fdi = "48"
        raw = toolkit.extraction_risk_near_anatomy(abs_image_path, fdi=lower_fdi, proximity_pixels=10.0, detections_json=detections_json, config=config)
        data = json.loads(raw)
        if data.get("error") and "未找到" not in str(data.get("error", "")):
            results.append({"tool": "extraction_risk_near_anatomy", "status": "fail", "message": data.get("error")})
        else:
            ok = "risk_near" in data
            results.append({"tool": "extraction_risk_near_anatomy", "status": "ok" if ok else "fail", "message": f"risk_near={data.get('risk_near')}"})
    except Exception as e:
        results.append({"tool": "extraction_risk_near_anatomy", "status": "error", "message": str(e)})

    # 6. get_quadrant_teeth(quadrant_name="Q1") -> 应有 quadrant, teeth 列表，teeth 为 1x
    try:
        raw = toolkit.get_quadrant_teeth(abs_image_path, quadrant_name="Q1", detections_json=detections_json, config=config)
        data = json.loads(raw)
        if "error" in data:
            results.append({"tool": "get_quadrant_teeth", "status": "fail", "message": data.get("error")})
        else:
            ok = "quadrant" in data and "teeth" in data
            if ok and data.get("teeth"):
                ok = all(t.get("fdi", "").startswith("1") for t in data["teeth"])
            results.append({"tool": "get_quadrant_teeth", "status": "ok" if ok else "fail", "message": f"teeth 数={len(data.get('teeth', []))}"})
    except Exception as e:
        results.append({"tool": "get_quadrant_teeth", "status": "error", "message": str(e)})

    # 7. list_teeth_with_disease(disease_class="Caries") -> 应有 disease_class, fdi_list
    try:
        raw = toolkit.list_teeth_with_disease(abs_image_path, disease_class="Caries", detections_json=detections_json, config=config)
        data = json.loads(raw)
        if "error" in data and data.get("error"):
            results.append({"tool": "list_teeth_with_disease", "status": "fail", "message": data.get("error")})
        else:
            ok = "disease_class" in data and "fdi_list" in data
            results.append({"tool": "list_teeth_with_disease", "status": "ok" if ok else "fail", "message": f"fdi_list 数={len(data.get('fdi_list', []))}"})
    except Exception as e:
        results.append({"tool": "list_teeth_with_disease", "status": "error", "message": str(e)})

    # 8. get_bone_loss_description -> 应有 description、quadrants_involved、teeth_involved
    try:
        raw = toolkit.get_bone_loss_description(abs_image_path, detections_json=detections_json, iou_threshold=0.1, config=config)
        data = json.loads(raw)
        ok = "description" in data and "quadrants_involved" in data and "teeth_involved" in data
        results.append({"tool": "get_bone_loss_description", "status": "ok" if ok else "fail", "message": f"description={data.get('description', '')}, 象限={len(data.get('quadrants_involved', []))}, 牙位={len(data.get('teeth_involved', []))}"})
    except Exception as e:
        results.append({"tool": "get_bone_loss_description", "status": "error", "message": str(e)})

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="使用 reference detections 验证 7 个高级组合工具")
    parser.add_argument("--validation_dir", type=str, default=None, help="tools_validation 根目录（默认 Agent_v3/runs/tools_validation）")
    parser.add_argument("--test_data_dir", type=str, default=None, help="test_data 根目录（默认 workspace test_data）")
    parser.add_argument("--patient", type=str, default=None, help="只跑指定 patient_id")
    parser.add_argument("--output", type=str, default=None, help="结果 JSON 输出路径（默认 validation_dir/high_level_tools_report.json）")
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parent.parent.parent
    validation_dir = Path(args.validation_dir) if args.validation_dir else Path(__file__).resolve().parent.parent / "runs" / "tools_validation"
    test_data_dir = Path(args.test_data_dir) if args.test_data_dir else workspace / "test_data"
    output_path = Path(args.output) if args.output else validation_dir / "high_level_tools_report.json"

    if not validation_dir.exists():
        logger.error("validation 目录不存在: %s", validation_dir)
        sys.exit(1)

    patient_ids: List[str] = []
    if args.patient:
        det_dir = validation_dir / args.patient / "detections"
        if det_dir.exists() and (test_data_dir / args.patient).exists():
            patient_ids = [args.patient]
        else:
            logger.error("指定患者 %s 在 validation 或 test_data 中不完整", args.patient)
            sys.exit(1)
    else:
        for d in validation_dir.iterdir():
            if d.is_dir() and (d / "detections").exists():
                pid = d.name
                img = test_data_dir / pid / "image_1.png"
                if not img.exists():
                    img = next((test_data_dir / pid).glob("*.png"), None) or next((test_data_dir / pid).glob("*.jpg"), None)
                if img:
                    patient_ids.append(pid)
        patient_ids.sort()

    if not patient_ids:
        logger.error("未找到任何具备 validation detections 且 test_data 图像的患者")
        sys.exit(1)

    logger.info("待验证患者: %s", patient_ids)

    tools_config = load_tools_config()
    toolkit = DentalToolkit(tools_config)
    all_results: Dict[str, List[Dict[str, Any]]] = {}
    per_tool_summary: Dict[str, Dict[str, int]] = {t: {"ok": 0, "fail": 0, "error": 0, "skip": 0} for t in HIGH_LEVEL_TOOLS}

    for patient_id in patient_ids:
        detections_dir = validation_dir / patient_id / "detections"
        sample_dir = test_data_dir / patient_id
        image_path = str(sample_dir / "image_1.png")
        if not Path(image_path).exists():
            image_path = str(next(sample_dir.glob("*.png"), None) or next(sample_dir.glob("*.jpg"), None) or "")
        if not image_path or not Path(image_path).exists():
            logger.warning("跳过 %s：无图像", patient_id)
            continue

        assembled = build_run_all_detections_json(toolkit, detections_dir)
        if not assembled:
            logger.warning("跳过 %s：组装 detections 失败", patient_id)
            continue

        with open(detections_dir / "teeth_fdi.json", "r", encoding="utf-8") as f:
            ref_teeth_fdi = json.load(f)

        config = RunnableConfig(configurable={"current_image_path": image_path})
        token = current_image_path_ctx.set(str(Path(image_path).resolve()))
        try:
            detections_json_str = json.dumps(assembled, ensure_ascii=False)
            results_log = run_high_level_tools(
                toolkit, image_path, detections_json_str, config, patient_id, ref_teeth_fdi
            )
            all_results[patient_id] = results_log
            for r in results_log:
                t = r.get("tool", "")
                s = r.get("status", "?")
                per_tool_summary[t][s] = per_tool_summary.get(t, {}).get(s, 0) + 1
            for r in results_log:
                logger.info("  %s %s -> %s  %s", patient_id, r.get("tool"), r.get("status"), r.get("message"))
        finally:
            current_image_path_ctx.reset(token)

    validation_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "patients": list(all_results.keys()),
        "per_patient": all_results,
        "per_tool_summary": per_tool_summary,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("报告已保存: %s", output_path)

    failed = sum(1 for per in all_results.values() for r in per if r.get("status") in ("fail", "error"))
    if failed > 0:
        logger.warning("存在失败或错误: %d 条", failed)
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
