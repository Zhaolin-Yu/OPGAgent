"""
对照回归：不传 detections_json（按需组合） vs 传 run_all_detections 的 detections_json。

目标（中文说明）：
- 验证同一张 OPG 图像下，两条路径产出的“高级工具结果”是否一致；
- 并验证高级工具的关键字段与 run_all_detections 的同名字段保持可对照：
  - get_tooth_by_fdi ↔ detections_json["teeth_fdi"][fdi]
  - get_quadrant ↔ detections_json["quadrants"]（归一化后） + teeth_fdi 前缀过滤
  - get_diseases_on_tooth ↔ detections_json["yolo_matched"/"tvem_matched"]（再加 reported_as 规范化）
  - get_bone_loss_description ↔ detections_json["bone_loss"]（描述是高级工具聚合的派生结果，不直接存在于 JSON）

注意：
- get_tooth_mask / extraction_risk_near_anatomy 都会调用 segment_object（分割），run_all_detections 不包含分割结果；
  因此我们只对比“传/不传 detections_json”两条高级工具输出是否一致，而不与 run_all_detections 直接字段对照。
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from opgagent.tools.dental_tools import DentalToolkit, current_image_path_ctx  # noqa: E402
from langchain_core.runnables import RunnableConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_tools_config() -> Dict[str, Any]:
    """加载 tools_config.yaml。"""
    config_path = Path(__file__).parent.parent / "src" / "opgagent" / "config" / "tools_config.yaml"
    if not config_path.exists():
        return {}
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pick_sample_image(test_data_dir: Path, patient_id: str) -> str:
    """选择样例图像路径。"""
    sample_dir = test_data_dir / patient_id
    img = sample_dir / "image_1.png"
    if img.exists():
        return str(img)
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        found = next(sample_dir.glob(ext), None)
        if found:
            return str(found)
    raise FileNotFoundError(f"image not found for patient {patient_id} under {sample_dir}")


def _safe_load(s: str) -> Any:
    """解析 JSON 字符串。"""
    return json.loads(s)


def _strip_reported_as(diseases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """去掉高阶派生字段 reported_as，用于与 run_all_detections 的 matched 结果对照。"""
    out = []
    for d in diseases:
        dd = dict(d)
        dd.pop("reported_as", None)
        out.append(dd)
    return out


def _json_equal(a: Any, b: Any) -> bool:
    """宽松一致性：用 canonical JSON 比较，避免 key 顺序影响。"""
    try:
        return json.dumps(a, ensure_ascii=False, sort_keys=True) == json.dumps(b, ensure_ascii=False, sort_keys=True)
    except Exception:
        return a == b


def _compare(name: str, a: Any, b: Any) -> Dict[str, Any]:
    """生成对比结果摘要。"""
    eq = _json_equal(a, b)
    return {"name": name, "equal": eq, "a": a if not eq else None, "b": b if not eq else None}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compare: live (no detections_json) vs run_all_detections path")
    parser.add_argument("--patient", type=str, default="4a14f295-4a12-422c-93e3-46f5c064fe4d")
    parser.add_argument("--fdi", type=str, default="11")
    parser.add_argument("--test_data_dir", type=str, default=None)
    parser.add_argument("--output", type=str, default="runs/tools_validation/live_vs_all_detections_report.json")
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parent.parent.parent
    test_data_dir = Path(args.test_data_dir) if args.test_data_dir else workspace / "test_data"
    image_path = pick_sample_image(test_data_dir, args.patient)
    abs_image_path = str(Path(image_path).resolve())
    fdi = str(args.fdi).strip()
    if len(fdi) == 1:
        fdi = "0" + fdi

    tools_config = load_tools_config()
    toolkit = DentalToolkit(tools_config)

    token = current_image_path_ctx.set(abs_image_path)
    config = RunnableConfig(configurable={"current_image_path": abs_image_path})

    report: Dict[str, Any] = {
        "patient": args.patient,
        "image_path": abs_image_path,
        "fdi": fdi,
        "passed": False,
        "comparisons": [],
        "notes": [],
        "error": None,
    }

    try:
        # 1) 先跑 all_detections，生成 detections_json
        detections_json = toolkit.run_all_detections(abs_image_path, config=config)
        det = _safe_load(detections_json)

        # 2) 高级工具（两条路径）
        live_tooth = _safe_load(toolkit.get_tooth_by_fdi(abs_image_path, fdi=fdi, detections_json=None, config=config))
        json_tooth = _safe_load(toolkit.get_tooth_by_fdi(abs_image_path, fdi=fdi, detections_json=detections_json, config=config))
        report["comparisons"].append(_compare("get_tooth_by_fdi: live vs with_detections_json", live_tooth, json_tooth))

        # 与 all_detections 的 teeth_fdi 字段对照（如果存在）
        det_tooth = (det.get("teeth_fdi") or {}).get(fdi)
        if det_tooth is not None:
            report["comparisons"].append(_compare("get_tooth_by_fdi: with_detections_json vs detections_json.teeth_fdi[fdi]", json_tooth, det_tooth))
        else:
            report["notes"].append("detections_json.teeth_fdi[fdi] 缺失：无法与 run_all_detections 直接对照（可能该牙不存在）")

        # quadrant：用 Q1 固定对照
        live_q = _safe_load(toolkit.get_quadrant(abs_image_path, quadrant_name="Q1", detections_json=None, config=config))
        json_q = _safe_load(toolkit.get_quadrant(abs_image_path, quadrant_name="Q1", detections_json=detections_json, config=config))
        report["comparisons"].append(_compare("get_quadrant(Q1): live vs with_detections_json", live_q, json_q))

        # diseases：对比高级工具输出 + 对照 all_detections 的 matched（忽略 reported_as）
        live_dis = _safe_load(toolkit.get_diseases_on_tooth(abs_image_path, fdi=fdi, detections_json=None, config=config))
        json_dis = _safe_load(toolkit.get_diseases_on_tooth(abs_image_path, fdi=fdi, detections_json=detections_json, config=config))
        report["comparisons"].append(_compare("get_diseases_on_tooth: live vs with_detections_json", live_dis, json_dis))

        det_yolo = (det.get("yolo_matched") or {}).get(fdi, [])
        det_tvem = (det.get("tvem_matched") or {}).get(fdi, [])
        det_dis_combined = list(det_yolo) + list(det_tvem)
        if det_dis_combined:
            report["comparisons"].append(
                _compare(
                    "get_diseases_on_tooth: with_detections_json(no reported_as) vs detections_json.yolo_matched+tvem_matched",
                    _strip_reported_as(json_dis.get("diseases", [])),
                    det_dis_combined,
                )
            )
        else:
            report["notes"].append("detections_json 中该牙无 matched 疾病（yolo_matched/tvem_matched 为空或缺失）")

        # bone loss description：只对比两条高级路径一致性（run_all_detections 里没有 description 字段）
        live_bl = _safe_load(toolkit.get_bone_loss_description(abs_image_path, detections_json=None, iou_threshold=0.1, config=config))
        json_bl = _safe_load(toolkit.get_bone_loss_description(abs_image_path, detections_json=detections_json, iou_threshold=0.1, config=config))
        report["comparisons"].append(_compare("get_bone_loss_description: live vs with_detections_json", live_bl, json_bl))
        report["notes"].append("bone_loss_description 是派生聚合字段，不要求与 detections_json 的原始 bone_loss 逐字段一致")

        # mask / risk：只能对比两条高级路径一致性（run_all_detections 不含分割结果）
        live_mask = _safe_load(toolkit.get_tooth_mask(abs_image_path, fdi=fdi, detections_json=None, config=config))
        json_mask = _safe_load(toolkit.get_tooth_mask(abs_image_path, fdi=fdi, detections_json=detections_json, config=config))
        # 为避免轮廓点顺序导致的误判，这里只比较是否都有 mask_contour 以及点数量
        live_mask_sig = {"has_mask_contour": bool(live_mask.get("mask_contour")), "n_points": len(live_mask.get("mask_contour") or [])}
        json_mask_sig = {"has_mask_contour": bool(json_mask.get("mask_contour")), "n_points": len(json_mask.get("mask_contour") or [])}
        report["comparisons"].append(_compare("get_tooth_mask: live vs with_detections_json (signature)", live_mask_sig, json_mask_sig))
        report["notes"].append("get_tooth_mask 的完整 contour 点序可能不同，这里用 signature（是否存在+点数）对比")

        live_risk = _safe_load(toolkit.extraction_risk_near_anatomy(abs_image_path, fdi=fdi, detections_json=None, proximity_pixels=10.0, config=config))
        json_risk = _safe_load(toolkit.extraction_risk_near_anatomy(abs_image_path, fdi=fdi, detections_json=detections_json, proximity_pixels=10.0, config=config))
        # 风险字段必须一致
        live_risk_sig = {"risk_near": live_risk.get("risk_near"), "anatomy": live_risk.get("anatomy"), "fdi": live_risk.get("fdi")}
        json_risk_sig = {"risk_near": json_risk.get("risk_near"), "anatomy": json_risk.get("anatomy"), "fdi": json_risk.get("fdi")}
        report["comparisons"].append(_compare("extraction_risk_near_anatomy: live vs with_detections_json (signature)", live_risk_sig, json_risk_sig))

        report["passed"] = all(c["equal"] for c in report["comparisons"])

    except Exception as e:
        report["error"] = str(e)
        report["passed"] = False
    finally:
        current_image_path_ctx.reset(token)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if report["passed"]:
        logger.info("PASS: %s", out_path)
        sys.exit(0)
    logger.error("FAIL: %s  error=%s", out_path, report.get("error"))
    for c in report["comparisons"]:
        if not c.get("equal"):
            logger.error("DIFF: %s", c.get("name"))
    sys.exit(1)


if __name__ == "__main__":
    main()

