"""
Live regression test: run high-level tools WITHOUT detections_json.

目标（中文说明）：
- 不传 detections_json，完全走“按需组合基础工具 + 运行期缓存”路径；
- 连续调用多个高级工具，验证：
  1) 工具可正常运行（服务端口已启动时）；
  2) 基础工具被按需调用且被缓存复用（例如 quadrant/tooth 只调用一次）；
  3) 产出结构符合预期。
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from opgagent.tools.dental_tools import DentalToolkit, current_image_path_ctx  # noqa: E402
from langchain_core.runnables import RunnableConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CountingDentalToolkit(DentalToolkit):
    """计数版本 toolkit：统计基础工具调用次数，验证缓存是否生效。"""

    def __init__(self, tools_config: Dict[str, Any]):
        super().__init__(tools_config)
        self.call_counts: Dict[str, int] = {}

    def _inc(self, name: str) -> None:
        self.call_counts[name] = self.call_counts.get(name, 0) + 1

    # 基础工具计数（高级工具会间接调用这些方法）
    def quadrant_detection(self, *args, **kwargs) -> str:  # noqa: ANN001, D401
        self._inc("quadrant_detection")
        return super().quadrant_detection(*args, **kwargs)

    def tooth_enumeration(self, *args, **kwargs) -> str:  # noqa: ANN001
        self._inc("tooth_enumeration")
        return super().tooth_enumeration(*args, **kwargs)

    def disease_detection_tvem(self, *args, **kwargs) -> str:  # noqa: ANN001
        self._inc("disease_detection_tvem")
        return super().disease_detection_tvem(*args, **kwargs)

    def bone_loss_detection(self, *args, **kwargs) -> str:  # noqa: ANN001
        self._inc("bone_loss_detection")
        return super().bone_loss_detection(*args, **kwargs)

    def anatomy_detection(self, *args, **kwargs) -> str:  # noqa: ANN001
        self._inc("anatomy_detection")
        return super().anatomy_detection(*args, **kwargs)

    def segment_object(self, *args, **kwargs) -> str:  # noqa: ANN001
        self._inc("segment_object")
        return super().segment_object(*args, **kwargs)

    def match_disease_to_tooth(self, *args, **kwargs) -> str:  # noqa: ANN001
        self._inc("match_disease_to_tooth")
        return super().match_disease_to_tooth(*args, **kwargs)


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


def safe_json_loads(s: str) -> Dict[str, Any]:
    """解析 JSON 字符串，失败则抛错。"""
    return json.loads(s)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Live regression: high-level tools without detections_json")
    parser.add_argument("--patient", type=str, default="4a14f295-4a12-422c-93e3-46f5c064fe4d")
    parser.add_argument("--test_data_dir", type=str, default=None)
    parser.add_argument("--output", type=str, default="runs/tools_validation/high_level_live_report.json")
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parent.parent.parent
    test_data_dir = Path(args.test_data_dir) if args.test_data_dir else workspace / "test_data"
    image_path = pick_sample_image(test_data_dir, args.patient)
    abs_image_path = str(Path(image_path).resolve())

    tools_config = load_tools_config()
    toolkit = CountingDentalToolkit(tools_config)

    # 注入真实路径，保持与 Agent run 一致
    token = current_image_path_ctx.set(abs_image_path)
    config = RunnableConfig(configurable={"current_image_path": abs_image_path})

    report: Dict[str, Any] = {
        "patient": args.patient,
        "image_path": abs_image_path,
        "steps": [],
        "call_counts": {},
        "passed": False,
        "error": None,
    }

    try:
        # 1) 单牙信息（会触发 quadrant + tooth_enumeration + teeth_fdi 计算）
        tooth_11 = safe_json_loads(toolkit.get_tooth_by_fdi(abs_image_path, fdi="11", detections_json=None, config=config))
        report["steps"].append({"tool": "get_tooth_by_fdi(11)", "ok": "error" not in tooth_11})

        # 2) 象限信息（应复用缓存，不应再次调用 quadrant/tooth）
        q1 = safe_json_loads(toolkit.get_quadrant(abs_image_path, quadrant_name="Q1", detections_json=None, config=config))
        report["steps"].append({"tool": "get_quadrant(Q1)", "ok": "error" not in q1})

        # 3) 单牙 mask（触发 medsam）
        mask_11 = safe_json_loads(toolkit.get_tooth_mask(abs_image_path, fdi="11", detections_json=None, config=config))
        report["steps"].append({"tool": "get_tooth_mask(11)", "ok": bool(mask_11.get("mask_contour")) and not mask_11.get("error")})

        # 4) 单牙疾病（触发 tvem_disease + match）
        dis_11 = safe_json_loads(toolkit.get_diseases_on_tooth(abs_image_path, fdi="11", detections_json=None, config=config))
        report["steps"].append({"tool": "get_diseases_on_tooth(11)", "ok": "diseases" in dis_11 and dis_11.get("fdi") == "11"})

        # 5) 拔牙风险（触发 anatomy + medsam）
        risk_11 = safe_json_loads(toolkit.extraction_risk_near_anatomy(abs_image_path, fdi="11", detections_json=None, config=config))
        report["steps"].append({"tool": "extraction_risk_near_anatomy(11)", "ok": "risk_near" in risk_11})

        # 6) bone loss 描述（触发 bone_loss；应复用 quadrants/teeth_fdi）
        bl = safe_json_loads(toolkit.get_bone_loss_description(abs_image_path, detections_json=None, iou_threshold=0.1, config=config))
        report["steps"].append({"tool": "get_bone_loss_description", "ok": "description" in bl})

        report["call_counts"] = dict(toolkit.call_counts)

        # 断言：quadrant/tooth 只调一次（缓存生效）；疾病检测各一次；骨吸收/解剖至少一次
        expected_min = {
            "quadrant_detection": 1,
            "tooth_enumeration": 1,
            "disease_detection_tvem": 1,
            "bone_loss_detection": 1,
            "anatomy_detection": 1,
            "segment_object": 1,
            "match_disease_to_tooth": 1,
        }
        # 允许 segment_object 多次（mask + risk），但至少一次；quadrant/tooth 应为 1
        hard_equal = {"quadrant_detection": 1, "tooth_enumeration": 1}

        ok = all(step.get("ok") for step in report["steps"])
        for k, v in expected_min.items():
            if toolkit.call_counts.get(k, 0) < v:
                ok = False
        for k, v in hard_equal.items():
            if toolkit.call_counts.get(k, 0) != v:
                ok = False
        report["passed"] = ok

    except Exception as e:
        report["error"] = str(e)
        report["call_counts"] = dict(toolkit.call_counts)
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
    logger.error("FAIL: %s  error=%s  counts=%s", out_path, report.get("error"), report.get("call_counts"))
    sys.exit(1)


if __name__ == "__main__":
    main()

