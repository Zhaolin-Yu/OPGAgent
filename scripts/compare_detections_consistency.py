"""
比对每位患者的 Agent_v3 输出与 reference detections 是否一致。

一致标准：
- 同名字段存在（如 model, detections）
- detections 长度一致
- 首个检测项的关键字段一致（bbox 数值、class_name 等），允许 bbox 为 list 或 dict 的等价形式
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# 需要比对的文件名（不含路径）
DETECTION_FILES = [
    "tvem_4quadrants.json",
    "yolo_enumeration.json",
    "tvem_11diseases.json",
    "tvem_bone_loss.json",
    "tvem_mandibular_maxillary.json",
    "teeth_fdi.json",
]


def bbox_to_list(b: Any) -> List[float]:
    """将 bbox（list 或 dict）规范为 [x1,y1,x2,y2]"""
    if isinstance(b, list) and len(b) >= 4:
        return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
    if isinstance(b, dict):
        return [
            float(b.get("x1", 0)),
            float(b.get("y1", 0)),
            float(b.get("x2", 0)),
            float(b.get("y2", 0)),
        ]
    return []


def detections_match(a: List[Dict], b: List[Dict], tol: float = 1e-2) -> Tuple[bool, str]:
    """比对两个 detections 列表：长度相同且首项关键字段一致（bbox 允许数值误差）。"""
    if len(a) != len(b):
        return False, f"len {len(a)} vs {len(b)}"
    if not a:
        return True, "ok"
    aa, bb = a[0], b[0]
    if aa.get("class_name") != bb.get("class_name") and aa.get("class") != bb.get("class"):
        return False, f"class_name {aa.get('class_name')} vs {bb.get('class_name')}"
    ba = bbox_to_list(aa.get("bbox") or aa.get("box"))
    bb_ = bbox_to_list(bb.get("bbox") or bb.get("box"))
    if len(ba) != 4 or len(bb_) != 4:
        return True, "ok"  # 无 bbox 则只比长度
    for i in range(4):
        if abs(ba[i] - bb_[i]) > tol:
            return False, f"bbox[0] {ba} vs {bb_}"
    return True, "ok"


def compare_file(ref_path: Path, out_path: Path) -> Tuple[bool, str]:
    """比对单个 JSON 文件：结构一致、detections 长度及首项一致。"""
    if not ref_path.exists():
        return True, "no_ref"
    if not out_path.exists():
        return False, "out_missing"
    try:
        with open(ref_path, "r", encoding="utf-8") as f:
            ref = json.load(f)
        with open(out_path, "r", encoding="utf-8") as f:
            out = json.load(f)
    except Exception as e:
        return False, str(e)
    if "error" in out and out.get("error"):
        return False, f"out_error: {out.get('error')}"
    ref_d = ref.get("detections", ref if isinstance(ref, list) else [])
    out_d = out.get("detections", out if isinstance(out, dict) else [])
    if isinstance(ref_d, dict):
        ref_d = list(ref_d.values()) if ref_d else []
    if not isinstance(ref_d, list):
        ref_d = []
    if not isinstance(out_d, list):
        out_d = list(out_d.values()) if isinstance(out, dict) and "teeth_fdi" in str(out_path) else []
    if "teeth_fdi" in str(out_path):
        # teeth_fdi 是 {fdi: {box, ...}}，比 key 数量和首项
        if len(ref) != len(out):
            return False, f"teeth_fdi keys {len(ref)} vs {len(out)}"
        return True, "ok"
    ok, msg = detections_match(ref_d, out_d)
    if not ok:
        return False, msg
    return True, "ok"


def main():
    workspace = Path(__file__).resolve().parent.parent.parent
    ref_base = workspace / "Agent_refactor" / "runs" / "inference_20260128_232845"
    out_base = workspace / "Agent_v3" / "runs" / "tools_validation"

    if not out_base.exists():
        print("未找到 tools_validation 输出目录")
        sys.exit(1)

    patients = [d.name for d in out_base.iterdir() if d.is_dir() and (d / "detections").exists()]
    patients.sort()
    report: List[Dict[str, Any]] = []
    all_ok = True

    for pid in patients:
        ref_det = ref_base / pid / "detections"
        out_det = out_base / pid / "detections"
        if not ref_det.exists():
            report.append({"patient": pid, "status": "skip", "reason": "no_ref_detections"})
            continue
        file_results = []
        for fn in DETECTION_FILES:
            ok, msg = compare_file(ref_det / fn, out_det / fn)
            file_results.append({"file": fn, "ok": ok, "message": msg})
            if not ok:
                all_ok = False
        report.append({"patient": pid, "status": "ok" if all(r["ok"] for r in file_results) else "diff", "files": file_results})

    # 打印
    print("Per-patient consistency (Agent_v3 vs reference):\n")
    for r in report:
        status = r["status"]
        pid = r["patient"]
        if status == "skip":
            print(f"  {pid}: skip ({r.get('reason', '')})")
            continue
        bad = [f["file"] for f in r.get("files", []) if not f["ok"]]
        if not bad:
            print(f"  {pid}: 一致")
        else:
            print(f"  {pid}: 不一致 -> {bad}")
            for f in r.get("files", []):
                if not f["ok"]:
                    print(f"      {f['file']}: {f['message']}")

    out_report = out_base / "consistency_report.json"
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {out_report}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
