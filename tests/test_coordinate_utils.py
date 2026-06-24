"""
测试坐标工具
"""

from opgagent.tools.coordinate_utils import (
    calculate_iou,
    generate_fdi_notation,
    assign_teeth_to_quadrants,
    match_diseases_to_teeth,
    merge_detection_results
)


def test_calculate_iou():
    """测试 IoU 计算"""
    box1 = [0, 0, 10, 10]
    box2 = [5, 5, 15, 15]
    
    iou = calculate_iou(box1, box2)
    
    # 交集: [5, 5, 10, 10] = 25
    # 并集: 100 + 100 - 25 = 175
    # IoU = 25 / 175 ≈ 0.143
    assert 0 < iou < 1


def test_generate_fdi_notation():
    """测试 FDI 编号生成"""
    assert generate_fdi_notation("Q1", "1") == "11"
    assert generate_fdi_notation("Q2", "8") == "28"
    assert generate_fdi_notation("Upperright", "1") == "11"
    assert generate_fdi_notation("Lowerleft", "8") == "38"


def test_assign_teeth_to_quadrants():
    """测试牙齿分配到象限"""
    teeth = {
        "tooth1": {"box": [10, 10, 20, 20], "number": "1"},
        "tooth2": {"box": [30, 30, 40, 40], "number": "2"}
    }
    
    quadrants = {
        "q1": {"box": [0, 0, 50, 50], "name": "Q1"},
        "q2": {"box": [50, 0, 100, 50], "name": "Q2"}
    }
    
    assignments = assign_teeth_to_quadrants(teeth, quadrants, iou_threshold=0.1)
    
    assert len(assignments) == 2
    assert "tooth1" in assignments
    assert "tooth2" in assignments


def test_match_diseases_to_teeth():
    """测试疾病匹配到牙齿"""
    diseases = [
        {"box": [12, 12, 18, 18], "class": "caries", "confidence": 0.9}
    ]
    
    teeth = {
        "tooth1": {"box": [10, 10, 20, 20], "fdi": "11"}
    }
    
    matched = match_diseases_to_teeth(diseases, teeth, iou_threshold=0.1)
    
    assert "tooth1" in matched
    assert len(matched["tooth1"]) > 0
