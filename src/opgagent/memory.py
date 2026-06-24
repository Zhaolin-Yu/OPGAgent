"""
Agent Memory 管理
负责存储 agent 的推理状态和工具调用历史
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class AgentMemory:
    """Agent Memory 管理类"""
    
    def __init__(self, image_path: str, question: str):
        """
        初始化 Memory
        
        Args:
            image_path: OPG 图像路径
            question: 用户问题
        """
        self.image_path = image_path
        self.question = question
        
        # 工具调用历史
        self.tool_calls: List[Dict[str, Any]] = []
        
        # 检测结果缓存
        self.detection_results: Dict[str, Any] = {
            "quadrants": {},
            "teeth": {},
            "diseases": [],
            "bone_loss": [],
            "anatomy": []
        }
        
        # 推理步骤记录
        self.reasoning_steps: List[str] = []
        
        # 时间戳
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
    
    def add_tool_call(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: Any,
        iteration: int,
        reasoning: str = ""
    ) -> None:
        """
        记录工具调用
        
        Args:
            tool_name: 工具名称
            tool_input: 工具输入
            tool_output: 工具输出
            iteration: 迭代次数
            reasoning: Agent 的推理（Thought）内容
        """
        call_record = {
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_output": tool_output if isinstance(tool_output, (dict, list, str, int, float, bool)) else str(tool_output),
            "iteration": iteration,
            "timestamp": datetime.now().isoformat()
        }
        
        # 添加 reasoning（如果有）
        if reasoning:
            call_record["reasoning"] = reasoning
        
        self.tool_calls.append(call_record)
        self.updated_at = datetime.now().isoformat()
        
        # 更新检测结果缓存
        self._update_detection_cache(tool_name, tool_output)
        
        logger.debug(f"工具调用已记录: {tool_name} (迭代 {iteration})")
    
    def _update_detection_cache(self, tool_name: str, tool_output: Any) -> None:
        """
        更新检测结果缓存（基于 Agent_refactor 的结果格式）
        
        Args:
            tool_name: 工具名称
            tool_output: 工具输出
        """
        try:
            # 解析 JSON 字符串
            if isinstance(tool_output, str):
                try:
                    output_dict = json.loads(tool_output)
                except json.JSONDecodeError:
                    return
            else:
                output_dict = tool_output
            
            # 根据工具类型更新缓存（兼容 Agent_refactor 格式）
            if tool_name == "quadrant_detection":
                # TVEM 4quadrants 格式：{"detections": [{"class_name": "1", "bbox": [...], ...}]}
                if "detections" in output_dict:
                    quadrants = {}
                    for det in output_dict["detections"]:
                        class_name = det.get("class_name") or det.get("class") or ""
                        # 映射到标准象限名
                        quadrant_map = {
                            "1": "Upperright", "2": "Upperleft",
                            "3": "Lowerleft", "4": "Lowerright"
                        }
                        name = quadrant_map.get(class_name, class_name)
                        if name in {"Upperright", "Upperleft", "Lowerleft", "Lowerright"}:
                            bbox = det.get("bbox") or det.get("box") or []
                            quadrants[name] = {
                                "name": name,
                                "box": bbox,
                                "confidence": det.get("confidence", 0)
                            }
                    self.detection_results["quadrants"] = quadrants
                elif "quadrants" in output_dict:
                    self.detection_results["quadrants"] = output_dict["quadrants"]
            
            elif tool_name == "tooth_enumeration":
                # YOLO enumeration 格式：{"detections": [...]} 或 {"predictions": [...]}
                detections = output_dict.get("detections") or output_dict.get("predictions") or []
                teeth = {}
                for idx, det in enumerate(detections):
                    tooth_id = f"tooth_{idx}"
                    teeth[tooth_id] = {
                        "number": det.get("number") or det.get("class"),
                        "box": det.get("bbox") or det.get("box") or [],
                        "confidence": det.get("confidence", 0)
                    }
                self.detection_results["teeth"] = teeth

            elif tool_name == "disease_detection_tvem":
                # TVEM 11diseases 格式：{"detections": [{"class_name": "...", "bbox": [...], ...}]}
                detections = output_dict.get("detections") or []
                for det in detections:
                    det["source"] = "tvem_11diseases"
                self.detection_results["diseases"].extend(detections)
            
            elif tool_name == "bone_loss_detection":
                # TVEM bone_loss 格式：{"detections": [...]}
                detections = output_dict.get("detections") or []
                self.detection_results["bone_loss"].extend(detections)
            
            elif tool_name == "anatomy_detection":
                # TVEM mandibular_maxillary 格式：{"detections": [{"class_id": 0/1, ...}]}
                detections = output_dict.get("detections") or []
                self.detection_results["anatomy"].extend(detections)
            
            elif tool_name == "calculate_fdi":
                # calculate_fdi 输出：{"teeth": {fdi: {box, confidence, ...}}}
                if isinstance(output_dict, dict):
                    # 如果输出是字典，直接使用
                    if "teeth" in output_dict:
                        self.detection_results["teeth"] = output_dict["teeth"]
                    else:
                        # 如果整个输出就是 teeth 字典
                        self.detection_results["teeth"] = output_dict
            
            elif tool_name == "match_disease_to_tooth":
                # match_disease_to_tooth 输出：{fdi: [disease1, disease2, ...]}
                # 更新牙齿的疾病信息
                if isinstance(output_dict, dict):
                    for fdi, diseases in output_dict.items():
                        if fdi != "other_tooth" and fdi in self.detection_results.get("teeth", {}):
                            if "diseases" not in self.detection_results["teeth"][fdi]:
                                self.detection_results["teeth"][fdi]["diseases"] = []
                            self.detection_results["teeth"][fdi]["diseases"].extend(diseases)
        
        except Exception as e:
            logger.warning(f"更新检测结果缓存失败: {e}", exc_info=True)
    
    def add_reasoning_step(self, step: str) -> None:
        """
        添加推理步骤
        
        Args:
            step: 推理步骤描述
        """
        self.reasoning_steps.append({
            "step": step,
            "timestamp": datetime.now().isoformat()
        })
        self.updated_at = datetime.now().isoformat()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取 Memory 摘要
        
        Returns:
            Memory 摘要字典
        """
        return {
            "image_path": self.image_path,
            "question": self.question,
            "total_tool_calls": len(self.tool_calls),
            "detection_summary": {
                "quadrants": len(self.detection_results.get("quadrants", {})),
                "teeth": len(self.detection_results.get("teeth", {})),
                "diseases": len(self.detection_results.get("diseases", [])),
                "bone_loss": len(self.detection_results.get("bone_loss", [])),
                "anatomy": len(self.detection_results.get("anatomy", []))
            },
            "reasoning_steps_count": len(self.reasoning_steps),
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            Memory 字典
        """
        return {
            "image_path": self.image_path,
            "question": self.question,
            "tool_calls": self.tool_calls,
            "detection_results": self.detection_results,
            "reasoning_steps": self.reasoning_steps,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    def save_to_file(self, file_path: str) -> None:
        """
        保存到文件
        
        Args:
            file_path: 文件路径
        """
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"Memory 已保存到: {file_path}")
