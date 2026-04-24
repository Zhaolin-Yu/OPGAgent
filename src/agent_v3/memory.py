"""
Agent memory management
Stores the agent's reasoning state and tool-call history
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class AgentMemory:
    """Agent memory management class"""

    def __init__(self, image_path: str, question: str):
        """
        Initialize memory

        Args:
            image_path: OPG image path
            question: user question
        """
        self.image_path = image_path
        self.question = question

        # Tool-call history
        self.tool_calls: List[Dict[str, Any]] = []

        # Detection result cache
        self.detection_results: Dict[str, Any] = {
            "quadrants": {},
            "teeth": {},
            "diseases": [],
            "bone_loss": [],
            "anatomy": []
        }

        # Reasoning step log
        self.reasoning_steps: List[str] = []

        # Timestamps
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
        Record a tool call

        Args:
            tool_name: tool name
            tool_input: tool input
            tool_output: tool output
            iteration: iteration index
            reasoning: agent's reasoning (Thought) content
        """
        call_record = {
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_output": tool_output if isinstance(tool_output, (dict, list, str, int, float, bool)) else str(tool_output),
            "iteration": iteration,
            "timestamp": datetime.now().isoformat()
        }

        # Include reasoning if provided
        if reasoning:
            call_record["reasoning"] = reasoning

        self.tool_calls.append(call_record)
        self.updated_at = datetime.now().isoformat()

        # Update detection result cache
        self._update_detection_cache(tool_name, tool_output)

        logger.debug(f"Tool call recorded: {tool_name} (iteration {iteration})")
    
    def _update_detection_cache(self, tool_name: str, tool_output: Any) -> None:
        """
        Update the detection result cache (based on the Agent_refactor result format)

        Args:
            tool_name: tool name
            tool_output: tool output
        """
        try:
            # Parse JSON string
            if isinstance(tool_output, str):
                try:
                    output_dict = json.loads(tool_output)
                except json.JSONDecodeError:
                    return
            else:
                output_dict = tool_output

            # Update cache by tool type (compatible with Agent_refactor format)
            if tool_name == "quadrant_detection":
                # TVEM 4quadrants format: {"detections": [{"class_name": "1", "bbox": [...], ...}]}
                if "detections" in output_dict:
                    quadrants = {}
                    for det in output_dict["detections"]:
                        class_name = det.get("class_name") or det.get("class") or ""
                        # Map to standard quadrant names
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
                # YOLO enumeration format: {"detections": [...]} or {"predictions": [...]}
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

            elif tool_name == "disease_detection_yolo":
                # YOLO disease format: {"detections": [{"class_name": "...", "bbox": [...], ...}]}
                detections = output_dict.get("detections") or []
                for det in detections:
                    det["source"] = "yolo_disease"
                self.detection_results["diseases"].extend(detections)

            elif tool_name == "disease_detection_tvem":
                # TVEM 11diseases format: {"detections": [{"class_name": "...", "bbox": [...], ...}]}
                detections = output_dict.get("detections") or []
                for det in detections:
                    det["source"] = "tvem_11diseases"
                self.detection_results["diseases"].extend(detections)

            elif tool_name == "bone_loss_detection":
                # TVEM bone_loss format: {"detections": [...]}
                detections = output_dict.get("detections") or []
                self.detection_results["bone_loss"].extend(detections)

            elif tool_name == "anatomy_detection":
                # TVEM mandibular_maxillary format: {"detections": [{"class_id": 0/1, ...}]}
                detections = output_dict.get("detections") or []
                self.detection_results["anatomy"].extend(detections)

            elif tool_name == "calculate_fdi":
                # calculate_fdi output: {"teeth": {fdi: {box, confidence, ...}}}
                if isinstance(output_dict, dict):
                    # If output is a dict, use directly
                    if "teeth" in output_dict:
                        self.detection_results["teeth"] = output_dict["teeth"]
                    else:
                        # If the entire output is the teeth dict itself
                        self.detection_results["teeth"] = output_dict

            elif tool_name == "match_disease_to_tooth":
                # match_disease_to_tooth output: {fdi: [disease1, disease2, ...]}
                # Update per-tooth disease info
                if isinstance(output_dict, dict):
                    for fdi, diseases in output_dict.items():
                        if fdi != "other_tooth" and fdi in self.detection_results.get("teeth", {}):
                            if "diseases" not in self.detection_results["teeth"][fdi]:
                                self.detection_results["teeth"][fdi]["diseases"] = []
                            self.detection_results["teeth"][fdi]["diseases"].extend(diseases)

        except Exception as e:
            logger.warning(f"Failed to update detection cache: {e}", exc_info=True)
    
    def add_reasoning_step(self, step: str) -> None:
        """
        Add a reasoning step

        Args:
            step: reasoning step description
        """
        self.reasoning_steps.append({
            "step": step,
            "timestamp": datetime.now().isoformat()
        })
        self.updated_at = datetime.now().isoformat()

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of memory

        Returns:
            Memory summary dict
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
        Convert to dict

        Returns:
            Memory dict
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
        Save memory to file

        Args:
            file_path: file path
        """
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(f"Memory saved to: {file_path}")
