"""
分散式思考系統實現 (Distributed Thinking System Implementation)

This module implements a comprehensive distributed thinking system that demonstrates
"thinking without a brain" through various biological, engineering, and philosophical approaches.

無腦思考的科學實現 - 整合章魚觸手神經網、群體智慧、延展心智理論
Scientific implementation of brainless thinking - integrating octopus neural networks,
swarm intelligence, and extended mind theory.
"""

import time
import random
import math
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import threading
import queue
import json


class ThinkingMode(Enum):
    """思考模式 (Thinking Modes)"""
    BIOLOGICAL_DISTRIBUTED = "biological_distributed"  # 生物分散式
    SWARM_INTELLIGENCE = "swarm_intelligence"  # 群體智慧
    EXTENDED_MIND = "extended_mind"  # 延展心智
    EMBODIED_COGNITION = "embodied_cognition"  # 具身認知
    CHEMICAL_SIGNALING = "chemical_signaling"  # 化學信號
    EMERGENT_COLLECTIVE = "emergent_collective"  # 涌現集體


@dataclass
class Signal:
    """信號類 (Signal Class)"""
    signal_type: str
    intensity: float
    source_id: str
    timestamp: float
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentalStimulus:
    """環境刺激 (Environmental Stimulus)"""
    stimulus_type: str
    position: Tuple[float, float, float]
    intensity: float
    duration: float
    properties: Dict[str, Any] = field(default_factory=dict)


class DistributedNode(ABC):
    """分散式節點抽象基類 (Abstract Base Class for Distributed Nodes)"""
    
    def __init__(self, node_id: str, position: Tuple[float, float, float] = (0, 0, 0)):
        self.node_id = node_id
        self.position = position
        self.local_state = {}
        self.neighbors = []
        self.signal_queue = queue.Queue()
        self.memory = {}
        self.active = True
        self.last_update = time.time()
    
    @abstractmethod
    def process_signal(self, signal: Signal) -> Optional[Signal]:
        """處理接收到的信號 (Process received signal)"""
        pass
    
    @abstractmethod
    def generate_local_response(self, stimulus: EnvironmentalStimulus) -> Any:
        """生成局部響應 (Generate local response)"""
        pass
    
    def add_neighbor(self, neighbor: 'DistributedNode'):
        """添加鄰居節點 (Add neighbor node)"""
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)
    
    def send_signal(self, signal: Signal):
        """向鄰居發送信號 (Send signal to neighbors)"""
        for neighbor in self.neighbors:
            neighbor.receive_signal(signal)
    
    def receive_signal(self, signal: Signal):
        """接收信號 (Receive signal)"""
        self.signal_queue.put(signal)
    
    def update(self):
        """更新節點狀態 (Update node state)"""
        if not self.active:
            return
        
        # 處理所有待處理信號 (Process all pending signals)
        while not self.signal_queue.empty():
            try:
                signal = self.signal_queue.get_nowait()
                response = self.process_signal(signal)
                if response:
                    self.send_signal(response)
            except queue.Empty:
                break
        
        self.last_update = time.time()


class OctopusTentacleNode(DistributedNode):
    """章魚觸手節點 (Octopus Tentacle Node)"""
    
    def __init__(self, node_id: str, position: Tuple[float, float, float] = (0, 0, 0)):
        super().__init__(node_id, position)
        self.autonomy_level = 0.7  # 自主性程度
        self.tactile_sensitivity = 0.8
        self.local_memory_size = 100
        self.motor_response_threshold = 0.5
        self.learning_rate = 0.1
        
        # 觸手特有屬性 (Tentacle-specific properties)
        self.sucker_states = [0.0] * 20  # 20個吸盤狀態
        self.muscle_tension = 0.0
        self.proprioception = {"angle": 0.0, "extension": 0.0}
    
    def process_signal(self, signal: Signal) -> Optional[Signal]:
        """處理信號 - 章魚觸手式"""
        if signal.signal_type == "tactile":
            return self._process_tactile_signal(signal)
        elif signal.signal_type == "chemical":
            return self._process_chemical_signal(signal)
        elif signal.signal_type == "coordination":
            return self._process_coordination_signal(signal)
        return None
    
    def _process_tactile_signal(self, signal: Signal) -> Optional[Signal]:
        """處理觸覺信號 (Process tactile signal)"""
        intensity = signal.intensity * self.tactile_sensitivity
        
        # 局部決策：是否需要進一步探索 (Local decision: further exploration needed?)
        if intensity > self.motor_response_threshold:
            # 激活相應的吸盤 (Activate corresponding suckers)
            sucker_index = int(signal.payload.get("contact_point", 0)) % len(self.sucker_states)
            self.sucker_states[sucker_index] = intensity
            
            # 生成協調信號給其他觸手 (Generate coordination signal to other tentacles)
            return Signal(
                signal_type="coordination",
                intensity=intensity * 0.5,
                source_id=self.node_id,
                timestamp=time.time(),
                payload={
                    "action": "grasp_assistance",
                    "location": signal.payload.get("contact_point"),
                    "grip_strength": intensity
                }
            )
        return None
    
    def _process_chemical_signal(self, signal: Signal) -> Optional[Signal]:
        """處理化學信號 (Process chemical signal)"""
        chemical_type = signal.payload.get("chemical_type", "unknown")
        
        if chemical_type == "food":
            # 食物信號：增加探索行為 (Food signal: increase exploratory behavior)
            self.autonomy_level = min(1.0, self.autonomy_level + 0.1)
            return Signal(
                signal_type="exploration",
                intensity=signal.intensity,
                source_id=self.node_id,
                timestamp=time.time(),
                payload={"direction": signal.payload.get("direction", [0, 0, 0])}
            )
        elif chemical_type == "threat":
            # 威脅信號：收縮和警戒 (Threat signal: contract and alert)
            self.muscle_tension = min(1.0, self.muscle_tension + signal.intensity)
            return Signal(
                signal_type="alert",
                intensity=signal.intensity * 0.8,
                source_id=self.node_id,
                timestamp=time.time(),
                payload={"threat_level": signal.intensity}
            )
        return None
    
    def _process_coordination_signal(self, signal: Signal) -> Optional[Signal]:
        """處理協調信號 (Process coordination signal)"""
        action = signal.payload.get("action", "none")
        
        if action == "grasp_assistance":
            # 協助抓握 (Assist grasping)
            grip_strength = signal.payload.get("grip_strength", 0.5)
            self.muscle_tension = grip_strength * 0.7
            
            # 確認協助 (Confirm assistance)
            return Signal(
                signal_type="confirmation",
                intensity=grip_strength,
                source_id=self.node_id,
                timestamp=time.time(),
                payload={"action_confirmed": True}
            )
        return None
    
    def generate_local_response(self, stimulus: EnvironmentalStimulus) -> Dict[str, Any]:
        """生成局部響應 (Generate local response)"""
        response = {
            "node_id": self.node_id,
            "response_type": "tactile_exploration",
            "timestamp": time.time()
        }
        
        if stimulus.stimulus_type == "object_contact":
            # 物體接觸響應 (Object contact response)
            response.update({
                "action": "explore_texture",
                "sucker_activation": self.sucker_states.copy(),
                "muscle_tension": self.muscle_tension,
                "exploration_direction": self._calculate_exploration_direction(stimulus)
            })
        elif stimulus.stimulus_type == "environmental_change":
            # 環境變化響應 (Environmental change response)
            response.update({
                "action": "adaptive_positioning",
                "autonomy_adjustment": self.autonomy_level,
                "sensitivity_adjustment": self.tactile_sensitivity
            })
        
        return response
    
    def _calculate_exploration_direction(self, stimulus: EnvironmentalStimulus) -> List[float]:
        """計算探索方向 (Calculate exploration direction)"""
        # 基於刺激位置和當前本體感覺計算最佳探索方向
        stimulus_pos = np.array(stimulus.position)
        current_pos = np.array(self.position)
        direction = stimulus_pos - current_pos
        
        # 歸一化方向向量 (Normalize direction vector)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        
        return direction.tolist()


class SwarmIntelligenceNode(DistributedNode):
    """群體智慧節點 (Swarm Intelligence Node)"""
    
    def __init__(self, node_id: str, position: Tuple[float, float, float] = (0, 0, 0)):
        super().__init__(node_id, position)
        self.opinion = random.random()  # 初始意見 (Initial opinion)
        self.confidence = random.random()  # 信心程度 (Confidence level)
        self.influence_radius = 5.0  # 影響半徑 (Influence radius)
        self.social_weight = 0.3  # 社會影響權重 (Social influence weight)
        self.evidence_weight = 0.7  # 證據權重 (Evidence weight)
        self.consensus_threshold = 0.8  # 共識閾值 (Consensus threshold)
    
    def process_signal(self, signal: Signal) -> Optional[Signal]:
        """處理信號 - 群體智慧式"""
        if signal.signal_type == "opinion_share":
            return self._process_opinion_signal(signal)
        elif signal.signal_type == "evidence":
            return self._process_evidence_signal(signal)
        elif signal.signal_type == "consensus_check":
            return self._process_consensus_signal(signal)
        return None
    
    def _process_opinion_signal(self, signal: Signal) -> Optional[Signal]:
        """處理意見信號 (Process opinion signal)"""
        neighbor_opinion = signal.payload.get("opinion", 0.5)
        neighbor_confidence = signal.payload.get("confidence", 0.5)
        
        # 社會影響：調整自己的意見 (Social influence: adjust own opinion)
        influence_strength = self._calculate_influence_strength(signal)
        opinion_adjustment = (neighbor_opinion - self.opinion) * self.social_weight * influence_strength
        self.opinion = max(0, min(1, self.opinion + opinion_adjustment))
        
        # 如果意見變化顯著，分享給鄰居 (If opinion changes significantly, share with neighbors)
        if abs(opinion_adjustment) > 0.1:
            return Signal(
                signal_type="opinion_share",
                intensity=self.confidence,
                source_id=self.node_id,
                timestamp=time.time(),
                payload={
                    "opinion": self.opinion,
                    "confidence": self.confidence,
                    "change_amount": abs(opinion_adjustment)
                }
            )
        return None
    
    def _process_evidence_signal(self, signal: Signal) -> Optional[Signal]:
        """處理證據信號 (Process evidence signal)"""
        evidence_value = signal.payload.get("evidence_value", 0.5)
        evidence_quality = signal.payload.get("quality", 0.5)
        
        # 基於證據調整意見 (Adjust opinion based on evidence)
        evidence_impact = evidence_value * evidence_quality * self.evidence_weight
        self.opinion = max(0, min(1, self.opinion + evidence_impact))
        self.confidence = min(1, self.confidence + evidence_quality * 0.1)
        
        # 高質量證據觸發意見分享 (High-quality evidence triggers opinion sharing)
        if evidence_quality > 0.7:
            return Signal(
                signal_type="opinion_share",
                intensity=self.confidence,
                source_id=self.node_id,
                timestamp=time.time(),
                payload={
                    "opinion": self.opinion,
                    "confidence": self.confidence,
                    "evidence_based": True
                }
            )
        return None
    
    def _process_consensus_signal(self, signal: Signal) -> Optional[Signal]:
        """處理共識檢查信號 (Process consensus check signal)"""
        return Signal(
            signal_type="consensus_response",
            intensity=self.confidence,
            source_id=self.node_id,
            timestamp=time.time(),
            payload={
                "opinion": self.opinion,
                "confidence": self.confidence,
                "ready_for_consensus": self.confidence > self.consensus_threshold
            }
        )
    
    def _calculate_influence_strength(self, signal: Signal) -> float:
        """計算影響強度 (Calculate influence strength)"""
        # 基於距離和信心程度計算影響強度
        distance = self._calculate_distance_to_source(signal)
        distance_factor = max(0, 1 - distance / self.influence_radius)
        confidence_factor = signal.payload.get("confidence", 0.5)
        return distance_factor * confidence_factor
    
    def _calculate_distance_to_source(self, signal: Signal) -> float:
        """計算到信號源的距離 (Calculate distance to signal source)"""
        # 簡化實現：隨機距離 (Simplified: random distance)
        return random.uniform(0, 10)
    
    def generate_local_response(self, stimulus: EnvironmentalStimulus) -> Dict[str, Any]:
        """生成局部響應 (Generate local response)"""
        return {
            "node_id": self.node_id,
            "response_type": "swarm_decision",
            "opinion": self.opinion,
            "confidence": self.confidence,
            "consensus_ready": self.confidence > self.consensus_threshold,
            "timestamp": time.time()
        }


class ExtendedMindNode(DistributedNode):
    """延展心智節點 (Extended Mind Node)"""
    
    def __init__(self, node_id: str, position: Tuple[float, float, float] = (0, 0, 0)):
        super().__init__(node_id, position)
        self.external_tools = {}
        self.cognitive_load = 0.0
        self.tool_proficiency = {}
        self.task_queue = []
        self.delegation_threshold = 0.7
    
    def add_external_tool(self, tool_name: str, tool_interface: Callable):
        """添加外部工具 (Add external tool)"""
        self.external_tools[tool_name] = tool_interface
        self.tool_proficiency[tool_name] = 0.5  # 初始熟練度
    
    def process_signal(self, signal: Signal) -> Optional[Signal]:
        """處理信號 - 延展心智式"""
        if signal.signal_type == "cognitive_task":
            return self._process_cognitive_task(signal)
        elif signal.signal_type == "tool_request":
            return self._process_tool_request(signal)
        elif signal.signal_type == "memory_query":
            return self._process_memory_query(signal)
        return None
    
    def _process_cognitive_task(self, signal: Signal) -> Optional[Signal]:
        """處理認知任務 (Process cognitive task)"""
        task_complexity = signal.payload.get("complexity", 0.5)
        task_type = signal.payload.get("task_type", "general")
        
        # 評估是否需要工具輔助 (Assess if tool assistance is needed)
        if task_complexity > self.delegation_threshold:
            # 尋找合適的工具 (Find appropriate tool)
            suitable_tool = self._find_suitable_tool(task_type)
            if suitable_tool:
                result = self._delegate_to_tool(suitable_tool, signal.payload)
                return Signal(
                    signal_type="task_result",
                    intensity=1.0 - task_complexity + 0.5,  # 工具提升效能
                    source_id=self.node_id,
                    timestamp=time.time(),
                    payload={
                        "result": result,
                        "tool_used": suitable_tool,
                        "enhanced_capability": True
                    }
                )
        
        # 直接處理任務 (Process task directly)
        result = self._process_task_internally(signal.payload)
        return Signal(
            signal_type="task_result",
            intensity=1.0 - task_complexity,
            source_id=self.node_id,
            timestamp=time.time(),
            payload={"result": result, "internal_processing": True}
        )
    
    def _find_suitable_tool(self, task_type: str) -> Optional[str]:
        """尋找合適的工具 (Find suitable tool)"""
        tool_mapping = {
            "calculation": "calculator",
            "navigation": "gps",
            "translation": "translator",
            "memory": "external_storage",
            "visualization": "graph_tool"
        }
        return tool_mapping.get(task_type)
    
    def _delegate_to_tool(self, tool_name: str, task_data: Dict) -> Any:
        """委派任務給工具 (Delegate task to tool)"""
        if tool_name in self.external_tools:
            tool = self.external_tools[tool_name]
            try:
                result = tool(task_data)
                # 提升工具熟練度 (Improve tool proficiency)
                self.tool_proficiency[tool_name] = min(1.0, 
                    self.tool_proficiency[tool_name] + 0.05)
                return result
            except Exception as e:
                return f"Tool error: {str(e)}"
        return "Tool not available"
    
    def _process_task_internally(self, task_data: Dict) -> Any:
        """內部處理任務 (Process task internally)"""
        # 簡化的內部處理邏輯
        return f"Internal processing result for {task_data.get('task_type', 'unknown')}"
    
    def generate_local_response(self, stimulus: EnvironmentalStimulus) -> Dict[str, Any]:
        """生成局部響應 (Generate local response)"""
        return {
            "node_id": self.node_id,
            "response_type": "extended_cognition",
            "available_tools": list(self.external_tools.keys()),
            "tool_proficiencies": self.tool_proficiency.copy(),
            "cognitive_load": self.cognitive_load,
            "timestamp": time.time()
        }


class ChemicalSignalingNode(DistributedNode):
    """化學信號節點 (Chemical Signaling Node)"""
    
    def __init__(self, node_id: str, position: Tuple[float, float, float] = (0, 0, 0)):
        super().__init__(node_id, position)
        self.chemical_state = {}
        self.signal_thresholds = {
            "danger": 0.3,
            "food": 0.2,
            "mating": 0.4,
            "territory": 0.5
        }
        self.response_patterns = {}
        self.adaptation_rate = 0.1
        self.signal_decay_rate = 0.95
    
    def process_signal(self, signal: Signal) -> Optional[Signal]:
        """處理化學信號 (Process chemical signal)"""
        if signal.signal_type == "chemical":
            return self._process_chemical_signal(signal)
        return None
    
    def _process_chemical_signal(self, signal: Signal) -> Optional[Signal]:
        """處理化學信號 (Process chemical signal)"""
        chemical_type = signal.payload.get("chemical_type", "unknown")
        concentration = signal.intensity
        
        # 更新本地化學狀態 (Update local chemical state)
        self.chemical_state[chemical_type] = concentration
        
        # 檢查是否超過反應閾值 (Check if exceeds response threshold)
        threshold = self.signal_thresholds.get(chemical_type, 0.5)
        if concentration > threshold:
            response = self._generate_chemical_response(chemical_type, concentration)
            
            # 傳播信號給鄰居 (Propagate signal to neighbors)
            return Signal(
                signal_type="chemical",
                intensity=concentration * 0.8,  # 信號衰減
                source_id=self.node_id,
                timestamp=time.time(),
                payload={
                    "chemical_type": chemical_type,
                    "response_triggered": True,
                    "original_source": signal.source_id
                }
            )
        return None
    
    def _generate_chemical_response(self, chemical_type: str, concentration: float) -> Dict:
        """生成化學響應 (Generate chemical response)"""
        responses = {
            "danger": {"action": "defensive_posture", "intensity": concentration},
            "food": {"action": "approach_behavior", "intensity": concentration * 0.8},
            "mating": {"action": "courtship_display", "intensity": concentration * 0.6},
            "territory": {"action": "territorial_marking", "intensity": concentration * 0.9}
        }
        
        return responses.get(chemical_type, {"action": "no_response", "intensity": 0})
    
    def update_thresholds(self, chemical_type: str, new_threshold: float):
        """更新反應閾值 (Update response threshold)"""
        self.signal_thresholds[chemical_type] = new_threshold
    
    def decay_chemical_signals(self):
        """化學信號衰減 (Chemical signal decay)"""
        for chemical_type in list(self.chemical_state.keys()):
            self.chemical_state[chemical_type] *= self.signal_decay_rate
            if self.chemical_state[chemical_type] < 0.01:
                del self.chemical_state[chemical_type]
    
    def generate_local_response(self, stimulus: EnvironmentalStimulus) -> Dict[str, Any]:
        """生成局部響應 (Generate local response)"""
        return {
            "node_id": self.node_id,
            "response_type": "chemical_signaling",
            "chemical_state": self.chemical_state.copy(),
            "active_thresholds": self.signal_thresholds.copy(),
            "timestamp": time.time()
        }


class DistributedThinkingSystem:
    """
    分散式思考系統主類 (Main Distributed Thinking System Class)
    
    整合多種無腦思考模式，創建一個完整的分散式智能網絡
    Integrates multiple brainless thinking modes to create a complete distributed intelligence network
    """
    
    def __init__(self):
        self.nodes = {}
        self.thinking_modes = set()
        self.global_state = {}
        self.simulation_running = False
        self.time_step = 0.1  # 時間步長 (Time step)
        self.max_iterations = 1000
        
        # 性能監控 (Performance monitoring)
        self.performance_metrics = {
            "response_times": [],
            "consensus_achieved": [],
            "adaptation_events": [],
            "emergence_detected": []
        }
    
    def add_node(self, node: DistributedNode):
        """添加節點到系統 (Add node to system)"""
        self.nodes[node.node_id] = node
        
        # 確定節點類型並更新思考模式 (Determine node type and update thinking modes)
        if isinstance(node, OctopusTentacleNode):
            self.thinking_modes.add(ThinkingMode.BIOLOGICAL_DISTRIBUTED)
        elif isinstance(node, SwarmIntelligenceNode):
            self.thinking_modes.add(ThinkingMode.SWARM_INTELLIGENCE)
        elif isinstance(node, ExtendedMindNode):
            self.thinking_modes.add(ThinkingMode.EXTENDED_MIND)
        elif isinstance(node, ChemicalSignalingNode):
            self.thinking_modes.add(ThinkingMode.CHEMICAL_SIGNALING)
    
    def create_network_topology(self, topology_type: str = "small_world"):
        """創建網絡拓撲 (Create network topology)"""
        node_list = list(self.nodes.values())
        
        if topology_type == "fully_connected":
            self._create_fully_connected_topology(node_list)
        elif topology_type == "small_world":
            self._create_small_world_topology(node_list)
        elif topology_type == "random":
            self._create_random_topology(node_list)
        elif topology_type == "hierarchical":
            self._create_hierarchical_topology(node_list)
    
    def _create_small_world_topology(self, nodes: List[DistributedNode]):
        """創建小世界網絡拓撲 (Create small-world network topology)"""
        n = len(nodes)
        k = min(4, n - 1)  # 每個節點的平均連接數
        
        # 首先創建環形連接 (First create ring connections)
        for i, node in enumerate(nodes):
            for j in range(1, k // 2 + 1):
                neighbor = nodes[(i + j) % n]
                node.add_neighbor(neighbor)
                neighbor.add_neighbor(node)
        
        # 隨機重連以創建小世界特性 (Random rewiring for small-world properties)
        rewiring_probability = 0.3
        for node in nodes:
            for neighbor in node.neighbors[:]:
                if random.random() < rewiring_probability:
                    node.neighbors.remove(neighbor)
                    new_neighbor = random.choice([n for n in nodes if n != node and n not in node.neighbors])
                    node.add_neighbor(new_neighbor)
    
    def _create_fully_connected_topology(self, nodes: List[DistributedNode]):
        """創建全連接拓撲 (Create fully connected topology)"""
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i != j:
                    node1.add_neighbor(node2)
    
    def _create_random_topology(self, nodes: List[DistributedNode]):
        """創建隨機拓撲 (Create random topology)"""
        connection_probability = 0.3
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i != j and random.random() < connection_probability:
                    node1.add_neighbor(node2)
                    node2.add_neighbor(node1)
    
    def _create_hierarchical_topology(self, nodes: List[DistributedNode]):
        """創建層次拓撲 (Create hierarchical topology)"""
        # 簡化的層次結構實現
        n = len(nodes)
        layers = 3
        nodes_per_layer = n // layers
        
        for layer in range(layers):
            start_idx = layer * nodes_per_layer
            end_idx = min(start_idx + nodes_per_layer, n)
            layer_nodes = nodes[start_idx:end_idx]
            
            # 層內連接 (Intra-layer connections)
            for i, node1 in enumerate(layer_nodes):
                for j, node2 in enumerate(layer_nodes):
                    if i != j:
                        node1.add_neighbor(node2)
            
            # 層間連接 (Inter-layer connections)
            if layer < layers - 1:
                next_layer_start = end_idx
                next_layer_end = min(next_layer_start + nodes_per_layer, n)
                next_layer_nodes = nodes[next_layer_start:next_layer_end]
                
                for node1 in layer_nodes:
                    for node2 in next_layer_nodes[:2]:  # 限制連接數
                        node1.add_neighbor(node2)
                        node2.add_neighbor(node1)
    
    def inject_stimulus(self, stimulus: EnvironmentalStimulus, target_nodes: List[str] = None):
        """注入環境刺激 (Inject environmental stimulus)"""
        if target_nodes is None:
            target_nodes = list(self.nodes.keys())
        
        responses = {}
        for node_id in target_nodes:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                response = node.generate_local_response(stimulus)
                responses[node_id] = response
        
        return responses
    
    def simulate_thinking_process(self, duration: float = 10.0, stimulus_events: List[Tuple[float, EnvironmentalStimulus]] = None):
        """模擬思考過程 (Simulate thinking process)"""
        self.simulation_running = True
        start_time = time.time()
        iteration = 0
        
        if stimulus_events is None:
            stimulus_events = []
        
        print(f"開始分散式思考模擬 (Starting distributed thinking simulation)")
        print(f"節點數量: {len(self.nodes)}, 思考模式: {[mode.value for mode in self.thinking_modes]}")
        
        while self.simulation_running and (time.time() - start_time) < duration and iteration < self.max_iterations:
            current_time = time.time() - start_time
            
            # 處理定時刺激事件 (Process timed stimulus events)
            for event_time, stimulus in stimulus_events:
                if abs(current_time - event_time) < self.time_step:
                    print(f"時間 {current_time:.2f}: 注入刺激 {stimulus.stimulus_type}")
                    self.inject_stimulus(stimulus)
            
            # 更新所有節點 (Update all nodes)
            for node in self.nodes.values():
                node.update()
            
            # 檢測涌現行為 (Detect emergent behavior)
            emergent_patterns = self._detect_emergent_behavior()
            if emergent_patterns:
                print(f"時間 {current_time:.2f}: 檢測到涌現行為: {emergent_patterns}")
                self.performance_metrics["emergence_detected"].append(current_time)
            
            # 記錄性能指標 (Record performance metrics)
            if iteration % 50 == 0:  # 每50次迭代記錄一次
                self._record_performance_metrics(current_time)
            
            time.sleep(self.time_step)
            iteration += 1
        
        self.simulation_running = False
        print(f"模擬結束 (Simulation ended) - 迭代次數: {iteration}, 持續時間: {time.time() - start_time:.2f}秒")
        return self._generate_simulation_report()
    
    def _detect_emergent_behavior(self) -> List[str]:
        """檢測涌現行為 (Detect emergent behavior)"""
        patterns = []
        
        # 檢測群體同步 (Detect collective synchronization)
        if self._detect_synchronization():
            patterns.append("collective_synchronization")
        
        # 檢測信息級聯 (Detect information cascade)
        if self._detect_information_cascade():
            patterns.append("information_cascade")
        
        # 檢測適應性涌現 (Detect adaptive emergence)
        if self._detect_adaptive_emergence():
            patterns.append("adaptive_emergence")
        
        return patterns
    
    def _detect_synchronization(self) -> bool:
        """檢測同步現象 (Detect synchronization)"""
        # 簡化實現：檢查群體智慧節點的意見同步
        swarm_nodes = [node for node in self.nodes.values() if isinstance(node, SwarmIntelligenceNode)]
        if len(swarm_nodes) < 2:
            return False
        
        opinions = [node.opinion for node in swarm_nodes]
        opinion_variance = np.var(opinions)
        return opinion_variance < 0.05  # 意見變異度小於閾值
    
    def _detect_information_cascade(self) -> bool:
        """檢測信息級聯 (Detect information cascade)"""
        # 檢查信號傳播模式
        signal_activity = sum(1 for node in self.nodes.values() if not node.signal_queue.empty())
        total_nodes = len(self.nodes)
        return signal_activity / total_nodes > 0.7  # 70%以上節點活躍
    
    def _detect_adaptive_emergence(self) -> bool:
        """檢測適應性涌現 (Detect adaptive emergence)"""
        # 檢查節點狀態的適應性變化
        adaptive_nodes = 0
        for node in self.nodes.values():
            if hasattr(node, 'autonomy_level') and node.autonomy_level > 0.8:
                adaptive_nodes += 1
            elif hasattr(node, 'confidence') and node.confidence > 0.8:
                adaptive_nodes += 1
        
        return adaptive_nodes / len(self.nodes) > 0.5
    
    def _record_performance_metrics(self, current_time: float):
        """記錄性能指標 (Record performance metrics)"""
        # 計算平均響應時間 (Calculate average response time)
        active_nodes = [node for node in self.nodes.values() if node.active]
        if active_nodes:
            avg_response_time = np.mean([time.time() - node.last_update for node in active_nodes])
            self.performance_metrics["response_times"].append(avg_response_time)
        
        # 檢查共識達成 (Check consensus achievement)
        consensus_achieved = self._check_consensus()
        self.performance_metrics["consensus_achieved"].append(consensus_achieved)
    
    def _check_consensus(self) -> bool:
        """檢查是否達成共識 (Check if consensus is achieved)"""
        swarm_nodes = [node for node in self.nodes.values() if isinstance(node, SwarmIntelligenceNode)]
        if len(swarm_nodes) < 2:
            return False
        
        high_confidence_nodes = [node for node in swarm_nodes if node.confidence > 0.8]
        return len(high_confidence_nodes) / len(swarm_nodes) > 0.8
    
    def _generate_simulation_report(self) -> Dict[str, Any]:
        """生成模擬報告 (Generate simulation report)"""
        report = {
            "simulation_summary": {
                "total_nodes": len(self.nodes),
                "thinking_modes": [mode.value for mode in self.thinking_modes],
                "emergence_events": len(self.performance_metrics["emergence_detected"]),
                "consensus_rate": np.mean(self.performance_metrics["consensus_achieved"]) if self.performance_metrics["consensus_achieved"] else 0
            },
            "performance_metrics": self.performance_metrics,
            "network_analysis": {
                "average_connections": np.mean([len(node.neighbors) for node in self.nodes.values()]),
                "network_density": self._calculate_network_density(),
                "clustering_coefficient": self._calculate_clustering_coefficient()
            },
            "thinking_effectiveness": self._evaluate_thinking_effectiveness()
        }
        
        return report
    
    def _calculate_network_density(self) -> float:
        """計算網絡密度 (Calculate network density)"""
        total_possible_connections = len(self.nodes) * (len(self.nodes) - 1)
        actual_connections = sum(len(node.neighbors) for node in self.nodes.values())
        return actual_connections / total_possible_connections if total_possible_connections > 0 else 0
    
    def _calculate_clustering_coefficient(self) -> float:
        """計算聚類係數 (Calculate clustering coefficient)"""
        # 簡化實現
        clustering_coefficients = []
        for node in self.nodes.values():
            if len(node.neighbors) < 2:
                clustering_coefficients.append(0)
                continue
            
            neighbor_connections = 0
            for i, neighbor1 in enumerate(node.neighbors):
                for j, neighbor2 in enumerate(node.neighbors[i+1:], i+1):
                    if neighbor2 in neighbor1.neighbors:
                        neighbor_connections += 1
            
            possible_connections = len(node.neighbors) * (len(node.neighbors) - 1) / 2
            clustering_coefficients.append(neighbor_connections / possible_connections if possible_connections > 0 else 0)
        
        return np.mean(clustering_coefficients) if clustering_coefficients else 0
    
    def _evaluate_thinking_effectiveness(self) -> Dict[str, float]:
        """評估思考效果 (Evaluate thinking effectiveness)"""
        effectiveness = {}
        
        # 響應速度 (Response speed)
        if self.performance_metrics["response_times"]:
            avg_response_time = np.mean(self.performance_metrics["response_times"])
            effectiveness["response_speed"] = max(0, 1 - avg_response_time / 10)  # 歸一化到0-1
        
        # 涌現頻率 (Emergence frequency)
        emergence_rate = len(self.performance_metrics["emergence_detected"]) / max(1, len(self.performance_metrics["consensus_achieved"]))
        effectiveness["emergence_quality"] = min(1, emergence_rate)
        
        # 適應性 (Adaptability)
        adaptive_nodes = sum(1 for node in self.nodes.values() 
                           if (hasattr(node, 'autonomy_level') and node.autonomy_level > 0.6) or
                              (hasattr(node, 'confidence') and node.confidence > 0.6))
        effectiveness["adaptability"] = adaptive_nodes / len(self.nodes)
        
        # 總體效果 (Overall effectiveness)
        effectiveness["overall"] = np.mean(list(effectiveness.values()))
        
        return effectiveness
    
    def demonstrate_brainless_thinking(self):
        """演示無腦思考能力 (Demonstrate brainless thinking capabilities)"""
        print("\n=== 無腦思考系統演示 (Brainless Thinking System Demonstration) ===\n")
        
        # 創建演示場景 (Create demonstration scenario)
        demo_stimuli = [
            (1.0, EnvironmentalStimulus("object_contact", (1, 1, 1), 0.8, 2.0, {"texture": "rough"})),
            (3.0, EnvironmentalStimulus("chemical_trace", (2, 2, 2), 0.6, 3.0, {"chemical_type": "food"})),
            (5.0, EnvironmentalStimulus("social_signal", (3, 3, 3), 0.9, 1.0, {"opinion": 0.8})),
            (7.0, EnvironmentalStimulus("threat_detected", (0, 0, 0), 0.7, 2.0, {"threat_level": "medium"}))
        ]
        
        # 運行模擬 (Run simulation)
        report = self.simulate_thinking_process(duration=10.0, stimulus_events=demo_stimuli)
        
        # 打印結果 (Print results)
        print("\n=== 模擬結果報告 (Simulation Results Report) ===")
        print(f"網絡規模: {report['simulation_summary']['total_nodes']} 個節點")
        print(f"思考模式: {', '.join(report['simulation_summary']['thinking_modes'])}")
        print(f"涌現事件: {report['simulation_summary']['emergence_events']} 次")
        print(f"共識達成率: {report['simulation_summary']['consensus_rate']:.2%}")
        print(f"網絡密度: {report['network_analysis']['network_density']:.3f}")
        print(f"聚類係數: {report['network_analysis']['clustering_coefficient']:.3f}")
        
        effectiveness = report['thinking_effectiveness']
        print(f"\n思考效果評估:")
        print(f"  響應速度: {effectiveness.get('response_speed', 0):.2%}")
        print(f"  涌現品質: {effectiveness.get('emergence_quality', 0):.2%}")
        print(f"  適應能力: {effectiveness.get('adaptability', 0):.2%}")
        print(f"  總體效果: {effectiveness.get('overall', 0):.2%}")
        
        return report


def create_demo_system() -> DistributedThinkingSystem:
    """創建演示系統 (Create demonstration system)"""
    system = DistributedThinkingSystem()
    
    # 添加章魚觸手節點 (Add octopus tentacle nodes)
    for i in range(4):
        tentacle = OctopusTentacleNode(f"tentacle_{i}", (i, 0, 0))
        system.add_node(tentacle)
    
    # 添加群體智慧節點 (Add swarm intelligence nodes)
    for i in range(6):
        swarm_node = SwarmIntelligenceNode(f"swarm_{i}", (i, 1, 0))
        system.add_node(swarm_node)
    
    # 添加延展心智節點 (Add extended mind nodes)
    for i in range(3):
        extended_node = ExtendedMindNode(f"extended_{i}", (i, 2, 0))
        
        # 添加模擬工具 (Add mock tools)
        extended_node.add_external_tool("calculator", lambda x: f"計算結果: {x.get('value', 0) * 2}")
        extended_node.add_external_tool("gps", lambda x: f"導航到: {x.get('destination', '未知位置')}")
        extended_node.add_external_tool("translator", lambda x: f"翻譯: {x.get('text', '無文本')} -> 已翻譯")
        
        system.add_node(extended_node)
    
    # 添加化學信號節點 (Add chemical signaling nodes)
    for i in range(4):
        chemical_node = ChemicalSignalingNode(f"chemical_{i}", (i, 3, 0))
        system.add_node(chemical_node)
    
    # 創建網絡拓撲 (Create network topology)
    system.create_network_topology("small_world")
    
    return system


def main():
    """主函數 - 演示分散式思考系統 (Main function - demonstrate distributed thinking system)"""
    print("=== 分散式思考系統：無腦思考的科學實現 ===")
    print("=== Distributed Thinking System: Scientific Implementation of Brainless Cognition ===\n")
    
    # 創建並運行演示系統 (Create and run demonstration system)
    demo_system = create_demo_system()
    report = demo_system.demonstrate_brainless_thinking()
    
    # 展示特定思考模式的例子 (Show examples of specific thinking modes)
    print("\n=== 具體思考模式示例 (Specific Thinking Mode Examples) ===")
    
    # 章魚觸手式思考 (Octopus tentacle thinking)
    print("\n1. 章魚觸手式分散思考 (Octopus Tentacle Distributed Thinking):")
    tentacle = list(demo_system.nodes.values())[0]
    if isinstance(tentacle, OctopusTentacleNode):
        stimulus = EnvironmentalStimulus("object_contact", (0, 0, 0), 0.9, 1.0)
        response = tentacle.generate_local_response(stimulus)
        print(f"   觸手響應: {response}")
    
    # 群體智慧決策 (Swarm intelligence decision)
    print("\n2. 群體智慧決策 (Swarm Intelligence Decision):")
    swarm_nodes = [node for node in demo_system.nodes.values() if isinstance(node, SwarmIntelligenceNode)]
    if swarm_nodes:
        opinions = [node.opinion for node in swarm_nodes[:3]]
        confidences = [node.confidence for node in swarm_nodes[:3]]
        print(f"   節點意見: {[f'{op:.2f}' for op in opinions]}")
        print(f"   信心程度: {[f'{conf:.2f}' for conf in confidences]}")
    
    # 延展心智工具使用 (Extended mind tool usage)
    print("\n3. 延展心智工具整合 (Extended Mind Tool Integration):")
    extended_nodes = [node for node in demo_system.nodes.values() if isinstance(node, ExtendedMindNode)]
    if extended_nodes:
        extended_node = extended_nodes[0]
        print(f"   可用工具: {list(extended_node.external_tools.keys())}")
        print(f"   工具熟練度: {extended_node.tool_proficiency}")
    
    print("\n=== 結論 (Conclusion) ===")
    print("分散式思考系統成功演示了在沒有中央大腦的情況下進行複雜認知的可能性。")
    print("通過整合生物學、工程學和哲學的見解，我們創建了一個真正的'無腦智囊'。")
    print("\nThe distributed thinking system successfully demonstrates the possibility of complex cognition without a central brain.")
    print("By integrating insights from biology, engineering, and philosophy, we have created a true 'brainless intelligence'.")


if __name__ == "__main__":
    main()