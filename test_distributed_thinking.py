"""
測試分散式思考系統 (Test Distributed Thinking System)

Simple test suite for the distributed thinking system implementation.
"""

import unittest
import time
from DistributedThinkingSystem import (
    DistributedThinkingSystem,
    OctopusTentacleNode,
    SwarmIntelligenceNode,
    ExtendedMindNode,
    ChemicalSignalingNode,
    EnvironmentalStimulus,
    Signal
)


class TestDistributedThinkingSystem(unittest.TestCase):
    """分散式思考系統測試 (Distributed Thinking System Tests)"""
    
    def setUp(self):
        """設置測試環境 (Set up test environment)"""
        self.system = DistributedThinkingSystem()
    
    def test_node_creation(self):
        """測試節點創建 (Test node creation)"""
        # 測試不同類型的節點創建
        tentacle = OctopusTentacleNode("test_tentacle")
        swarm = SwarmIntelligenceNode("test_swarm")
        extended = ExtendedMindNode("test_extended")
        chemical = ChemicalSignalingNode("test_chemical")
        
        self.assertEqual(tentacle.node_id, "test_tentacle")
        self.assertEqual(swarm.node_id, "test_swarm")
        self.assertEqual(extended.node_id, "test_extended")
        self.assertEqual(chemical.node_id, "test_chemical")
    
    def test_system_node_addition(self):
        """測試系統節點添加 (Test system node addition)"""
        tentacle = OctopusTentacleNode("tentacle_1")
        self.system.add_node(tentacle)
        
        self.assertIn("tentacle_1", self.system.nodes)
        self.assertEqual(len(self.system.nodes), 1)
    
    def test_network_topology_creation(self):
        """測試網絡拓撲創建 (Test network topology creation)"""
        # 添加多個節點
        for i in range(5):
            node = SwarmIntelligenceNode(f"node_{i}")
            self.system.add_node(node)
        
        # 創建網絡拓撲
        self.system.create_network_topology("small_world")
        
        # 檢查節點是否有鄰居連接
        total_connections = sum(len(node.neighbors) for node in self.system.nodes.values())
        self.assertGreater(total_connections, 0)
    
    def test_stimulus_injection(self):
        """測試刺激注入 (Test stimulus injection)"""
        # 添加節點
        tentacle = OctopusTentacleNode("tentacle_test")
        self.system.add_node(tentacle)
        
        # 創建環境刺激
        stimulus = EnvironmentalStimulus("test_stimulus", (0, 0, 0), 0.5, 1.0)
        
        # 注入刺激
        responses = self.system.inject_stimulus(stimulus)
        
        self.assertIn("tentacle_test", responses)
        self.assertIsNotNone(responses["tentacle_test"])
    
    def test_signal_processing(self):
        """測試信號處理 (Test signal processing)"""
        # 創建章魚觸手節點
        tentacle = OctopusTentacleNode("tentacle_signal_test")
        
        # 創建觸覺信號
        signal = Signal(
            signal_type="tactile",
            intensity=0.8,
            source_id="test_source",
            timestamp=time.time(),
            payload={"contact_point": 5, "texture": "rough"}
        )
        
        # 處理信號
        response = tentacle.process_signal(signal)
        
        # 檢查是否產生響應
        if response:
            self.assertEqual(response.signal_type, "coordination")
            self.assertIn("action", response.payload)
    
    def test_swarm_opinion_processing(self):
        """測試群體意見處理 (Test swarm opinion processing)"""
        swarm_node = SwarmIntelligenceNode("swarm_test")
        
        # 多次嘗試直到看到意見變化 (Try multiple times until we see opinion change)
        initial_opinion = swarm_node.opinion
        opinion_changed = False
        
        for attempt in range(10):  # 最多嘗試10次
            # 創建意見分享信號，確保有足夠大的差異來觸發變化
            target_opinion = 1.0 if initial_opinion < 0.5 else 0.0
            opinion_signal = Signal(
                signal_type="opinion_share",
                intensity=0.9,
                source_id="neighbor",
                timestamp=time.time(),
                payload={"opinion": target_opinion, "confidence": 0.9}
            )
            
            # 重置初始意見並處理信號
            test_opinion = initial_opinion
            swarm_node.opinion = test_opinion
            
            # 處理信號
            response = swarm_node.process_signal(opinion_signal)
            
            # 檢查意見是否有變化
            opinion_change = abs(swarm_node.opinion - test_opinion)
            if opinion_change > 0.001:
                opinion_changed = True
                break
        
        self.assertTrue(opinion_changed, "Opinion should change when processing strong opinion signal after multiple attempts")
    
    def test_extended_mind_tool_usage(self):
        """測試延展心智工具使用 (Test extended mind tool usage)"""
        extended_node = ExtendedMindNode("extended_test")
        
        # 添加測試工具
        def test_tool(data):
            return f"Tool processed: {data.get('input', 'no input')}"
        
        extended_node.add_external_tool("test_tool", test_tool)
        
        # 檢查工具是否正確添加
        self.assertIn("test_tool", extended_node.external_tools)
        self.assertIn("test_tool", extended_node.tool_proficiency)
    
    def test_chemical_signaling(self):
        """測試化學信號 (Test chemical signaling)"""
        chemical_node = ChemicalSignalingNode("chemical_test")
        
        # 創建化學信號
        chemical_signal = Signal(
            signal_type="chemical",
            intensity=0.6,
            source_id="source",
            timestamp=time.time(),
            payload={"chemical_type": "danger"}
        )
        
        # 處理信號
        response = chemical_node.process_signal(chemical_signal)
        
        # 檢查化學狀態是否更新
        self.assertIn("danger", chemical_node.chemical_state)
    
    def test_short_simulation(self):
        """測試短時間模擬 (Test short simulation)"""
        # 創建簡單系統
        for i in range(3):
            node = SwarmIntelligenceNode(f"sim_node_{i}")
            self.system.add_node(node)
        
        self.system.create_network_topology("fully_connected")
        
        # 運行短時間模擬
        report = self.system.simulate_thinking_process(duration=1.0)
        
        # 檢查報告結構
        self.assertIn("simulation_summary", report)
        self.assertIn("performance_metrics", report)
        self.assertIn("network_analysis", report)
        self.assertIn("thinking_effectiveness", report)
    
    def test_performance_metrics(self):
        """測試性能指標 (Test performance metrics)"""
        # 添加節點並運行模擬
        for i in range(4):
            node = SwarmIntelligenceNode(f"perf_node_{i}")
            self.system.add_node(node)
        
        self.system.create_network_topology("small_world")
        
        # 運行模擬
        report = self.system.simulate_thinking_process(duration=0.5)
        
        # 檢查性能指標
        effectiveness = report["thinking_effectiveness"]
        self.assertIn("overall", effectiveness)
        self.assertIsInstance(effectiveness["overall"], (int, float))


class TestNodeInteractions(unittest.TestCase):
    """節點交互測試 (Node Interaction Tests)"""
    
    def test_neighbor_communication(self):
        """測試鄰居通信 (Test neighbor communication)"""
        node1 = SwarmIntelligenceNode("node1")
        node2 = SwarmIntelligenceNode("node2")
        
        # 建立鄰居關係
        node1.add_neighbor(node2)
        node2.add_neighbor(node1)
        
        # 檢查鄰居關係
        self.assertIn(node2, node1.neighbors)
        self.assertIn(node1, node2.neighbors)
        
        # 測試信號發送
        test_signal = Signal("test", 0.5, "node1", time.time())
        node1.send_signal(test_signal)
        
        # 檢查node2是否收到信號
        self.assertFalse(node2.signal_queue.empty())
    
    def test_multi_node_coordination(self):
        """測試多節點協調 (Test multi-node coordination)"""
        nodes = []
        for i in range(4):
            node = OctopusTentacleNode(f"coord_node_{i}")
            nodes.append(node)
        
        # 建立全連接網絡
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i != j:
                    node1.add_neighbor(node2)
        
        # 從一個節點發送協調信號
        coord_signal = Signal(
            signal_type="coordination",
            intensity=0.8,
            source_id="coord_node_0",
            timestamp=time.time(),
            payload={"action": "grasp_assistance", "grip_strength": 0.8}
        )
        
        nodes[0].send_signal(coord_signal)
        
        # 檢查其他節點是否收到信號
        for node in nodes[1:]:
            self.assertFalse(node.signal_queue.empty())


def run_tests():
    """運行所有測試 (Run all tests)"""
    print("=== 分散式思考系統測試 (Distributed Thinking System Tests) ===\n")
    
    # 創建測試套件
    test_suite = unittest.TestSuite()
    
    # 添加測試類
    test_suite.addTest(unittest.makeSuite(TestDistributedThinkingSystem))
    test_suite.addTest(unittest.makeSuite(TestNodeInteractions))
    
    # 運行測試
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 打印測試結果摘要
    print(f"\n=== 測試結果摘要 (Test Results Summary) ===")
    print(f"運行測試: {result.testsRun}")
    print(f"失敗: {len(result.failures)}")
    print(f"錯誤: {len(result.errors)}")
    print(f"成功率: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n失敗的測試:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}")
    
    if result.errors:
        print(f"\n錯誤的測試:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)