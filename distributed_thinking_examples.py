"""
分散式思考系統應用示例 (Distributed Thinking System Application Examples)

This module provides practical examples of how the distributed thinking system
can be applied to real-world scenarios and integrated with existing evaluation frameworks.

無腦思考的實際應用案例 - 與人類表達評估框架的整合
Practical applications of brainless thinking - integration with human expression evaluation frameworks
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional
from DistributedThinkingSystem import (
    DistributedThinkingSystem, 
    OctopusTentacleNode, 
    SwarmIntelligenceNode,
    ExtendedMindNode,
    ChemicalSignalingNode,
    EnvironmentalStimulus,
    Signal
)

try:
    from HumanExpressionEvaluator import (
        HumanExpressionEvaluator,
        ExpressionContext,
        EvaluationDimension
    )
    HUMAN_EVALUATOR_AVAILABLE = True
except ImportError:
    print("Warning: HumanExpressionEvaluator not available. Some features will be limited.")
    HUMAN_EVALUATOR_AVAILABLE = False


class DistributedThinkingEvaluationIntegrator:
    """
    分散式思考與人類表達評估整合器 
    (Distributed Thinking and Human Expression Evaluation Integrator)
    """
    
    def __init__(self):
        self.distributed_system = DistributedThinkingSystem()
        if HUMAN_EVALUATOR_AVAILABLE:
            self.human_evaluator = HumanExpressionEvaluator()
        else:
            self.human_evaluator = None
    
    def evaluate_distributed_vs_centralized_thinking(self, problem_statement: str) -> Dict[str, Any]:
        """
        比較分散式思考與集中式思考的效果
        Compare effectiveness of distributed vs centralized thinking
        """
        results = {
            "problem_statement": problem_statement,
            "distributed_approach": self._simulate_distributed_thinking(problem_statement),
            "centralized_approach": self._simulate_centralized_thinking(problem_statement),
            "comparison_metrics": {}
        }
        
        # 計算比較指標 (Calculate comparison metrics)
        dist_score = results["distributed_approach"]["effectiveness_score"]
        cent_score = results["centralized_approach"]["effectiveness_score"]
        
        results["comparison_metrics"] = {
            "distributed_advantage": max(0, dist_score - cent_score),
            "centralized_advantage": max(0, cent_score - dist_score),
            "optimal_approach": "distributed" if dist_score > cent_score else "centralized",
            "effectiveness_ratio": dist_score / cent_score if cent_score > 0 else float('inf')
        }
        
        return results
    
    def _simulate_distributed_thinking(self, problem: str) -> Dict[str, Any]:
        """模擬分散式思考過程 (Simulate distributed thinking process)"""
        # 創建專門的問題解決網絡 (Create specialized problem-solving network)
        system = DistributedThinkingSystem()
        
        # 添加不同類型的節點來處理問題的不同方面
        # Add different types of nodes to handle different aspects of the problem
        
        # 感知節點 - 分析問題特徵 (Perception nodes - analyze problem features)
        for i in range(3):
            tentacle = OctopusTentacleNode(f"perceiver_{i}")
            system.add_node(tentacle)
        
        # 決策節點 - 集體決策 (Decision nodes - collective decision making)
        for i in range(4):
            swarm_node = SwarmIntelligenceNode(f"decider_{i}")
            system.add_node(swarm_node)
        
        # 工具節點 - 外部資源利用 (Tool nodes - external resource utilization)
        for i in range(2):
            extended_node = ExtendedMindNode(f"tool_user_{i}")
            # 添加問題解決工具 (Add problem-solving tools)
            extended_node.add_external_tool("analyzer", self._create_problem_analyzer())
            extended_node.add_external_tool("synthesizer", self._create_solution_synthesizer())
            system.add_node(extended_node)
        
        system.create_network_topology("small_world")
        
        # 將問題轉換為環境刺激 (Convert problem to environmental stimulus)
        problem_stimulus = EnvironmentalStimulus(
            stimulus_type="complex_problem",
            position=(0, 0, 0),
            intensity=self._assess_problem_complexity(problem),
            duration=5.0,
            properties={"problem_text": problem, "domain": "general"}
        )
        
        # 運行分散式處理 (Run distributed processing)
        start_time = time.time()
        responses = system.inject_stimulus(problem_stimulus)
        processing_time = time.time() - start_time
        
        # 評估分散式方法的效果 (Evaluate distributed approach effectiveness)
        effectiveness = self._evaluate_distributed_effectiveness(responses, processing_time)
        
        return {
            "approach": "distributed",
            "node_count": len(system.nodes),
            "processing_time": processing_time,
            "responses": responses,
            "effectiveness_score": effectiveness,
            "advantages": [
                "並行處理 (Parallel processing)",
                "多角度分析 (Multi-perspective analysis)", 
                "容錯性高 (High fault tolerance)",
                "適應性強 (High adaptability)"
            ]
        }
    
    def _simulate_centralized_thinking(self, problem: str) -> Dict[str, Any]:
        """模擬集中式思考過程 (Simulate centralized thinking process)"""
        start_time = time.time()
        
        # 模擬傳統的串行思考過程 (Simulate traditional serial thinking process)
        steps = [
            "問題理解 (Problem understanding)",
            "信息收集 (Information gathering)",
            "方案生成 (Solution generation)",
            "方案評估 (Solution evaluation)",
            "決策制定 (Decision making)"
        ]
        
        processing_results = []
        for step in steps:
            # 模擬每個步驟的處理時間 (Simulate processing time for each step)
            step_time = np.random.uniform(0.1, 0.3)
            time.sleep(step_time)
            processing_results.append({
                "step": step,
                "duration": step_time,
                "result": f"Completed {step.split('(')[0].strip()}"
            })
        
        total_time = time.time() - start_time
        
        # 評估集中式方法的效果 (Evaluate centralized approach effectiveness)
        effectiveness = self._evaluate_centralized_effectiveness(processing_results, total_time)
        
        return {
            "approach": "centralized",
            "processing_steps": processing_results,
            "processing_time": total_time,
            "effectiveness_score": effectiveness,
            "advantages": [
                "邏輯一致性 (Logical consistency)",
                "決策明確 (Clear decision making)",
                "易於控制 (Easy to control)",
                "結果可預測 (Predictable results)"
            ]
        }
    
    def _assess_problem_complexity(self, problem: str) -> float:
        """評估問題複雜度 (Assess problem complexity)"""
        # 基於問題長度、專業詞彙、語法結構等評估複雜度
        complexity_factors = {
            "length": min(len(problem) / 500, 1.0),
            "vocabulary": len(set(problem.split())) / len(problem.split()) if problem.split() else 0,
            "punctuation": problem.count('?') + problem.count('!') + problem.count(';')
        }
        
        return np.mean(list(complexity_factors.values()))
    
    def _create_problem_analyzer(self):
        """創建問題分析工具 (Create problem analysis tool)"""
        def analyze(task_data):
            problem = task_data.get("problem_text", "")
            analysis = {
                "problem_type": "analytical" if "?" in problem else "creative",
                "key_concepts": problem.split()[:5],  # 前5個關鍵詞
                "complexity_level": self._assess_problem_complexity(problem),
                "recommended_approach": "distributed" if len(problem) > 100 else "centralized"
            }
            return analysis
        return analyze
    
    def _create_solution_synthesizer(self):
        """創建解決方案合成工具 (Create solution synthesis tool)"""
        def synthesize(task_data):
            problem = task_data.get("problem_text", "")
            solutions = [
                "分解問題為子問題 (Break down into sub-problems)",
                "尋找類似案例 (Find similar cases)",
                "應用已知模式 (Apply known patterns)",
                "創新性組合 (Innovative combination)"
            ]
            return {
                "possible_solutions": solutions,
                "synthesis_confidence": np.random.uniform(0.6, 0.9),
                "implementation_difficulty": self._assess_problem_complexity(problem)
            }
        return synthesize
    
    def _evaluate_distributed_effectiveness(self, responses: Dict, processing_time: float) -> float:
        """評估分散式方法效果 (Evaluate distributed approach effectiveness)"""
        factors = {
            "response_diversity": len(set(str(r) for r in responses.values())),
            "processing_speed": max(0, 1 - processing_time / 10),  # 歸一化處理速度
            "node_participation": len([r for r in responses.values() if r]) / len(responses)
        }
        return np.mean(list(factors.values()))
    
    def _evaluate_centralized_effectiveness(self, results: List, processing_time: float) -> float:
        """評估集中式方法效果 (Evaluate centralized approach effectiveness)"""
        factors = {
            "completeness": len(results) / 5,  # 完成所有5個步驟
            "processing_speed": max(0, 1 - processing_time / 5),
            "logical_consistency": 0.8  # 假設集中式方法邏輯一致性較高
        }
        return np.mean(list(factors.values()))


class BiologicalThinkingDemo:
    """生物學思考演示 (Biological Thinking Demonstration)"""
    
    def __init__(self):
        self.system = DistributedThinkingSystem()
        self._setup_biological_network()
    
    def _setup_biological_network(self):
        """設置生物學網絡 (Setup biological network)"""
        # 模擬章魚的8隻觸手 (Simulate octopus's 8 tentacles)
        for i in range(8):
            tentacle = OctopusTentacleNode(f"tentacle_{i}", (i * 45, 0, 0))  # 45度間隔分布
            self.system.add_node(tentacle)
        
        # 創建觸手間的協調網絡 (Create coordination network between tentacles)
        self.system.create_network_topology("small_world")
    
    def demonstrate_octopus_exploration(self):
        """演示章魚式探索 (Demonstrate octopus-style exploration)"""
        print("=== 章魚式分散探索演示 (Octopus-style Distributed Exploration Demo) ===")
        
        # 模擬遇到複雜物體 (Simulate encountering a complex object)
        exploration_stimuli = [
            (0.5, EnvironmentalStimulus("object_contact", (1, 0, 0), 0.7, 1.0, {"texture": "smooth", "temperature": "warm"})),
            (1.0, EnvironmentalStimulus("object_contact", (2, 0, 0), 0.8, 1.0, {"texture": "rough", "hardness": "soft"})),
            (1.5, EnvironmentalStimulus("object_contact", (3, 0, 0), 0.6, 1.0, {"texture": "bumpy", "size": "large"})),
            (2.0, EnvironmentalStimulus("chemical_trace", (2, 0, 0), 0.5, 2.0, {"chemical_type": "food"}))
        ]
        
        print("開始探索未知物體...")
        report = self.system.simulate_thinking_process(duration=3.0, stimulus_events=exploration_stimuli)
        
        print(f"探索結果:")
        print(f"  響應速度: {report['thinking_effectiveness'].get('response_speed', 0):.2%}")
        print(f"  適應能力: {report['thinking_effectiveness'].get('adaptability', 0):.2%}")
        print(f"  協調效果: {report['network_analysis']['clustering_coefficient']:.3f}")
        
        return report
    
    def demonstrate_plant_signaling(self):
        """演示植物式化學信號 (Demonstrate plant-style chemical signaling)"""
        print("\n=== 植物式化學信號演示 (Plant-style Chemical Signaling Demo) ===")
        
        # 添加化學信號節點模擬植物根系網絡 (Add chemical nodes to simulate plant root network)
        for i in range(6):
            chemical_node = ChemicalSignalingNode(f"root_{i}", (i, i, 0))
            self.system.add_node(chemical_node)
        
        # 重新建立網絡 (Rebuild network)
        self.system.create_network_topology("hierarchical")
        
        # 模擬營養物質發現和威脅警報 (Simulate nutrient discovery and threat alert)
        chemical_events = [
            (0.5, EnvironmentalStimulus("chemical_trace", (2, 2, 0), 0.8, 3.0, {"chemical_type": "food"})),
            (2.0, EnvironmentalStimulus("chemical_trace", (4, 4, 0), 0.9, 2.0, {"chemical_type": "danger"}))
        ]
        
        print("開始化學信號傳播...")
        report = self.system.simulate_thinking_process(duration=4.0, stimulus_events=chemical_events)
        
        print(f"信號傳播結果:")
        print(f"  網絡密度: {report['network_analysis']['network_density']:.3f}")
        print(f"  涌現事件: {report['simulation_summary']['emergence_events']} 次")
        
        return report


class SwarmIntelligenceDemo:
    """群體智慧演示 (Swarm Intelligence Demonstration)"""
    
    def __init__(self):
        self.system = DistributedThinkingSystem()
        self._setup_swarm_network()
    
    def _setup_swarm_network(self):
        """設置群體網絡 (Setup swarm network)"""
        # 創建蜂群式決策網絡 (Create bee colony decision network)
        for i in range(12):
            bee_node = SwarmIntelligenceNode(f"bee_{i}")
            self.system.add_node(bee_node)
        
        self.system.create_network_topology("small_world")
    
    def demonstrate_collective_decision_making(self):
        """演示集體決策 (Demonstrate collective decision making)"""
        print("=== 蜂群式集體決策演示 (Bee Colony Collective Decision Making Demo) ===")
        
        # 模擬多個巢穴選址選項 (Simulate multiple nest site options)
        decision_stimuli = [
            (1.0, EnvironmentalStimulus("site_evaluation", (1, 0, 0), 0.7, 2.0, {"site_quality": 0.7, "option": "A"})),
            (1.5, EnvironmentalStimulus("site_evaluation", (2, 0, 0), 0.8, 2.0, {"site_quality": 0.8, "option": "B"})),
            (2.0, EnvironmentalStimulus("site_evaluation", (3, 0, 0), 0.6, 2.0, {"site_quality": 0.6, "option": "C"}))
        ]
        
        print("開始集體決策過程...")
        
        # 初始化蜂群意見 (Initialize swarm opinions)
        swarm_nodes = [node for node in self.system.nodes.values() if isinstance(node, SwarmIntelligenceNode)]
        for node in swarm_nodes:
            node.opinion = np.random.uniform(0.3, 0.9)  # 隨機初始意見
            node.confidence = np.random.uniform(0.4, 0.8)
        
        report = self.system.simulate_thinking_process(duration=4.0, stimulus_events=decision_stimuli)
        
        # 分析最終決策結果 (Analyze final decision results)
        final_opinions = [node.opinion for node in swarm_nodes]
        final_confidences = [node.confidence for node in swarm_nodes]
        
        print(f"決策結果分析:")
        print(f"  平均意見: {np.mean(final_opinions):.3f}")
        print(f"  意見一致性: {1 - np.std(final_opinions):.3f}")
        print(f"  平均信心度: {np.mean(final_confidences):.3f}")
        print(f"  共識達成率: {report['simulation_summary']['consensus_rate']:.2%}")
        
        return report


class ExtendedMindDemo:
    """延展心智演示 (Extended Mind Demonstration)"""
    
    def __init__(self):
        self.system = DistributedThinkingSystem()
        self._setup_extended_mind_network()
    
    def _setup_extended_mind_network(self):
        """設置延展心智網絡 (Setup extended mind network)"""
        # 創建人-工具協作網絡 (Create human-tool collaboration network)
        for i in range(5):
            extended_node = ExtendedMindNode(f"cognitive_agent_{i}")
            
            # 為每個節點配置不同的工具組合 (Configure different tool combinations for each node)
            if i == 0:  # 計算專家 (Calculation expert)
                extended_node.add_external_tool("calculator", self._create_calculator())
                extended_node.add_external_tool("statistics", self._create_statistics_tool())
            elif i == 1:  # 導航專家 (Navigation expert)
                extended_node.add_external_tool("gps", self._create_gps_tool())
                extended_node.add_external_tool("map", self._create_map_tool())
            elif i == 2:  # 語言專家 (Language expert)
                extended_node.add_external_tool("translator", self._create_translator())
                extended_node.add_external_tool("dictionary", self._create_dictionary())
            elif i == 3:  # 記憶專家 (Memory expert)
                extended_node.add_external_tool("database", self._create_database())
                extended_node.add_external_tool("search", self._create_search_tool())
            else:  # 通用專家 (General expert)
                extended_node.add_external_tool("assistant", self._create_general_assistant())
            
            self.system.add_node(extended_node)
        
        self.system.create_network_topology("fully_connected")
    
    def _create_calculator(self):
        """創建計算器工具 (Create calculator tool)"""
        def calculate(data):
            expression = data.get("expression", "1+1")
            try:
                # 安全的數學表達式評估 (Safe mathematical expression evaluation)
                result = eval(expression, {"__builtins__": {}}, {"sin": np.sin, "cos": np.cos, "pi": np.pi})
                return f"計算結果: {expression} = {result}"
            except:
                return f"計算錯誤: 無法計算 {expression}"
        return calculate
    
    def _create_statistics_tool(self):
        """創建統計工具 (Create statistics tool)"""
        def analyze(data):
            numbers = data.get("numbers", [1, 2, 3, 4, 5])
            return {
                "mean": np.mean(numbers),
                "std": np.std(numbers),
                "min": np.min(numbers),
                "max": np.max(numbers)
            }
        return analyze
    
    def _create_gps_tool(self):
        """創建GPS工具 (Create GPS tool)"""
        def navigate(data):
            destination = data.get("destination", "未知位置")
            return f"導航至 {destination}: 預估時間 {np.random.randint(5, 30)} 分鐘"
        return navigate
    
    def _create_map_tool(self):
        """創建地圖工具 (Create map tool)"""
        def show_map(data):
            location = data.get("location", "當前位置")
            return f"地圖顯示: {location} 附近的興趣點和路線"
        return show_map
    
    def _create_translator(self):
        """創建翻譯工具 (Create translator tool)"""
        def translate(data):
            text = data.get("text", "Hello")
            target_lang = data.get("target", "中文")
            translations = {
                "Hello": "你好",
                "Thank you": "謝謝",
                "Goodbye": "再見"
            }
            return f"翻譯為{target_lang}: {translations.get(text, f'[翻譯]{text}')}"
        return translate
    
    def _create_dictionary(self):
        """創建字典工具 (Create dictionary tool)"""
        def lookup(data):
            word = data.get("word", "intelligence")
            definitions = {
                "intelligence": "智力；智慧；情報",
                "distributed": "分佈的；分散的",
                "thinking": "思考；思維"
            }
            return f"定義: {word} - {definitions.get(word, '未找到定義')}"
        return lookup
    
    def _create_database(self):
        """創建數據庫工具 (Create database tool)"""
        def query(data):
            query_text = data.get("query", "SELECT * FROM knowledge")
            return f"數據庫查詢結果: {query_text} -> 找到 {np.random.randint(1, 100)} 條記錄"
        return query
    
    def _create_search_tool(self):
        """創建搜索工具 (Create search tool)"""
        def search(data):
            keywords = data.get("keywords", ["distributed", "thinking"])
            return f"搜索結果: 找到 {len(keywords) * np.random.randint(10, 50)} 條相關信息"
        return search
    
    def _create_general_assistant(self):
        """創建通用助手工具 (Create general assistant tool)"""
        def assist(data):
            task = data.get("task", "一般任務")
            return f"助手回應: 正在處理 '{task}'，建議下一步行動"
        return assist
    
    def demonstrate_collaborative_problem_solving(self):
        """演示協作問題解決 (Demonstrate collaborative problem solving)"""
        print("=== 延展心智協作問題解決演示 (Extended Mind Collaborative Problem Solving Demo) ===")
        
        # 模擬複雜的多學科問題 (Simulate complex interdisciplinary problem)
        problem_stimuli = [
            (1.0, EnvironmentalStimulus("math_problem", (0, 0, 0), 0.8, 2.0, {"expression": "2*pi*3.14159", "complexity": "medium"})),
            (2.0, EnvironmentalStimulus("navigation_query", (0, 0, 0), 0.7, 1.5, {"destination": "最近的圖書館", "urgency": "high"})),
            (3.0, EnvironmentalStimulus("translation_request", (0, 0, 0), 0.6, 1.0, {"text": "Thank you", "target": "中文"})),
            (4.0, EnvironmentalStimulus("information_search", (0, 0, 0), 0.9, 2.0, {"keywords": ["artificial", "intelligence"], "depth": "comprehensive"}))
        ]
        
        print("開始協作問題解決...")
        report = self.system.simulate_thinking_process(duration=5.0, stimulus_events=problem_stimuli)
        
        # 分析工具使用效果 (Analyze tool usage effectiveness)
        extended_nodes = [node for node in self.system.nodes.values() if isinstance(node, ExtendedMindNode)]
        tool_usage_stats = {}
        
        for node in extended_nodes:
            for tool_name, proficiency in node.tool_proficiency.items():
                if tool_name not in tool_usage_stats:
                    tool_usage_stats[tool_name] = []
                tool_usage_stats[tool_name].append(proficiency)
        
        print(f"協作解決結果:")
        print(f"  參與節點: {len(extended_nodes)} 個")
        print(f"  使用工具: {len(tool_usage_stats)} 種")
        print(f"  平均工具熟練度: {np.mean([np.mean(profs) for profs in tool_usage_stats.values()]):.3f}")
        print(f"  認知卸載效率: {report['thinking_effectiveness'].get('overall', 0):.2%}")
        
        return report


def demonstrate_integration_with_human_evaluation():
    """演示與人類表達評估的整合 (Demonstrate integration with human expression evaluation)"""
    print("\n=== 分散式思考與人類表達評估整合演示 ===")
    print("=== Distributed Thinking and Human Expression Evaluation Integration Demo ===")
    
    if not HUMAN_EVALUATOR_AVAILABLE:
        print("人類表達評估器不可用，跳過整合演示")
        return
    
    integrator = DistributedThinkingEvaluationIntegrator()
    
    # 測試不同複雜度的問題 (Test problems of different complexity)
    test_problems = [
        "1 + 1 等於多少？",  # 簡單問題
        "如何優化團隊協作的效率？",  # 中等複雜度問題
        "在複雜的多文化環境中，如何建立有效的跨文化溝通機制並解決文化衝突？"  # 複雜問題
    ]
    
    for i, problem in enumerate(test_problems, 1):
        print(f"\n--- 測試案例 {i}: {problem[:20]}... ---")
        
        result = integrator.evaluate_distributed_vs_centralized_thinking(problem)
        
        print(f"分散式方法效果: {result['distributed_approach']['effectiveness_score']:.3f}")
        print(f"集中式方法效果: {result['centralized_approach']['effectiveness_score']:.3f}")
        print(f"推薦方法: {result['comparison_metrics']['optimal_approach']}")
        print(f"效果比率: {result['comparison_metrics']['effectiveness_ratio']:.2f}")


def main():
    """主演示函數 (Main demonstration function)"""
    print("=== 分散式思考系統綜合演示 (Comprehensive Distributed Thinking System Demo) ===\n")
    
    # 1. 生物學思考演示 (Biological thinking demo)
    bio_demo = BiologicalThinkingDemo()
    bio_demo.demonstrate_octopus_exploration()
    bio_demo.demonstrate_plant_signaling()
    
    # 2. 群體智慧演示 (Swarm intelligence demo)
    swarm_demo = SwarmIntelligenceDemo()
    swarm_demo.demonstrate_collective_decision_making()
    
    # 3. 延展心智演示 (Extended mind demo)
    extended_demo = ExtendedMindDemo()
    extended_demo.demonstrate_collaborative_problem_solving()
    
    # 4. 整合評估演示 (Integration evaluation demo)
    demonstrate_integration_with_human_evaluation()
    
    print("\n=== 總結 (Summary) ===")
    print("所有演示完成！分散式思考系統成功展示了多種無腦思考模式：")
    print("1. 生物學啟發的分散式神經處理")
    print("2. 群體智慧的集體決策機制")
    print("3. 延展心智的工具協作模式")
    print("4. 與現有評估框架的有效整合")
    print("\nAll demonstrations completed! The distributed thinking system successfully showcased multiple brainless thinking modes:")
    print("1. Biologically-inspired distributed neural processing")
    print("2. Swarm intelligence collective decision mechanisms")
    print("3. Extended mind tool collaboration models")
    print("4. Effective integration with existing evaluation frameworks")


if __name__ == "__main__":
    main()