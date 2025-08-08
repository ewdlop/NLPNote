"""
路徑積分自然語言處理 (Path Integral Natural Language Processing)
依照天道的路徑積分 (Following the Path Integral of the Heavenly Way)

This module implements a path integral approach to natural language processing
that follows the philosophical principles of 天道 (Heavenly Way), emphasizing
natural flow, harmony, and emergent optimal solutions in linguistic analysis.

The approach combines:
1. Mathematical path integrals from quantum mechanics
2. Chinese philosophical principles of 天道 (natural order and harmony)
3. Advanced NLP techniques for semantic analysis
4. Integration with existing human expression evaluation frameworks
"""

import math
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import itertools

try:
    from HumanExpressionEvaluator import HumanExpressionEvaluator, ExpressionContext, EvaluationResult
    from AStarNLP import AStarNLP
    EVALUATOR_AVAILABLE = True
except ImportError:
    EVALUATOR_AVAILABLE = False


class TianDaoPath(Enum):
    """天道路徑類型 (Heavenly Way Path Types)"""
    WU_WEI = "wu_wei"  # 無為 - Natural action through non-action
    YIN_YANG = "yin_yang"  # 陰陽 - Balance and complementarity
    WU_XING = "wu_xing"  # 五行 - Five elements interaction
    TAI_CHI = "tai_chi"  # 太極 - Supreme ultimate harmony
    NATURAL_FLOW = "natural_flow"  # 自然流動 - Natural flowing


@dataclass
class SemanticPath:
    """語義路徑 (Semantic Path)"""
    start_concept: str
    end_concept: str
    intermediate_concepts: List[str]
    path_weight: float
    harmony_score: float
    naturalness_score: float
    tian_dao_alignment: float
    path_type: TianDaoPath


@dataclass
class PathIntegrationResult:
    """路徑積分結果 (Path Integration Result)"""
    optimal_path: SemanticPath
    all_paths: List[SemanticPath]
    integration_value: float
    harmony_index: float
    naturalness_index: float
    tian_dao_index: float
    convergence_achieved: bool


class TianDaoCalculator:
    """天道計算器 (Heavenly Way Calculator)"""
    
    def __init__(self):
        """初始化天道計算器"""
        self.wu_wei_weight = 0.3  # 無為權重
        self.yin_yang_weight = 0.25  # 陰陽權重
        self.wu_xing_weight = 0.2  # 五行權重
        self.tai_chi_weight = 0.15  # 太極權重
        self.natural_flow_weight = 0.1  # 自然流動權重
    
    def calculate_wu_wei_score(self, concept1: str, concept2: str) -> float:
        """
        計算無為分數 (Calculate Wu Wei Score)
        無為: 順其自然，不強求，自然而然的行動
        """
        # 計算概念間的自然連接強度
        # 較短的概念通常更自然，較長的可能過於造作
        length_factor = 1.0 / (1.0 + abs(len(concept1) - len(concept2)) * 0.1)
        
        # 計算字符相似性（自然關聯）
        common_chars = set(concept1.lower()) & set(concept2.lower())
        similarity = len(common_chars) / max(len(set(concept1.lower())), len(set(concept2.lower())), 1)
        
        # 無為強調自然，不強求
        wu_wei_score = (length_factor * 0.6 + similarity * 0.4) * math.exp(-abs(len(concept1) - len(concept2)) * 0.05)
        
        return min(max(wu_wei_score, 0.0), 1.0)
    
    def calculate_yin_yang_score(self, concept1: str, concept2: str) -> float:
        """
        計算陰陽分數 (Calculate Yin Yang Score)
        陰陽: 互補、平衡、對立統一
        """
        # 檢查概念的互補性
        opposing_pairs = {
            '大': '小', '高': '低', '快': '慢', '明': '暗', '強': '弱',
            '好': '壞', '新': '舊', '熱': '冷', '正': '負', '是': '非',
            'big': 'small', 'high': 'low', 'fast': 'slow', 'light': 'dark',
            'strong': 'weak', 'good': 'bad', 'new': 'old', 'hot': 'cold'
        }
        
        # 計算對立統一性
        complementarity = 0.0
        for char1 in concept1:
            if char1 in opposing_pairs:
                if opposing_pairs[char1] in concept2:
                    complementarity += 1.0
        
        for char2 in concept2:
            if char2 in opposing_pairs:
                if opposing_pairs[char2] in concept1:
                    complementarity += 1.0
        
        # 正規化
        max_possible_complementarity = len(concept1) + len(concept2)
        if max_possible_complementarity > 0:
            yin_yang_score = complementarity / max_possible_complementarity
        else:
            yin_yang_score = 0.0
        
        # 加入平衡因子
        balance_factor = 1.0 - abs(len(concept1) - len(concept2)) / max(len(concept1), len(concept2), 1)
        yin_yang_score = (yin_yang_score * 0.7 + balance_factor * 0.3)
        
        return min(max(yin_yang_score, 0.0), 1.0)
    
    def calculate_wu_xing_score(self, concepts: List[str]) -> float:
        """
        計算五行分數 (Calculate Wu Xing Score)
        五行: 木火土金水的相生相剋關係
        """
        wu_xing_elements = {
            '木': {'生': '火', '剋': '土', '被生': '水', '被剋': '金'},
            '火': {'生': '土', '剋': '金', '被生': '木', '被剋': '水'},
            '土': {'生': '金', '剋': '水', '被生': '火', '被剋': '木'},
            '金': {'生': '水', '剋': '木', '被生': '土', '被剋': '火'},
            '水': {'生': '木', '剋': '火', '被生': '金', '被剋': '土'}
        }
        
        # 將概念映射到五行
        element_mapping = {
            '成長': '木', '創新': '木', '開始': '木',
            '熱情': '火', '活力': '火', '光明': '火',
            '穩定': '土', '中心': '土', '平衡': '土',
            '堅定': '金', '決斷': '金', '收穫': '金',
            '流動': '水', '智慧': '水', '適應': '水'
        }
        
        # 計算五行和諧度
        harmony_score = 0.0
        element_count = 0
        
        for i, concept in enumerate(concepts):
            for keyword, element in element_mapping.items():
                if keyword in concept:
                    element_count += 1
                    # 檢查與其他概念的五行關係
                    for j, other_concept in enumerate(concepts):
                        if i != j:
                            for other_keyword, other_element in element_mapping.items():
                                if other_keyword in other_concept and element in wu_xing_elements:
                                    if other_element == wu_xing_elements[element]['生']:
                                        harmony_score += 1.0  # 相生關係
                                    elif other_element == wu_xing_elements[element]['被生']:
                                        harmony_score += 0.8  # 被生關係
                                    elif other_element == wu_xing_elements[element]['剋']:
                                        harmony_score -= 0.5  # 相剋關係
                                    elif other_element == wu_xing_elements[element]['被剋']:
                                        harmony_score -= 0.3  # 被剋關係
        
        if element_count > 0:
            wu_xing_score = harmony_score / element_count
            wu_xing_score = (wu_xing_score + 1.0) / 2.0  # 正規化到 [0, 1]
        else:
            wu_xing_score = 0.5  # 中性分數
        
        return min(max(wu_xing_score, 0.0), 1.0)
    
    def calculate_tai_chi_score(self, concepts: List[str]) -> float:
        """
        計算太極分數 (Calculate Tai Chi Score)
        太極: 最高的和諧與統一
        """
        if len(concepts) < 2:
            return 0.5
        
        # 計算整體和諧度
        total_harmony = 0.0
        pair_count = 0
        
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                # 計算每對概念的和諧度
                yin_yang = self.calculate_yin_yang_score(concepts[i], concepts[j])
                wu_wei = self.calculate_wu_wei_score(concepts[i], concepts[j])
                
                # 太極強調整體統一
                pair_harmony = (yin_yang + wu_wei) / 2.0
                total_harmony += pair_harmony
                pair_count += 1
        
        if pair_count > 0:
            average_harmony = total_harmony / pair_count
        else:
            average_harmony = 0.5
        
        # 加入統一性因子
        concept_lengths = [len(c) for c in concepts]
        if concept_lengths:
            length_variance = np.var(concept_lengths)
            unity_factor = 1.0 / (1.0 + length_variance * 0.1)
        else:
            unity_factor = 1.0
        
        tai_chi_score = average_harmony * 0.8 + unity_factor * 0.2
        
        return min(max(tai_chi_score, 0.0), 1.0)
    
    def calculate_natural_flow_score(self, path: List[str]) -> float:
        """
        計算自然流動分數 (Calculate Natural Flow Score)
        自然流動: 順暢、無阻礙的概念轉換
        """
        if len(path) < 2:
            return 1.0
        
        flow_score = 0.0
        
        for i in range(len(path) - 1):
            current_concept = path[i]
            next_concept = path[i + 1]
            
            # 計算概念間的轉換流暢度
            # 1. 長度漸變（自然過渡）
            length_transition = 1.0 - abs(len(current_concept) - len(next_concept)) / max(len(current_concept), len(next_concept), 1) * 0.5
            
            # 2. 字符重疊（連續性）
            common_chars = set(current_concept.lower()) & set(next_concept.lower())
            overlap_score = len(common_chars) / max(len(set(current_concept.lower())), len(set(next_concept.lower())), 1)
            
            # 3. 語音相似性（簡化版）
            phonetic_similarity = self._calculate_phonetic_similarity(current_concept, next_concept)
            
            # 綜合流暢度
            transition_flow = (length_transition * 0.4 + overlap_score * 0.3 + phonetic_similarity * 0.3)
            flow_score += transition_flow
        
        # 正規化
        average_flow = flow_score / (len(path) - 1)
        
        return min(max(average_flow, 0.0), 1.0)
    
    def _calculate_phonetic_similarity(self, concept1: str, concept2: str) -> float:
        """計算語音相似性（簡化版）"""
        # 簡化的語音相似性計算
        # 基於首字母和元音
        vowels = set('aeiouAEIOU')
        
        # 提取首字母
        first_char_similarity = 1.0 if (concept1 and concept2 and concept1[0].lower() == concept2[0].lower()) else 0.0
        
        # 提取元音模式
        vowels1 = [c for c in concept1 if c in vowels]
        vowels2 = [c for c in concept2 if c in vowels]
        
        if vowels1 and vowels2:
            vowel_similarity = len(set(vowels1) & set(vowels2)) / max(len(set(vowels1)), len(set(vowels2)))
        else:
            vowel_similarity = 0.5
        
        return (first_char_similarity * 0.6 + vowel_similarity * 0.4)
    
    def calculate_tian_dao_alignment(self, path: List[str], path_type: TianDaoPath) -> float:
        """
        計算天道對齊度 (Calculate Heavenly Way Alignment)
        綜合所有天道原則的對齊程度
        """
        if len(path) < 2:
            return 0.5
        
        # 根據路徑類型計算對應分數
        if path_type == TianDaoPath.WU_WEI:
            primary_score = np.mean([self.calculate_wu_wei_score(path[i], path[i+1]) for i in range(len(path)-1)])
            weight = self.wu_wei_weight
        elif path_type == TianDaoPath.YIN_YANG:
            primary_score = np.mean([self.calculate_yin_yang_score(path[i], path[i+1]) for i in range(len(path)-1)])
            weight = self.yin_yang_weight
        elif path_type == TianDaoPath.WU_XING:
            primary_score = self.calculate_wu_xing_score(path)
            weight = self.wu_xing_weight
        elif path_type == TianDaoPath.TAI_CHI:
            primary_score = self.calculate_tai_chi_score(path)
            weight = self.tai_chi_weight
        elif path_type == TianDaoPath.NATURAL_FLOW:
            primary_score = self.calculate_natural_flow_score(path)
            weight = self.natural_flow_weight
        else:
            primary_score = 0.5
            weight = 0.2
        
        # 計算輔助分數
        natural_flow = self.calculate_natural_flow_score(path)
        tai_chi = self.calculate_tai_chi_score(path)
        
        # 綜合對齊度
        alignment = primary_score * weight + natural_flow * 0.3 + tai_chi * 0.2
        
        return min(max(alignment, 0.0), 1.0)


class PathIntegralNLP:
    """
    路徑積分自然語言處理器 (Path Integral Natural Language Processor)
    依照天道原則的語言路徑積分分析
    """
    
    def __init__(self, max_path_length: int = 10, integration_steps: int = 100):
        """
        初始化路徑積分NLP處理器
        
        Args:
            max_path_length: 最大路徑長度
            integration_steps: 積分步數
        """
        self.max_path_length = max_path_length
        self.integration_steps = integration_steps
        self.tian_dao_calculator = TianDaoCalculator()
        
        # 整合現有評估器（如果可用）
        if EVALUATOR_AVAILABLE:
            self.expression_evaluator = HumanExpressionEvaluator()
            self.astar_nlp = AStarNLP()
        else:
            self.expression_evaluator = None
            self.astar_nlp = None
    
    def generate_semantic_paths(self, start_concept: str, end_concept: str, 
                              intermediate_concepts: Optional[List[str]] = None) -> List[SemanticPath]:
        """
        生成語義路徑 (Generate Semantic Paths)
        在起始概念和結束概念之間生成所有可能的語義路徑
        """
        if intermediate_concepts is None:
            intermediate_concepts = self._generate_intermediate_concepts(start_concept, end_concept)
        
        paths = []
        
        # 生成不同長度的路徑
        for path_length in range(2, min(self.max_path_length + 1, len(intermediate_concepts) + 3)):
            # 生成該長度的所有可能路徑
            for combo in itertools.combinations(intermediate_concepts, path_length - 2):
                full_path = [start_concept] + list(combo) + [end_concept]
                
                # 為每種天道路徑類型創建路徑
                for path_type in TianDaoPath:
                    semantic_path = self._create_semantic_path(full_path, path_type)
                    if semantic_path.tian_dao_alignment > 0.1:  # 只保留有意義的路徑
                        paths.append(semantic_path)
        
        # 排序並限制路徑數量
        paths.sort(key=lambda p: p.tian_dao_alignment, reverse=True)
        return paths[:50]  # 保留前50個最佳路徑
    
    def _generate_intermediate_concepts(self, start_concept: str, end_concept: str) -> List[str]:
        """生成中間概念"""
        # 基於起始和結束概念生成相關的中間概念
        intermediate_concepts = []
        
        # 基於字符分析生成概念
        start_chars = set(start_concept)
        end_chars = set(end_concept)
        common_chars = start_chars & end_chars
        
        # 生成包含共同字符的概念
        for char in common_chars:
            if len(char.strip()) > 0:
                intermediate_concepts.extend([
                    f"包含{char}的概念",
                    f"關於{char}",
                    f"{char}相關",
                    f"涉及{char}"
                ])
        
        # 生成過渡概念
        transition_concepts = [
            "轉換", "演變", "發展", "變化", "過程", "階段",
            "橋接", "連接", "關聯", "相關", "類似", "對比",
            "平衡", "和諧", "統一", "整合", "融合", "結合"
        ]
        
        # 生成語義相關概念
        semantic_concepts = [
            f"{start_concept}的特徵", f"{end_concept}的本質",
            f"從{start_concept}到{end_concept}", f"{start_concept}與{end_concept}",
            "中間狀態", "過渡階段", "轉換點", "關鍵節點"
        ]
        
        intermediate_concepts.extend(transition_concepts)
        intermediate_concepts.extend(semantic_concepts)
        
        # 如果有A*NLP可用，使用其生成更多概念
        if self.astar_nlp:
            preprocessed_start = self.astar_nlp.preprocess_text(start_concept)
            preprocessed_end = self.astar_nlp.preprocess_text(end_concept)
            
            for word_start in preprocessed_start:
                for word_end in preprocessed_end:
                    intermediate_concepts.extend([
                        f"{word_start}到{word_end}",
                        f"{word_start}與{word_end}的關係",
                        f"類似{word_start}",
                        f"接近{word_end}"
                    ])
        
        # 去重並限制數量
        unique_concepts = list(set(intermediate_concepts))
        return unique_concepts[:30]  # 限制為30個中間概念
    
    def _create_semantic_path(self, path_concepts: List[str], path_type: TianDaoPath) -> SemanticPath:
        """創建語義路徑"""
        if len(path_concepts) < 2:
            raise ValueError("路徑至少需要兩個概念")
        
        start_concept = path_concepts[0]
        end_concept = path_concepts[-1]
        intermediate_concepts = path_concepts[1:-1]
        
        # 計算路徑權重
        path_weight = self._calculate_path_weight(path_concepts)
        
        # 計算和諧度
        harmony_score = self.tian_dao_calculator.calculate_tai_chi_score(path_concepts)
        
        # 計算自然度
        naturalness_score = self.tian_dao_calculator.calculate_natural_flow_score(path_concepts)
        
        # 計算天道對齊度
        tian_dao_alignment = self.tian_dao_calculator.calculate_tian_dao_alignment(path_concepts, path_type)
        
        return SemanticPath(
            start_concept=start_concept,
            end_concept=end_concept,
            intermediate_concepts=intermediate_concepts,
            path_weight=path_weight,
            harmony_score=harmony_score,
            naturalness_score=naturalness_score,
            tian_dao_alignment=tian_dao_alignment,
            path_type=path_type
        )
    
    def _calculate_path_weight(self, path_concepts: List[str]) -> float:
        """計算路徑權重"""
        if len(path_concepts) < 2:
            return 0.0
        
        total_weight = 0.0
        
        for i in range(len(path_concepts) - 1):
            # 計算相鄰概念間的權重
            concept1 = path_concepts[i]
            concept2 = path_concepts[i + 1]
            
            # 基於無為原則的權重
            wu_wei_weight = self.tian_dao_calculator.calculate_wu_wei_score(concept1, concept2)
            
            # 基於陰陽平衡的權重
            yin_yang_weight = self.tian_dao_calculator.calculate_yin_yang_score(concept1, concept2)
            
            # 綜合權重
            segment_weight = (wu_wei_weight + yin_yang_weight) / 2.0
            total_weight += segment_weight
        
        # 正規化
        average_weight = total_weight / (len(path_concepts) - 1)
        
        return average_weight
    
    def path_integral_evaluation(self, start_concept: str, end_concept: str,
                                intermediate_concepts: Optional[List[str]] = None) -> PathIntegrationResult:
        """
        執行路徑積分評估 (Perform Path Integral Evaluation)
        依照天道原則計算最優語義路徑
        """
        # 生成所有可能的語義路徑
        semantic_paths = self.generate_semantic_paths(start_concept, end_concept, intermediate_concepts)
        
        if not semantic_paths:
            # 如果沒有找到路徑，創建直接路徑
            direct_path = self._create_semantic_path([start_concept, end_concept], TianDaoPath.WU_WEI)
            semantic_paths = [direct_path]
        
        # 執行路徑積分計算
        integration_value = self._compute_path_integral(semantic_paths)
        
        # 找到最優路徑
        optimal_path = max(semantic_paths, key=lambda p: p.tian_dao_alignment)
        
        # 計算整體指標
        harmony_index = np.mean([path.harmony_score for path in semantic_paths])
        naturalness_index = np.mean([path.naturalness_score for path in semantic_paths])
        tian_dao_index = np.mean([path.tian_dao_alignment for path in semantic_paths])
        
        # 檢查收斂性
        convergence_achieved = self._check_convergence(semantic_paths)
        
        return PathIntegrationResult(
            optimal_path=optimal_path,
            all_paths=semantic_paths,
            integration_value=integration_value,
            harmony_index=harmony_index,
            naturalness_index=naturalness_index,
            tian_dao_index=tian_dao_index,
            convergence_achieved=convergence_achieved
        )
    
    def _compute_path_integral(self, paths: List[SemanticPath]) -> float:
        """計算路徑積分"""
        if not paths:
            return 0.0
        
        # 使用蒙特卡羅方法近似路徑積分
        total_integral = 0.0
        
        for path in paths:
            # 計算每條路徑的貢獻
            path_length = len(path.intermediate_concepts) + 2
            
            # 路徑的行動量 (action)
            action = self._calculate_path_action(path)
            
            # 費曼路徑積分的權重因子
            weight_factor = math.exp(-action / (path_length * 0.1))
            
            # 天道對齊度作為額外權重
            tian_dao_weight = path.tian_dao_alignment
            
            # 路徑貢獻
            path_contribution = weight_factor * tian_dao_weight * path.path_weight
            total_integral += path_contribution
        
        # 正規化
        if len(paths) > 0:
            integration_value = total_integral / len(paths)
        else:
            integration_value = 0.0
        
        return integration_value
    
    def _calculate_path_action(self, path: SemanticPath) -> float:
        """計算路徑的行動量 (action)"""
        # 在物理學中，行動量是拉格朗日量的時間積分
        # 在此處，我們將其定義為路徑偏離"自然流動"的程度
        
        full_path = [path.start_concept] + path.intermediate_concepts + [path.end_concept]
        
        action = 0.0
        
        for i in range(len(full_path) - 1):
            # 計算每個轉換的"成本"
            current_concept = full_path[i]
            next_concept = full_path[i + 1]
            
            # 長度變化成本（偏離自然）
            length_cost = abs(len(current_concept) - len(next_concept)) * 0.1
            
            # 語義跳躍成本
            semantic_distance = self._calculate_semantic_distance(current_concept, next_concept)
            semantic_cost = semantic_distance * 0.5
            
            # 總成本
            transition_cost = length_cost + semantic_cost
            action += transition_cost
        
        return action
    
    def _calculate_semantic_distance(self, concept1: str, concept2: str) -> float:
        """計算語義距離"""
        # 簡化的語義距離計算
        # 基於字符重疊和長度差異
        
        set1 = set(concept1.lower())
        set2 = set(concept2.lower())
        
        # 雅卡德距離
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union > 0:
            jaccard_similarity = intersection / union
            jaccard_distance = 1.0 - jaccard_similarity
        else:
            jaccard_distance = 1.0
        
        # 長度距離
        length_distance = abs(len(concept1) - len(concept2)) / max(len(concept1), len(concept2), 1)
        
        # 綜合距離
        semantic_distance = (jaccard_distance + length_distance) / 2.0
        
        return semantic_distance
    
    def _check_convergence(self, paths: List[SemanticPath]) -> bool:
        """檢查路徑積分是否收斂"""
        if len(paths) < 5:
            return False
        
        # 檢查天道對齊度的方差
        alignments = [path.tian_dao_alignment for path in paths[:10]]  # 取前10條路徑
        alignment_variance = np.var(alignments)
        
        # 如果方差較小，說明收斂
        convergence_threshold = 0.05
        convergence_achieved = alignment_variance < convergence_threshold
        
        return convergence_achieved
    
    def natural_language_flow_analysis(self, text: str, context: Optional[ExpressionContext] = None) -> Dict[str, Any]:
        """
        自然語言流動分析 (Natural Language Flow Analysis)
        分析文本中的自然流動和天道對齊
        """
        if context is None:
            context = ExpressionContext()
        
        # 分解文本為概念
        if self.astar_nlp:
            concepts = self.astar_nlp.preprocess_text(text)
        else:
            # 改進的中文分詞
            import re
            
            # 先移除標點符號並分割
            clean_text = re.sub(r'[，。！？；：、]', ' ', text)  # 中文標點
            clean_text = re.sub(r'[,.!?;:]', ' ', clean_text)   # 英文標點
            
            # 分別處理中文字符和英文單詞
            concepts = []
            
            # 使用正則表達式分割中英文
            tokens = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', clean_text)
            
            for token in tokens:
                if re.match(r'[\u4e00-\u9fff]+', token):  # 中文
                    # 中文按字符分割，然後組合成詞
                    if len(token) >= 2:
                        # 按兩個字符一組分割
                        for i in range(0, len(token), 2):
                            if i + 1 < len(token):
                                concepts.append(token[i:i+2])
                            else:
                                concepts.append(token[i])
                    else:
                        concepts.append(token)
                else:  # 英文
                    concepts.append(token.lower())
            
            # 過濾空白
            concepts = [word.strip() for word in concepts if word.strip()]
        
        if len(concepts) < 2:
            return {
                'flow_analysis': '文本過短，無法進行流動分析',
                'concepts_analyzed': concepts,
                'tian_dao_alignment': 0.5,
                'natural_flow_score': 0.5,
                'harmony_index': 0.5,
                'detailed_flow_scores': [],
                'detailed_harmony_scores': [],
                'recommendations': ['建議增加更多內容以便分析'],
                'integrated_analysis': {}
            }
        
        # 分析概念間的流動
        flow_scores = []
        harmony_scores = []
        
        for i in range(len(concepts) - 1):
            # 計算相鄰概念的流動分數
            flow_score = self.tian_dao_calculator.calculate_natural_flow_score([concepts[i], concepts[i+1]])
            flow_scores.append(flow_score)
            
            # 計算和諧分數
            harmony_score = self.tian_dao_calculator.calculate_yin_yang_score(concepts[i], concepts[i+1])
            harmony_scores.append(harmony_score)
        
        # 整體分析
        overall_flow = np.mean(flow_scores) if flow_scores else 0.5
        overall_harmony = np.mean(harmony_scores) if harmony_scores else 0.5
        
        # 計算天道對齊度
        tian_dao_alignment = self.tian_dao_calculator.calculate_tian_dao_alignment(concepts, TianDaoPath.NATURAL_FLOW)
        
        # 生成建議
        recommendations = self._generate_flow_recommendations(overall_flow, overall_harmony, tian_dao_alignment)
        
        # 整合現有評估器的結果（如果可用）
        integrated_analysis = {}
        if self.expression_evaluator and EVALUATOR_AVAILABLE:
            try:
                evaluation_result = self.expression_evaluator.comprehensive_evaluation(text, context)
                integrated_analysis = {
                    'human_expression_evaluation': evaluation_result,
                    'integration_notes': '已整合人類表達評估框架的結果'
                }
            except Exception as e:
                integrated_analysis = {
                    'integration_error': f'整合評估時出錯: {str(e)}'
                }
        
        return {
            'flow_analysis': f'文本包含 {len(concepts)} 個概念，整體流動分數: {overall_flow:.3f}',
            'concepts_analyzed': concepts,
            'tian_dao_alignment': tian_dao_alignment,
            'natural_flow_score': overall_flow,
            'harmony_index': overall_harmony,
            'detailed_flow_scores': flow_scores,
            'detailed_harmony_scores': harmony_scores,
            'recommendations': recommendations,
            'integrated_analysis': integrated_analysis
        }
    
    def _generate_flow_recommendations(self, flow_score: float, harmony_score: float, 
                                     tian_dao_alignment: float) -> List[str]:
        """生成流動改善建議"""
        recommendations = []
        
        if flow_score < 0.3:
            recommendations.append("建議改善概念間的自然過渡，使表達更流暢")
            recommendations.append("考慮使用更相近的概念或添加過渡詞語")
        
        if harmony_score < 0.3:
            recommendations.append("建議增強概念間的和諧性，尋找互補或平衡的表達")
            recommendations.append("考慮陰陽平衡原則，避免過於極端的對比")
        
        if tian_dao_alignment < 0.4:
            recommendations.append("建議更好地遵循天道原則，讓表達更自然")
            recommendations.append("嘗試無為的表達方式，避免過度造作")
        
        if flow_score > 0.7 and harmony_score > 0.7 and tian_dao_alignment > 0.7:
            recommendations.append("優秀！您的表達已經很好地遵循了天道原則")
            recommendations.append("繼續保持這種自然流暢的表達風格")
        
        if not recommendations:
            recommendations.append("您的表達基本符合自然流動原則")
            recommendations.append("可以嘗試進一步優化概念間的連接")
        
        return recommendations