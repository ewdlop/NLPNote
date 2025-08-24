"""
人格格論模型 (Personality Lattice Model)

This module implements a mathematical framework for modeling personality traits
using lattice theory, integrating with the existing NLP human expression
evaluation system.

Based on the theoretical framework outlined in 數學/人格與格論.md
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import itertools
from collections import defaultdict
import networkx as nx


class PersonalityTrait(Enum):
    """基礎人格特質枚舉 (Basic Personality Traits Enumeration)"""
    # 基礎核心特質 (Core Foundation Traits)
    BASIC_COGNITIVE = "basic_cognitive"
    BASIC_BEHAVIORAL = "basic_behavioral" 
    BASIC_SOCIAL = "basic_social"
    CORE_PERSONALITY = "core_personality"
    
    # 外向性相關 (Extraversion-related)
    FRIENDLINESS = "friendliness"
    EXPRESSIVENESS = "expressiveness"
    COMPETITIVENESS = "competitiveness"
    CONFIDENCE = "confidence"
    SOCIABILITY = "sociability"
    DOMINANCE = "dominance"
    SOCIAL_LEADERSHIP = "social_leadership"
    
    # 情緒穩定性相關 (Emotional Stability-related)
    ANXIETY_TOLERANCE = "anxiety_tolerance"
    ADAPTABILITY = "adaptability"
    CALMNESS = "calmness"
    OPTIMISM = "optimism"
    STRESS_MANAGEMENT = "stress_management"
    EMOTION_CONTROL = "emotion_control"
    COMPLETE_EMOTION_REGULATION = "complete_emotion_regulation"
    
    # 開放性相關 (Openness-related)
    CREATIVITY = "creativity"
    INTELLIGENCE = "intelligence"
    COGNITIVE_OPENNESS = "cognitive_openness"
    
    # 責任心相關 (Conscientiousness-related)
    RESPONSIBILITY = "responsibility"
    SYSTEMATIC_THINKING = "systematic_thinking"
    
    # 宜人性相關 (Agreeableness-related)
    COOPERATION = "cooperation"
    EMPATHY = "empathy"
    AGREEABLENESS = "agreeableness"
    
    # 高層整合特質 (High-level Integrated Traits)
    COMPLETE_PERSONALITY = "complete_personality"


@dataclass
class LatticeNode:
    """格節點 (Lattice Node)"""
    trait: PersonalityTrait
    level: int
    parents: Set[PersonalityTrait] = field(default_factory=set)
    children: Set[PersonalityTrait] = field(default_factory=set)
    intensity: float = 0.0  # 特質強度 (0.0 to 1.0)


@dataclass
class SituationalContext:
    """情境上下文 (Situational Context)"""
    situation_type: str = "general"  # professional, social, academic, intimate
    formality_level: float = 0.5  # 0.0 (informal) to 1.0 (formal)
    stress_level: float = 0.0  # 0.0 (relaxed) to 1.0 (high stress)
    social_hierarchy: str = "equal"  # subordinate, equal, superior
    cultural_context: str = "universal"
    emotional_valence: float = 0.0  # -1.0 (negative) to 1.0 (positive)


class PersonalityLattice:
    """人格特質格 (Personality Trait Lattice)"""
    
    def __init__(self):
        self.nodes: Dict[PersonalityTrait, LatticeNode] = {}
        self.relations: Set[Tuple[PersonalityTrait, PersonalityTrait]] = set()
        self.graph = nx.DiGraph()
        self._initialize_lattice()
    
    def _initialize_lattice(self):
        """初始化人格格結構 (Initialize personality lattice structure)"""
        
        # 定義層次關係 (Define hierarchical relationships)
        # Format: (child_trait, parent_trait)
        relationships = [
            # 基礎層級關係 (Foundation level relationships)
            (PersonalityTrait.BASIC_COGNITIVE, PersonalityTrait.CORE_PERSONALITY),
            (PersonalityTrait.BASIC_BEHAVIORAL, PersonalityTrait.CORE_PERSONALITY),
            (PersonalityTrait.BASIC_SOCIAL, PersonalityTrait.CORE_PERSONALITY),
            
            # 外向性樹狀結構 (Extraversion tree structure)
            (PersonalityTrait.FRIENDLINESS, PersonalityTrait.BASIC_SOCIAL),
            (PersonalityTrait.EXPRESSIVENESS, PersonalityTrait.BASIC_SOCIAL),
            (PersonalityTrait.COMPETITIVENESS, PersonalityTrait.BASIC_BEHAVIORAL),
            (PersonalityTrait.CONFIDENCE, PersonalityTrait.BASIC_BEHAVIORAL),
            
            (PersonalityTrait.SOCIABILITY, PersonalityTrait.FRIENDLINESS),
            (PersonalityTrait.SOCIABILITY, PersonalityTrait.EXPRESSIVENESS),
            (PersonalityTrait.DOMINANCE, PersonalityTrait.COMPETITIVENESS),
            (PersonalityTrait.DOMINANCE, PersonalityTrait.CONFIDENCE),
            
            (PersonalityTrait.SOCIAL_LEADERSHIP, PersonalityTrait.SOCIABILITY),
            (PersonalityTrait.SOCIAL_LEADERSHIP, PersonalityTrait.DOMINANCE),
            
            # 情緒穩定性結構 (Emotional stability structure)
            (PersonalityTrait.ANXIETY_TOLERANCE, PersonalityTrait.BASIC_BEHAVIORAL),
            (PersonalityTrait.ADAPTABILITY, PersonalityTrait.BASIC_BEHAVIORAL),
            (PersonalityTrait.CALMNESS, PersonalityTrait.BASIC_BEHAVIORAL),
            (PersonalityTrait.OPTIMISM, PersonalityTrait.BASIC_COGNITIVE),
            
            (PersonalityTrait.STRESS_MANAGEMENT, PersonalityTrait.ANXIETY_TOLERANCE),
            (PersonalityTrait.STRESS_MANAGEMENT, PersonalityTrait.ADAPTABILITY),
            (PersonalityTrait.EMOTION_CONTROL, PersonalityTrait.CALMNESS),
            (PersonalityTrait.EMOTION_CONTROL, PersonalityTrait.OPTIMISM),
            
            (PersonalityTrait.COMPLETE_EMOTION_REGULATION, PersonalityTrait.STRESS_MANAGEMENT),
            (PersonalityTrait.COMPLETE_EMOTION_REGULATION, PersonalityTrait.EMOTION_CONTROL),
            
            # 開放性結構 (Openness structure)
            (PersonalityTrait.CREATIVITY, PersonalityTrait.BASIC_COGNITIVE),
            (PersonalityTrait.INTELLIGENCE, PersonalityTrait.BASIC_COGNITIVE),
            (PersonalityTrait.COGNITIVE_OPENNESS, PersonalityTrait.CREATIVITY),
            (PersonalityTrait.COGNITIVE_OPENNESS, PersonalityTrait.INTELLIGENCE),
            
            # 責任心結構 (Conscientiousness structure)
            (PersonalityTrait.RESPONSIBILITY, PersonalityTrait.BASIC_BEHAVIORAL),
            (PersonalityTrait.SYSTEMATIC_THINKING, PersonalityTrait.BASIC_COGNITIVE),
            
            # 宜人性結構 (Agreeableness structure)
            (PersonalityTrait.COOPERATION, PersonalityTrait.BASIC_SOCIAL),
            (PersonalityTrait.EMPATHY, PersonalityTrait.BASIC_SOCIAL),
            (PersonalityTrait.AGREEABLENESS, PersonalityTrait.COOPERATION),
            (PersonalityTrait.AGREEABLENESS, PersonalityTrait.EMPATHY),
            
            # 頂層整合 (Top-level integration)
            (PersonalityTrait.COMPLETE_PERSONALITY, PersonalityTrait.SOCIAL_LEADERSHIP),
            (PersonalityTrait.COMPLETE_PERSONALITY, PersonalityTrait.COMPLETE_EMOTION_REGULATION),
            (PersonalityTrait.COMPLETE_PERSONALITY, PersonalityTrait.COGNITIVE_OPENNESS),
            (PersonalityTrait.COMPLETE_PERSONALITY, PersonalityTrait.RESPONSIBILITY),
            (PersonalityTrait.COMPLETE_PERSONALITY, PersonalityTrait.AGREEABLENESS),
        ]
        
        # 創建節點 (Create nodes)
        for trait in PersonalityTrait:
            self.nodes[trait] = LatticeNode(trait=trait, level=0)
        
        # 建立關係 (Establish relationships)
        for child, parent in relationships:
            self.add_relation(child, parent)
        
        # 計算層級 (Calculate levels)
        self._calculate_levels()
        
        # 構建圖結構 (Build graph structure)
        self._build_graph()
    
    def add_relation(self, child: PersonalityTrait, parent: PersonalityTrait):
        """添加偏序關係 (Add partial order relation)"""
        self.relations.add((child, parent))
        self.nodes[child].parents.add(parent)
        self.nodes[parent].children.add(child)
    
    def _calculate_levels(self):
        """計算每個節點在格中的層級 (Calculate level of each node in lattice)"""
        # 使用拓撲排序計算層級
        visited = set()
        
        def dfs_level(trait: PersonalityTrait) -> int:
            if trait in visited:
                return self.nodes[trait].level
            
            visited.add(trait)
            max_parent_level = -1
            
            for parent in self.nodes[trait].parents:
                parent_level = dfs_level(parent)
                max_parent_level = max(max_parent_level, parent_level)
            
            self.nodes[trait].level = max_parent_level + 1
            return self.nodes[trait].level
        
        for trait in PersonalityTrait:
            dfs_level(trait)
    
    def _build_graph(self):
        """構建NetworkX圖結構 (Build NetworkX graph structure)"""
        for child, parent in self.relations:
            self.graph.add_edge(child.value, parent.value)
    
    def join(self, trait_a: PersonalityTrait, trait_b: PersonalityTrait) -> PersonalityTrait:
        """計算兩個特質的並（最小上界）(Calculate join/supremum of two traits)"""
        return self._supremum([trait_a, trait_b])
    
    def meet(self, trait_a: PersonalityTrait, trait_b: PersonalityTrait) -> PersonalityTrait:
        """計算兩個特質的交（最大下界）(Calculate meet/infimum of two traits)"""
        return self._infimum([trait_a, trait_b])
    
    def _supremum(self, traits: List[PersonalityTrait]) -> PersonalityTrait:
        """計算特質集合的最小上界 (Calculate supremum of trait set)"""
        if not traits:
            return PersonalityTrait.CORE_PERSONALITY
        
        if len(traits) == 1:
            return traits[0]
        
        # 找到所有特質的共同上界
        common_upper_bounds = set(PersonalityTrait)
        
        for trait in traits:
            upper_bounds = self._get_upper_bounds(trait)
            common_upper_bounds &= upper_bounds
        
        # 在共同上界中找到最小的
        if not common_upper_bounds:
            return PersonalityTrait.COMPLETE_PERSONALITY
        
        min_level = min(self.nodes[t].level for t in common_upper_bounds)
        candidates = [t for t in common_upper_bounds if self.nodes[t].level == min_level]
        
        return candidates[0] if candidates else PersonalityTrait.COMPLETE_PERSONALITY
    
    def _infimum(self, traits: List[PersonalityTrait]) -> PersonalityTrait:
        """計算特質集合的最大下界 (Calculate infimum of trait set)"""
        if not traits:
            return PersonalityTrait.CORE_PERSONALITY
        
        if len(traits) == 1:
            return traits[0]
        
        # 找到所有特質的共同下界
        common_lower_bounds = set(PersonalityTrait)
        
        for trait in traits:
            lower_bounds = self._get_lower_bounds(trait)
            common_lower_bounds &= lower_bounds
        
        # 在共同下界中找到最大的
        if not common_lower_bounds:
            return PersonalityTrait.CORE_PERSONALITY
        
        max_level = max(self.nodes[t].level for t in common_lower_bounds)
        candidates = [t for t in common_lower_bounds if self.nodes[t].level == max_level]
        
        return candidates[0] if candidates else PersonalityTrait.CORE_PERSONALITY
    
    def _get_upper_bounds(self, trait: PersonalityTrait) -> Set[PersonalityTrait]:
        """獲取特質的所有上界 (Get all upper bounds of a trait)"""
        upper_bounds = {trait}
        stack = [trait]
        
        while stack:
            current = stack.pop()
            for parent in self.nodes[current].parents:
                if parent not in upper_bounds:
                    upper_bounds.add(parent)
                    stack.append(parent)
        
        return upper_bounds
    
    def _get_lower_bounds(self, trait: PersonalityTrait) -> Set[PersonalityTrait]:
        """獲取特質的所有下界 (Get all lower bounds of a trait)"""
        lower_bounds = {trait}
        stack = [trait]
        
        while stack:
            current = stack.pop()
            for child in self.nodes[current].children:
                if child not in lower_bounds:
                    lower_bounds.add(child)
                    stack.append(child)
        
        return lower_bounds
    
    def calculate_trait_intensity(self, trait: PersonalityTrait) -> float:
        """計算特質強度 (Calculate trait intensity)"""
        upper_bounds = self._get_upper_bounds(trait)
        lower_bounds = self._get_lower_bounds(trait)
        total_traits = len(PersonalityTrait)
        
        # 使用公式：I(t) = |下界|/|P| × |上界|/|P|
        intensity = (len(lower_bounds) / total_traits) * (len(upper_bounds) / total_traits)
        return min(max(intensity, 0.0), 1.0)
    
    def get_situational_ideal(self, context: SituationalContext) -> Set[PersonalityTrait]:
        """獲取情境對應的特質理想 (Get situational trait ideal)"""
        ideal = set()
        
        # 根據情境類型選擇相關特質
        if context.situation_type == "professional":
            ideal.update([
                PersonalityTrait.RESPONSIBILITY,
                PersonalityTrait.SYSTEMATIC_THINKING,
                PersonalityTrait.CONFIDENCE,
                PersonalityTrait.STRESS_MANAGEMENT,
            ])
        elif context.situation_type == "social":
            ideal.update([
                PersonalityTrait.SOCIABILITY,
                PersonalityTrait.FRIENDLINESS,
                PersonalityTrait.COOPERATION,
                PersonalityTrait.EMPATHY,
            ])
        elif context.situation_type == "academic":
            ideal.update([
                PersonalityTrait.INTELLIGENCE,
                PersonalityTrait.CREATIVITY,
                PersonalityTrait.SYSTEMATIC_THINKING,
                PersonalityTrait.RESPONSIBILITY,
            ])
        elif context.situation_type == "intimate":
            ideal.update([
                PersonalityTrait.EMPATHY,
                PersonalityTrait.EMOTION_CONTROL,
                PersonalityTrait.FRIENDLINESS,
                PersonalityTrait.COOPERATION,
            ])
        
        # 根據正式程度調整
        if context.formality_level > 0.7:
            ideal.add(PersonalityTrait.RESPONSIBILITY)
            ideal.add(PersonalityTrait.CALMNESS)
        elif context.formality_level < 0.3:
            ideal.add(PersonalityTrait.EXPRESSIVENESS)
            ideal.add(PersonalityTrait.CREATIVITY)
        
        # 根據壓力水平調整
        if context.stress_level > 0.6:
            ideal.add(PersonalityTrait.STRESS_MANAGEMENT)
            ideal.add(PersonalityTrait.ADAPTABILITY)
        
        # 擴展為理想（包含所有下界）
        expanded_ideal = set()
        for trait in ideal:
            expanded_ideal.update(self._get_lower_bounds(trait))
        
        return expanded_ideal
    
    def evaluate_personality_consistency(self, activated_traits: Set[PersonalityTrait], 
                                       context: SituationalContext) -> float:
        """評估人格一致性 (Evaluate personality consistency)"""
        situational_ideal = self.get_situational_ideal(context)
        
        if not activated_traits or not situational_ideal:
            return 0.0
        
        intersection = activated_traits & situational_ideal
        
        # 計算一致性分數
        recall = len(intersection) / len(situational_ideal) if situational_ideal else 0.0
        precision = len(intersection) / len(activated_traits) if activated_traits else 0.0
        
        # F1分數作為一致性度量
        if recall + precision == 0:
            return 0.0
        
        consistency = 2 * (recall * precision) / (recall + precision)
        return consistency
    
    def infer_personality_profile(self, linguistic_features: Dict[str, float]) -> Dict[PersonalityTrait, float]:
        """從語言特徵推斷人格檔案 (Infer personality profile from linguistic features)"""
        profile = {}
        
        # 基於語言特徵的簡化映射
        # 實際應用中應該使用更複雜的機器學習模型
        
        for trait in PersonalityTrait:
            # 計算基礎激活分數
            activation_score = 0.0
            
            # 外向性特質映射
            if trait in [PersonalityTrait.SOCIABILITY, PersonalityTrait.EXPRESSIVENESS]:
                activation_score += linguistic_features.get('first_person_plural', 0.0) * 0.3
                activation_score += linguistic_features.get('exclamation_marks', 0.0) * 0.2
                activation_score += linguistic_features.get('question_marks', 0.0) * 0.2
            
            # 情緒穩定性特質映射
            elif trait in [PersonalityTrait.CALMNESS, PersonalityTrait.OPTIMISM]:
                activation_score += (1.0 - linguistic_features.get('negative_words', 0.0)) * 0.4
                activation_score += linguistic_features.get('positive_words', 0.0) * 0.3
            
            # 開放性特質映射
            elif trait in [PersonalityTrait.CREATIVITY, PersonalityTrait.INTELLIGENCE]:
                activation_score += linguistic_features.get('complex_words', 0.0) * 0.3
                activation_score += linguistic_features.get('metaphors', 0.0) * 0.4
                activation_score += linguistic_features.get('abstract_concepts', 0.0) * 0.3
            
            # 責任心特質映射
            elif trait in [PersonalityTrait.RESPONSIBILITY, PersonalityTrait.SYSTEMATIC_THINKING]:
                activation_score += linguistic_features.get('temporal_references', 0.0) * 0.3
                activation_score += linguistic_features.get('logical_connectors', 0.0) * 0.4
            
            # 宜人性特質映射
            elif trait in [PersonalityTrait.COOPERATION, PersonalityTrait.EMPATHY]:
                activation_score += linguistic_features.get('other_references', 0.0) * 0.3
                activation_score += linguistic_features.get('inclusive_language', 0.0) * 0.4
            
            # 歸一化到0-1範圍
            profile[trait] = min(max(activation_score, 0.0), 1.0)
        
        return profile
    
    def generate_hasse_diagram_data(self) -> Dict[str, Any]:
        """生成Hasse圖數據用於可視化 (Generate Hasse diagram data for visualization)"""
        nodes = []
        edges = []
        
        for trait in PersonalityTrait:
            node = self.nodes[trait]
            nodes.append({
                'id': trait.value,
                'label': trait.value.replace('_', ' ').title(),
                'level': node.level,
                'intensity': self.calculate_trait_intensity(trait)
            })
        
        for child, parent in self.relations:
            edges.append({
                'source': child.value,
                'target': parent.value
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'levels': max(node.level for node in self.nodes.values()) + 1
        }


class PersonalityLatticeEvaluator:
    """人格格評估器 (Personality Lattice Evaluator)"""
    
    def __init__(self):
        self.lattice = PersonalityLattice()
    
    def evaluate_expression_personality(self, expression: str, context: SituationalContext) -> Dict[str, Any]:
        """評估表達式的人格特徵 (Evaluate personality characteristics of expression)"""
        
        # 1. 提取語言特徵（簡化版本）
        linguistic_features = self._extract_linguistic_features(expression)
        
        # 2. 推斷人格檔案
        personality_profile = self.lattice.infer_personality_profile(linguistic_features)
        
        # 3. 找到激活的特質
        activated_traits = {trait for trait, score in personality_profile.items() if score > 0.3}
        
        # 4. 計算一致性
        consistency = self.lattice.evaluate_personality_consistency(activated_traits, context)
        
        # 5. 找到主導特質組合
        dominant_traits = [trait for trait, score in personality_profile.items() if score > 0.6]
        
        if len(dominant_traits) >= 2:
            # 計算特質組合的並
            combined_trait = self.lattice.join(dominant_traits[0], dominant_traits[1])
            for trait in dominant_traits[2:]:
                combined_trait = self.lattice.join(combined_trait, trait)
        else:
            combined_trait = dominant_traits[0] if dominant_traits else PersonalityTrait.CORE_PERSONALITY
        
        return {
            'linguistic_features': linguistic_features,
            'personality_profile': {trait.value: score for trait, score in personality_profile.items()},
            'activated_traits': [trait.value for trait in activated_traits],
            'dominant_traits': [trait.value for trait in dominant_traits],
            'combined_personality': combined_trait.value,
            'situational_consistency': consistency,
            'overall_personality_score': np.mean(list(personality_profile.values())),
            'context_adaptation': self._calculate_context_adaptation(personality_profile, context)
        }
    
    def _extract_linguistic_features(self, expression: str) -> Dict[str, float]:
        """提取語言特徵（簡化實現）(Extract linguistic features - simplified implementation)"""
        
        # 這是一個簡化的特徵提取器
        # 實際應用中應該使用更sophisticated的NLP技術
        
        words = expression.lower().split()
        total_words = len(words) if words else 1
        
        features = {
            'first_person_plural': len([w for w in words if w in ['we', 'us', 'our', 'ours', '我們', '咱們']]) / total_words,
            'exclamation_marks': expression.count('!') / len(expression) if expression else 0.0,
            'question_marks': expression.count('?') / len(expression) if expression else 0.0,
            'negative_words': len([w for w in words if w in ['not', 'no', 'never', 'bad', 'terrible', '不', '沒', '糟']]) / total_words,
            'positive_words': len([w for w in words if w in ['good', 'great', 'excellent', 'wonderful', '好', '棒', '優秀']]) / total_words,
            'complex_words': len([w for w in words if len(w) > 6]) / total_words,
            'metaphors': len([w for w in words if w in ['like', 'as', 'metaphor', '如', '似', '像']]) / total_words,
            'abstract_concepts': len([w for w in words if w in ['concept', 'idea', 'theory', 'philosophy', '概念', '理論', '哲學']]) / total_words,
            'temporal_references': len([w for w in words if w in ['when', 'then', 'time', 'schedule', '時間', '時候', '排程']]) / total_words,
            'logical_connectors': len([w for w in words if w in ['because', 'therefore', 'however', 'thus', '因為', '所以', '然而']]) / total_words,
            'other_references': len([w for w in words if w in ['you', 'they', 'others', 'people', '你', '他們', '別人']]) / total_words,
            'inclusive_language': len([w for w in words if w in ['together', 'share', 'collaborate', '一起', '分享', '合作']]) / total_words,
        }
        
        return features
    
    def _calculate_context_adaptation(self, personality_profile: Dict[PersonalityTrait, float], 
                                    context: SituationalContext) -> float:
        """計算情境適應度 (Calculate context adaptation)"""
        
        situational_ideal = self.lattice.get_situational_ideal(context)
        
        if not situational_ideal:
            return 0.5  # 中性適應度
        
        # 計算人格檔案與情境理想的重疊度
        relevant_scores = [personality_profile[trait] for trait in situational_ideal if trait in personality_profile]
        
        if not relevant_scores:
            return 0.0
        
        # 返回相關特質的平均激活分數
        return np.mean(relevant_scores)


# 使用示例和測試函數
def demonstrate_personality_lattice():
    """演示人格格的使用 (Demonstrate personality lattice usage)"""
    
    print("=== 人格格論模型演示 (Personality Lattice Model Demonstration) ===\n")
    
    # 創建評估器
    evaluator = PersonalityLatticeEvaluator()
    lattice = evaluator.lattice
    
    # 演示格運算
    print("1. 格運算演示 (Lattice Operations Demonstration):")
    trait_a = PersonalityTrait.FRIENDLINESS
    trait_b = PersonalityTrait.COMPETITIVENESS
    
    join_result = lattice.join(trait_a, trait_b)
    meet_result = lattice.meet(trait_a, trait_b)
    
    print(f"   {trait_a.value} ∨ {trait_b.value} = {join_result.value}")
    print(f"   {trait_a.value} ∧ {trait_b.value} = {meet_result.value}")
    
    # 計算特質強度
    print(f"\n2. 特質強度計算 (Trait Intensity Calculation):")
    for trait in [PersonalityTrait.SOCIABILITY, PersonalityTrait.SOCIAL_LEADERSHIP, PersonalityTrait.CORE_PERSONALITY]:
        intensity = lattice.calculate_trait_intensity(trait)
        print(f"   {trait.value}: {intensity:.3f}")
    
    # 情境適應分析
    print(f"\n3. 情境適應分析 (Situational Adaptation Analysis):")
    contexts = [
        SituationalContext(situation_type="professional", formality_level=0.8),
        SituationalContext(situation_type="social", formality_level=0.3),
        SituationalContext(situation_type="academic", formality_level=0.6),
    ]
    
    for context in contexts:
        ideal = lattice.get_situational_ideal(context)
        print(f"   {context.situation_type} (正式度: {context.formality_level}):")
        print(f"     理想特質: {[t.value for t in list(ideal)[:5]]}...")
    
    # 表達式評估
    print(f"\n4. 表達式人格評估 (Expression Personality Evaluation):")
    expressions = [
        ("我們需要一起合作完成這個專案", SituationalContext(situation_type="professional")),
        ("哇！這個想法真的很棒，充滿創意！", SituationalContext(situation_type="social")),
        ("根據理論分析，我認為這個方法更系統化", SituationalContext(situation_type="academic")),
    ]
    
    for expr, ctx in expressions:
        result = evaluator.evaluate_expression_personality(expr, ctx)
        print(f"\n   表達式: \"{expr}\"")
        print(f"   主導特質: {result['dominant_traits']}")
        print(f"   組合人格: {result['combined_personality']}")
        print(f"   情境一致性: {result['situational_consistency']:.3f}")
        print(f"   整體人格分數: {result['overall_personality_score']:.3f}")


if __name__ == "__main__":
    demonstrate_personality_lattice()