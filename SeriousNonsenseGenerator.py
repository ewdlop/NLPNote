"""
一本正經地「胡說八道」生成器 (Serious Nonsense Generator)

This module implements a sophisticated generator that creates academic-sounding
text that appears meaningful but is actually nonsensical - embodying the concept
of "一本正經地「胡說八道」" (seriously talking nonsense).

The generator combines real academic terminology and structures with meaningless
connections to create plausible-sounding but ultimately empty content.
"""

import random
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class AcademicStyle(Enum):
    """Academic writing styles"""
    SCIENTIFIC = "scientific"
    PHILOSOPHICAL = "philosophical"
    TECHNICAL = "technical"
    THEORETICAL = "theoretical"
    LINGUISTIC = "linguistic"


class Language(Enum):
    """Supported languages"""
    ENGLISH = "en"
    CHINESE = "zh"


@dataclass
class GenerationContext:
    """Context for generating nonsense"""
    style: AcademicStyle = AcademicStyle.SCIENTIFIC
    language: Language = Language.ENGLISH
    topic: str = "general"
    complexity: float = 0.7  # 0.0 to 1.0
    length: str = "medium"  # short, medium, long


class SeriousNonsenseGenerator:
    """
    Generator for academic-sounding nonsense text
    
    This class creates sophisticated-sounding text that uses proper academic
    vocabulary and structure but combines concepts in meaningless ways.
    """
    
    def __init__(self):
        """Initialize the nonsense generator with vocabulary banks"""
        
        # English academic vocabulary banks
        self.en_scientific_terms = [
            "quantum entanglement", "neural plasticity", "cognitive dissonance",
            "thermodynamic equilibrium", "epistemological framework", "dialectical synthesis",
            "morphological analysis", "stochastic processes", "phenomenological reduction",
            "hermeneutic interpretation", "paradigmatic shift", "algorithmic complexity",
            "metacognitive awareness", "differential equations", "probabilistic inference",
            "syntactic structures", "semantic networks", "pragmatic implications",
            "dimensional analysis", "fractal geometry", "chaos theory",
            "information entropy", "computational linguistics", "neurochemical pathways"
        ]
        
        self.en_academic_connectors = [
            "Furthermore", "However", "Nevertheless", "Consequently", "In contrast",
            "Specifically", "Particularly", "Moreover", "Additionally", "Subsequently",
            "Therefore", "Thus", "Hence", "Accordingly", "In other words",
            "That is to say", "More precisely", "In essence", "Fundamentally",
            "Interestingly", "Notably", "Remarkably", "Significantly"
        ]
        
        self.en_abstract_concepts = [
            "manifestation", "paradigm", "framework", "construct", "mechanism",
            "phenomenon", "principle", "methodology", "approach", "perspective",
            "dimension", "aspect", "component", "element", "factor",
            "variable", "parameter", "criterion", "indicator", "measure",
            "conceptualization", "formulation", "articulation", "representation"
        ]
        
        self.en_academic_verbs = [
            "demonstrates", "illustrates", "elucidates", "encompasses", "manifests",
            "embodies", "facilitates", "necessitates", "presupposes", "transcends",
            "encompasses", "delineates", "substantiates", "corroborates", "exemplifies",
            "conceptualizes", "operationalizes", "synthesizes", "contextualizes"
        ]
        
        # Chinese academic vocabulary banks
        self.zh_academic_terms = [
            "認知語言學", "結構主義", "後現代主義", "解構主義", "現象學",
            "詮釋學", "符號學", "語用學", "語義學", "句法學",
            "形態學", "音韻學", "語音學", "語言哲學", "語言心理學",
            "神經語言學", "社會語言學", "歷史語言學", "比較語言學",
            "計算語言學", "語料庫語言學", "心理語言學", "神經科學",
            "認知科學", "人工智能", "機器學習", "深度學習"
        ]
        
        self.zh_connectors = [
            "然而", "因此", "從而", "進而", "換言之", "也就是說",
            "具體而言", "總而言之", "此外", "另外", "同時",
            "與此同時", "相對而言", "就此而言", "在此基礎上",
            "基於此", "由此可見", "顯而易見", "毫無疑問"
        ]
        
        self.zh_abstract_concepts = [
            "範式", "框架", "體系", "機制", "原理", "理論", "概念",
            "方法", "途徑", "視角", "層面", "維度", "要素", "因素",
            "變量", "參數", "標準", "指標", "測度", "表徵", "建構"
        ]
        
        self.zh_academic_verbs = [
            "體現", "反映", "揭示", "闡明", "說明", "表明", "顯示",
            "證明", "論證", "驗證", "確認", "支持", "強化", "深化",
            "拓展", "延伸", "涵蓋", "包含", "融合", "整合", "統一"
        ]
        
    def generate_nonsense(self, context: GenerationContext = None) -> str:
        """
        Generate academic-sounding nonsense text
        
        Args:
            context: Generation context specifying style, language, etc.
            
        Returns:
            Generated nonsense text
        """
        if context is None:
            context = GenerationContext()
            
        if context.language == Language.ENGLISH:
            return self._generate_english_nonsense(context)
        else:
            return self._generate_chinese_nonsense(context)
    
    def _generate_english_nonsense(self, context: GenerationContext) -> str:
        """Generate English academic nonsense"""
        
        # Choose vocabulary based on style
        terms = self.en_scientific_terms.copy()
        if context.style == AcademicStyle.PHILOSOPHICAL:
            terms.extend(["ontological", "epistemological", "metaphysical", "teleological"])
        elif context.style == AcademicStyle.TECHNICAL:
            terms.extend(["algorithmic", "computational", "systematic", "methodological"])
        
        # Generate paragraphs based on length
        paragraphs = []
        num_paragraphs = {"short": 1, "medium": 2, "long": 3}.get(context.length, 2)
        
        for _ in range(num_paragraphs):
            paragraph = self._generate_english_paragraph(terms, context)
            paragraphs.append(paragraph)
        
        return "\n\n".join(paragraphs)
    
    def _generate_english_paragraph(self, terms: List[str], context: GenerationContext) -> str:
        """Generate a single English paragraph"""
        sentences = []
        num_sentences = int(3 + context.complexity * 4)  # 3-7 sentences
        
        for i in range(num_sentences):
            # Choose sentence structure based on complexity
            if context.complexity > 0.7 and random.random() > 0.5:
                sentence = self._generate_complex_english_sentence(terms)
            else:
                sentence = self._generate_simple_english_sentence(terms)
            
            # Add connector for sentences after the first
            if i > 0 and random.random() > 0.3:
                connector = random.choice(self.en_academic_connectors)
                sentence = f"{connector}, {sentence.lower()}"
            
            sentences.append(sentence)
        
        return " ".join(sentences)
    
    def _generate_simple_english_sentence(self, terms: List[str]) -> str:
        """Generate a simple academic sentence"""
        templates = [
            "The {concept} of {term1} {verb} the {dimension} of {term2}.",
            "This {framework} {verb} a novel {approach} to {term1}.",
            "The {analysis} {verb} that {term1} is {adjective} to {term2}.",
            "Such {methodology} {verb} the {relationship} between {term1} and {term2}.",
            "The {investigation} {verb} significant {implications} for {term1}."
        ]
        
        template = random.choice(templates)
        
        # Fill in the template
        filled = template.format(
            concept=random.choice(self.en_abstract_concepts),
            term1=random.choice(terms),
            term2=random.choice(terms),
            verb=random.choice(self.en_academic_verbs),
            dimension=random.choice(["aspects", "properties", "characteristics", "features"]),
            framework=random.choice(["framework", "paradigm", "model", "theory"]),
            approach=random.choice(["approach", "methodology", "strategy", "technique"]),
            analysis=random.choice(["analysis", "investigation", "examination", "study"]),
            adjective=random.choice(["fundamental", "crucial", "essential", "integral"]),
            methodology=random.choice(["methodology", "approach", "framework", "system"]),
            relationship=random.choice(["relationship", "connection", "correlation", "interaction"]),
            investigation=random.choice(["investigation", "research", "study", "analysis"]),
            implications=random.choice(["implications", "consequences", "ramifications", "effects"])
        )
        
        return filled
    
    def _generate_complex_english_sentence(self, terms: List[str]) -> str:
        """Generate a complex academic sentence with multiple clauses"""
        templates = [
            "While the {concept1} of {term1} {verb1} the {framework} through which {term2} can be {analyzed}, it is essential to consider how such {methodology} {verb2} the broader {implications} of {term3} within the {context} of {term4}.",
            "The {investigation} of {term1} not only {verb1} the {theoretical} {foundations} of {term2}, but also {verb2} new {perspectives} on the {relationship} between {term3} and the {paradigmatic} {shifts} in {term4}.",
            "Through a comprehensive {analysis} of {term1}, we can observe that the {manifestation} of {term2} {verb1} both the {structural} and {functional} {dimensions} of {term3}, thereby {verb2} our understanding of {term4}."
        ]
        
        template = random.choice(templates)
        
        filled = template.format(
            concept1=random.choice(self.en_abstract_concepts),
            term1=random.choice(terms),
            term2=random.choice(terms),
            term3=random.choice(terms),
            term4=random.choice(terms),
            verb1=random.choice(self.en_academic_verbs),
            verb2=random.choice(["enhances", "transforms", "challenges", "redefines"]),
            framework=random.choice(["framework", "paradigm", "model", "structure"]),
            analyzed=random.choice(["understood", "conceptualized", "interpreted", "analyzed"]),
            methodology=random.choice(["methodology", "approach", "framework", "perspective"]),
            implications=random.choice(["implications", "consequences", "significance", "meaning"]),
            context=random.choice(["context", "domain", "sphere", "realm"]),
            investigation=random.choice(["examination", "exploration", "investigation", "analysis"]),
            theoretical=random.choice(["theoretical", "conceptual", "methodological", "epistemological"]),
            foundations=random.choice(["foundations", "underpinnings", "basis", "groundwork"]),
            perspectives=random.choice(["perspectives", "insights", "understandings", "viewpoints"]),
            relationship=random.choice(["relationship", "interaction", "connection", "correlation"]),
            paradigmatic=random.choice(["paradigmatic", "fundamental", "structural", "conceptual"]),
            shifts=random.choice(["shifts", "changes", "transformations", "developments"]),
            analysis=random.choice(["analysis", "examination", "investigation", "study"]),
            manifestation=random.choice(["manifestation", "expression", "representation", "embodiment"]),
            structural=random.choice(["structural", "organizational", "systematic", "architectural"]),
            functional=random.choice(["functional", "operational", "practical", "applied"]),
            dimensions=random.choice(["dimensions", "aspects", "components", "elements"])
        )
        
        return filled
    
    def _generate_chinese_nonsense(self, context: GenerationContext) -> str:
        """Generate Chinese academic nonsense"""
        paragraphs = []
        num_paragraphs = {"short": 1, "medium": 2, "long": 3}.get(context.length, 2)
        
        for _ in range(num_paragraphs):
            paragraph = self._generate_chinese_paragraph(context)
            paragraphs.append(paragraph)
        
        return "\n\n".join(paragraphs)
    
    def _generate_chinese_paragraph(self, context: GenerationContext) -> str:
        """Generate a single Chinese paragraph"""
        sentences = []
        num_sentences = int(3 + context.complexity * 4)
        
        for i in range(num_sentences):
            if context.complexity > 0.7 and random.random() > 0.5:
                sentence = self._generate_complex_chinese_sentence()
            else:
                sentence = self._generate_simple_chinese_sentence()
            
            # Add connector for sentences after the first
            if i > 0 and random.random() > 0.3:
                connector = random.choice(self.zh_connectors)
                sentence = f"{connector}，{sentence}"
            
            sentences.append(sentence)
        
        return "".join(sentences)
    
    def _generate_simple_chinese_sentence(self) -> str:
        """Generate a simple Chinese academic sentence"""
        templates = [
            "{term1}的{concept1}{verb1}了{term2}的{concept2}。",
            "這種{framework}{verb1}了{term1}的{methodology}。",
            "研究{verb1}{term1}與{term2}之間的{relationship}具有重要意義。",
            "通過{analysis}，我們發現{term1}{verb1}了{term2}的{dimension}。",
            "該{theory}{verb1}了{term1}在{term2}中的{role}。"
        ]
        
        template = random.choice(templates)
        
        filled = template.format(
            term1=random.choice(self.zh_academic_terms),
            term2=random.choice(self.zh_academic_terms),
            concept1=random.choice(self.zh_abstract_concepts),
            concept2=random.choice(self.zh_abstract_concepts),
            verb1=random.choice(self.zh_academic_verbs),
            framework=random.choice(["框架", "範式", "理論", "模型"]),
            methodology=random.choice(["方法", "途徑", "策略", "技術"]),
            relationship=random.choice(["關係", "聯繫", "相關性", "互動"]),
            analysis=random.choice(["分析", "研究", "探討", "考察"]),
            dimension=random.choice(["層面", "維度", "特性", "屬性"]),
            theory=random.choice(["理論", "學說", "觀點", "假設"]),
            role=random.choice(["作用", "功能", "地位", "意義"])
        )
        
        return filled
    
    def _generate_complex_chinese_sentence(self) -> str:
        """Generate a complex Chinese academic sentence"""
        templates = [
            "基於{term1}的{theoretical}基礎，本研究通過{methodology}來{analyze}{term2}與{term3}之間的{relationship}，從而{reveal}其在{term4}中的{significance}。",
            "從{perspective}的角度來看，{term1}不僅{verb1}了{term2}的{structure}，而且還{verb2}了我們對{term3}的{understanding}。",
            "通過深入{analysis}{term1}的{characteristics}，我們可以發現其{manifestation}在{term2}中具有{particular}的{meaning}，這對理解{term3}具有重要的{implications}。"
        ]
        
        template = random.choice(templates)
        
        filled = template.format(
            term1=random.choice(self.zh_academic_terms),
            term2=random.choice(self.zh_academic_terms),
            term3=random.choice(self.zh_academic_terms),
            term4=random.choice(self.zh_academic_terms),
            theoretical=random.choice(["理論", "概念", "方法", "認知"]),
            methodology=random.choice(["定性分析", "定量研究", "比較方法", "系統分析"]),
            analyze=random.choice(["探討", "分析", "研究", "考察"]),
            relationship=random.choice(["關係", "聯繫", "相互作用", "相關性"]),
            reveal=random.choice(["揭示", "闡明", "說明", "證明"]),
            significance=random.choice(["重要性", "意義", "作用", "價值"]),
            perspective=random.choice(["理論", "方法論", "認知", "哲學"]),
            verb1=random.choice(["改變", "影響", "塑造", "重構"]),
            verb2=random.choice(["豐富", "深化", "拓展", "提升"]),
            structure=random.choice(["結構", "體系", "框架", "模式"]),
            understanding=random.choice(["理解", "認識", "把握", "詮釋"]),
            analysis=random.choice(["分析", "研究", "探討", "考察"]),
            characteristics=random.choice(["特徵", "特性", "屬性", "本質"]),
            manifestation=random.choice(["表現", "體現", "反映", "展現"]),
            particular=random.choice(["特殊", "獨特", "特定", "特有"]),
            meaning=random.choice(["意義", "含義", "意涵", "價值"]),
            implications=random.choice(["啟示", "意義", "價值", "貢獻"])
        )
        
        return filled
    
    def generate_academic_title(self, context: GenerationContext = None) -> str:
        """Generate an academic-sounding title"""
        if context is None:
            context = GenerationContext()
        
        if context.language == Language.ENGLISH:
            return self._generate_english_title(context)
        else:
            return self._generate_chinese_title(context)
    
    def _generate_english_title(self, context: GenerationContext) -> str:
        """Generate English academic title"""
        templates = [
            "A {adjective} Analysis of {term1} in the Context of {term2}",
            "Towards a {adjective} Understanding of {term1}: {framework} and {implications}",
            "The {relationship} between {term1} and {term2}: A {methodological} Approach",
            "{term1} and {term2}: {perspective} Perspectives on {term3}",
            "Rethinking {term1}: {implications} for {term2} Theory"
        ]
        
        template = random.choice(templates)
        
        filled = template.format(
            adjective=random.choice(["Comprehensive", "Systematic", "Critical", "Theoretical", "Empirical"]),
            term1=random.choice(self.en_scientific_terms),
            term2=random.choice(self.en_scientific_terms),
            term3=random.choice(self.en_scientific_terms),
            framework=random.choice(["Frameworks", "Methodologies", "Approaches", "Perspectives"]),
            implications=random.choice(["Implications", "Applications", "Consequences", "Ramifications"]),
            relationship=random.choice(["Relationship", "Connection", "Interaction", "Correlation"]),
            methodological=random.choice(["Methodological", "Theoretical", "Conceptual", "Analytical"]),
            perspective=random.choice(["Theoretical", "Methodological", "Conceptual", "Analytical"])
        )
        
        return filled
    
    def _generate_chinese_title(self, context: GenerationContext) -> str:
        """Generate Chinese academic title"""
        templates = [
            "{term1}與{term2}關係的{adjective}研究",
            "基於{framework}的{term1}{analysis}",
            "{term1}在{term2}中的{role}：理論與實踐",
            "論{term1}的{characteristics}及其{implications}",
            "{term1}研究的{perspective}思考"
        ]
        
        template = random.choice(templates)
        
        filled = template.format(
            term1=random.choice(self.zh_academic_terms),
            term2=random.choice(self.zh_academic_terms),
            adjective=random.choice(["系統性", "綜合性", "批判性", "理論性", "實證性"]),
            framework=random.choice(["認知框架", "理論體系", "分析模式", "研究範式"]),
            analysis=random.choice(["分析", "研究", "探討", "考察"]),
            role=random.choice(["作用", "功能", "地位", "意義"]),
            characteristics=random.choice(["特徵", "特性", "本質", "規律"]),
            implications=random.choice(["意義", "啟示", "影響", "作用"]),
            perspective=random.choice(["方法論", "認識論", "理論", "批判性"])
        )
        
        return filled


def main():
    """Demonstration of the Serious Nonsense Generator"""
    generator = SeriousNonsenseGenerator()
    
    print("=== 一本正經地「胡說八道」生成器演示 ===\n")
    print("=== Serious Nonsense Generator Demo ===\n")
    
    # English examples
    print("【English Academic Nonsense】\n")
    
    en_context = GenerationContext(
        style=AcademicStyle.SCIENTIFIC,
        language=Language.ENGLISH,
        complexity=0.8,
        length="medium"
    )
    
    title = generator.generate_academic_title(en_context)
    print(f"Title: {title}\n")
    
    content = generator.generate_nonsense(en_context)
    print(f"{content}\n")
    
    print("-" * 60)
    
    # Chinese examples
    print("\n【中文學術胡話】\n")
    
    zh_context = GenerationContext(
        style=AcademicStyle.THEORETICAL,
        language=Language.CHINESE,
        complexity=0.7,
        length="medium"
    )
    
    zh_title = generator.generate_academic_title(zh_context)
    print(f"標題：{zh_title}\n")
    
    zh_content = generator.generate_nonsense(zh_context)
    print(f"{zh_content}\n")
    
    print("-" * 60)
    
    # Different styles
    print("\n【Different Styles】\n")
    
    styles = [AcademicStyle.PHILOSOPHICAL, AcademicStyle.TECHNICAL, AcademicStyle.LINGUISTIC]
    
    for style in styles:
        context = GenerationContext(
            style=style,
            language=Language.ENGLISH,
            complexity=0.6,
            length="short"
        )
        
        title = generator.generate_academic_title(context)
        content = generator.generate_nonsense(context)
        
        print(f"Style: {style.value.upper()}")
        print(f"Title: {title}")
        print(f"Content: {content[:200]}...")
        print()


if __name__ == "__main__":
    main()