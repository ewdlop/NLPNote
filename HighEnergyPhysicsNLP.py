"""
High Energy Physics NLP Module
高能物理自然語言處理模組

This module implements natural language processing techniques specifically designed
for high energy physics texts, combining particle physics concepts with NLP methods.

高能物理與自然語言處理的結合，專門處理粒子物理學文本的NLP技術。
"""

import re
import math
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json

# Try to import existing modules for integration
try:
    from HumanExpressionEvaluator import HumanExpressionEvaluator, ExpressionContext
    EXPRESSION_EVALUATOR_AVAILABLE = True
except ImportError:
    EXPRESSION_EVALUATOR_AVAILABLE = False

try:
    from AStarNLP import AStarNLP
    ASTAR_NLP_AVAILABLE = True
except ImportError:
    ASTAR_NLP_AVAILABLE = False


class ParticleType(Enum):
    """粒子類型 (Particle Types)"""
    LEPTON = "lepton"
    QUARK = "quark"
    BOSON = "boson"
    HADRON = "hadron"
    MESON = "meson"
    BARYON = "baryon"


class PhysicsConceptType(Enum):
    """物理概念類型 (Physics Concept Types)"""
    PARTICLE = "particle"
    FORCE = "force"
    FIELD = "field"
    INTERACTION = "interaction"
    CONSERVATION = "conservation"
    SYMMETRY = "symmetry"
    DECAY = "decay"
    SCATTERING = "scattering"


@dataclass
class PhysicsEntity:
    """物理實體 (Physics Entity)"""
    name: str
    entity_type: PhysicsConceptType
    particle_type: Optional[ParticleType] = None
    properties: Dict[str, Any] = None
    confidence: float = 1.0
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class PhysicsEquation:
    """物理方程 (Physics Equation)"""
    latex: str
    description: str
    variables: List[str]
    constants: List[str]
    domain: str  # e.g., "quantum_mechanics", "relativity", "particle_physics"
    confidence: float = 1.0


class HighEnergyPhysicsNLP:
    """
    High Energy Physics Natural Language Processing Class
    高能物理自然語言處理類
    
    Combines particle physics concepts with NLP techniques for processing
    scientific texts, equations, and theoretical descriptions.
    """
    
    def __init__(self):
        """Initialize the High Energy Physics NLP system"""
        self.physics_vocabulary = self._build_physics_vocabulary()
        self.equation_patterns = self._build_equation_patterns()
        self.particle_signatures = self._build_particle_signatures()
        
        # Initialize integration with existing modules
        if EXPRESSION_EVALUATOR_AVAILABLE:
            self.expression_evaluator = HumanExpressionEvaluator()
        if ASTAR_NLP_AVAILABLE:
            self.astar_nlp = AStarNLP()
    
    def _build_physics_vocabulary(self) -> Dict[str, PhysicsConceptType]:
        """Build a comprehensive physics vocabulary"""
        return {
            # Particles
            "electron": PhysicsConceptType.PARTICLE,
            "muon": PhysicsConceptType.PARTICLE,
            "tau": PhysicsConceptType.PARTICLE,
            "neutrino": PhysicsConceptType.PARTICLE,
            "quark": PhysicsConceptType.PARTICLE,
            "photon": PhysicsConceptType.PARTICLE,
            "gluon": PhysicsConceptType.PARTICLE,
            "higgs": PhysicsConceptType.PARTICLE,
            "boson": PhysicsConceptType.PARTICLE,
            "proton": PhysicsConceptType.PARTICLE,
            "neutron": PhysicsConceptType.PARTICLE,
            "positron": PhysicsConceptType.PARTICLE,
            "antiproton": PhysicsConceptType.PARTICLE,
            
            # Forces and Interactions
            "electromagnetic": PhysicsConceptType.FORCE,
            "weak": PhysicsConceptType.FORCE,
            "strong": PhysicsConceptType.FORCE,
            "gravitational": PhysicsConceptType.FORCE,
            "interaction": PhysicsConceptType.INTERACTION,
            "scattering": PhysicsConceptType.SCATTERING,
            "decay": PhysicsConceptType.DECAY,
            
            # Fields
            "field": PhysicsConceptType.FIELD,
            "gauge": PhysicsConceptType.FIELD,
            "scalar": PhysicsConceptType.FIELD,
            "vector": PhysicsConceptType.FIELD,
            "tensor": PhysicsConceptType.FIELD,
            
            # Conservation laws
            "conservation": PhysicsConceptType.CONSERVATION,
            "energy": PhysicsConceptType.CONSERVATION,
            "momentum": PhysicsConceptType.CONSERVATION,
            "charge": PhysicsConceptType.CONSERVATION,
            "baryon": PhysicsConceptType.CONSERVATION,
            "lepton": PhysicsConceptType.CONSERVATION,
            
            # Symmetries
            "symmetry": PhysicsConceptType.SYMMETRY,
            "parity": PhysicsConceptType.SYMMETRY,
            "chirality": PhysicsConceptType.SYMMETRY,
            "supersymmetry": PhysicsConceptType.SYMMETRY,
            
            # Chinese physics terms
            "粒子": PhysicsConceptType.PARTICLE,
            "電子": PhysicsConceptType.PARTICLE,
            "質子": PhysicsConceptType.PARTICLE,
            "中子": PhysicsConceptType.PARTICLE,
            "光子": PhysicsConceptType.PARTICLE,
            "玻色子": PhysicsConceptType.PARTICLE,
            "費米子": PhysicsConceptType.PARTICLE,
            "夸克": PhysicsConceptType.PARTICLE,
            "中微子": PhysicsConceptType.PARTICLE,
            "能量": PhysicsConceptType.CONSERVATION,
            "動量": PhysicsConceptType.CONSERVATION,
            "守恆": PhysicsConceptType.CONSERVATION,
            "場": PhysicsConceptType.FIELD,
            "電磁場": PhysicsConceptType.FIELD,
            "電磁力": PhysicsConceptType.FORCE,
            "弱力": PhysicsConceptType.FORCE,
            "強力": PhysicsConceptType.FORCE,
            "重力": PhysicsConceptType.FORCE,
            "相互作用": PhysicsConceptType.INTERACTION,
            "對稱性": PhysicsConceptType.SYMMETRY,
            "衰變": PhysicsConceptType.DECAY,
            "散射": PhysicsConceptType.SCATTERING,
        }
    
    def _build_equation_patterns(self) -> List[Dict[str, str]]:
        """Build patterns for recognizing physics equations"""
        return [
            {
                "pattern": r"E\s*=\s*mc\^?2|E\s*=\s*mc²",
                "name": "Einstein mass-energy relation",
                "domain": "special_relativity"
            },
            {
                "pattern": r"[FfΓγ]\s*=\s*[Gg]m[₁₂]*m[₁₂]*/r\^?2",
                "name": "Newton's law of gravitation",
                "domain": "classical_mechanics"
            },
            {
                "pattern": r"ψ|Ψ|\|ψ⟩|\|Ψ⟩",
                "name": "Quantum state vector",
                "domain": "quantum_mechanics"
            },
            {
                "pattern": r"Ĥ|H\^\s*ψ|Ĥψ",
                "name": "Hamiltonian operator",
                "domain": "quantum_mechanics"
            },
            {
                "pattern": r"∇\s*×\s*[BEH]|curl\s*[BEH]",
                "name": "Maxwell equation (curl)",
                "domain": "electromagnetism"
            },
            {
                "pattern": r"α\s*≈\s*1/137|α\s*=\s*e\^?2",
                "name": "Fine structure constant",
                "domain": "quantum_electrodynamics"
            },
            {
                "pattern": r"g\s*=\s*8\.314|R\s*=\s*8\.314",
                "name": "Universal gas constant",
                "domain": "thermodynamics"
            }
        ]
    
    def _build_particle_signatures(self) -> Dict[str, Dict[str, Any]]:
        """Build particle signature database"""
        return {
            "electron": {
                "symbol": "e⁻",
                "charge": -1,
                "spin": 0.5,
                "mass_mev": 0.511,
                "type": ParticleType.LEPTON
            },
            "muon": {
                "symbol": "μ⁻",
                "charge": -1,
                "spin": 0.5,
                "mass_mev": 105.7,
                "type": ParticleType.LEPTON
            },
            "proton": {
                "symbol": "p⁺",
                "charge": 1,
                "spin": 0.5,
                "mass_mev": 938.3,
                "type": ParticleType.BARYON
            },
            "neutron": {
                "symbol": "n⁰",
                "charge": 0,
                "spin": 0.5,
                "mass_mev": 939.6,
                "type": ParticleType.BARYON
            },
            "photon": {
                "symbol": "γ",
                "charge": 0,
                "spin": 1,
                "mass_mev": 0,
                "type": ParticleType.BOSON
            },
            "higgs": {
                "symbol": "H",
                "charge": 0,
                "spin": 0,
                "mass_mev": 125000,
                "type": ParticleType.BOSON
            }
        }
    
    def extract_physics_entities(self, text: str) -> List[PhysicsEntity]:
        """
        Extract physics entities from text
        從文本中提取物理實體
        """
        entities = []
        
        # Handle both English (word boundaries) and Chinese (character-based) text
        # First try English word tokenization
        english_words = re.findall(r'\b\w+\b', text.lower())
        
        # Then try Chinese character sequences (for multi-character terms)
        chinese_terms = []
        for term in self.physics_vocabulary.keys():
            if any('\u4e00' <= char <= '\u9fff' for char in term):  # Chinese characters
                if term in text:
                    chinese_terms.append(term)
        
        # Combine all found terms
        all_terms = english_words + chinese_terms
        
        for term in all_terms:
            if term in self.physics_vocabulary:
                entity = PhysicsEntity(
                    name=term,
                    entity_type=self.physics_vocabulary[term],
                    confidence=0.9
                )
                
                # Add particle-specific information if available
                if term in self.particle_signatures:
                    signature = self.particle_signatures[term]
                    entity.particle_type = signature.get("type")
                    entity.properties = {
                        "symbol": signature.get("symbol"),
                        "charge": signature.get("charge"),
                        "spin": signature.get("spin"),
                        "mass_mev": signature.get("mass_mev")
                    }
                
                entities.append(entity)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            entity_key = (entity.name, entity.entity_type)
            if entity_key not in seen:
                seen.add(entity_key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def detect_physics_equations(self, text: str) -> List[PhysicsEquation]:
        """
        Detect and classify physics equations in text
        檢測並分類文本中的物理方程
        """
        equations = []
        
        for eq_pattern in self.equation_patterns:
            matches = re.finditer(eq_pattern["pattern"], text, re.IGNORECASE)
            for match in matches:
                equation = PhysicsEquation(
                    latex=match.group(),
                    description=eq_pattern["name"],
                    variables=self._extract_variables(match.group()),
                    constants=self._extract_constants(match.group()),
                    domain=eq_pattern["domain"],
                    confidence=0.8
                )
                equations.append(equation)
        
        return equations
    
    def _extract_variables(self, equation: str) -> List[str]:
        """Extract variables from equation"""
        # Simple heuristic - single letters that aren't known constants
        known_constants = {'c', 'e', 'π', 'h', 'ħ', 'α', 'G'}
        variables = re.findall(r'\b[a-zA-Z]\b', equation)
        return [v for v in variables if v not in known_constants]
    
    def _extract_constants(self, equation: str) -> List[str]:
        """Extract physical constants from equation"""
        known_constants = {'c', 'e', 'π', 'h', 'ħ', 'α', 'G'}
        constants = re.findall(r'\b[a-zA-Z]\b', equation)
        return [c for c in constants if c in known_constants]
    
    def calculate_physics_concept_similarity(self, concept1: str, concept2: str) -> float:
        """
        Calculate semantic similarity between physics concepts
        計算物理概念之間的語義相似度
        """
        # Use concept types and properties for similarity
        if concept1 in self.physics_vocabulary and concept2 in self.physics_vocabulary:
            type1 = self.physics_vocabulary[concept1]
            type2 = self.physics_vocabulary[concept2]
            
            if type1 == type2:
                base_similarity = 0.7
            else:
                base_similarity = 0.3
            
            # Add particle-specific similarity if both are particles
            if (concept1 in self.particle_signatures and 
                concept2 in self.particle_signatures):
                
                sig1 = self.particle_signatures[concept1]
                sig2 = self.particle_signatures[concept2]
                
                # Similar particle types get higher similarity
                if sig1.get("type") == sig2.get("type"):
                    base_similarity += 0.2
                
                # Similar charges get slight boost
                if sig1.get("charge") == sig2.get("charge"):
                    base_similarity += 0.1
            
            return min(base_similarity, 1.0)
        
        # Use string similarity as fallback
        return self._string_similarity(concept1, concept2)
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity using simple metrics"""
        if s1 == s2:
            return 1.0
        
        # Simple Levenshtein-like similarity
        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0
        
        common_chars = set(s1) & set(s2)
        return len(common_chars) / max(len1, len2)
    
    def analyze_physics_paper_abstract(self, abstract: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of a physics paper abstract
        全面分析物理論文摘要
        """
        entities = self.extract_physics_entities(abstract)
        equations = self.detect_physics_equations(abstract)
        
        # Categorize entities by type
        entity_types = {}
        for entity in entities:
            entity_type = entity.entity_type.value
            if entity_type not in entity_types:
                entity_types[entity_type] = []
            entity_types[entity_type].append(entity.name)
        
        # Calculate physics complexity score
        complexity_score = self._calculate_complexity_score(entities, equations)
        
        # Determine main physics domain
        domains = [eq.domain for eq in equations]
        main_domain = max(set(domains), key=domains.count) if domains else "unknown"
        
        analysis = {
            "entities": {
                "total_count": len(entities),
                "by_type": entity_types,
                "unique_particles": len([e for e in entities if e.particle_type])
            },
            "equations": {
                "total_count": len(equations),
                "domains": list(set(domains)),
                "main_domain": main_domain
            },
            "analysis": {
                "complexity_score": complexity_score,
                "physics_density": len(entities) / max(len(abstract.split()), 1),
                "theoretical_level": self._assess_theoretical_level(entities, equations)
            },
            "recommendations": self._generate_recommendations(entities, equations)
        }
        
        return analysis
    
    def _calculate_complexity_score(self, entities: List[PhysicsEntity], 
                                  equations: List[PhysicsEquation]) -> float:
        """Calculate physics complexity score"""
        entity_score = len(entities) * 0.1
        equation_score = len(equations) * 0.3
        
        # Bonus for advanced concepts
        advanced_concepts = ["supersymmetry", "gauge", "field", "symmetry"]
        advanced_bonus = sum(0.2 for e in entities 
                           if any(concept in e.name for concept in advanced_concepts))
        
        return min(entity_score + equation_score + advanced_bonus, 1.0)
    
    def _assess_theoretical_level(self, entities: List[PhysicsEntity], 
                                equations: List[PhysicsEquation]) -> str:
        """Assess the theoretical level of the content"""
        if any("quantum" in eq.domain for eq in equations):
            return "advanced"
        elif any(e.entity_type == PhysicsConceptType.FIELD for e in entities):
            return "intermediate"
        elif entities:
            return "basic"
        else:
            return "minimal"
    
    def _generate_recommendations(self, entities: List[PhysicsEntity], 
                                equations: List[PhysicsEquation]) -> List[str]:
        """Generate recommendations for further analysis"""
        recommendations = []
        
        if not entities:
            recommendations.append("Consider adding more physics terminology")
        
        if not equations:
            recommendations.append("Mathematical formulations could enhance the content")
        
        particle_count = len([e for e in entities if e.particle_type])
        if particle_count > 5:
            recommendations.append("Rich particle physics content - consider detailed analysis")
        
        if len(set(eq.domain for eq in equations)) > 2:
            recommendations.append("Interdisciplinary physics - potential for cross-domain insights")
        
        return recommendations
    
    def physics_informed_text_generation(self, seed_concepts: List[str], 
                                       target_length: int = 100) -> str:
        """
        Generate physics-informed text based on seed concepts
        基於種子概念生成物理學相關文本
        """
        if not seed_concepts:
            return "Physics is the fundamental science that seeks to understand the universe."
        
        # Build text around physics concepts
        text_parts = []
        
        for concept in seed_concepts[:3]:  # Limit to first 3 concepts
            if concept in self.physics_vocabulary:
                concept_type = self.physics_vocabulary[concept]
                
                if concept_type == PhysicsConceptType.PARTICLE:
                    if concept in self.particle_signatures:
                        sig = self.particle_signatures[concept]
                        text_parts.append(
                            f"The {concept} ({sig.get('symbol', concept)}) is a "
                            f"{sig.get('type', 'particle').value} with charge {sig.get('charge', 'unknown')} "
                            f"and mass {sig.get('mass_mev', 'unknown')} MeV/c²."
                        )
                    else:
                        text_parts.append(f"The {concept} is an important particle in high energy physics.")
                
                elif concept_type == PhysicsConceptType.FORCE:
                    text_parts.append(f"The {concept} force plays a crucial role in particle interactions.")
                
                elif concept_type == PhysicsConceptType.CONSERVATION:
                    text_parts.append(f"Conservation of {concept} is a fundamental principle in physics.")
        
        # Join and truncate to target length
        generated_text = " ".join(text_parts)
        words = generated_text.split()
        if len(words) > target_length:
            generated_text = " ".join(words[:target_length]) + "..."
        
        return generated_text
    
    def integrate_with_expression_evaluator(self, physics_text: str) -> Optional[Dict[str, Any]]:
        """
        Integrate physics analysis with human expression evaluator
        將物理分析與人類表達評估器整合
        """
        if not EXPRESSION_EVALUATOR_AVAILABLE:
            return None
        
        # Create physics-specific context
        context = ExpressionContext(
            situation="scientific",
            formality_level="formal",
            cultural_background="scientific_community"
        )
        
        # Analyze physics content first
        physics_analysis = self.analyze_physics_paper_abstract(physics_text)
        
        # Then analyze as human expression
        expression_result = self.expression_evaluator.comprehensive_evaluation(
            physics_text, context
        )
        
        # Combine results
        integrated_result = {
            "physics_analysis": physics_analysis,
            "expression_evaluation": expression_result,
            "integration_insights": {
                "physics_formality_alignment": (
                    physics_analysis["analysis"]["complexity_score"] * 
                    expression_result.get("social", {}).get("overall_score", 0.5)
                ),
                "scientific_communication_score": (
                    physics_analysis["analysis"]["physics_density"] + 
                    expression_result.get("cognitive", {}).get("overall_score", 0.5)
                ) / 2
            }
        }
        
        return integrated_result


def demonstrate_physics_nlp():
    """
    Demonstration function for High Energy Physics NLP
    高能物理NLP演示函數
    """
    print("🔬 High Energy Physics NLP Demonstration")
    print("=" * 50)
    
    # Initialize the system
    physics_nlp = HighEnergyPhysicsNLP()
    
    # Sample physics text
    sample_text = """
    The Standard Model describes the electromagnetic, weak, and strong interactions 
    between fundamental particles. Electrons and muons are leptons that interact 
    via the electromagnetic force. The Higgs boson gives mass to particles through 
    the Higgs field. Energy-momentum conservation applies: E = mc².
    """
    
    print("\n📄 Sample Text:")
    print(sample_text.strip())
    
    # Extract entities
    print("\n🔍 Physics Entities:")
    entities = physics_nlp.extract_physics_entities(sample_text)
    for entity in entities:
        print(f"  • {entity.name} ({entity.entity_type.value})")
        if entity.particle_type:
            print(f"    Particle type: {entity.particle_type.value}")
            if entity.properties:
                print(f"    Properties: {entity.properties}")
    
    # Detect equations
    print("\n📐 Physics Equations:")
    equations = physics_nlp.detect_physics_equations(sample_text)
    for eq in equations:
        print(f"  • {eq.latex} - {eq.description} ({eq.domain})")
    
    # Comprehensive analysis
    print("\n📊 Comprehensive Analysis:")
    analysis = physics_nlp.analyze_physics_paper_abstract(sample_text)
    print(f"  Complexity Score: {analysis['analysis']['complexity_score']:.2f}")
    print(f"  Physics Density: {analysis['analysis']['physics_density']:.2f}")
    print(f"  Theoretical Level: {analysis['analysis']['theoretical_level']}")
    print(f"  Main Domain: {analysis['equations']['main_domain']}")
    
    # Generate physics-informed text
    print("\n✍️ Physics-Informed Text Generation:")
    concepts = ["electron", "photon", "energy"]
    generated = physics_nlp.physics_informed_text_generation(concepts, 50)
    print(f"  {generated}")
    
    # Concept similarity
    print("\n🔗 Concept Similarity:")
    similarity = physics_nlp.calculate_physics_concept_similarity("electron", "muon")
    print(f"  electron ↔ muon: {similarity:.2f}")
    
    print("\n✅ Demonstration complete!")


if __name__ == "__main__":
    demonstrate_physics_nlp()