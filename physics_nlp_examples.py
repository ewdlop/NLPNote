#!/usr/bin/env python3
"""
High Energy Physics NLP Examples and Demonstrations
高能物理NLP示例與演示

This script provides comprehensive examples of how to use the High Energy Physics NLP module
for various tasks including research paper analysis, educational content generation,
and integration with existing NLP tools.

本腳本提供了如何使用高能物理NLP模組進行各種任務的全面示例，
包括研究論文分析、教育內容生成，以及與現有NLP工具的整合。
"""

import sys
import json
from typing import Dict, Any
from HighEnergyPhysicsNLP import HighEnergyPhysicsNLP, ParticleType, PhysicsConceptType

def example_1_basic_entity_extraction():
    """Example 1: Basic Physics Entity Extraction 基本物理實體提取"""
    print("\n" + "="*60)
    print("🔬 Example 1: Basic Physics Entity Extraction")
    print("示例1：基本物理實體提取")
    print("="*60)
    
    physics_nlp = HighEnergyPhysicsNLP()
    
    sample_texts = [
        "The electron has a charge of -1 and interacts via electromagnetic force.",
        "Higgs bosons decay into two photons with high probability.",
        "Quantum field theory describes particle interactions through gauge symmetries.",
        "電子和質子通過電磁力相互作用。",  # Chinese text
    ]
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n📝 Text {i}: {text}")
        entities = physics_nlp.extract_physics_entities(text)
        
        if entities:
            print("🔍 Detected Physics Entities:")
            for entity in entities:
                print(f"  • {entity.name} [{entity.entity_type.value}]")
                if entity.particle_type:
                    print(f"    ↳ Particle type: {entity.particle_type.value}")
                if entity.properties:
                    properties_str = ", ".join([f"{k}={v}" for k, v in entity.properties.items() if v is not None])
                    print(f"    ↳ Properties: {properties_str}")
        else:
            print("❌ No physics entities detected")

def example_2_equation_detection():
    """Example 2: Physics Equation Detection 物理方程檢測"""
    print("\n" + "="*60)
    print("📐 Example 2: Physics Equation Detection")
    print("示例2：物理方程檢測")
    print("="*60)
    
    physics_nlp = HighEnergyPhysicsNLP()
    
    texts_with_equations = [
        "Einstein's famous equation E = mc² relates mass and energy.",
        "The Schrödinger equation iℏ∂ψ/∂t = Ĥψ governs quantum evolution.",
        "Maxwell's equations include ∇×E = -∂B/∂t for electromagnetic induction.",
        "Newton's law F = ma connects force, mass, and acceleration.",
        "The fine structure constant α ≈ 1/137 characterizes electromagnetic interactions.",
    ]
    
    for i, text in enumerate(texts_with_equations, 1):
        print(f"\n📝 Text {i}: {text}")
        equations = physics_nlp.detect_physics_equations(text)
        
        if equations:
            print("🔍 Detected Equations:")
            for eq in equations:
                print(f"  • {eq.latex}")
                print(f"    ↳ Description: {eq.description}")
                print(f"    ↳ Domain: {eq.domain}")
                if eq.variables:
                    print(f"    ↳ Variables: {', '.join(eq.variables)}")
                if eq.constants:
                    print(f"    ↳ Constants: {', '.join(eq.constants)}")
        else:
            print("❌ No equations detected")

def example_3_paper_analysis():
    """Example 3: Complete Physics Paper Analysis 完整物理論文分析"""
    print("\n" + "="*60)
    print("📊 Example 3: Complete Physics Paper Analysis")
    print("示例3：完整物理論文分析")
    print("="*60)
    
    physics_nlp = HighEnergyPhysicsNLP()
    
    # Sample physics paper abstracts
    abstracts = [
        {
            "title": "Standard Model Analysis",
            "abstract": """
            We present a comprehensive analysis of the Standard Model of particle physics,
            focusing on the electromagnetic, weak, and strong interactions between fundamental
            particles. Our study examines the behavior of leptons, quarks, and gauge bosons
            within the framework of quantum field theory. The Higgs mechanism provides mass
            to particles through spontaneous symmetry breaking. Conservation laws including
            energy, momentum, and charge conservation play crucial roles in particle
            interactions. The fine structure constant α ≈ 1/137 characterizes the strength
            of electromagnetic interactions.
            """
        },
        {
            "title": "Quantum Entanglement Research",
            "abstract": """
            This paper investigates quantum entanglement phenomena in two-qubit systems.
            We analyze Bell states |Φ⁺⟩ and |Ψ⁻⟩ using the density matrix formalism.
            The concurrence C measures entanglement strength, with C = 1 indicating
            maximal entanglement. Our experimental setup employs photons generated
            through spontaneous parametric down-conversion. Quantum state tomography
            reveals non-local correlations violating Bell inequalities.
            """
        }
    ]
    
    for paper in abstracts:
        print(f"\n📄 Paper: {paper['title']}")
        print(f"Abstract excerpt: {paper['abstract'][:100]}...")
        
        analysis = physics_nlp.analyze_physics_paper_abstract(paper['abstract'])
        
        print("\n📊 Analysis Results:")
        print(f"  🧮 Complexity Score: {analysis['analysis']['complexity_score']:.2f}")
        print(f"  📏 Physics Density: {analysis['analysis']['physics_density']:.2f}")
        print(f"  🎓 Theoretical Level: {analysis['analysis']['theoretical_level']}")
        print(f"  🏷️ Main Domain: {analysis['equations']['main_domain']}")
        
        print(f"\n📈 Entity Statistics:")
        print(f"  Total entities: {analysis['entities']['total_count']}")
        print(f"  Unique particles: {analysis['entities']['unique_particles']}")
        
        print(f"\n📐 Equation Statistics:")
        print(f"  Total equations: {analysis['equations']['total_count']}")
        print(f"  Domains covered: {', '.join(analysis['equations']['domains'])}")
        
        if analysis['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in analysis['recommendations']:
                print(f"  • {rec}")

def example_4_concept_similarity():
    """Example 4: Physics Concept Similarity Analysis 物理概念相似度分析"""
    print("\n" + "="*60)
    print("🔗 Example 4: Physics Concept Similarity Analysis")
    print("示例4：物理概念相似度分析")
    print("="*60)
    
    physics_nlp = HighEnergyPhysicsNLP()
    
    concept_pairs = [
        ("electron", "muon"),        # Same type particles
        ("proton", "neutron"),       # Similar composite particles  
        ("photon", "gluon"),         # Different gauge bosons
        ("electromagnetic", "weak"), # Related forces
        ("energy", "momentum"),      # Related conservation laws
        ("field", "particle"),       # Different concept types
        ("電子", "electron"),         # Cross-language similarity
    ]
    
    print("🔍 Calculating similarity between physics concepts:")
    print("計算物理概念之間的相似度：")
    
    for concept1, concept2 in concept_pairs:
        similarity = physics_nlp.calculate_physics_concept_similarity(concept1, concept2)
        print(f"  {concept1:15} ↔ {concept2:15} : {similarity:.2f}")

def example_5_text_generation():
    """Example 5: Physics-Informed Text Generation 物理知識文本生成"""
    print("\n" + "="*60)
    print("✍️ Example 5: Physics-Informed Text Generation")
    print("示例5：物理知識文本生成")
    print("="*60)
    
    physics_nlp = HighEnergyPhysicsNLP()
    
    concept_sets = [
        ["electron", "photon", "electromagnetic"],
        ["higgs", "boson", "field", "symmetry"],
        ["quark", "gluon", "strong"],
        ["neutrino", "weak", "interaction"],
        ["energy", "momentum", "conservation"],
    ]
    
    print("🤖 Generating physics-informed text from seed concepts:")
    print("從種子概念生成物理知識文本：")
    
    for i, concepts in enumerate(concept_sets, 1):
        print(f"\n📝 Generation {i}:")
        print(f"Seed concepts: {', '.join(concepts)}")
        
        generated_text = physics_nlp.physics_informed_text_generation(concepts, 60)
        print(f"Generated text: {generated_text}")

def example_6_integration_demo():
    """Example 6: Integration with Existing NLP Tools 與現有NLP工具整合"""
    print("\n" + "="*60)
    print("🔧 Example 6: Integration with Existing NLP Tools")
    print("示例6：與現有NLP工具整合")
    print("="*60)
    
    physics_nlp = HighEnergyPhysicsNLP()
    
    # Test integration with HumanExpressionEvaluator
    scientific_texts = [
        "The experimental results strongly suggest the discovery of a new particle.",
        "Our theoretical framework provides compelling evidence for supersymmetry.",
        "The data clearly indicates violations of the Standard Model predictions.",
    ]
    
    print("🤝 Testing integration with HumanExpressionEvaluator:")
    
    for i, text in enumerate(scientific_texts, 1):
        print(f"\n📝 Text {i}: {text}")
        
        # Try integration
        try:
            integrated_result = physics_nlp.integrate_with_expression_evaluator(text)
            if integrated_result:
                physics_score = integrated_result["physics_analysis"]["analysis"]["complexity_score"]
                formality_score = integrated_result["integration_insights"]["physics_formality_alignment"]
                comm_score = integrated_result["integration_insights"]["scientific_communication_score"]
                
                print(f"  🔬 Physics complexity: {physics_score:.2f}")
                print(f"  📝 Formality alignment: {formality_score:.2f}")
                print(f"  💬 Communication score: {comm_score:.2f}")
            else:
                print("  ❌ Integration not available (HumanExpressionEvaluator missing)")
        except Exception as e:
            print(f"  ⚠️ Integration error: {str(e)}")

def example_7_multilingual_support():
    """Example 7: Multilingual Physics NLP 多語言物理NLP"""
    print("\n" + "="*60)
    print("🌍 Example 7: Multilingual Physics NLP")
    print("示例7：多語言物理NLP")
    print("="*60)
    
    physics_nlp = HighEnergyPhysicsNLP()
    
    multilingual_texts = [
        ("English", "Electrons interact with photons through electromagnetic force."),
        ("中文", "電子通過電磁力與光子相互作用。能量守恆是基本物理定律。"),
        ("Mixed", "The 質子 has positive charge and interacts via strong force."),
    ]
    
    print("🌐 Testing multilingual physics entity extraction:")
    print("測試多語言物理實體提取：")
    
    for language, text in multilingual_texts:
        print(f"\n🌏 Language: {language}")
        print(f"📝 Text: {text}")
        
        entities = physics_nlp.extract_physics_entities(text)
        if entities:
            print("🔍 Detected entities:")
            for entity in entities:
                print(f"  • {entity.name} [{entity.entity_type.value}]")
        else:
            print("❌ No entities detected")

def example_8_performance_analysis():
    """Example 8: Performance and Accuracy Analysis 性能與準確度分析"""
    print("\n" + "="*60)
    print("⚡ Example 8: Performance and Accuracy Analysis")
    print("示例8：性能與準確度分析")
    print("="*60)
    
    physics_nlp = HighEnergyPhysicsNLP()
    
    # Test with various complexity levels
    test_cases = [
        {
            "complexity": "Basic",
            "text": "Electrons have negative charge and orbit the nucleus."
        },
        {
            "complexity": "Intermediate", 
            "text": "The Standard Model includes gauge bosons that mediate fundamental forces."
        },
        {
            "complexity": "Advanced",
            "text": "Supersymmetric partners of Standard Model particles could explain dark matter through R-parity conservation."
        },
        {
            "complexity": "Mathematical",
            "text": "The Lagrangian L = ψ̄(iγμDμ - m)ψ describes fermion dynamics in gauge theory."
        }
    ]
    
    print("📊 Analyzing performance across complexity levels:")
    print("分析不同複雜度級別的性能：")
    
    for case in test_cases:
        print(f"\n🎓 Complexity: {case['complexity']}")
        print(f"📝 Text: {case['text']}")
        
        # Analyze the text
        analysis = physics_nlp.analyze_physics_paper_abstract(case['text'])
        entities = physics_nlp.extract_physics_entities(case['text'])
        equations = physics_nlp.detect_physics_equations(case['text'])
        
        print(f"📈 Results:")
        print(f"  Entities found: {len(entities)}")
        print(f"  Equations found: {len(equations)}")
        print(f"  Complexity score: {analysis['analysis']['complexity_score']:.2f}")
        print(f"  Theoretical level: {analysis['analysis']['theoretical_level']}")

def run_all_examples():
    """Run all examples in sequence 按順序運行所有示例"""
    print("🚀 High Energy Physics NLP - Comprehensive Examples")
    print("高能物理NLP - 綜合示例")
    print("=" * 80)
    
    examples = [
        example_1_basic_entity_extraction,
        example_2_equation_detection,
        example_3_paper_analysis,
        example_4_concept_similarity,
        example_5_text_generation,
        example_6_integration_demo,
        example_7_multilingual_support,
        example_8_performance_analysis,
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"\n❌ Error in example {i}: {str(e)}")
            print("Continuing with next example...")
    
    print("\n" + "="*80)
    print("✅ All examples completed!")
    print("所有示例完成！")
    print("="*80)

def interactive_demo():
    """Interactive demonstration mode 互動演示模式"""
    print("\n🎮 Interactive Physics NLP Demo")
    print("互動物理NLP演示")
    print("-" * 40)
    
    physics_nlp = HighEnergyPhysicsNLP()
    
    while True:
        print("\nEnter physics text to analyze (or 'quit' to exit):")
        print("輸入物理文本進行分析（或輸入'quit'退出）：")
        
        user_input = input("> ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Thanks for using Physics NLP! 感謝使用物理NLP！")
            break
        
        if not user_input:
            continue
        
        print(f"\n🔍 Analyzing: {user_input}")
        
        # Entity extraction
        entities = physics_nlp.extract_physics_entities(user_input)
        if entities:
            print("\n📋 Physics Entities:")
            for entity in entities:
                print(f"  • {entity.name} [{entity.entity_type.value}]")
        
        # Equation detection
        equations = physics_nlp.detect_physics_equations(user_input)
        if equations:
            print("\n📐 Equations:")
            for eq in equations:
                print(f"  • {eq.latex} - {eq.description}")
        
        # Quick analysis
        analysis = physics_nlp.analyze_physics_paper_abstract(user_input)
        print(f"\n📊 Quick Stats:")
        print(f"  Complexity: {analysis['analysis']['complexity_score']:.2f}")
        print(f"  Physics density: {analysis['analysis']['physics_density']:.2f}")
        print(f"  Level: {analysis['analysis']['theoretical_level']}")

def main():
    """Main function with menu options 帶菜單選項的主函數"""
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == 'interactive':
            interactive_demo()
        elif mode == 'all':
            run_all_examples()
        elif mode.isdigit() and 1 <= int(mode) <= 8:
            example_num = int(mode)
            examples = [
                example_1_basic_entity_extraction,
                example_2_equation_detection,
                example_3_paper_analysis,
                example_4_concept_similarity,
                example_5_text_generation,
                example_6_integration_demo,
                example_7_multilingual_support,
                example_8_performance_analysis,
            ]
            examples[example_num - 1]()
        else:
            print("Invalid option. Use: all, interactive, or 1-8")
    else:
        # Default: run all examples
        run_all_examples()

if __name__ == "__main__":
    main()