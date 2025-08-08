#!/usr/bin/env python3
"""
Simple Lie Bracket Example
簡單李括號示例

A concise demonstration of the Lie bracket framework answering:
"Computer lie bracket of Math and physics(physical mathematics - mathematical physics =?)"
"""

from LieBracket import LieBracketFramework
from MathematicalExpressionAnalyzer import MathematicalExpressionAnalyzer


def main():
    print("🔬 Computer Lie Bracket: Physical Mathematics - Mathematical Physics = ?")
    print("計算李括號：物理數學 - 數學物理 = ？")
    print("=" * 70)
    
    # Create the framework
    framework = LieBracketFramework()
    
    # Demonstrate the philosophical insight
    comparison = framework.demonstrate_physical_vs_mathematical()
    
    print("\n🧮 THE ANSWER:")
    print("答案：")
    insight = comparison['lie_bracket_insight']
    print(f"Formula: {insight['formula']}")
    print(f"Meaning: {insight['interpretation']}")
    print(f"Synthesis: {insight['synthesis']}")
    
    print("\n📊 EXAMPLES:")
    print("示例：")
    
    # Physical Mathematics Example
    print("\n1️⃣ Physical Mathematics (物理數學):")
    print("   Start: Quantum spin phenomenon")
    print("   Derive: [σ_x, σ_y] = 2iσ_z (Pauli matrices)")
    print("   Result: SU(2) Lie algebra structure")
    
    # Mathematical Physics Example
    print("\n2️⃣ Mathematical Physics (數學物理):")
    print("   Start: SO(3) rotation group")
    print("   Apply: [R_x, R_y] = R_z (vector fields)")
    print("   Result: Describes physical rotations")
    
    # Show NLP integration
    print("\n🔤 NLP ANALYSIS:")
    analyzer = MathematicalExpressionAnalyzer()
    
    sample_text = "The commutator [A, B] = AB - BA represents quantum non-commutativity"
    analysis = analyzer.analyze_lie_bracket_expression(sample_text)
    
    print(f"Text: {sample_text}")
    print(f"Concepts found: {[c.name for c in analysis['mathematical_concepts']]}")
    print(f"Contains bracket notation: {'Yes' if analysis['lie_bracket_structure'] else 'No'}")
    
    # The final insight
    print("\n💡 CONCLUSION:")
    print("結論：")
    print("Physical Mathematics and Mathematical Physics are like position and momentum:")
    print("They don't commute: [Physical_Math, Mathematical_Physics] ≠ 0")
    print("This non-commutativity is what makes science dynamic and progressive!")
    print("Complete understanding = synthesis of both approaches")
    
    print("\n🎯 Framework successfully addresses the question!")
    print("框架成功回答了這個問題！")


if __name__ == "__main__":
    main()