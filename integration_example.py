#!/usr/bin/env python3
"""
Integration Example: 降維打擊 with NLP Tools
==============================================

This example demonstrates how to integrate the Dimensional Reduction Attack
functionality with existing NLP tools in the repository.

Author: NLP Note Project
Date: 2024-12-22
"""

import sys
import os
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def demonstrate_integration():
    """演示降維攻擊與現有NLP工具的整合"""
    
    print("🔗 降維打擊與NLP工具整合演示")
    print("🔗 Dimensional Reduction Attack Integration with NLP Tools")
    print("=" * 60)
    
    try:
        from DimensionalReductionAttack import DimensionalAttackOrchestrator, EarthFlattener
        
        # 測試基本功能
        print("\n1. 基本降維攻擊測試...")
        orchestrator = DimensionalAttackOrchestrator()
        sample_data = np.random.rand(100, 20)
        
        if 'pca' in orchestrator.list_available_attacks():
            result = orchestrator.execute_dimensional_attack(sample_data, 'pca', 3)
            print(f"   ✅ PCA攻擊成功: {result.original_dimensions}D → {result.reduced_dimensions}D")
        
        # 測試扁平化功能
        print("\n2. 地球扁平化測試...")
        flattener = EarthFlattener()
        
        test_dict = {
            'nlp': {
                'tasks': ['classification', 'ner', 'sentiment'],
                'models': {'bert': 'transformer', 'lstm': 'rnn'}
            }
        }
        
        flat_result = flattener.flatten_nested_dict(test_dict)
        print(f"   ✅ 字典扁平化成功: 複雜度減少 {flat_result.complexity_reduction:.1%}")
        
        # 嘗試與現有工具整合
        print("\n3. 嘗試與現有NLP工具整合...")
        
        try:
            from HumanExpressionEvaluator import HumanExpressionEvaluator, ExpressionContext
            evaluator = HumanExpressionEvaluator()
            print("   ✅ 成功導入 HumanExpressionEvaluator")
            
            # 演示可能的整合方式
            context = ExpressionContext(
                formality_level='formal',
                situation='academic'
            )
            
            # 這裡可以添加更多整合邏輯
            print("   💡 可以整合表達評估與降維技術")
            
        except ImportError:
            print("   ℹ️  HumanExpressionEvaluator 未找到，跳過整合測試")
        
        try:
            from SubtextAnalyzer import SubtextAnalyzer
            analyzer = SubtextAnalyzer()
            print("   ✅ 成功導入 SubtextAnalyzer")
            print("   💡 可以整合潛文本分析與數據降維")
            
        except ImportError:
            print("   ℹ️  SubtextAnalyzer 未找到，跳過整合測試")
        
        print("\n4. 整合應用場景建議:")
        print("   📊 高維語言特徵降維可視化")
        print("   🗂️  複雜NLP流水線結果扁平化")
        print("   🎯 多維評估結果的降維分析")
        print("   🌍 嵌套語言結構的簡化處理")
        
        print("\n✅ 整合演示完成!")
        
    except ImportError as e:
        print(f"❌ 導入錯誤: {e}")
        print("請確保 DimensionalReductionAttack.py 在當前目錄")


def show_conceptual_connections():
    """展示概念聯繫"""
    print("\n" + "=" * 60)
    print("🧠 概念聯繫: 降維打擊在NLP中的哲學意義")
    print("🧠 Conceptual Connections: Philosophical Meaning of Dimensional Reduction Attack in NLP")
    print("=" * 60)
    
    connections = [
        {
            "概念": "降維打擊 (Dimensional Reduction Attack)",
            "NLP應用": "將高維詞向量壓縮到可理解的低維空間",
            "哲學意義": "從複雜到簡潔的信息轉換藝術"
        },
        {
            "概念": "扁平化地球 (Flattening Earth)", 
            "NLP應用": "將嵌套的語言結構展開為線性表示",
            "哲學意義": "將立體思維映射到平面理解"
        },
        {
            "概念": "維度攻擊效果 (Attack Effectiveness)",
            "NLP應用": "評估降維後信息保留的程度",
            "哲學意義": "衡量簡化過程中的智慧損失"
        }
    ]
    
    for i, conn in enumerate(connections, 1):
        print(f"\n{i}. {conn['概念']}")
        print(f"   🔬 NLP應用: {conn['NLP應用']}")
        print(f"   🤔 哲學意義: {conn['哲學意義']}")
    
    print(f"\n💭 總結思考:")
    print("   降維打擊不僅是一種技術手段，更是一種認知模式的轉換。")
    print("   它幫助我們理解如何在保持本質的同時簡化複雜性。")
    print("   在NLP領域，這種思維方式對於處理高維語言數據特別有價值。")


if __name__ == "__main__":
    demonstrate_integration()
    show_conceptual_connections()
    
    print("\n🎯 降維打擊技術已準備就緒，可用於各種NLP任務!")
    print("🌍 Dimensional Reduction Attack technology is ready for various NLP tasks!")