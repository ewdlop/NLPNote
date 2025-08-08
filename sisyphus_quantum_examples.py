"""
薛西弗斯量子分析器示例集 (Sisyphus Quantum Analyzer Example Collection)

這個模組提供各種示例來展示薛西弗斯量子分析器的功能，
包括循環邏輯檢測和量子隧穿突破時刻的識別。

This module provides various examples to demonstrate the functionality of
the Sisyphus Quantum Analyzer, including circular logic detection and 
quantum tunneling breakthrough moment identification.
"""

from SisyphusQuantumAnalyzer import SisyphusQuantumAnalyzer, SisyphusPattern, QuantumTunnelingMoment
from SubtextAnalyzer import SubtextAnalyzer
from HumanExpressionEvaluator import HumanExpressionEvaluator, ExpressionContext
import json


class SisyphusQuantumExampleRunner:
    """運行薛西弗斯量子分析器的示例 (Example runner for Sisyphus Quantum Analyzer)"""
    
    def __init__(self):
        self.sisyphus_analyzer = SisyphusQuantumAnalyzer()
        self.subtext_analyzer = SubtextAnalyzer()
        self.expression_evaluator = HumanExpressionEvaluator()
    
    def get_example_texts(self) -> dict:
        """獲取示例文本 (Get example texts)"""
        return {
            'circular_logic_examples': [
                # 經典循環邏輯 (Classic circular logic)
                """
                這個法律是公正的，因為它保護正義。而正義就是這個法律所代表的。
                因此，這個法律必然是正確的，因為正確的法律就是公正的法律。
                """,
                
                # 宗教循環論證 (Religious circular argument)  
                """
                聖經是真理，因為聖經是上帝的話語。我們知道聖經是上帝的話語，
                因為聖經自己這麼說。而聖經不會說謊，因為它是真理。
                """,
                
                # 權威循環 (Authority circular)
                """
                專家說這是對的，所以這是對的。為什麼專家是對的？
                因為他們是專家。為什麼他們是專家？因為他們總是對的。
                """,
                
                # 定義循環 (Definition circular)
                """
                民主是好的，因為民主意味著人民統治。人民統治是好的，
                因為這就是民主。民主制度之所以優越，就是因為它是民主的。
                """
            ],
            
            'quantum_tunneling_examples': [
                # 悖論解決 (Paradox resolution)
                """
                這個矛盾看似無解：如果我說"我正在說謊"，那麼如果這是真的，
                它就是假的；如果這是假的，它就是真的。但是，突然意識到，
                問題不在於語句本身，而在於我們對真假的二元框架。
                也許真相在於超越真假的第三種狀態。
                """,
                
                # 創新突破 (Innovation breakthrough)
                """
                傳統方法告訴我們，要解決衝突就必須找到妥協。
                但是，如果我們完全重新定義問題呢？也許衝突本身就是答案，
                不是要消除衝突，而是要學會在衝突中創造新的可能性。
                這種轉換超越了解決問題的傳統思維。
                """,
                
                # 哲學洞察 (Philosophical insight)
                """
                我們一直在尋找生命的意義，彷彿意義是一個隱藏的寶藏。
                然而，也許意義不是被發現的，而是被創造的。
                當我們停止尋找意義，開始創造意義，我們就超越了
                "生命有沒有意義"這個問題本身。
                """,
                
                # 科學範式轉換 (Scientific paradigm shift)
                """
                牛頓物理學無法解釋某些現象，這些異常被視為錯誤。
                但愛因斯坦意識到，問題不在於測量錯誤，而在於我們對
                時間和空間的基本假設。通過質疑這些假設，相對論誕生了，
                完全改變了我們理解宇宙的方式。
                """
            ],
            
            'mixed_examples': [
                # 循環中的突破 (Breakthrough within circularity)
                """
                我們總是重複同樣的模式：遇到問題，應用舊方法，失敗，
                然後責怪外部因素。這個循環似乎永無止境。
                但是，如果我們把失敗本身看作是一種數據，
                而不是需要避免的東西呢？突然間，失敗變成了學習的機會，
                循環變成了螺旋式上升。
                """,
                
                # 矛盾中的和諧 (Harmony within contradiction)
                """
                藝術必須既要創新又要傳統，這似乎是矛盾的。
                創新意味著打破傳統，而傳統意味著保持不變。
                然而，最偉大的藝術作品往往同時做到了這兩點：
                它們在傳統的基礎上創新，通過創新來延續傳統。
                這種矛盾的統一超越了非此即彼的邏輯。
                """,
                
                # 邏輯與直覺的融合 (Logic and intuition fusion)
                """
                科學方法強調邏輯和實證，這與直覺和靈感似乎相對立。
                但許多重大發現都來自於直覺的閃光，然後用邏輯來驗證。
                也許真正的創新來自於邏輯與直覺的舞蹈，
                而不是其中任何一方的獨舞。
                """
            ]
        }
    
    def run_comprehensive_analysis(self, text: str, title: str = "Unknown") -> dict:
        """運行綜合分析 (Run comprehensive analysis)"""
        print(f"\n{'='*60}")
        print(f"分析標題 (Analysis Title): {title}")
        print(f"{'='*60}")
        print("文本內容 (Text Content):")
        print(text.strip())
        print(f"\n{'='*60}")
        
        # 薛西弗斯量子分析 (Sisyphus Quantum Analysis)
        sq_result = self.sisyphus_analyzer.analyze(text)
        
        # 潛文本分析 (Subtext Analysis)
        subtext_result = self.subtext_analyzer.calculate_subtext_probability(text)
        
        # 人類表達評估 (Human Expression Evaluation)
        context = ExpressionContext(
            situation='analytical_text',
            formality_level='neutral'
        )
        expr_result = self.expression_evaluator.comprehensive_evaluation(text, context)
        
        # 整合分析結果 (Integrate analysis results)
        integrated_result = self._integrate_all_analyses(sq_result, subtext_result, expr_result)
        
        # 顯示結果 (Display results)
        self._display_results(sq_result, subtext_result, expr_result, integrated_result)
        
        return {
            'sisyphus_quantum': sq_result,
            'subtext': subtext_result,
            'expression_evaluation': expr_result,
            'integrated': integrated_result
        }
    
    def _integrate_all_analyses(self, sq_result: dict, subtext_result: dict, expr_result: dict) -> dict:
        """整合所有分析結果 (Integrate all analysis results)"""
        # 提取關鍵分數 (Extract key scores)
        sisyphus_score = sq_result['sisyphus_analysis']['score']
        quantum_score = sq_result['quantum_analysis']['score']
        subtext_score = subtext_result['probability']
        expr_score = expr_result['integrated']['overall_score']
        
        # 計算整合洞察 (Calculate integrated insights)
        insights = []
        
        # 基於分數組合的洞察 (Insights based on score combinations)
        if sisyphus_score > 0.6 and quantum_score > 0.6:
            insights.append("文本展現了從循環邏輯到突破性思維的轉化過程")
        elif sisyphus_score > 0.6 and quantum_score < 0.3:
            insights.append("文本陷入循環邏輯，缺乏突破性思維")
        elif sisyphus_score < 0.3 and quantum_score > 0.6:
            insights.append("文本展現清晰的突破性思維，避免了循環論證")
        
        if subtext_score > 0.7:
            insights.append("文本具有豐富的潛在含義和隱喻層次")
        
        if expr_score > 0.7:
            insights.append("文本在語言表達質量方面表現優秀")
        
        # 特殊模式識別 (Special pattern recognition)
        if sisyphus_score > 0.5 and subtext_score > 0.5:
            insights.append("文本可能使用循環結構來強化深層含義")
        
        if quantum_score > 0.5 and expr_score > 0.6:
            insights.append("文本成功運用高質量表達來傳達創新概念")
        
        return {
            'overall_creativity_score': (quantum_score + subtext_score) / 2,
            'overall_logic_score': (1 - sisyphus_score + expr_score) / 2,
            'balanced_assessment': self._assess_balance(sisyphus_score, quantum_score),
            'integrated_insights': insights,
            'recommendation': self._generate_recommendation(sisyphus_score, quantum_score, subtext_score, expr_score)
        }
    
    def _assess_balance(self, sisyphus_score: float, quantum_score: float) -> str:
        """評估薛西弗斯和量子得分的平衡 (Assess balance between Sisyphus and quantum scores)"""
        if quantum_score > sisyphus_score * 2:
            return "高度創新型：突破性思維佔主導地位"
        elif quantum_score > sisyphus_score:
            return "創新導向型：創新思維略勝於循環模式"
        elif sisyphus_score > quantum_score * 2:
            return "循環模式型：重複論證佔主導地位"
        else:
            return "平衡型：循環模式與創新思維並存"
    
    def _generate_recommendation(self, sisyphus: float, quantum: float, subtext: float, expr: float) -> str:
        """生成改進建議 (Generate improvement recommendations)"""
        recommendations = []
        
        if sisyphus > 0.6:
            recommendations.append("建議減少重複論證，增加新穎論點")
        
        if quantum < 0.4:
            recommendations.append("建議加入更多創新思維和概念突破")
        
        if subtext < 0.4:
            recommendations.append("建議增加隱喻和象徵性表達來豐富含義層次")
        
        if expr < 0.5:
            recommendations.append("建議改善語言表達的清晰度和適當性")
        
        if not recommendations:
            recommendations.append("文本整體表現良好，可保持現有風格")
        
        return " | ".join(recommendations)
    
    def _display_results(self, sq_result: dict, subtext_result: dict, expr_result: dict, integrated: dict):
        """顯示分析結果 (Display analysis results)"""
        print("\n🔄 薛西弗斯分析 (Sisyphus Analysis):")
        sisyphus = sq_result['sisyphus_analysis']
        print(f"   循環邏輯分數: {sisyphus['score']:.2f}")
        print(f"   解釋: {sisyphus['interpretation']}")
        
        if sisyphus['patterns']:
            print("   檢測到的模式:")
            for pattern in sisyphus['patterns']:
                print(f"     • {pattern.explanation}")
        
        print("\n⚡ 量子隧穿分析 (Quantum Tunneling Analysis):")
        quantum = sq_result['quantum_analysis']
        print(f"   突破性思維分數: {quantum['score']:.2f}")
        print(f"   解釋: {quantum['interpretation']}")
        
        if quantum['moments']:
            print("   檢測到的突破時刻:")
            for moment in quantum['moments']:
                print(f"     • {moment.moment_type}: 超越了 {moment.barrier_transcended}")
        
        print(f"\n📊 潛文本分析分數: {subtext_result['probability']:.2f}")
        print(f"🎯 表達評估分數: {expr_result['integrated']['overall_score']:.2f}")
        
        print(f"\n🔮 整合評估 (Integrated Assessment):")
        print(f"   創造力分數: {integrated['overall_creativity_score']:.2f}")
        print(f"   邏輯清晰度: {integrated['overall_logic_score']:.2f}")
        print(f"   平衡類型: {integrated['balanced_assessment']}")
        
        print("\n💡 整合洞察 (Integrated Insights):")
        for insight in integrated['integrated_insights']:
            print(f"   • {insight}")
        
        print(f"\n📝 改進建議: {integrated['recommendation']}")
    
    def run_all_examples(self):
        """運行所有示例 (Run all examples)"""
        examples = self.get_example_texts()
        
        print("🚀 薛西弗斯量子分析器 - 完整示例集")
        print("🚀 Sisyphus Quantum Analyzer - Complete Example Collection")
        print("="*80)
        
        # 循環邏輯示例 (Circular logic examples)
        print("\n🔄 循環邏輯示例 (Circular Logic Examples)")
        print("="*50)
        for i, text in enumerate(examples['circular_logic_examples'], 1):
            self.run_comprehensive_analysis(text, f"循環邏輯示例 {i}")
        
        # 量子隧穿示例 (Quantum tunneling examples)
        print("\n\n⚡ 量子隧穿示例 (Quantum Tunneling Examples)")
        print("="*50)
        for i, text in enumerate(examples['quantum_tunneling_examples'], 1):
            self.run_comprehensive_analysis(text, f"量子隧穿示例 {i}")
        
        # 混合示例 (Mixed examples)
        print("\n\n🔀 混合示例 (Mixed Examples)")
        print("="*50)
        for i, text in enumerate(examples['mixed_examples'], 1):
            self.run_comprehensive_analysis(text, f"混合示例 {i}")
    
    def interactive_analysis(self):
        """互動式分析 (Interactive analysis)"""
        print("\n🤖 薛西弗斯量子分析器 - 互動模式")
        print("🤖 Sisyphus Quantum Analyzer - Interactive Mode")
        print("="*60)
        print("輸入文本進行分析，輸入 'quit' 退出")
        print("Enter text for analysis, type 'quit' to exit")
        print("-"*60)
        
        while True:
            text = input("\n請輸入要分析的文本 (Enter text to analyze): ")
            
            if text.lower().strip() == 'quit':
                print("感謝使用薛西弗斯量子分析器！")
                break
            
            if not text.strip():
                print("請輸入有效的文本")
                continue
            
            self.run_comprehensive_analysis(text, "用戶輸入文本")


def main():
    """主函數 (Main function)"""
    runner = SisyphusQuantumExampleRunner()
    
    print("選擇運行模式 (Choose run mode):")
    print("1. 運行所有示例 (Run all examples)")
    print("2. 互動式分析 (Interactive analysis)")
    print("3. 僅運行循環邏輯示例 (Run only circular logic examples)")
    print("4. 僅運行量子隧穿示例 (Run only quantum tunneling examples)")
    
    choice = input("\n請選擇 (1-4): ").strip()
    
    if choice == '1':
        runner.run_all_examples()
    elif choice == '2':
        runner.interactive_analysis()
    elif choice == '3':
        examples = runner.get_example_texts()
        for i, text in enumerate(examples['circular_logic_examples'], 1):
            runner.run_comprehensive_analysis(text, f"循環邏輯示例 {i}")
    elif choice == '4':
        examples = runner.get_example_texts()
        for i, text in enumerate(examples['quantum_tunneling_examples'], 1):
            runner.run_comprehensive_analysis(text, f"量子隧穿示例 {i}")
    else:
        print("無效選擇，運行所有示例...")
        runner.run_all_examples()


if __name__ == "__main__":
    main()