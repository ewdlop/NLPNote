#!/usr/bin/env python3
"""
π表達式互動示例 (Pi Expression Interactive Examples)

這個腳本提供了π表達式評估的互動示例，展示了不同數學方法
如何計算和評估π的值，類似於程式語言中的表達式評估。

This script provides interactive examples of π expression evaluation,
demonstrating how different mathematical methods can calculate and 
evaluate π values, similar to expression evaluation in programming languages.
"""

import math
import sys
import time
from decimal import Decimal, getcontext

# 嘗試導入pi_evaluator，如果失敗則提供基本實現
try:
    from pi_evaluator import PiExpressionEvaluator
    FULL_EVALUATOR_AVAILABLE = True
except ImportError:
    FULL_EVALUATOR_AVAILABLE = False
    print("注意: 完整π評估器不可用，使用基本實現")
    print("Note: Full π evaluator not available, using basic implementation")


class BasicPiCalculator:
    """基本π計算器（如果完整版本不可用時使用）"""
    
    @staticmethod
    def leibniz_series(terms: int) -> dict:
        """萊布尼茲級數計算π"""
        start_time = time.time()
        pi_approx = 0
        
        for k in range(terms):
            sign = (-1) ** k
            term = sign / (2 * k + 1)
            pi_approx += term
        
        pi_approx *= 4
        error = abs(pi_approx - math.pi)
        
        return {
            'method': 'Leibniz Series (Basic)',
            'result': pi_approx,
            'error': error,
            'computation_time': time.time() - start_time,
            'terms_used': terms
        }
    
    @staticmethod
    def monte_carlo_pi(n_points: int) -> dict:
        """蒙特卡羅方法計算π"""
        import random
        start_time = time.time()
        
        inside_circle = 0
        for _ in range(n_points):
            x, y = random.uniform(-1, 1), random.uniform(-1, 1)
            if x*x + y*y <= 1:
                inside_circle += 1
        
        pi_approx = 4 * inside_circle / n_points
        error = abs(pi_approx - math.pi)
        
        return {
            'method': 'Monte Carlo (Basic)',
            'result': pi_approx,
            'error': error,
            'computation_time': time.time() - start_time,
            'points_used': n_points,
            'hit_rate': inside_circle / n_points
        }


def display_pi_introduction():
    """顯示π的介紹資訊"""
    print("π (Pi) - 數學中最重要的常數之一")
    print("=" * 50)
    print("定義: π = 圓周長 / 直徑")
    print("數值: π ≈ 3.14159265358979323846...")
    print("性質: 無理數、超越數")
    print()
    print("π is one of the most important constants in mathematics")
    print("Definition: π = circumference / diameter")
    print("Value: π ≈ 3.14159265358979323846...")
    print("Properties: irrational, transcendental number")
    print()


def demonstrate_pi_series_expressions():
    """演示π的級數表達式"""
    print("π的級數表達式演示 (Pi Series Expressions Demo)")
    print("=" * 60)
    print()
    
    # 演示不同的級數表達式
    expressions = [
        {
            'name': '萊布尼茲級數 (Leibniz Series)',
            'formula': 'π/4 = 1 - 1/3 + 1/5 - 1/7 + 1/9 - ...',
            'mathematical': 'π = 4 × Σ(k=0→∞) [(-1)^k / (2k+1)]',
            'convergence': '收斂慢 (Slow convergence)'
        },
        {
            'name': '尼爾森級數 (Nilakantha Series)',
            'formula': 'π = 3 + 4/(2×3×4) - 4/(4×5×6) + 4/(6×7×8) - ...',
            'mathematical': 'π = 3 + 4 × Σ(k=1→∞) [(-1)^(k+1) / ((2k)(2k+1)(2k+2))]',
            'convergence': '比萊布尼茲快 (Faster than Leibniz)'
        },
        {
            'name': '馬欽公式 (Machin Formula)',
            'formula': 'π/4 = 4×arctan(1/5) - arctan(1/239)',
            'mathematical': 'π/4 = 4×arctan(1/5) - arctan(1/239)',
            'convergence': '快速收斂 (Fast convergence)'
        },
        {
            'name': '沃利斯乘積 (Wallis Product)',
            'formula': 'π/2 = (2×2)/(1×3) × (4×4)/(3×5) × (6×6)/(5×7) × ...',
            'mathematical': 'π/2 = ∏(k=1→∞) [(2k)² / ((2k-1)(2k+1))]',
            'convergence': '收斂較慢 (Slower convergence)'
        },
        {
            'name': '拉馬努金級數 (Ramanujan Series)',
            'formula': '1/π = (2√2/9801) × Σ[(4k)!(1103+26390k)] / [((k!)⁴)(396^(4k))]',
            'mathematical': '拉馬努金的神奇公式 (Ramanujan\'s miraculous formula)',
            'convergence': '極快收斂 (Extremely fast convergence)'
        }
    ]
    
    for i, expr in enumerate(expressions, 1):
        print(f"{i}. {expr['name']}")
        print(f"   公式: {expr['formula']}")
        print(f"   數學表示: {expr['mathematical']}")
        print(f"   收斂性: {expr['convergence']}")
        print()


def demonstrate_pi_calculation_comparison():
    """演示π計算方法的比較"""
    print("π計算方法比較演示 (Pi Calculation Methods Comparison)")
    print("=" * 65)
    print()
    
    if FULL_EVALUATOR_AVAILABLE:
        evaluator = PiExpressionEvaluator(precision=15)
        
        # 測試不同方法
        test_cases = [
            ('leibniz_series', 10000, '萊布尼茲級數 (10,000項)'),
            ('monte_carlo_pi', 100000, '蒙特卡羅方法 (100,000點)'),
            ('agm_algorithm', 5, 'AGM演算法 (5次迭代)'),
            ('machin_formula', 20, '馬欽公式 (20項/arctan)')
        ]
        
        results = []
        for method_name, param, description in test_cases:
            print(f"計算中: {description}...")
            try:
                if method_name == 'leibniz_series':
                    result = evaluator.leibniz_series(param)
                elif method_name == 'monte_carlo_pi':
                    result = evaluator.monte_carlo_pi(param, seed=42)
                elif method_name == 'agm_algorithm':
                    result = evaluator.agm_algorithm(param)
                elif method_name == 'machin_formula':
                    result = evaluator.machin_formula(param)
                
                results.append((description, result))
            except Exception as e:
                print(f"錯誤: {e}")
                continue
        
        # 顯示結果
        print("\n結果比較:")
        print("-" * 80)
        print(f"{'方法':<25} {'π值':<20} {'誤差':<15} {'時間(秒)':<10} {'準確位數'}")
        print("-" * 80)
        
        for description, result in results:
            pi_str = f"{result['result']:.10f}"
            error_str = f"{result['error']:.2e}"
            time_str = f"{result['computation_time']:.4f}"
            accuracy = result.get('decimal_accuracy', 'N/A')
            
            print(f"{description:<25} {pi_str:<20} {error_str:<15} {time_str:<10} {accuracy}")
    
    else:
        # 使用基本計算器
        calculator = BasicPiCalculator()
        
        print("使用基本π計算器進行演示...")
        print()
        
        # 萊布尼茲級數測試
        print("1. 萊布尼茲級數測試:")
        for terms in [100, 1000, 10000]:
            result = calculator.leibniz_series(terms)
            print(f"   {terms:>5}項: π ≈ {result['result']:.8f}, "
                  f"誤差: {result['error']:.2e}, "
                  f"時間: {result['computation_time']:.4f}秒")
        
        print()
        
        # 蒙特卡羅方法測試
        print("2. 蒙特卡羅方法測試:")
        for points in [1000, 10000, 100000]:
            result = calculator.monte_carlo_pi(points)
            print(f"   {points:>6}點: π ≈ {result['result']:.8f}, "
                  f"誤差: {result['error']:.2e}, "
                  f"命中率: {result['hit_rate']:.4f}")


def demonstrate_pi_expression_evaluation():
    """演示π表達式的評估過程"""
    print("π表達式評估過程演示 (Pi Expression Evaluation Process)")
    print("=" * 65)
    print()
    
    # 模擬表達式評估過程
    expressions = [
        {
            'expression': 'π',
            'evaluation_steps': [
                '1. 識別π常數',
                '2. 查找數值近似',
                '3. 返回: 3.14159265...'
            ],
            'result': math.pi
        },
        {
            'expression': '2 × π',
            'evaluation_steps': [
                '1. 識別乘法表達式',
                '2. 評估左運算元: 2',
                '3. 評估右運算元: π ≈ 3.14159265...',
                '4. 執行乘法: 2 × 3.14159265...',
                '5. 返回: 6.28318530...'
            ],
            'result': 2 * math.pi
        },
        {
            'expression': 'π²',
            'evaluation_steps': [
                '1. 識別冪運算表達式',
                '2. 評估底數: π ≈ 3.14159265...',
                '3. 評估指數: 2',
                '4. 執行冪運算: (3.14159265...)²',
                '5. 返回: 9.86960440...'
            ],
            'result': math.pi ** 2
        },
        {
            'expression': 'sin(π)',
            'evaluation_steps': [
                '1. 識別函數調用',
                '2. 評估參數: π ≈ 3.14159265...',
                '3. 調用sin函數',
                '4. 返回: 0 (理論上)',
                '5. 實際: ≈ 1.22e-16 (浮點誤差)'
            ],
            'result': math.sin(math.pi)
        }
    ]
    
    for i, expr_info in enumerate(expressions, 1):
        print(f"表達式 {i}: {expr_info['expression']}")
        print("評估步驟:")
        for step in expr_info['evaluation_steps']:
            print(f"   {step}")
        print(f"最終結果: {expr_info['result']}")
        print()


def demonstrate_pi_convergence_analysis():
    """演示π級數的收斂分析"""
    print("π級數收斂分析演示 (Pi Series Convergence Analysis)")
    print("=" * 60)
    print()
    
    print("萊布尼茲級數收斂分析:")
    print("π/4 = 1 - 1/3 + 1/5 - 1/7 + 1/9 - ...")
    print()
    
    # 使用基本計算示範收斂
    partial_sums = []
    pi_quarter = 0
    actual_pi_quarter = math.pi / 4
    
    print(f"{'項數':<8} {'部分和':<15} {'π估計':<15} {'誤差':<15} {'收斂狀況'}")
    print("-" * 65)
    
    for k in range(20):
        term = (-1) ** k / (2 * k + 1)
        pi_quarter += term
        pi_estimate = 4 * pi_quarter
        error = abs(pi_estimate - math.pi)
        
        if k in [0, 4, 9, 19] or k < 5:
            convergence_status = "收斂中..." if k < 19 else "持續收斂"
            print(f"{k+1:<8} {pi_quarter:<15.8f} {pi_estimate:<15.8f} {error:<15.2e} {convergence_status}")
    
    print()
    print("觀察: 萊布尼茲級數收斂緩慢，需要大量項才能獲得高精度")
    print("Observation: Leibniz series converges slowly, requiring many terms for high precision")


def demonstrate_practical_applications():
    """演示π的實際應用"""
    print("π的實際應用演示 (Practical Applications of π)")
    print("=" * 55)
    print()
    
    applications = [
        {
            'area': '幾何學 (Geometry)',
            'examples': [
                '圓面積: A = πr²',
                '圓周長: C = 2πr',
                '球體積: V = (4/3)πr³',
                '球表面積: S = 4πr²'
            ]
        },
        {
            'area': '物理學 (Physics)',
            'examples': [
                '簡諧運動週期: T = 2π√(m/k)',
                '交流電頻率: f = ω/(2π)',
                '量子力學波函數正規化',
                '統計力學中的相空間積分'
            ]
        },
        {
            'area': '工程學 (Engineering)',
            'examples': [
                '信號處理中的傅立葉變換',
                '數位濾波器設計',
                '控制系統分析',
                '電磁場計算'
            ]
        },
        {
            'area': '數值分析 (Numerical Analysis)',
            'examples': [
                '算法性能基準測試',
                '浮點數精度測試',
                '隨機數生成器驗證',
                '高精度計算驗證'
            ]
        }
    ]
    
    for app in applications:
        print(f"{app['area']}:")
        for example in app['examples']:
            print(f"  • {example}")
        print()


def interactive_pi_calculator():
    """互動式π計算器"""
    print("互動式π計算器 (Interactive Pi Calculator)")
    print("=" * 50)
    print()
    print("選擇計算方法:")
    print("1. 萊布尼茲級數")
    print("2. 蒙特卡羅方法")
    print("3. 顯示π的更多位數")
    print("4. π的表達式計算")
    print("0. 返回主選單")
    print()
    
    while True:
        try:
            choice = input("請選擇 (0-4): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                terms = int(input("輸入級數項數 (建議1000-50000): "))
                calculator = BasicPiCalculator()
                result = calculator.leibniz_series(terms)
                print(f"結果: π ≈ {result['result']:.10f}")
                print(f"誤差: {result['error']:.2e}")
                print(f"計算時間: {result['computation_time']:.4f}秒")
                print()
            
            elif choice == '2':
                points = int(input("輸入隨機點數量 (建議10000-1000000): "))
                calculator = BasicPiCalculator()
                result = calculator.monte_carlo_pi(points)
                print(f"結果: π ≈ {result['result']:.10f}")
                print(f"誤差: {result['error']:.2e}")
                print(f"命中率: {result['hit_rate']:.4f}")
                print(f"計算時間: {result['computation_time']:.4f}秒")
                print()
            
            elif choice == '3':
                # 使用Decimal提供更多位數
                getcontext().prec = 100
                pi_decimal = Decimal(str(math.pi))
                print(f"π的高精度值:")
                print(f"{pi_decimal}")
                print()
            
            elif choice == '4':
                expr = input("輸入包含π的表達式 (例如: 2*pi, pi**2): ").strip()
                try:
                    # 簡單的表達式評估 (安全起見，只允許基本運算)
                    expr_safe = expr.replace('pi', str(math.pi))
                    expr_safe = expr_safe.replace('^', '**')  # 支援^作為冪運算
                    
                    # 基本安全檢查
                    allowed_chars = set('0123456789+-*/.() ')
                    allowed_chars.update(str(math.pi))
                    
                    if all(c in allowed_chars for c in expr_safe):
                        result = eval(expr_safe)
                        print(f"表達式: {expr}")
                        print(f"計算結果: {result}")
                        print()
                    else:
                        print("表達式包含不允許的字符")
                        print()
                except Exception as e:
                    print(f"計算錯誤: {e}")
                    print()
            
            else:
                print("無效選擇，請重試")
                
        except ValueError:
            print("請輸入有效的數字")
        except KeyboardInterrupt:
            print("\n退出計算器")
            break
        except Exception as e:
            print(f"錯誤: {e}")


def main_menu():
    """主選單函數"""
    while True:
        print("\n" + "=" * 70)
        print("π表達式與評估互動演示 (Pi Expressions & Evaluation Interactive Demo)")
        print("=" * 70)
        print()
        print("選擇演示內容:")
        print("1. π的基本介紹")
        print("2. π的級數表達式演示")
        print("3. π計算方法比較")
        print("4. π表達式評估過程")
        print("5. π級數收斂分析")
        print("6. π的實際應用")
        print("7. 互動式π計算器")
        print("8. 完整演示 (運行所有模組)")
        print("0. 退出")
        print()
        
        try:
            choice = input("請選擇 (0-8): ").strip()
            
            if choice == '0':
                print("謝謝使用！Thank you for using!")
                break
            elif choice == '1':
                print("\n")
                display_pi_introduction()
            elif choice == '2':
                print("\n")
                demonstrate_pi_series_expressions()
            elif choice == '3':
                print("\n")
                demonstrate_pi_calculation_comparison()
            elif choice == '4':
                print("\n")
                demonstrate_pi_expression_evaluation()
            elif choice == '5':
                print("\n")
                demonstrate_pi_convergence_analysis()
            elif choice == '6':
                print("\n")
                demonstrate_practical_applications()
            elif choice == '7':
                print("\n")
                interactive_pi_calculator()
            elif choice == '8':
                print("\n全面演示開始...")
                display_pi_introduction()
                input("\n按Enter繼續...")
                demonstrate_pi_series_expressions()
                input("\n按Enter繼續...")
                demonstrate_pi_calculation_comparison()
                input("\n按Enter繼續...")
                demonstrate_pi_expression_evaluation()
                input("\n按Enter繼續...")
                demonstrate_pi_convergence_analysis()
                input("\n按Enter繼續...")
                demonstrate_practical_applications()
                print("\n全面演示完成！")
            else:
                print("無效選擇，請重試")
                
            if choice != '0':
                input("\n按Enter返回主選單...")
                
        except KeyboardInterrupt:
            print("\n\n程式中斷。再見！")
            break
        except Exception as e:
            print(f"錯誤: {e}")


if __name__ == "__main__":
    print("啟動π表達式互動演示...")
    print("Starting Pi Expressions Interactive Demo...")
    main_menu()