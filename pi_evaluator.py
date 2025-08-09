#!/usr/bin/env python3
"""
π表達式評估器 (Pi Expression Evaluator)

這個模組提供了多種計算π的方法和表達式評估功能，展示了不同數學級數和演算法
如何被用來近似這個重要的數學常數。

This module provides various methods for calculating π and expression evaluation 
capabilities, demonstrating how different mathematical series and algorithms 
can be used to approximate this important mathematical constant.
"""

import math
import time
from typing import Dict, List, Tuple, Any, Optional
from decimal import Decimal, getcontext
import random


class PiExpressionEvaluator:
    """π表達式評估器類別"""
    
    def __init__(self, precision: int = 50):
        """
        初始化π評估器
        
        Args:
            precision: 計算精度（小數位數）
        """
        self.precision = precision
        getcontext().prec = precision + 10  # 額外精度以避免捨入誤差
    
    def leibniz_series(self, terms: int) -> Dict[str, Any]:
        """
        萊布尼茲級數計算π
        π/4 = 1 - 1/3 + 1/5 - 1/7 + 1/9 - ...
        
        Args:
            terms: 級數項數
            
        Returns:
            包含結果和元資訊的字典
        """
        start_time = time.time()
        
        pi_quarter = Decimal(0)
        for k in range(terms):
            sign = (-1) ** k
            term = Decimal(sign) / Decimal(2 * k + 1)
            pi_quarter += term
        
        pi_approx = 4 * pi_quarter
        
        computation_time = time.time() - start_time
        actual_pi = Decimal(str(math.pi))
        error = abs(pi_approx - actual_pi)
        
        return {
            'method': 'Leibniz Series',
            'expression': 'π = 4 × Σ(k=0 to n) [(-1)^k / (2k+1)]',
            'terms_used': terms,
            'result': float(pi_approx),
            'pi_approximation': str(pi_approx)[:self.precision + 2],
            'error': float(error),
            'computation_time': computation_time,
            'convergence_rate': 'Linear (slow)',
            'decimal_accuracy': self._count_accurate_digits(pi_approx, actual_pi)
        }
    
    def nilakantha_series(self, terms: int) -> Dict[str, Any]:
        """
        尼爾森級數計算π
        π = 3 + 4/(2×3×4) - 4/(4×5×6) + 4/(6×7×8) - ...
        
        Args:
            terms: 級數項數
            
        Returns:
            包含結果和元資訊的字典
        """
        start_time = time.time()
        
        pi_approx = Decimal(3)
        
        for k in range(1, terms + 1):
            sign = (-1) ** (k + 1)
            denominator = (2 * k) * (2 * k + 1) * (2 * k + 2)
            term = Decimal(4 * sign) / Decimal(denominator)
            pi_approx += term
        
        computation_time = time.time() - start_time
        actual_pi = Decimal(str(math.pi))
        error = abs(pi_approx - actual_pi)
        
        return {
            'method': 'Nilakantha Series',
            'expression': 'π = 3 + 4 × Σ(k=1 to n) [(-1)^(k+1) / ((2k)(2k+1)(2k+2))]',
            'terms_used': terms,
            'result': float(pi_approx),
            'pi_approximation': str(pi_approx)[:self.precision + 2],
            'error': float(error),
            'computation_time': computation_time,
            'convergence_rate': 'Faster than Leibniz',
            'decimal_accuracy': self._count_accurate_digits(pi_approx, actual_pi)
        }
    
    def machin_formula(self, terms_per_arctan: int = 100) -> Dict[str, Any]:
        """
        馬欽公式計算π
        π/4 = 4×arctan(1/5) - arctan(1/239)
        
        Args:
            terms_per_arctan: 每個反正切函數使用的級數項數
            
        Returns:
            包含結果和元資訊的字典
        """
        start_time = time.time()
        
        # arctan級數: arctan(x) = x - x³/3 + x⁵/5 - x⁷/7 + ...
        def arctan_series(x: Decimal, terms: int) -> Decimal:
            result = Decimal(0)
            x_power = x
            x_squared = x * x
            
            for n in range(terms):
                sign = (-1) ** n
                coefficient = Decimal(1) / Decimal(2 * n + 1)
                result += sign * coefficient * x_power
                x_power *= x_squared
            
            return result
        
        arctan_1_5 = arctan_series(Decimal(1) / Decimal(5), terms_per_arctan)
        arctan_1_239 = arctan_series(Decimal(1) / Decimal(239), terms_per_arctan)
        
        pi_quarter = 4 * arctan_1_5 - arctan_1_239
        pi_approx = 4 * pi_quarter
        
        computation_time = time.time() - start_time
        actual_pi = Decimal(str(math.pi))
        error = abs(pi_approx - actual_pi)
        
        return {
            'method': 'Machin Formula',
            'expression': 'π/4 = 4×arctan(1/5) - arctan(1/239)',
            'terms_per_arctan': terms_per_arctan,
            'result': float(pi_approx),
            'pi_approximation': str(pi_approx)[:self.precision + 2],
            'error': float(error),
            'computation_time': computation_time,
            'convergence_rate': 'Fast (exponential in series)',
            'decimal_accuracy': self._count_accurate_digits(pi_approx, actual_pi)
        }
    
    def agm_algorithm(self, iterations: int = 10) -> Dict[str, Any]:
        """
        AGM演算法計算π (Gauss-Legendre Algorithm)
        
        Args:
            iterations: 迭代次數
            
        Returns:
            包含結果和元資訊的字典
        """
        start_time = time.time()
        
        a = Decimal(1)
        b = Decimal(1) / Decimal(2).sqrt()
        t = Decimal(1) / Decimal(4)
        p = Decimal(1)
        
        for _ in range(iterations):
            a_new = (a + b) / 2
            b_new = (a * b).sqrt()
            t_new = t - p * (a - a_new) ** 2
            p_new = 2 * p
            
            a, b, t, p = a_new, b_new, t_new, p_new
        
        pi_approx = (a + b) ** 2 / (4 * t)
        
        computation_time = time.time() - start_time
        actual_pi = Decimal(str(math.pi))
        error = abs(pi_approx - actual_pi)
        
        return {
            'method': 'AGM Algorithm (Gauss-Legendre)',
            'expression': 'Iterative: aₙ₊₁=(aₙ+bₙ)/2, bₙ₊₁=√(aₙbₙ), etc.',
            'iterations': iterations,
            'result': float(pi_approx),
            'pi_approximation': str(pi_approx)[:self.precision + 2],
            'error': float(error),
            'computation_time': computation_time,
            'convergence_rate': 'Quadratic (very fast)',
            'decimal_accuracy': self._count_accurate_digits(pi_approx, actual_pi)
        }
    
    def monte_carlo_pi(self, n_points: int, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        蒙特卡羅方法計算π
        
        Args:
            n_points: 隨機點數量
            seed: 隨機種子
            
        Returns:
            包含結果和元資訊的字典
        """
        start_time = time.time()
        
        if seed is not None:
            random.seed(seed)
        
        inside_circle = 0
        points_inside = []  # 儲存一些點以供視覺化（如果需要）
        
        for i in range(n_points):
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            
            if x * x + y * y <= 1:
                inside_circle += 1
                if len(points_inside) < 100:  # 只儲存前100個點
                    points_inside.append((x, y))
        
        pi_approx = 4 * inside_circle / n_points
        
        computation_time = time.time() - start_time
        actual_pi = math.pi
        error = abs(pi_approx - actual_pi)
        
        # 計算標準誤差
        p = inside_circle / n_points  # 命中率
        standard_error = 4 * math.sqrt(p * (1 - p) / n_points)
        
        return {
            'method': 'Monte Carlo Method',
            'expression': 'π ≈ 4 × (points inside unit circle) / (total points)',
            'points_used': n_points,
            'points_inside_circle': inside_circle,
            'hit_rate': inside_circle / n_points,
            'result': pi_approx,
            'pi_approximation': f"{pi_approx:.{min(10, self.precision)}f}",
            'error': error,
            'standard_error': standard_error,
            'computation_time': computation_time,
            'convergence_rate': 'Probabilistic (√n)',
            'decimal_accuracy': self._count_accurate_digits(Decimal(str(pi_approx)), Decimal(str(actual_pi)))
        }
    
    def wallis_product(self, terms: int) -> Dict[str, Any]:
        """
        沃利斯乘積公式計算π
        π/2 = ∏(k=1 to ∞) [(2k)²]/[(2k-1)(2k+1)]
        
        Args:
            terms: 乘積項數
            
        Returns:
            包含結果和元資訊的字典
        """
        start_time = time.time()
        
        product = Decimal(1)
        
        for k in range(1, terms + 1):
            numerator = (2 * k) ** 2
            denominator = (2 * k - 1) * (2 * k + 1)
            term = Decimal(numerator) / Decimal(denominator)
            product *= term
        
        pi_approx = 2 * product
        
        computation_time = time.time() - start_time
        actual_pi = Decimal(str(math.pi))
        error = abs(pi_approx - actual_pi)
        
        return {
            'method': 'Wallis Product',
            'expression': 'π/2 = ∏(k=1 to n) [(2k)²]/[(2k-1)(2k+1)]',
            'terms_used': terms,
            'result': float(pi_approx),
            'pi_approximation': str(pi_approx)[:self.precision + 2],
            'error': float(error),
            'computation_time': computation_time,
            'convergence_rate': 'Slow (logarithmic)',
            'decimal_accuracy': self._count_accurate_digits(pi_approx, actual_pi)
        }
    
    def bbp_formula(self, terms: int) -> Dict[str, Any]:
        """
        BBP公式計算π
        π = Σ(k=0 to ∞) [1/16^k × (4/(8k+1) - 2/(8k+4) - 1/(8k+5) - 1/(8k+6))]
        
        Args:
            terms: 級數項數
            
        Returns:
            包含結果和元資訊的字典
        """
        start_time = time.time()
        
        pi_approx = Decimal(0)
        
        for k in range(terms):
            power_16 = Decimal(16) ** k
            term = (Decimal(4) / Decimal(8 * k + 1) - 
                   Decimal(2) / Decimal(8 * k + 4) - 
                   Decimal(1) / Decimal(8 * k + 5) - 
                   Decimal(1) / Decimal(8 * k + 6)) / power_16
            pi_approx += term
        
        computation_time = time.time() - start_time
        actual_pi = Decimal(str(math.pi))
        error = abs(pi_approx - actual_pi)
        
        return {
            'method': 'BBP Formula',
            'expression': 'π = Σ(k=0 to n) [1/16^k × (4/(8k+1) - 2/(8k+4) - 1/(8k+5) - 1/(8k+6))]',
            'terms_used': terms,
            'result': float(pi_approx),
            'pi_approximation': str(pi_approx)[:self.precision + 2],
            'error': float(error),
            'computation_time': computation_time,
            'convergence_rate': 'Fast (can compute specific digits)',
            'decimal_accuracy': self._count_accurate_digits(pi_approx, actual_pi),
            'special_property': 'Can compute π in hexadecimal digit by digit'
        }
    
    def evaluate_all_methods(self, quick_demo: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        評估所有方法並比較結果
        
        Args:
            quick_demo: 是否使用較小的參數進行快速演示
            
        Returns:
            包含所有方法結果的字典
        """
        if quick_demo:
            params = {
                'leibniz_terms': 10000,
                'nilakantha_terms': 1000,
                'machin_terms': 50,
                'agm_iterations': 5,
                'monte_carlo_points': 100000,
                'wallis_terms': 1000,
                'bbp_terms': 100
            }
        else:
            params = {
                'leibniz_terms': 100000,
                'nilakantha_terms': 10000,
                'machin_terms': 100,
                'agm_iterations': 10,
                'monte_carlo_points': 1000000,
                'wallis_terms': 10000,
                'bbp_terms': 1000
            }
        
        results = {}
        
        print("正在評估π的各種表達式和計算方法...")
        print("Evaluating various expressions and calculation methods for π...")
        print()
        
        # 評估每種方法
        methods = [
            ('leibniz', lambda: self.leibniz_series(params['leibniz_terms'])),
            ('nilakantha', lambda: self.nilakantha_series(params['nilakantha_terms'])),
            ('machin', lambda: self.machin_formula(params['machin_terms'])),
            ('agm', lambda: self.agm_algorithm(params['agm_iterations'])),
            ('monte_carlo', lambda: self.monte_carlo_pi(params['monte_carlo_points'], seed=42)),
            ('wallis', lambda: self.wallis_product(params['wallis_terms'])),
            ('bbp', lambda: self.bbp_formula(params['bbp_terms']))
        ]
        
        for method_name, method_func in methods:
            print(f"計算中: {method_name}...")
            try:
                result = method_func()
                results[method_name] = result
            except Exception as e:
                print(f"錯誤於 {method_name}: {e}")
                results[method_name] = {'error': str(e)}
        
        # 添加比較分析
        results['comparison'] = self._compare_methods(results)
        
        return results
    
    def _count_accurate_digits(self, approx: Decimal, actual: Decimal) -> int:
        """計算近似值的準確位數"""
        if approx == actual:
            return self.precision
        
        error = abs(approx - actual)
        if error == 0:
            return self.precision
        
        # 計算準確的小數位數
        accurate_digits = -int(error.log10()) - 1
        return max(0, min(accurate_digits, self.precision))
    
    def _compare_methods(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """比較各種方法的性能"""
        valid_results = {k: v for k, v in results.items() 
                        if 'error' not in v and 'result' in v}
        
        if not valid_results:
            return {'error': 'No valid results to compare'}
        
        # 找出最準確的方法
        most_accurate = min(valid_results.items(), 
                           key=lambda x: x[1].get('error', float('inf')))
        
        # 找出最快的方法
        fastest = min(valid_results.items(), 
                     key=lambda x: x[1].get('computation_time', float('inf')))
        
        # 計算平均誤差
        avg_error = sum(r.get('error', 0) for r in valid_results.values()) / len(valid_results)
        
        # 效率分析
        efficiency_scores = {}
        for name, result in valid_results.items():
            accuracy = result.get('decimal_accuracy', 0)
            time_taken = result.get('computation_time', 1)
            efficiency_scores[name] = accuracy / time_taken if time_taken > 0 else 0
        
        best_efficiency = max(efficiency_scores.items(), key=lambda x: x[1])
        
        return {
            'most_accurate_method': {
                'name': most_accurate[0],
                'error': most_accurate[1].get('error'),
                'accuracy': most_accurate[1].get('decimal_accuracy')
            },
            'fastest_method': {
                'name': fastest[0],
                'time': fastest[1].get('computation_time'),
                'result': fastest[1].get('result')
            },
            'most_efficient_method': {
                'name': best_efficiency[0],
                'efficiency_score': best_efficiency[1],
                'description': 'accuracy per second'
            },
            'average_error': avg_error,
            'total_methods_evaluated': len(valid_results),
            'actual_pi': math.pi,
            'precision_used': self.precision
        }


def demonstrate_pi_expressions():
    """演示π表達式評估器的功能"""
    print("=" * 70)
    print("π表達式與評估演示 (Pi Expressions and Evaluation Demonstration)")
    print("=" * 70)
    print()
    
    # 初始化評估器
    evaluator = PiExpressionEvaluator(precision=20)
    
    # 運行所有方法的評估
    results = evaluator.evaluate_all_methods(quick_demo=True)
    
    print("\n" + "=" * 70)
    print("評估結果 (Evaluation Results)")
    print("=" * 70)
    
    # 顯示每種方法的結果
    for method_name, result in results.items():
        if method_name == 'comparison':
            continue
            
        if 'error' in result:
            print(f"\n{method_name.upper()} - 錯誤: {result['error']}")
            continue
        
        print(f"\n{result['method'].upper()}")
        print("-" * 50)
        print(f"表達式: {result['expression']}")
        print(f"π近似值: {result['pi_approximation']}")
        print(f"誤差: {result['error']:.2e}")
        print(f"準確位數: {result['decimal_accuracy']}")
        print(f"計算時間: {result['computation_time']:.4f}秒")
        print(f"收斂率: {result['convergence_rate']}")
        
        # 顯示方法特定的資訊
        if 'terms_used' in result:
            print(f"使用項數: {result['terms_used']}")
        elif 'iterations' in result:
            print(f"迭代次數: {result['iterations']}")
        elif 'points_used' in result:
            print(f"使用點數: {result['points_used']}")
            print(f"命中率: {result['hit_rate']:.4f}")
    
    # 顯示比較結果
    if 'comparison' in results and 'error' not in results['comparison']:
        print("\n" + "=" * 70)
        print("方法比較分析 (Method Comparison Analysis)")
        print("=" * 70)
        
        comp = results['comparison']
        print(f"最準確方法: {comp['most_accurate_method']['name']}")
        print(f"  - 準確位數: {comp['most_accurate_method']['accuracy']}")
        print(f"  - 誤差: {comp['most_accurate_method']['error']:.2e}")
        
        print(f"\n最快方法: {comp['fastest_method']['name']}")
        print(f"  - 計算時間: {comp['fastest_method']['time']:.4f}秒")
        
        print(f"\n最高效方法: {comp['most_efficient_method']['name']}")
        print(f"  - 效率分數: {comp['most_efficient_method']['efficiency_score']:.2f}")
        
        print(f"\n平均誤差: {comp['average_error']:.2e}")
        print(f"評估方法總數: {comp['total_methods_evaluated']}")
        print(f"實際π值: {comp['actual_pi']}")
    
    print("\n" + "=" * 70)
    print("演示完成 (Demonstration Complete)")
    print("=" * 70)
    print()
    print("這個演示展示了多種計算π的數學表達式和方法，")
    print("每種方法都有其獨特的收斂特性和計算效率。")
    print()
    print("This demonstration shows various mathematical expressions and methods")
    print("for calculating π, each with its unique convergence properties and efficiency.")


if __name__ == "__main__":
    demonstrate_pi_expressions()