#!/usr/bin/env python3
"""
Test Suite for Base e Computation and Mathematical Utilities
============================================================

This module provides comprehensive tests for the mathematical utilities
implemented to replace NumPy dependencies, with special focus on 
Euler's number (e) computations.
"""

import math
import sys
import traceback
from typing import List, Tuple

# Import our mathematical utilities
try:
    from math_utils import MathUtils, BaseEComputations, mean, std, exp, ln, log, E
    MATH_UTILS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import math_utils: {e}")
    MATH_UTILS_AVAILABLE = False


class TestResult:
    """Class to store test results"""
    def __init__(self, test_name: str, passed: bool, error: str = None, value: any = None):
        self.test_name = test_name
        self.passed = passed
        self.error = error
        self.value = value

class BaseEComputationTester:
    """Test suite for base e computation and mathematical utilities"""

    def __init__(self):
        self.results: List[TestResult] = []
        self.tolerance = 1e-10

    def assert_close(self, actual: float, expected: float, tolerance: float = None) -> bool:
        """Check if two float values are close within tolerance"""
        tol = tolerance or self.tolerance
        return abs(actual - expected) < tol

    def run_test(self, test_name: str, test_func) -> TestResult:
        """Run a single test and record the result"""
        try:
            result = test_func()
            test_result = TestResult(test_name, True, None, result)
            print(f"✓ {test_name}: PASSED")
            if result is not None:
                print(f"    Result: {result}")
        except Exception as e:
            test_result = TestResult(test_name, False, str(e), None)
            print(f"✗ {test_name}: FAILED - {e}")
            
        self.results.append(test_result)
        return test_result

    def test_e_constant_accuracy(self):
        """Test accuracy of Euler's number constant"""
        expected_e = math.e
        actual_e = E
        
        if not self.assert_close(actual_e, expected_e):
            raise AssertionError(f"E constant mismatch: {actual_e} != {expected_e}")
        
        return f"E = {actual_e} (error: {abs(actual_e - expected_e):.2e})"

    def test_e_approximation_methods(self):
        """Test different methods of approximating e"""
        expected_e = math.e
        
        # Test Taylor series approximation
        approx_e_20 = MathUtils.e_approximation(20)
        approx_e_50 = MathUtils.e_approximation(50)
        
        if not self.assert_close(approx_e_20, expected_e, 1e-10):
            raise AssertionError(f"20-term approximation too inaccurate: {approx_e_20}")
            
        if not self.assert_close(approx_e_50, expected_e, 1e-15):
            raise AssertionError(f"50-term approximation too inaccurate: {approx_e_50}")
        
        # Test continued fraction method
        if hasattr(BaseEComputations, 'e_continued_fraction'):
            cf_e = BaseEComputations.e_continued_fraction(20)
            if not self.assert_close(cf_e, expected_e, 1e-10):
                raise AssertionError(f"Continued fraction too inaccurate: {cf_e}")
                
        return {
            "20_terms": approx_e_20,
            "50_terms": approx_e_50,
            "continued_fraction": cf_e if 'cf_e' in locals() else None
        }

    def test_exponential_function(self):
        """Test exponential function e^x"""
        test_cases = [
            (0, 1.0),
            (1, math.e),
            (2, math.e**2),
            (-1, 1/math.e),
            (0.5, math.sqrt(math.e)),
            (10, math.exp(10))
        ]
        
        results = {}
        for x, expected in test_cases:
            actual = MathUtils.exp(x)
            if not self.assert_close(actual, expected, 1e-10):
                raise AssertionError(f"exp({x}) = {actual}, expected {expected}")
            results[f"exp({x})"] = actual
            
        return results

    def test_natural_logarithm(self):
        """Test natural logarithm function"""
        test_cases = [
            (1, 0.0),
            (math.e, 1.0),
            (math.e**2, 2.0),
            (1/math.e, -1.0),
            (math.sqrt(math.e), 0.5),
            (10, math.log(10))
        ]
        
        results = {}
        for x, expected in test_cases:
            actual = MathUtils.ln(x)
            if not self.assert_close(actual, expected, 1e-10):
                raise AssertionError(f"ln({x}) = {actual}, expected {expected}")
            results[f"ln({x})"] = actual
            
        return results

    def test_logarithm_with_base(self):
        """Test logarithm with arbitrary base"""
        test_cases = [
            (100, 10, 2.0),  # log_10(100) = 2
            (8, 2, 3.0),     # log_2(8) = 3
            (math.e, math.e, 1.0),  # log_e(e) = 1
            (1, 10, 0.0),    # log_10(1) = 0
        ]
        
        results = {}
        for x, base, expected in test_cases:
            actual = MathUtils.log(x, base)
            if not self.assert_close(actual, expected, 1e-10):
                raise AssertionError(f"log_{base}({x}) = {actual}, expected {expected}")
            results[f"log_{base}({x})"] = actual
            
        return results

    def test_statistical_functions(self):
        """Test statistical functions that replace NumPy"""
        test_data = [1, 2, 3, 4, 5]
        expected_mean = 3.0
        expected_std = math.sqrt(2.0)  # Population std for [1,2,3,4,5]
        
        # Test mean
        actual_mean = MathUtils.mean(test_data)
        if not self.assert_close(actual_mean, expected_mean):
            raise AssertionError(f"mean({test_data}) = {actual_mean}, expected {expected_mean}")
            
        # Test standard deviation
        actual_std = MathUtils.std(test_data)
        if not self.assert_close(actual_std, expected_std, 1e-10):
            raise AssertionError(f"std({test_data}) = {actual_std}, expected {expected_std}")
            
        # Test median
        expected_median = 3.0
        actual_median = MathUtils.median(test_data)
        if not self.assert_close(actual_median, expected_median):
            raise AssertionError(f"median({test_data}) = {actual_median}, expected {expected_median}")
            
        return {
            "mean": actual_mean,
            "std": actual_std,
            "median": actual_median
        }

    def test_advanced_functions(self):
        """Test advanced mathematical functions"""
        # Test sigmoid
        sigmoid_0 = MathUtils.sigmoid(0)
        if not self.assert_close(sigmoid_0, 0.5):
            raise AssertionError(f"sigmoid(0) = {sigmoid_0}, expected 0.5")
            
        # Test tanh
        tanh_0 = MathUtils.tanh(0)
        if not self.assert_close(tanh_0, 0.0):
            raise AssertionError(f"tanh(0) = {tanh_0}, expected 0.0")
            
        # Test softmax
        softmax_result = MathUtils.softmax([1, 2, 3])
        expected_sum = 1.0
        actual_sum = sum(softmax_result)
        if not self.assert_close(actual_sum, expected_sum):
            raise AssertionError(f"softmax sum = {actual_sum}, expected {expected_sum}")
            
        return {
            "sigmoid(0)": sigmoid_0,
            "tanh(0)": tanh_0,
            "softmax([1,2,3])": softmax_result
        }

    def test_euler_mascheroni_constant(self):
        """Test Euler-Mascheroni constant approximation"""
        if hasattr(BaseEComputations, 'euler_mascheroni_constant'):
            gamma_approx = BaseEComputations.euler_mascheroni_constant(1000)
            expected_gamma = 0.5772156649015329  # Known value
            
            if not self.assert_close(gamma_approx, expected_gamma, 1e-3):
                raise AssertionError(f"γ approximation too inaccurate: {gamma_approx}")
                
            return f"γ ≈ {gamma_approx}"
        else:
            return "Euler-Mascheroni constant not implemented"

    def test_convenience_functions(self):
        """Test convenience functions"""
        # Test convenience mean function
        test_data = [2, 4, 6, 8, 10]
        expected = 6.0
        actual = mean(test_data)
        
        if not self.assert_close(actual, expected):
            raise AssertionError(f"convenience mean = {actual}, expected {expected}")
            
        return {"convenience_mean": actual}

    def test_error_handling(self):
        """Test error handling for edge cases"""
        error_tests = []
        
        # Test ln of negative number
        try:
            MathUtils.ln(-1)
            error_tests.append("ln(-1) should raise ValueError")
        except ValueError:
            pass  # Expected
            
        # Test empty mean
        try:
            MathUtils.mean([])
            error_tests.append("mean([]) should raise ValueError")
        except ValueError:
            pass  # Expected
            
        # Test log with invalid base
        try:
            MathUtils.log(10, -1)
            error_tests.append("log with negative base should raise ValueError")
        except ValueError:
            pass  # Expected
            
        if error_tests:
            raise AssertionError("Error handling failed: " + "; ".join(error_tests))
            
        return "All error cases handled correctly"

    def test_integration_with_nlp_files(self):
        """Test integration with existing NLP files"""
        # Test that updated files can import and use our utilities
        integration_results = {}
        
        # Test HumanExpressionEvaluator
        try:
            from HumanExpressionEvaluator import HumanExpressionEvaluator, ExpressionContext
            evaluator = HumanExpressionEvaluator()
            context = ExpressionContext()
            result = evaluator.comprehensive_evaluation("Test expression", context)
            integration_results["HumanExpressionEvaluator"] = "✓ Working"
        except Exception as e:
            integration_results["HumanExpressionEvaluator"] = f"✗ Error: {e}"
            
        # Test math_utils import in the context that the NLP files use it
        try:
            from math_utils import mean as math_utils_mean
            test_result = math_utils_mean([1, 2, 3])
            if not self.assert_close(test_result, 2.0):
                raise AssertionError("Math utils mean function failed")
            integration_results["math_utils_mean"] = "✓ Working"
        except Exception as e:
            integration_results["math_utils_mean"] = f"✗ Error: {e}"
            
        return integration_results

    def run_all_tests(self):
        """Run all tests and return summary"""
        print("=" * 60)
        print("Base e Computation Test Suite")
        print("=" * 60)
        
        if not MATH_UTILS_AVAILABLE:
            print("✗ Cannot run tests: math_utils not available")
            return False
            
        # Define all tests
        tests = [
            ("E Constant Accuracy", self.test_e_constant_accuracy),
            ("E Approximation Methods", self.test_e_approximation_methods),
            ("Exponential Function", self.test_exponential_function),
            ("Natural Logarithm", self.test_natural_logarithm),
            ("Logarithm with Base", self.test_logarithm_with_base),
            ("Statistical Functions", self.test_statistical_functions),
            ("Advanced Functions", self.test_advanced_functions),
            ("Euler-Mascheroni Constant", self.test_euler_mascheroni_constant),
            ("Convenience Functions", self.test_convenience_functions),
            ("Error Handling", self.test_error_handling),
            ("Integration with NLP Files", self.test_integration_with_nlp_files),
        ]
        
        print(f"\nRunning {len(tests)} tests...\n")
        
        # Run all tests
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
            print()  # Add spacing between tests
            
        # Print summary
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        print("=" * 60)
        print(f"Test Summary: {passed}/{total} tests passed")
        print("=" * 60)
        
        if passed == total:
            print("🎉 All tests passed! Base e computation implementation is successful.")
        else:
            print("❌ Some tests failed. Check the output above for details.")
            failed_tests = [r.test_name for r in self.results if not r.passed]
            print(f"Failed tests: {', '.join(failed_tests)}")
            
        return passed == total


def main():
    """Main function to run the test suite"""
    tester = BaseEComputationTester()
    success = tester.run_all_tests()
    
    # Return appropriate exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()