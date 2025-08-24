"""
Mathematical Utilities Module for Base e Computation and More
============================================================

This module provides lightweight mathematical functions and constants,
particularly focusing on Euler's number (e) and related computations,
designed to replace NumPy dependencies for basic mathematical operations.

Features:
- Euler's number (e) constant and approximations
- Exponential and logarithmic functions
- Statistical functions (mean, standard deviation, etc.)
- Trigonometric functions
- Array-like operations
"""

import math
from typing import List, Union, Optional, Iterable
from functools import reduce


# Mathematical Constants
E = math.e  # Euler's number
PI = math.pi  # Pi
TAU = 2 * math.pi  # Tau (2π)
GOLDEN_RATIO = (1 + math.sqrt(5)) / 2  # Golden ratio φ


class MathUtils:
    """
    Mathematical utilities class providing various mathematical operations
    without requiring NumPy or other heavy dependencies.
    """

    @staticmethod
    def e_approximation(terms: int = 20) -> float:
        """
        Compute Euler's number using Taylor series approximation.
        
        e = Σ(1/n!) for n=0 to ∞
        
        Args:
            terms: Number of terms in the series (default: 20)
            
        Returns:
            Approximation of Euler's number
            
        Example:
            >>> MathUtils.e_approximation(10)
            2.7182815255731922
        """
        result = 0.0
        factorial = 1
        
        for n in range(terms):
            if n > 0:
                factorial *= n
            result += 1.0 / factorial
            
        return result

    @staticmethod
    def exp(x: float, terms: int = 50) -> float:
        """
        Compute e^x using Taylor series expansion.
        
        e^x = Σ(x^n/n!) for n=0 to ∞
        
        Args:
            x: Exponent
            terms: Number of terms in the series
            
        Returns:
            e^x approximation
            
        Example:
            >>> MathUtils.exp(1.0)
            2.718281828459045
        """
        if abs(x) < 1e-10:
            return 1.0
            
        # Use math.exp for better precision when available
        if hasattr(math, 'exp'):
            return math.exp(x)
            
        # Fallback to series expansion
        result = 0.0
        term = 1.0
        
        for n in range(terms):
            result += term
            term *= x / (n + 1)
            
        return result

    @staticmethod
    def ln(x: float, precision: float = 1e-10) -> float:
        """
        Compute natural logarithm using Newton's method or built-in function.
        
        Args:
            x: Value to compute ln of (must be positive)
            precision: Precision for iterative method
            
        Returns:
            Natural logarithm of x
            
        Raises:
            ValueError: If x <= 0
            
        Example:
            >>> MathUtils.ln(MathUtils.E)
            1.0
        """
        if x <= 0:
            raise ValueError("Natural logarithm undefined for non-positive values")
            
        # Use built-in for better precision
        if hasattr(math, 'log'):
            return math.log(x)
            
        # Fallback implementation using Newton's method
        # ln(x) = y such that e^y = x
        if x == 1.0:
            return 0.0
            
        # Initial guess
        y = x - 1 if x < 2 else math.log10(x) / math.log10(E)
        
        while True:
            exp_y = MathUtils.exp(y)
            new_y = y + (x - exp_y) / exp_y
            
            if abs(new_y - y) < precision:
                break
                
            y = new_y
            
        return y

    @staticmethod
    def log(x: float, base: float = 10) -> float:
        """
        Compute logarithm with arbitrary base.
        
        log_base(x) = ln(x) / ln(base)
        
        Args:
            x: Value to compute log of
            base: Base of logarithm (default: 10)
            
        Returns:
            Logarithm of x in the given base
        """
        if base <= 0 or base == 1:
            raise ValueError("Base must be positive and not equal to 1")
            
        return MathUtils.ln(x) / MathUtils.ln(base)

    @staticmethod
    def mean(values: Iterable[Union[int, float]]) -> float:
        """
        Calculate arithmetic mean of a sequence of numbers.
        
        Args:
            values: Iterable of numeric values
            
        Returns:
            Arithmetic mean
            
        Raises:
            ValueError: If values is empty
            
        Example:
            >>> MathUtils.mean([1, 2, 3, 4, 5])
            3.0
        """
        values_list = list(values)
        if not values_list:
            raise ValueError("Cannot compute mean of empty sequence")
            
        return sum(values_list) / len(values_list)

    @staticmethod
    def std(values: Iterable[Union[int, float]], ddof: int = 0) -> float:
        """
        Calculate standard deviation.
        
        Args:
            values: Iterable of numeric values
            ddof: Delta degrees of freedom (0 for population, 1 for sample)
            
        Returns:
            Standard deviation
        """
        values_list = list(values)
        if len(values_list) <= ddof:
            raise ValueError("Not enough values for standard deviation calculation")
            
        mean_val = MathUtils.mean(values_list)
        variance = sum((x - mean_val) ** 2 for x in values_list) / (len(values_list) - ddof)
        
        return math.sqrt(variance)

    @staticmethod
    def var(values: Iterable[Union[int, float]], ddof: int = 0) -> float:
        """
        Calculate variance.
        
        Args:
            values: Iterable of numeric values
            ddof: Delta degrees of freedom
            
        Returns:
            Variance
        """
        values_list = list(values)
        if len(values_list) <= ddof:
            raise ValueError("Not enough values for variance calculation")
            
        mean_val = MathUtils.mean(values_list)
        return sum((x - mean_val) ** 2 for x in values_list) / (len(values_list) - ddof)

    @staticmethod
    def median(values: Iterable[Union[int, float]]) -> float:
        """
        Calculate median of a sequence.
        
        Args:
            values: Iterable of numeric values
            
        Returns:
            Median value
        """
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        if n == 0:
            raise ValueError("Cannot compute median of empty sequence")
            
        if n % 2 == 1:
            return float(sorted_values[n // 2])
        else:
            return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2.0

    @staticmethod
    def sum(values: Iterable[Union[int, float]]) -> Union[int, float]:
        """
        Calculate sum of values (more explicit than built-in sum).
        
        Args:
            values: Iterable of numeric values
            
        Returns:
            Sum of values
        """
        return reduce(lambda x, y: x + y, values, 0)

    @staticmethod
    def prod(values: Iterable[Union[int, float]]) -> Union[int, float]:
        """
        Calculate product of values.
        
        Args:
            values: Iterable of numeric values
            
        Returns:
            Product of values
        """
        return reduce(lambda x, y: x * y, values, 1)

    @staticmethod
    def max(values: Iterable[Union[int, float]]) -> Union[int, float]:
        """
        Find maximum value.
        
        Args:
            values: Iterable of numeric values
            
        Returns:
            Maximum value
        """
        return max(values)

    @staticmethod
    def min(values: Iterable[Union[int, float]]) -> Union[int, float]:
        """
        Find minimum value.
        
        Args:
            values: Iterable of numeric values
            
        Returns:
            Minimum value
        """
        return min(values)

    @staticmethod
    def abs(x: Union[int, float]) -> Union[int, float]:
        """
        Absolute value.
        
        Args:
            x: Numeric value
            
        Returns:
            Absolute value of x
        """
        return abs(x)

    @staticmethod
    def power(base: float, exponent: float) -> float:
        """
        Compute base^exponent using mathematical properties.
        
        For positive base: base^exp = e^(exp * ln(base))
        
        Args:
            base: Base value
            exponent: Exponent value
            
        Returns:
            base^exponent
        """
        if base == 0:
            return 0.0 if exponent > 0 else float('inf')
        if base < 0 and not isinstance(exponent, int):
            raise ValueError("Cannot raise negative number to non-integer power")
            
        if hasattr(math, 'pow'):
            return math.pow(base, exponent)
            
        # Use exponential and logarithm properties
        if base > 0:
            return MathUtils.exp(exponent * MathUtils.ln(base))
        else:
            # Handle negative base with integer exponent
            return (-1) ** exponent * ((-base) ** exponent)

    @staticmethod
    def sigmoid(x: float) -> float:
        """
        Compute sigmoid function: 1 / (1 + e^(-x))
        
        Args:
            x: Input value
            
        Returns:
            Sigmoid of x
        """
        if x > 500:  # Prevent overflow
            return 1.0
        elif x < -500:
            return 0.0
        else:
            return 1.0 / (1.0 + MathUtils.exp(-x))

    @staticmethod
    def tanh(x: float) -> float:
        """
        Compute hyperbolic tangent: (e^x - e^(-x)) / (e^x + e^(-x))
        
        Args:
            x: Input value
            
        Returns:
            Hyperbolic tangent of x
        """
        if hasattr(math, 'tanh'):
            return math.tanh(x)
            
        if x > 500:
            return 1.0
        elif x < -500:
            return -1.0
        else:
            exp_x = MathUtils.exp(x)
            exp_neg_x = MathUtils.exp(-x)
            return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)

    @staticmethod
    def softmax(values: List[float]) -> List[float]:
        """
        Compute softmax function.
        
        Args:
            values: List of numeric values
            
        Returns:
            List of softmax probabilities
        """
        if not values:
            return []
            
        # Subtract max for numerical stability
        max_val = max(values)
        exp_values = [MathUtils.exp(x - max_val) for x in values]
        sum_exp = sum(exp_values)
        
        return [x / sum_exp for x in exp_values]

    @staticmethod
    def clamp(x: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """
        Clamp value to a range.
        
        Args:
            x: Value to clamp
            min_val: Minimum value
            max_val: Maximum value
            
        Returns:
            Clamped value
        """
        return max(min_val, min(max_val, x))

    @staticmethod
    def normalize(values: List[float], min_val: float = 0.0, max_val: float = 1.0) -> List[float]:
        """
        Normalize values to a specified range.
        
        Args:
            values: List of values to normalize
            min_val: Target minimum value
            max_val: Target maximum value
            
        Returns:
            Normalized values
        """
        if not values:
            return []
            
        val_min = min(values)
        val_max = max(values)
        
        if val_max == val_min:
            return [min_val] * len(values)
            
        scale = (max_val - min_val) / (val_max - val_min)
        return [min_val + (x - val_min) * scale for x in values]


# Convenience functions for backward compatibility
def mean(values: Iterable[Union[int, float]]) -> float:
    """Convenience function for arithmetic mean."""
    return MathUtils.mean(values)


def std(values: Iterable[Union[int, float]], ddof: int = 0) -> float:
    """Convenience function for standard deviation."""
    return MathUtils.std(values, ddof)


def exp(x: float) -> float:
    """Convenience function for exponential."""
    return MathUtils.exp(x)


def ln(x: float) -> float:
    """Convenience function for natural logarithm."""
    return MathUtils.ln(x)


def log(x: float, base: float = 10) -> float:
    """Convenience function for logarithm."""
    return MathUtils.log(x, base)


# Additional mathematical constants and functions related to base e
class BaseEComputations:
    """
    Specialized class for computations involving Euler's number (e).
    """
    
    @staticmethod
    def e_power_series(x: float, terms: int = 50) -> float:
        """
        Compute e^x using power series expansion with high precision.
        
        Args:
            x: Exponent
            terms: Number of terms
            
        Returns:
            e^x computed using series
        """
        return MathUtils.exp(x, terms)
    
    @staticmethod
    def e_continued_fraction(iterations: int = 20) -> float:
        """
        Compute e using continued fraction representation.
        
        e = [2; 1, 2, 1, 1, 4, 1, 1, 6, 1, 1, 8, ...]
        
        Args:
            iterations: Number of iterations
            
        Returns:
            Approximation of e
        """
        if iterations <= 0:
            return 2.0
            
        # Build continued fraction from bottom up
        result = 0.0
        
        for i in range(iterations - 1, -1, -1):
            if i == 0:
                a_i = 2
            elif i % 3 == 2:
                a_i = 2 * ((i + 1) // 3)
            else:
                a_i = 1
                
            if i == iterations - 1:
                result = a_i
            else:
                result = a_i + 1.0 / result
                
        return result
    
    @staticmethod
    def natural_exp_integral(x: float, terms: int = 100) -> float:
        """
        Compute the exponential integral Ei(x) for positive x.
        
        Ei(x) = ∫_{-∞}^x (e^t/t) dt
        
        Args:
            x: Input value (should be positive)
            terms: Number of terms in series
            
        Returns:
            Exponential integral approximation
        """
        if x <= 0:
            raise ValueError("Exponential integral undefined for non-positive values")
            
        # Use series expansion for small x
        if x < 1:
            # Ei(x) = γ + ln(x) + Σ(x^n / (n * n!)) for n=1,2,3...
            gamma = 0.5772156649015329  # Euler-Mascheroni constant
            result = gamma + MathUtils.ln(x)
            
            term = x
            for n in range(1, terms + 1):
                result += term / (n * math.factorial(n))
                term *= x
                
            return result
        else:
            # Use asymptotic expansion for large x
            result = MathUtils.exp(x) / x
            correction = 0.0
            term = 1.0
            
            for n in range(1, terms + 1):
                term *= n / x
                correction += term
                
            return result * (1 + correction)
    
    @staticmethod
    def euler_mascheroni_constant(terms: int = 1000) -> float:
        """
        Approximate the Euler-Mascheroni constant γ.
        
        γ = lim_{n→∞} (Σ(1/k) - ln(n)) for k=1 to n
        
        Args:
            terms: Number of terms to use
            
        Returns:
            Approximation of γ
        """
        harmonic_sum = sum(1.0 / k for k in range(1, terms + 1))
        return harmonic_sum - MathUtils.ln(terms)


if __name__ == "__main__":
    # Demo and basic tests
    print("Mathematical Utilities Demo")
    print("=" * 50)
    
    print(f"Euler's number (e): {E}")
    print(f"e approximation (20 terms): {MathUtils.e_approximation(20)}")
    print(f"e continued fraction: {BaseEComputations.e_continued_fraction(20)}")
    print(f"exp(1): {MathUtils.exp(1.0)}")
    print(f"ln(e): {MathUtils.ln(E)}")
    
    print(f"\nStatistical functions:")
    test_data = [1, 2, 3, 4, 5]
    print(f"Mean of {test_data}: {MathUtils.mean(test_data)}")
    print(f"Std of {test_data}: {MathUtils.std(test_data)}")
    print(f"Median of {test_data}: {MathUtils.median(test_data)}")
    
    print(f"\nOther functions:")
    print(f"sigmoid(0): {MathUtils.sigmoid(0)}")
    print(f"tanh(1): {MathUtils.tanh(1)}")
    print(f"softmax([1,2,3]): {MathUtils.softmax([1,2,3])}")