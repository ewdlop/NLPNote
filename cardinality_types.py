#!/usr/bin/env python3
"""
Cardinality Types and Operations for NLP Evaluation

This module provides explicit types and operations for handling cardinalities
(integral counts) and derived non-integral metrics (ratios, probabilities).

The key distinction is:
- Cardinality: How many discrete items (always integral)
- Ratio/Proportion: Derived from cardinalities (inherently non-integral)

This addresses the issue of "Non-integral cardinality represent by integral?"
by providing clear type separation and validation.
"""

from typing import Union, NewType, Optional
from dataclasses import dataclass
import math

# Explicit type aliases for clarity
Cardinality = NewType('Cardinality', int)  # Always integral
Ratio = NewType('Ratio', float)  # Always non-integral (0.0 to 1.0)
Probability = NewType('Probability', float)  # Always non-integral (0.0 to 1.0)
Score = NewType('Score', float)  # Non-integral evaluation metric


@dataclass
class CardinalityMetrics:
    """Container for cardinality-based metrics with explicit types"""
    total_count: Cardinality
    positive_count: Cardinality
    negative_count: Cardinality
    
    def __post_init__(self):
        """Validate that cardinalities are non-negative integers"""
        for field_name, value in [
            ('total_count', self.total_count),
            ('positive_count', self.positive_count), 
            ('negative_count', self.negative_count)
        ]:
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer, got {value}")
        
        if self.positive_count + self.negative_count > self.total_count:
            raise ValueError("Sum of positive and negative counts cannot exceed total count")
    
    @property
    def positive_ratio(self) -> Ratio:
        """Calculate positive ratio (non-integral derived metric)"""
        if self.total_count == 0:
            return Ratio(0.0)
        return Ratio(self.positive_count / self.total_count)
    
    @property
    def negative_ratio(self) -> Ratio:
        """Calculate negative ratio (non-integral derived metric)"""
        if self.total_count == 0:
            return Ratio(0.0)
        return Ratio(self.negative_count / self.total_count)
    
    @property
    def balance_score(self) -> Score:
        """Calculate balance score (non-integral evaluation metric)"""
        if self.total_count == 0:
            return Score(0.5)  # Neutral when no data
        
        ratio_diff = abs(self.positive_ratio - self.negative_ratio)
        # Higher balance score when positive and negative are more balanced
        return Score(1.0 - ratio_diff)


def safe_cardinality_ratio(numerator: Cardinality, 
                          denominator: Cardinality, 
                          default: float = 0.0) -> Ratio:
    """
    Safely compute ratio from two cardinalities
    
    Args:
        numerator: Count in numerator (integral)
        denominator: Count in denominator (integral)
        default: Value to return when denominator is 0
        
    Returns:
        Ratio: Non-integral ratio value
        
    Raises:
        ValueError: If inputs are not valid cardinalities
    """
    if not isinstance(numerator, int) or numerator < 0:
        raise ValueError(f"Numerator must be non-negative integer, got {numerator}")
    if not isinstance(denominator, int) or denominator < 0:
        raise ValueError(f"Denominator must be non-negative integer, got {denominator}")
    
    if denominator == 0:
        return Ratio(default)
    
    if numerator > denominator:
        raise ValueError(f"Numerator ({numerator}) cannot exceed denominator ({denominator})")
    
    return Ratio(numerator / denominator)


def safe_weighted_cardinality_ratio(counts: list[tuple[Cardinality, float]], 
                                   total_count: Cardinality,
                                   default: float = 0.0) -> Score:
    """
    Compute weighted ratio from multiple cardinalities
    
    Args:
        counts: List of (count, weight) tuples
        total_count: Total count for normalization
        default: Value to return when total_count is 0
        
    Returns:
        Score: Weighted non-integral score
    """
    if not isinstance(total_count, int) or total_count < 0:
        raise ValueError(f"Total count must be non-negative integer, got {total_count}")
    
    if total_count == 0:
        return Score(default)
    
    weighted_sum = 0.0
    count_sum = 0
    
    for count, weight in counts:
        if not isinstance(count, int) or count < 0:
            raise ValueError(f"Count must be non-negative integer, got {count}")
        if not isinstance(weight, (int, float)):
            raise ValueError(f"Weight must be numeric, got {weight}")
        
        weighted_sum += count * weight
        count_sum += count
    
    if count_sum > total_count:
        raise ValueError(f"Sum of counts ({count_sum}) exceeds total ({total_count})")
    
    return Score(weighted_sum / total_count)


def normalize_score_to_probability(score: Score) -> Probability:
    """
    Convert a score to a valid probability (0.0 to 1.0)
    
    Args:
        score: Input score (can be any float)
        
    Returns:
        Probability: Normalized probability value
    """
    if not isinstance(score, (int, float)):
        raise ValueError(f"Score must be numeric, got {score}")
    
    # Clamp to [0, 1] range
    normalized = max(0.0, min(1.0, float(score)))
    return Probability(normalized)


def cardinality_complexity_score(logical_count: Cardinality,
                                quantifier_count: Cardinality, 
                                total_words: Cardinality) -> Score:
    """
    Improved version of logical complexity calculation with explicit types
    
    This replaces the original calculation in HumanExpressionEvaluator.py
    that mixed integral counts with non-integral results.
    
    Args:
        logical_count: Count of logical operators (integral)
        quantifier_count: Count of quantifiers (integral)
        total_words: Total word count (integral)
        
    Returns:
        Score: Complexity score (non-integral, 0.0 to 1.0)
    """
    # Validate input cardinalities
    for name, value in [
        ('logical_count', logical_count),
        ('quantifier_count', quantifier_count),
        ('total_words', total_words)
    ]:
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be non-negative integer, got {value}")
    
    if total_words == 0:
        return Score(0.0)
    
    # Calculate complexity indicators per word (non-integral ratios)
    logical_density = safe_cardinality_ratio(logical_count, total_words)
    quantifier_density = safe_cardinality_ratio(quantifier_count, total_words)
    
    # Combine densities with scaling factor
    combined_density = logical_density + quantifier_density
    complexity_score = min(combined_density * 2.0, 1.0)
    
    return Score(complexity_score)


def cardinality_clarity_score(vague_count: Cardinality,
                             definite_count: Cardinality,
                             total_words: Cardinality) -> Score:
    """
    Improved version of truth value clarity calculation with explicit types
    
    Args:
        vague_count: Count of vague terms (integral)
        definite_count: Count of definite terms (integral)  
        total_words: Total word count (integral)
        
    Returns:
        Score: Clarity score (non-integral, 0.0 to 1.0)
    """
    # Validate input cardinalities
    for name, value in [
        ('vague_count', vague_count),
        ('definite_count', definite_count),
        ('total_words', total_words)
    ]:
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be non-negative integer, got {value}")
    
    if total_words == 0:
        return Score(0.5)  # Neutral when no data
    
    # Calculate effect of vague vs definite terms per word
    vague_ratio = safe_cardinality_ratio(vague_count, total_words)
    definite_ratio = safe_cardinality_ratio(definite_count, total_words)
    
    # Base clarity of 0.5, adjusted by the balance of definite vs vague
    clarity_adjustment = definite_ratio - vague_ratio
    clarity_score = 0.5 + clarity_adjustment
    
    # Ensure valid range
    return Score(max(0.0, min(1.0, clarity_score)))


def cardinality_lexical_density(content_word_count: Cardinality,
                               total_word_count: Cardinality) -> Ratio:
    """
    Calculate lexical density with explicit cardinality types
    
    Args:
        content_word_count: Count of content words (integral)
        total_word_count: Total word count (integral)
        
    Returns:
        Ratio: Lexical density (non-integral, 0.0 to 1.0)
    """
    return safe_cardinality_ratio(content_word_count, total_word_count, default=0.0)


def validate_cardinality_operation(operation_name: str, **kwargs) -> None:
    """
    Validate that all arguments to a cardinality operation are valid
    
    Args:
        operation_name: Name of the operation for error messages
        **kwargs: Named arguments to validate
        
    Raises:
        ValueError: If any argument is not a valid cardinality
    """
    for name, value in kwargs.items():
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{operation_name}: {name} must be non-negative integer, got {value}")


# Example usage and testing
def main():
    """Demonstrate proper cardinality handling"""
    
    print("=== Cardinality Types Demo ===")
    
    # Example 1: Basic cardinality metrics
    print("\n1. Basic Cardinality Metrics:")
    try:
        metrics = CardinalityMetrics(
            total_count=Cardinality(100),
            positive_count=Cardinality(60),
            negative_count=Cardinality(40)
        )
        print(f"Total count: {metrics.total_count} (integral)")
        print(f"Positive ratio: {metrics.positive_ratio:.3f} (non-integral)")
        print(f"Negative ratio: {metrics.negative_ratio:.3f} (non-integral)")
        print(f"Balance score: {metrics.balance_score:.3f} (non-integral)")
        
    except ValueError as e:
        print(f"Validation error: {e}")
    
    # Example 2: Safe ratio calculation
    print("\n2. Safe Ratio Calculation:")
    ratio1 = safe_cardinality_ratio(Cardinality(25), Cardinality(100))
    ratio2 = safe_cardinality_ratio(Cardinality(0), Cardinality(0))  # Edge case
    print(f"25/100 = {ratio1:.3f}")
    print(f"0/0 = {ratio2:.3f} (default)")
    
    # Example 3: Complexity scoring
    print("\n3. Complexity Scoring:")
    complexity = cardinality_complexity_score(
        logical_count=Cardinality(3),
        quantifier_count=Cardinality(2), 
        total_words=Cardinality(20)
    )
    print(f"Complexity score: {complexity:.3f}")
    
    # Example 4: Clarity scoring
    print("\n4. Clarity Scoring:")
    clarity = cardinality_clarity_score(
        vague_count=Cardinality(2),
        definite_count=Cardinality(5),
        total_words=Cardinality(30)
    )
    print(f"Clarity score: {clarity:.3f}")
    
    # Example 5: Error handling
    print("\n5. Error Handling:")
    try:
        # This should raise an error
        invalid_metrics = CardinalityMetrics(
            total_count=Cardinality(10),
            positive_count=Cardinality(15),  # Invalid: exceeds total
            negative_count=Cardinality(5)
        )
    except ValueError as e:
        print(f"Caught expected error: {e}")


if __name__ == "__main__":
    main()