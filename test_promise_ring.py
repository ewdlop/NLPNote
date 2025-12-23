#!/usr/bin/env python3
"""
Test suite for Promise Ring implementation
==========================================

This module contains unit tests for the promise ring mathematical concept.
"""

import unittest
from promise_ring_demo import (
    PromiseElement, PromiseProperty, BooleanPromiseRing, 
    IntegerPromiseRing, PromiseRing
)


class TestPromiseElement(unittest.TestCase):
    """Test cases for PromiseElement class."""
    
    def test_element_creation(self):
        """Test promise element creation."""
        promises = {PromiseProperty.COMMUTATIVE, PromiseProperty.STABLE}
        elem = PromiseElement(42, promises)
        
        self.assertEqual(elem.value, 42)
        self.assertEqual(elem.promises, promises)
    
    def test_has_promise(self):
        """Test promise checking."""
        promises = {PromiseProperty.ZERO, PromiseProperty.IDEMPOTENT}
        elem = PromiseElement(0, promises)
        
        self.assertTrue(elem.has_promise(PromiseProperty.ZERO))
        self.assertTrue(elem.has_promise(PromiseProperty.IDEMPOTENT))
        self.assertFalse(elem.has_promise(PromiseProperty.UNIT))
    
    def test_equality(self):
        """Test element equality."""
        promises = {PromiseProperty.COMMUTATIVE}
        elem1 = PromiseElement(5, promises)
        elem2 = PromiseElement(5, promises)
        elem3 = PromiseElement(6, promises)
        
        self.assertEqual(elem1, elem2)
        self.assertNotEqual(elem1, elem3)


class TestBooleanPromiseRing(unittest.TestCase):
    """Test cases for Boolean Promise Ring."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ring = BooleanPromiseRing()
        self.false_elem = self.ring.zero
        self.true_elem = self.ring.one
    
    def test_elements_creation(self):
        """Test boolean elements have correct promises."""
        self.assertTrue(self.false_elem.has_promise(PromiseProperty.ZERO))
        self.assertTrue(self.false_elem.has_promise(PromiseProperty.IDEMPOTENT))
        
        self.assertTrue(self.true_elem.has_promise(PromiseProperty.UNIT))
        self.assertTrue(self.true_elem.has_promise(PromiseProperty.IDEMPOTENT))
    
    def test_addition_or_operation(self):
        """Test Boolean OR operation."""
        # False OR False = False
        result = self.ring.add(self.false_elem, self.false_elem)
        self.assertEqual(result.value, False)
        self.assertTrue(result.has_promise(PromiseProperty.IDEMPOTENT))
        
        # False OR True = True
        result = self.ring.add(self.false_elem, self.true_elem)
        self.assertEqual(result.value, True)
        
        # True OR True = True
        result = self.ring.add(self.true_elem, self.true_elem)
        self.assertEqual(result.value, True)
        self.assertTrue(result.has_promise(PromiseProperty.IDEMPOTENT))
    
    def test_multiplication_and_operation(self):
        """Test Boolean AND operation."""
        # False AND False = False
        result = self.ring.multiply(self.false_elem, self.false_elem)
        self.assertEqual(result.value, False)
        self.assertTrue(result.has_promise(PromiseProperty.IDEMPOTENT))
        
        # False AND True = False
        result = self.ring.multiply(self.false_elem, self.true_elem)
        self.assertEqual(result.value, False)
        
        # True AND True = True
        result = self.ring.multiply(self.true_elem, self.true_elem)
        self.assertEqual(result.value, True)
        self.assertTrue(result.has_promise(PromiseProperty.IDEMPOTENT))


class TestIntegerPromiseRing(unittest.TestCase):
    """Test cases for Integer Promise Ring."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ring = IntegerPromiseRing()
    
    def test_element_promise_assignment(self):
        """Test that elements get appropriate promises."""
        # Zero should have ZERO promise
        zero = self.ring.create_element(0)
        self.assertTrue(zero.has_promise(PromiseProperty.ZERO))
        
        # ±1 should have UNIT promise
        one = self.ring.create_element(1)
        neg_one = self.ring.create_element(-1)
        self.assertTrue(one.has_promise(PromiseProperty.UNIT))
        self.assertTrue(neg_one.has_promise(PromiseProperty.UNIT))
        
        # Primes should have COMMUTATIVE promise
        prime = self.ring.create_element(7)
        self.assertTrue(prime.has_promise(PromiseProperty.COMMUTATIVE))
        
        # Composites should have STABLE promise
        composite = self.ring.create_element(6)
        self.assertTrue(composite.has_promise(PromiseProperty.STABLE))
    
    def test_addition(self):
        """Test integer addition."""
        a = self.ring.create_element(3)
        b = self.ring.create_element(5)
        result = self.ring.add(a, b)
        
        self.assertEqual(result.value, 8)
    
    def test_multiplication(self):
        """Test integer multiplication."""
        a = self.ring.create_element(3)
        b = self.ring.create_element(4)
        result = self.ring.multiply(a, b)
        
        self.assertEqual(result.value, 12)
    
    def test_promise_consistency(self):
        """Test promise consistency axiom."""
        # Create two elements with common promises
        prime1 = self.ring.create_element(7)   # has COMMUTATIVE
        prime2 = self.ring.create_element(11)  # has COMMUTATIVE
        
        result = self.ring.add(prime1, prime2)
        
        # Promise consistency: P(a + b) ⊇ P(a) ∩ P(b)
        intersection = prime1.promises & prime2.promises
        self.assertTrue(intersection.issubset(result.promises))


class TestPromiseRingAxioms(unittest.TestCase):
    """Test promise ring axioms and properties."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.bool_ring = BooleanPromiseRing()
        self.int_ring = IntegerPromiseRing()
    
    def test_promise_intersection(self):
        """Test promise intersection operation."""
        promises_a = {PromiseProperty.COMMUTATIVE, PromiseProperty.STABLE}
        promises_b = {PromiseProperty.COMMUTATIVE, PromiseProperty.ZERO}
        
        intersection = self.int_ring.promise_intersection(promises_a, promises_b)
        expected = {PromiseProperty.COMMUTATIVE}
        
        self.assertEqual(intersection, expected)
    
    def test_promise_combination(self):
        """Test promise combination for multiplication."""
        promises_a = {PromiseProperty.COMMUTATIVE}
        promises_b = {PromiseProperty.COMMUTATIVE}
        
        combination = self.int_ring.promise_combination(promises_a, promises_b)
        
        # Both commutative should result in commutative
        self.assertIn(PromiseProperty.COMMUTATIVE, combination)
    
    def test_zero_absorption(self):
        """Test zero absorption property."""
        promises_zero = {PromiseProperty.ZERO}
        promises_other = {PromiseProperty.STABLE}
        
        combination = self.int_ring.promise_combination(promises_zero, promises_other)
        
        # Zero should create absorbing property
        self.assertIn(PromiseProperty.ABSORBING, combination)


def run_promise_ring_tests():
    """Run all promise ring tests and display results."""
    print("Promise Ring Test Suite")
    print("=" * 50)
    
    # Create test suite
    test_classes = [
        TestPromiseElement,
        TestBooleanPromiseRing, 
        TestIntegerPromiseRing,
        TestPromiseRingAxioms
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✓ All tests passed successfully!")
        print(f"Ran {result.testsRun} tests")
    else:
        print(f"✗ {len(result.failures)} test(s) failed")
        print(f"✗ {len(result.errors)} test(s) had errors")
        print(f"Ran {result.testsRun} tests")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_promise_ring_tests()
    exit(0 if success else 1)