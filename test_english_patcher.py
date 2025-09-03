#!/usr/bin/env python3
"""
Test cases for English Language Patcher

This file contains comprehensive test cases to validate the English patching functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from EnglishPatcher import EnglishPatcher, PatchType
import unittest


class TestEnglishPatcher(unittest.TestCase):
    """Test cases for the English language patcher"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.patcher = EnglishPatcher()
    
    def test_spelling_corrections(self):
        """Test spelling correction functionality"""
        test_cases = [
            ("teh quick brown fox", "the quick brown fox"),
            ("I can't beleive it", "I can't believe it"),
            ("recieve the package", "receive the package"),
            ("seperate the items", "separate the items"),
            ("definately worth it", "definitely worth it"),
        ]
        
        for original, expected in test_cases:
            with self.subTest(original=original):
                result = self.patcher.patch_text(original)
                self.assertEqual(result.patched_text.lower(), expected.lower())
    
    def test_grammar_corrections(self):
        """Test grammar correction functionality"""
        test_cases = [
            ("I is going home", "I am going home"),
            ("You is correct", "You are correct"),
            ("This is a example", "This is an example"),
            ("That is an book", "That is a book"),
        ]
        
        for original, expected in test_cases:
            with self.subTest(original=original):
                result = self.patcher.patch_text(original)
                self.assertEqual(result.patched_text, expected)
    
    def test_punctuation_corrections(self):
        """Test punctuation correction functionality"""
        test_cases = [
            ("Hello , world", "Hello, world"),
            ("What time is it?I don't know", "What time is it? I don't know"),
            ("This  has   multiple    spaces", "This has multiple spaces"),
        ]
        
        for original, expected in test_cases:
            with self.subTest(original=original):
                result = self.patcher.patch_text(original)
                self.assertEqual(result.patched_text, expected)
    
    def test_capitalization_corrections(self):
        """Test capitalization correction functionality"""
        test_cases = [
            ("hello world", "Hello world"),
            ("the cat ran.the dog followed", "The cat ran. The dog followed"),
        ]
        
        for original, expected in test_cases:
            with self.subTest(original=original):
                result = self.patcher.patch_text(original)
                self.assertEqual(result.patched_text, expected)
    
    def test_style_corrections_aggressive(self):
        """Test style corrections in aggressive mode"""
        test_cases = [
            ("I can't do it", "I cannot do it"),
            ("It won't work", "It will not work"),
            ("They don't know", "They do not know"),
        ]
        
        for original, expected in test_cases:
            with self.subTest(original=original):
                result = self.patcher.patch_text(original, aggressive=True)
                self.assertEqual(result.patched_text, expected)
    
    def test_double_negative_fixes(self):
        """Test double negative corrections"""
        test_cases = [
            ("I don't have no money", "I don't have any money"),
            ("Can't get no satisfaction", "Can't get any satisfaction"),
        ]
        
        for original, expected in test_cases:
            with self.subTest(original=original):
                result = self.patcher.patch_text(original)
                self.assertEqual(result.patched_text, expected)
    
    def test_patch_confidence(self):
        """Test that patches have appropriate confidence scores"""
        text = "teh quick brown fox"
        result = self.patcher.patch_text(text)
        
        self.assertTrue(len(result.patches) > 0)
        for patch in result.patches:
            self.assertTrue(0.0 <= patch.confidence <= 1.0)
            self.assertIsInstance(patch.patch_type, PatchType)
    
    def test_no_patches_for_correct_text(self):
        """Test that correct text doesn't get patched"""
        correct_text = "The quick brown fox jumps over the lazy dog."
        result = self.patcher.patch_text(correct_text)
        
        self.assertEqual(result.original_text, result.patched_text)
        self.assertEqual(len(result.patches), 0)
    
    def test_patch_summary(self):
        """Test patch summary generation"""
        text = "teh quick brown fox.can you beleive it?"
        result = self.patcher.patch_text(text)
        summary = self.patcher.get_patch_summary(result)
        
        self.assertIsInstance(summary, str)
        self.assertTrue(len(summary) > 0)
        self.assertIn("patches", summary.lower())
    
    def test_preserve_formatting(self):
        """Test that original formatting is preserved where appropriate"""
        # Test with punctuation
        result = self.patcher._preserve_formatting("teh.", "the")
        self.assertEqual(result, "the.")
        
        # Test with capitalization
        result = self.patcher._preserve_formatting("TEH", "the")
        self.assertEqual(result, "THE")
        
        # Test with mixed case
        result = self.patcher._preserve_formatting("Teh", "the")
        self.assertEqual(result, "The")
    
    def test_text_simplification(self):
        """Test text simplification functionality"""
        test_cases = [
            ("I need to utilize this tool", "I need to use this tool"),
            ("We must commence the project", "We must begin the project"),
            ("Please facilitate the process", "Please help the process"),
            ("The results demonstrate success", "The results show success"),
        ]
        
        for original, expected in test_cases:
            with self.subTest(original=original):
                result = self.patcher.simplify_text(original)
                self.assertEqual(result.patched_text, expected)
                self.assertTrue(any(patch.patch_type == PatchType.SIMPLIFICATION for patch in result.patches))
    
    def test_simplification_with_patch_text(self):
        """Test simplification using patch_text with simplify=True"""
        text = "Subsequently, we will utilize advanced methodology to facilitate the process"
        result = self.patcher.patch_text(text, simplify=True)
        
        # Check that simplifications were applied
        self.assertTrue(any(patch.patch_type == PatchType.SIMPLIFICATION for patch in result.patches))
        self.assertNotEqual(result.original_text, result.patched_text)
        
        # Check specific simplifications
        simplified = result.patched_text.lower()
        self.assertIn("later", simplified)  # subsequently → later
        self.assertIn("use", simplified)    # utilize → use
        self.assertIn("method", simplified) # methodology → method
        self.assertIn("help", simplified)   # facilitate → help
    
    def test_capitalization_preservation_in_simplification(self):
        """Test that capitalization is preserved during simplification"""
        test_cases = [
            ("UTILIZE this tool", "USE this tool"),
            ("Utilize this tool", "Use this tool"),
            # Note: lowercase at sentence start gets capitalized by capitalization rules first
        ]
        
        for original, expected in test_cases:
            with self.subTest(original=original):
                result = self.patcher.patch_text(original, simplify=True)
                self.assertEqual(result.patched_text, expected)
        
        # Test lowercase in middle of sentence (won't be auto-capitalized)
        result = self.patcher.patch_text("I will utilize this tool", simplify=True)
        self.assertEqual(result.patched_text, "I will use this tool")
    
    def test_paradox_detection(self):
        """Test paradox and contradiction detection"""
        test_cases = [
            "Comfort is a pain.",
            "Love is hate.",
            "Peace is war.",
            "This is easy and difficult.",
            "Hot, cold weather today.",
            "The big small car.",
        ]
        
        for test_case in test_cases:
            with self.subTest(test_case=test_case):
                result = self.patcher.patch_text(test_case)
                paradox_patches = [p for p in result.patches if p.patch_type == PatchType.PARADOX]
                self.assertTrue(len(paradox_patches) > 0, f"Should detect paradox in: {test_case}")
                self.assertEqual(result.original_text, result.patched_text, "Paradox detection should not modify text")
    
    def test_comfort_is_a_pain_specific(self):
        """Test the specific phrase from the issue: 'Comfort is a pain.'"""
        result = self.patcher.patch_text("Comfort is a pain.")
        
        # Should detect the paradox
        paradox_patches = [p for p in result.patches if p.patch_type == PatchType.PARADOX]
        self.assertEqual(len(paradox_patches), 1)
        
        # Should not modify the text (paradox detection is analytical only)
        self.assertEqual(result.original_text, result.patched_text)
        
        # Check patch details
        paradox_patch = paradox_patches[0]
        self.assertIn("comfort", paradox_patch.explanation.lower())
        self.assertIn("pain", paradox_patch.explanation.lower())
        self.assertIn("opposing", paradox_patch.explanation.lower())
        self.assertTrue(paradox_patch.confidence > 0.5)
    
    def test_no_false_paradox_detection(self):
        """Test that non-paradoxical text doesn't trigger paradox detection"""
        normal_texts = [
            "The weather is nice today.",
            "I love reading books.",
            "This is a simple sentence.",
            "Programming is challenging but rewarding.",
            "The cat sat on the mat.",
        ]
        
        for text in normal_texts:
            with self.subTest(text=text):
                result = self.patcher.patch_text(text)
                paradox_patches = [p for p in result.patches if p.patch_type == PatchType.PARADOX]
                self.assertEqual(len(paradox_patches), 0, f"Should not detect paradox in: {text}")
    
    def test_complex_paradox_patterns(self):
        """Test detection of complex paradoxical patterns"""
        complex_cases = [
            ("Never always do that.", "Contradiction between 'never' and 'always'"),
            ("It's impossible to be possible.", "Contradiction between possible and impossible"),
            ("Everything is nothing.", "Contradiction between everything and nothing"),
        ]
        
        for text, expected_explanation_keyword in complex_cases:
            with self.subTest(text=text):
                result = self.patcher.patch_text(text)
                paradox_patches = [p for p in result.patches if p.patch_type == PatchType.PARADOX]
                self.assertTrue(len(paradox_patches) > 0)
                # Check that the explanation contains relevant keywords
                explanation = paradox_patches[0].explanation.lower()
                self.assertTrue(any(keyword in explanation for keyword in expected_explanation_keyword.lower().split()))


class TestEnglishPatcherIntegration(unittest.TestCase):
    """Integration tests for comprehensive English patching"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.patcher = EnglishPatcher()
    
    def test_comprehensive_patching(self):
        """Test comprehensive patching with multiple error types"""
        original = "teh quick brown fox jumps over the lazy dog.can you beleive it?this is a example of an text that need corrections."
        result = self.patcher.patch_text(original, aggressive=True)
        
        # Should have multiple patches of different types
        patch_types = set(patch.patch_type for patch in result.patches)
        self.assertTrue(len(patch_types) > 1)
        
        # Should have improved the text significantly
        self.assertNotEqual(result.original_text, result.patched_text)
        self.assertTrue(result.success_rate > 0)
    
    def test_real_world_examples(self):
        """Test with real-world examples of common errors"""
        examples = [
            "I was wandering if you could help me with this problem?",
            "There going to the store to buy groceries for there family.",
            "Its been along day and I cant wait to get home.",
            "The manager told the team that there performance was excellent.",
        ]
        
        for example in examples:
            with self.subTest(example=example):
                result = self.patcher.patch_text(example, aggressive=True)
                # Should make some improvements
                self.assertTrue(len(result.patches) > 0 or result.patched_text != example)


def run_performance_tests():
    """Run performance tests for the English patcher"""
    import time
    
    patcher = EnglishPatcher()
    
    # Test with various text lengths
    short_text = "teh quick brown fox"
    medium_text = " ".join([short_text] * 10)
    long_text = " ".join([short_text] * 100)
    
    texts = [
        ("Short text", short_text),
        ("Medium text", medium_text),
        ("Long text", long_text),
    ]
    
    print("\nPerformance Tests:")
    print("==================")
    
    for name, text in texts:
        start_time = time.time()
        result = patcher.patch_text(text, aggressive=True)
        end_time = time.time()
        
        duration = end_time - start_time
        words_per_second = len(text.split()) / duration if duration > 0 else float('inf')
        
        print(f"{name}: {duration:.4f}s ({words_per_second:.1f} words/sec)")
        print(f"  Applied {len(result.patches)} patches")


def main():
    """Run all tests"""
    print("Running English Patcher Tests")
    print("=" * 40)
    
    # Run unit tests
    unittest.main(verbosity=2, exit=False)
    
    # Run performance tests
    run_performance_tests()


if __name__ == "__main__":
    main()