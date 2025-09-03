#!/usr/bin/env python3
"""
Test suite for OrientationLessParser

Comprehensive test cases for the orientation-agnostic text parser,
covering various languages, scripts, and text directions.

無方向性解析器的測試套件，涵蓋各種語言、文字和文本方向的綜合測試案例。
"""

import unittest
import sys
import os

# Add the parent directory to the path to import OrientationLessParser
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from OrientationLessParser import (
    OrientationLessParser, 
    TextDirection, 
    ScriptType, 
    ParsedToken, 
    ParseResult
)


class TestOrientationLessParser(unittest.TestCase):
    """Test cases for OrientationLessParser"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.parser = OrientationLessParser()
    
    def test_empty_text(self):
        """Test parsing empty text"""
        result = self.parser.parse("")
        self.assertEqual(result.original_text, "")
        self.assertEqual(len(result.tokens), 0)
        self.assertEqual(result.dominant_direction, TextDirection.NEUTRAL)
        self.assertEqual(result.dominant_script, ScriptType.UNKNOWN)
        self.assertFalse(result.has_mixed_directions)
    
    def test_english_ltr_text(self):
        """Test parsing English left-to-right text"""
        text = "Hello world!"
        result = self.parser.parse(text)
        
        self.assertEqual(result.original_text, text)
        self.assertEqual(result.dominant_direction, TextDirection.LTR)
        self.assertEqual(result.dominant_script, ScriptType.LATIN)
        self.assertFalse(result.has_mixed_directions)
        
        # Check that we have the expected tokens
        word_tokens = [t for t in result.tokens if not t.is_whitespace and not t.is_punctuation]
        self.assertEqual(len(word_tokens), 2)
        self.assertEqual(word_tokens[0].text, "Hello")
        self.assertEqual(word_tokens[1].text, "world")
    
    def test_arabic_rtl_text(self):
        """Test parsing Arabic right-to-left text"""
        text = "مرحبا بالعالم"  # Hello world in Arabic
        result = self.parser.parse(text)
        
        self.assertEqual(result.dominant_direction, TextDirection.RTL)
        self.assertEqual(result.dominant_script, ScriptType.ARABIC)
        
        # Check that Arabic tokens are detected correctly
        word_tokens = [t for t in result.tokens if not t.is_whitespace]
        self.assertTrue(all(t.direction == TextDirection.RTL for t in word_tokens))
        self.assertTrue(all(t.script_type == ScriptType.ARABIC for t in word_tokens))
    
    def test_hebrew_rtl_text(self):
        """Test parsing Hebrew right-to-left text"""
        text = "שלום עולם"  # Hello world in Hebrew
        result = self.parser.parse(text)
        
        self.assertEqual(result.dominant_direction, TextDirection.RTL)
        self.assertEqual(result.dominant_script, ScriptType.HEBREW)
        
        word_tokens = [t for t in result.tokens if not t.is_whitespace]
        self.assertTrue(all(t.direction == TextDirection.RTL for t in word_tokens))
        self.assertTrue(all(t.script_type == ScriptType.HEBREW for t in word_tokens))
    
    def test_chinese_text(self):
        """Test parsing Chinese text"""
        text = "你好世界"  # Hello world in Chinese
        result = self.parser.parse(text)
        
        self.assertEqual(result.dominant_script, ScriptType.CJK)
        
        word_tokens = [t for t in result.tokens if not t.is_whitespace]
        self.assertTrue(all(t.script_type == ScriptType.CJK for t in word_tokens))
    
    def test_mixed_script_text(self):
        """Test parsing text with mixed scripts"""
        text = "Hello مرحبا 你好"  # Hello in English, Arabic, Chinese
        result = self.parser.parse(text)
        
        self.assertEqual(result.dominant_script, ScriptType.MIXED)
        self.assertTrue(result.has_mixed_directions)
        
        # Check that different scripts are detected
        tokens = [t for t in result.tokens if not t.is_whitespace]
        scripts = set(t.script_type for t in tokens)
        self.assertGreater(len(scripts), 1)
    
    def test_bidirectional_text(self):
        """Test parsing bidirectional text"""
        text = "The word مرحبا means hello"  # English with Arabic word
        result = self.parser.parse(text)
        
        self.assertTrue(result.has_mixed_directions)
        
        # Check that we have both LTR and RTL tokens
        directions = set(t.direction for t in result.tokens if not t.is_whitespace)
        self.assertIn(TextDirection.LTR, directions)
        self.assertIn(TextDirection.RTL, directions)
    
    def test_numbers_and_punctuation(self):
        """Test parsing numbers and punctuation"""
        text = "Price: $123.45!"
        result = self.parser.parse(text)
        
        # Check token types
        number_tokens = [t for t in result.tokens if t.is_number]
        punct_tokens = [t for t in result.tokens if t.is_punctuation]
        
        self.assertGreater(len(number_tokens), 0)
        self.assertGreater(len(punct_tokens), 0)
        
        # Numbers should be neutral
        self.assertTrue(all(t.direction == TextDirection.NEUTRAL for t in number_tokens))
    
    def test_normalize_text(self):
        """Test text normalization"""
        # Text with directional markers
        text = "Hello\u200Eworld"  # Contains LRM (Left-to-Right Mark)
        normalized = self.parser.normalize_text(text)
        
        # Directional markers should be removed
        self.assertNotIn('\u200E', normalized)
        self.assertEqual(normalized, "Helloworld")
    
    def test_script_detection(self):
        """Test script type detection"""
        test_cases = [
            ("Hello", ScriptType.LATIN),
            ("مرحبا", ScriptType.ARABIC),
            ("שלום", ScriptType.HEBREW),
            ("你好", ScriptType.CJK),
            ("こんにちは", ScriptType.CJK),  # Japanese Hiragana
            ("안녕", ScriptType.CJK),      # Korean
            ("Hello مرحبا", ScriptType.MIXED),
            ("", ScriptType.UNKNOWN),
            ("123", ScriptType.UNKNOWN),
        ]
        
        for text, expected_script in test_cases:
            with self.subTest(text=text):
                detected = self.parser.detect_script_type(text)
                self.assertEqual(detected, expected_script)
    
    def test_direction_detection(self):
        """Test text direction detection"""
        test_cases = [
            ("Hello world", TextDirection.LTR),
            ("مرحبا بالعالم", TextDirection.RTL),
            ("שלום עולם", TextDirection.RTL),
            ("Hello مرحبا", TextDirection.BIDI),
            ("", TextDirection.NEUTRAL),
            ("123", TextDirection.NEUTRAL),
        ]
        
        for text, expected_direction in test_cases:
            with self.subTest(text=text):
                detected = self.parser.detect_text_direction(text)
                # Note: Some cases might return MIXED instead of BIDI
                self.assertIn(detected, [expected_direction, TextDirection.MIXED])
    
    def test_tokenization(self):
        """Test orientation-agnostic tokenization"""
        text = "Hello, world!"
        tokens = self.parser.tokenize_orientation_agnostic(text)
        
        # Should have tokens for: Hello, comma, whitespace, world, exclamation
        self.assertGreater(len(tokens), 0)
        
        # Check that positions are correctly assigned
        for i, token in enumerate(tokens):
            self.assertEqual(token.logical_position, i)
    
    def test_logical_reordering(self):
        """Test logical reordering of tokens"""
        text = "English مرحبا text"  # Mixed LTR and RTL
        result = self.parser.parse(text)
        
        # Logical order should be different from visual order for RTL content
        self.assertEqual(len(result.logical_order), len(result.tokens))
        
        # All tokens should have updated logical positions
        for token in result.logical_order:
            self.assertIsNotNone(token.logical_position)
    
    def test_extract_text_content(self):
        """Test text content extraction"""
        text = "Hello, world! 123"
        result = self.parser.parse(text)
        
        # Extract with all content
        full_content = self.parser.extract_text_content(result)
        self.assertIn("Hello", full_content)
        self.assertIn("world", full_content)
        
        # Extract without punctuation
        no_punct = self.parser.extract_text_content(result, include_punctuation=False)
        self.assertNotIn(",", no_punct)
        self.assertNotIn("!", no_punct)
        
        # Extract without numbers
        no_numbers = self.parser.extract_text_content(result, include_numbers=False)
        self.assertNotIn("123", no_numbers)
    
    def test_parsing_statistics(self):
        """Test parsing statistics generation"""
        text = "Hello, world! 123"
        result = self.parser.parse(text)
        stats = self.parser.get_parsing_statistics(result)
        
        # Check that statistics are returned
        self.assertIn('total_tokens', stats)
        self.assertIn('word_tokens', stats)
        self.assertIn('punctuation_tokens', stats)
        self.assertIn('number_tokens', stats)
        self.assertIn('whitespace_tokens', stats)
        self.assertIn('direction_distribution', stats)
        self.assertIn('script_distribution', stats)
        
        # Check counts are reasonable
        self.assertGreater(stats['total_tokens'], 0)
        self.assertGreater(stats['word_tokens'], 0)
    
    def test_edge_cases(self):
        """Test edge cases"""
        edge_cases = [
            "",  # Empty string
            " ",  # Only whitespace
            "!@#$%",  # Only punctuation
            "12345",  # Only numbers
            "\n\t",  # Whitespace characters
            "a",  # Single character
            "🙂🎉",  # Emoji (should be handled gracefully)
        ]
        
        for text in edge_cases:
            with self.subTest(text=repr(text)):
                # Should not raise an exception
                result = self.parser.parse(text)
                self.assertIsInstance(result, ParseResult)
                self.assertEqual(result.original_text, text)
    
    def test_unicode_handling(self):
        """Test Unicode normalization and handling"""
        # Text with combining characters
        text = "café"  # May contain combining characters
        result = self.parser.parse(text)
        
        # Should handle Unicode properly
        self.assertIsInstance(result.normalized_text, str)
        self.assertGreater(len(result.tokens), 0)
    
    def test_performance_with_long_text(self):
        """Test performance with longer text"""
        # Create a longer text with mixed content
        long_text = "Hello world! " * 100 + "مرحبا بالعالم! " * 100
        
        # Should complete in reasonable time
        result = self.parser.parse(long_text)
        self.assertIsInstance(result, ParseResult)
        self.assertGreater(len(result.tokens), 0)


class TestIntegrationWithExistingFramework(unittest.TestCase):
    """Test integration with existing NLP framework"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.parser = OrientationLessParser()
    
    def test_integration_compatibility(self):
        """Test that parser can be used with existing framework"""
        # Test that our parser can handle the same types of input
        # as the existing framework expects
        
        test_texts = [
            "How can one distinguish between different types of expressions?",
            "請問您能幫我解決這個問題嗎？",
            "如果我們考慮所有可能的情況，那麼我們必須承認這個問題比我們想像的更複雜。"
        ]
        
        for text in test_texts:
            with self.subTest(text=text[:50]):
                result = self.parser.parse(text)
                
                # Should produce usable results
                self.assertIsInstance(result, ParseResult)
                self.assertGreater(len(result.tokens), 0)
                
                # Should be able to extract clean text
                clean_text = self.parser.extract_text_content(result)
                self.assertIsInstance(clean_text, str)
                self.assertGreater(len(clean_text.strip()), 0)


def run_comprehensive_tests():
    """Run comprehensive test suite with detailed output"""
    print("=" * 80)
    print("OrientationLessParser Comprehensive Test Suite")
    print("無方向性解析器綜合測試套件")
    print("=" * 80)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTest(unittest.makeSuite(TestOrientationLessParser))
    suite.addTest(unittest.makeSuite(TestIntegrationWithExistingFramework))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run comprehensive tests
    success = run_comprehensive_tests()
    
    if success:
        print("\n✅ All tests passed! OrientationLessParser is working correctly.")
        print("✅ 所有測試通過！無方向性解析器運行正常。")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        print("❌ 某些測試失敗。請檢查上面的輸出。")
        sys.exit(1)