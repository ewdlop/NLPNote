#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unicode Validation and Testing Utilities for NLPNote Repository

This module provides utilities to validate Unicode handling, test multilingual
content processing, and ensure proper display of Unicode content on GitHub.
"""

import os
import sys
import unicodedata
import re
from typing import List, Dict, Any, Tuple, Set
from pathlib import Path


class UnicodeValidator:
    """Unicode validation and testing utilities"""
    
    def __init__(self):
        self.encoding_issues = []
        self.display_issues = []
        self.filename_issues = []
        
    def validate_file_encoding(self, file_path: str) -> Dict[str, Any]:
        """Validate file encoding and Unicode content"""
        result = {
            'file_path': file_path,
            'encoding_valid': True,
            'is_utf8': False,
            'has_unicode': False,
            'unicode_categories': set(),
            'potential_issues': [],
            'char_count': 0
        }
        
        try:
            with open(file_path, 'rb') as f:
                raw_content = f.read()
            
            # Try to decode as UTF-8
            try:
                content = raw_content.decode('utf-8')
                result['is_utf8'] = True
                result['char_count'] = len(content)
                
                # Analyze Unicode content
                for char in content:
                    if ord(char) > 127:  # Non-ASCII
                        result['has_unicode'] = True
                        category = unicodedata.category(char)
                        result['unicode_categories'].add(category)
                        
                        # Check for potentially problematic characters
                        if category.startswith('C'):  # Control characters
                            result['potential_issues'].append(f'Control character: {repr(char)}')
                        elif category == 'Cf':  # Format characters
                            result['potential_issues'].append(f'Format character: {repr(char)}')
                
            except UnicodeDecodeError as e:
                result['encoding_valid'] = False
                result['potential_issues'].append(f'UTF-8 decode error: {e}')
                
        except Exception as e:
            result['encoding_valid'] = False
            result['potential_issues'].append(f'File read error: {e}')
            
        return result
    
    def validate_filename_unicode(self, file_path: str) -> Dict[str, Any]:
        """Validate Unicode characters in filename"""
        filename = os.path.basename(file_path)
        result = {
            'filename': filename,
            'has_unicode': False,
            'unicode_categories': set(),
            'github_safe': True,
            'potential_issues': []
        }
        
        for char in filename:
            if ord(char) > 127:  # Non-ASCII
                result['has_unicode'] = True
                category = unicodedata.category(char)
                result['unicode_categories'].add(category)
                
                # Check for GitHub display issues
                if category.startswith('C'):  # Control characters
                    result['github_safe'] = False
                    result['potential_issues'].append(f'Control character in filename: {repr(char)}')
                elif category in ['Zs', 'Zl', 'Zp']:  # Various space characters
                    result['potential_issues'].append(f'Special space character: {repr(char)}')
        
        return result
    
    def test_unicode_processing(self) -> Dict[str, Any]:
        """Test Unicode processing capabilities"""
        test_strings = {
            'chinese_simplified': '你好世界！这是简体中文测试。',
            'chinese_traditional': '你好世界！這是繁體中文測試。',
            'japanese_hiragana': 'こんにちは世界！これはひらがなテストです。',
            'japanese_katakana': 'コンニチハ世界！これはカタカナテストです。',
            'korean': '안녕하세요 세계! 이것은 한국어 테스트입니다.',
            'arabic': 'مرحبا بالعالم! هذا اختبار باللغة العربية.',
            'hebrew': 'שלום עולם! זה מבחן בעברית.',
            'russian': 'Привет мир! Это тест на русском языке.',
            'thai': 'สวัสดีชาวโลก! นี่คือการทดสอบภาษาไทย',
            'emoji': '🌍🌎🌏 Hello World! 👋 Testing emojis 🧪🔬',
            'mixed_scripts': 'Hello 你好 こんにちは 안녕하세요 मनमोहन',
            'mathematical': '∀x∈ℝ: x² ≥ 0 ∧ π ≈ 3.14159 ∫₀^∞ e^(-x²)dx',
            'symbols': '★☆♠♥♦♣♪♫☎☏☺☻☼♀♂'
        }
        
        results = {}
        
        for test_name, test_string in test_strings.items():
            test_result = {
                'original': test_string,
                'length': len(test_string),
                'byte_length': len(test_string.encode('utf-8')),
                'categories': set(),
                'scripts': set(),
                'processing_tests': {}
            }
            
            # Analyze characters
            for char in test_string:
                if ord(char) > 31:  # Skip control characters for analysis
                    category = unicodedata.category(char)
                    test_result['categories'].add(category)
                    
                    try:
                        script = unicodedata.name(char).split()[0]
                        test_result['scripts'].add(script)
                    except ValueError:
                        pass  # Some characters don't have names
            
            # Test various operations
            try:
                test_result['processing_tests']['upper'] = test_string.upper()
                test_result['processing_tests']['lower'] = test_string.lower()
                test_result['processing_tests']['normalized_nfc'] = unicodedata.normalize('NFC', test_string)
                test_result['processing_tests']['normalized_nfd'] = unicodedata.normalize('NFD', test_string)
                test_result['processing_tests']['ascii_errors'] = False
            except Exception as e:
                test_result['processing_tests']['error'] = str(e)
            
            # Test ASCII encoding (should fail gracefully)
            try:
                test_string.encode('ascii')
                test_result['processing_tests']['ascii_encodable'] = True
            except UnicodeEncodeError:
                test_result['processing_tests']['ascii_encodable'] = False
            
            results[test_name] = test_result
        
        return results
    
    def scan_repository(self, repo_path: str = '.') -> Dict[str, Any]:
        """Scan entire repository for Unicode issues"""
        repo_path = Path(repo_path)
        results = {
            'total_files': 0,
            'files_with_unicode': 0,
            'files_with_unicode_names': 0,
            'encoding_issues': [],
            'filename_issues': [],
            'file_results': [],
            'summary': {}
        }
        
        # File extensions to check
        text_extensions = {'.md', '.txt', '.py', '.js', '.html', '.css', '.json', '.xml', '.yaml', '.yml'}
        
        for file_path in repo_path.rglob('*'):
            if file_path.is_file() and not file_path.name.startswith('.'):
                results['total_files'] += 1
                
                # Check filename
                filename_result = self.validate_filename_unicode(str(file_path))
                if filename_result['has_unicode']:
                    results['files_with_unicode_names'] += 1
                if not filename_result['github_safe']:
                    results['filename_issues'].append(filename_result)
                
                # Check file content for text files
                if file_path.suffix.lower() in text_extensions:
                    encoding_result = self.validate_file_encoding(str(file_path))
                    
                    if encoding_result['has_unicode']:
                        results['files_with_unicode'] += 1
                    
                    if not encoding_result['encoding_valid']:
                        results['encoding_issues'].append(encoding_result)
                    
                    results['file_results'].append({
                        'filename': filename_result,
                        'encoding': encoding_result
                    })
        
        # Generate summary
        results['summary'] = {
            'unicode_support_score': self._calculate_unicode_score(results),
            'recommendations': self._generate_recommendations(results)
        }
        
        return results
    
    def _calculate_unicode_score(self, results: Dict[str, Any]) -> float:
        """Calculate a Unicode support score (0-100)"""
        score = 100.0
        
        # Deduct points for encoding issues
        if results['encoding_issues']:
            score -= len(results['encoding_issues']) * 10
        
        # Deduct points for filename issues
        if results['filename_issues']:
            score -= len(results['filename_issues']) * 5
        
        # Bonus for Unicode usage (shows internationalization)
        if results['files_with_unicode'] > 0:
            unicode_ratio = results['files_with_unicode'] / max(results['total_files'], 1)
            score += unicode_ratio * 10
        
        return max(0, min(100, score))
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for Unicode improvements"""
        recommendations = []
        
        if results['encoding_issues']:
            recommendations.append("Fix encoding issues in files - ensure all text files are UTF-8 encoded")
        
        if results['filename_issues']:
            recommendations.append("Review filenames with problematic Unicode characters")
        
        if results['files_with_unicode'] == 0:
            recommendations.append("Consider adding Unicode test cases for better internationalization support")
        
        recommendations.append("Ensure all Python files declare UTF-8 encoding with # -*- coding: utf-8 -*-")
        recommendations.append("Test Unicode processing in CI/CD pipeline")
        recommendations.append("Add Unicode-specific documentation")
        
        return recommendations


def main():
    """Main function for command-line usage"""
    validator = UnicodeValidator()
    
    print("🌍 Unicode Validation Report for NLPNote Repository")
    print("=" * 60)
    
    # Test Unicode processing capabilities
    print("\n📋 Testing Unicode Processing Capabilities...")
    processing_results = validator.test_unicode_processing()
    
    for test_name, result in processing_results.items():
        print(f"\n{test_name.replace('_', ' ').title()}:")
        print(f"  Text: {result['original'][:50]}{'...' if len(result['original']) > 50 else ''}")
        print(f"  Length: {result['length']} chars, {result['byte_length']} bytes")
        print(f"  Categories: {', '.join(sorted(result['categories']))}")
        
        if result['processing_tests'].get('error'):
            print(f"  ❌ Processing error: {result['processing_tests']['error']}")
        else:
            print(f"  ✅ Processing successful")
    
    # Scan repository
    print("\n📁 Scanning Repository...")
    repo_results = validator.scan_repository()
    
    print(f"\nRepository Summary:")
    print(f"  Total files scanned: {repo_results['total_files']}")
    print(f"  Files with Unicode content: {repo_results['files_with_unicode']}")
    print(f"  Files with Unicode names: {repo_results['files_with_unicode_names']}")
    print(f"  Encoding issues: {len(repo_results['encoding_issues'])}")
    print(f"  Filename issues: {len(repo_results['filename_issues'])}")
    print(f"  Unicode support score: {repo_results['summary']['unicode_support_score']:.1f}/100")
    
    # Show issues
    if repo_results['encoding_issues']:
        print("\n⚠️ Encoding Issues:")
        for issue in repo_results['encoding_issues'][:5]:  # Show first 5
            print(f"  {issue['file_path']}: {', '.join(issue['potential_issues'])}")
    
    if repo_results['filename_issues']:
        print("\n⚠️ Filename Issues:")
        for issue in repo_results['filename_issues'][:5]:  # Show first 5
            print(f"  {issue['filename']}: {', '.join(issue['potential_issues'])}")
    
    # Show recommendations
    print("\n💡 Recommendations:")
    for rec in repo_results['summary']['recommendations']:
        print(f"  • {rec}")
    
    print(f"\n✅ Unicode validation complete!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())