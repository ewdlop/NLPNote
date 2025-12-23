"""
字符變換器 (Character Transformer)

Enhanced character transformation tools building on the existing Unicode vector
matrix multiplication work, providing advanced character encoding transformations.

基於現有Unicode向量矩陣乘法工作的增強字符變換工具，提供高級字符編碼變換。
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from CharacterAnalyzer import CharacterAnalyzer, WritingSystem


@dataclass
class TransformationResult:
    """變換結果 (Transformation Result)"""
    original_text: str
    transformed_text: str
    transformation_matrix: np.ndarray
    character_mappings: List[Dict[str, Any]]
    target_writing_system: WritingSystem


class CharacterTransformer:
    """
    字符變換器 (Character Transformer)
    
    Advanced character transformation using Unicode vectors and matrix operations.
    使用Unicode向量和矩陣運算的高級字符變換。
    """
    
    def __init__(self):
        self.analyzer = CharacterAnalyzer()
        
        # Unicode block mappings for target transformations
        self.target_ranges = {
            WritingSystem.CJK: {
                'start': 0x4E00,
                'end': 0x9FFF,
                'name': 'CJK Unified Ideographs'
            },
            WritingSystem.LATIN: {
                'start': 0x0041,
                'end': 0x007A,
                'name': 'Basic Latin'
            },
            WritingSystem.CYRILLIC: {
                'start': 0x0400,
                'end': 0x04FF,
                'name': 'Cyrillic'
            },
            WritingSystem.ARABIC: {
                'start': 0x0600,
                'end': 0x06FF,
                'name': 'Arabic'
            },
            WritingSystem.HEBREW: {
                'start': 0x0590,
                'end': 0x05FF,
                'name': 'Hebrew'
            },
            WritingSystem.GREEK: {
                'start': 0x0370,
                'end': 0x03FF,
                'name': 'Greek and Coptic'
            }
        }
    
    def create_transformation_matrix(self, 
                                   source_text: str, 
                                   target_system: WritingSystem,
                                   method: str = 'random') -> np.ndarray:
        """
        創建變換矩陣 (Create transformation matrix)
        """
        unicode_vectors = np.array([ord(char) for char in source_text if not char.isspace()])
        
        if len(unicode_vectors) == 0:
            return np.eye(1)
        
        matrix_size = len(unicode_vectors)
        
        if method == 'random':
            # Random transformation with reproducible seed based on text
            seed = sum(unicode_vectors) % 1000
            np.random.seed(seed)
            matrix = np.random.randint(1, 5, (matrix_size, matrix_size))
            
        elif method == 'linear_shift':
            # Linear shift transformation
            matrix = np.eye(matrix_size)
            shift_value = self._calculate_shift_value(source_text, target_system)
            matrix += shift_value
            
        elif method == 'phonetic':
            # Phonetic-based transformation (simplified)
            matrix = self._create_phonetic_matrix(unicode_vectors, target_system)
            
        elif method == 'radical_based':
            # Chinese radical-based transformation
            matrix = self._create_radical_matrix(unicode_vectors, target_system)
            
        else:
            # Default to identity matrix
            matrix = np.eye(matrix_size)
        
        return matrix.astype(np.int32)
    
    def _calculate_shift_value(self, text: str, target_system: WritingSystem) -> int:
        """計算位移值 (Calculate shift value)"""
        if target_system not in self.target_ranges:
            return 1
        
        source_avg = np.mean([ord(c) for c in text if not c.isspace()])
        target_range = self.target_ranges[target_system]
        target_center = (target_range['start'] + target_range['end']) / 2
        
        return max(1, int(abs(target_center - source_avg) / 1000))
    
    def _create_phonetic_matrix(self, unicode_vectors: np.ndarray, target_system: WritingSystem) -> np.ndarray:
        """創建語音變換矩陣 (Create phonetic transformation matrix)"""
        size = len(unicode_vectors)
        matrix = np.eye(size)
        
        # Simple phonetic rules - this is a simplified example
        phonetic_shift = {
            WritingSystem.CJK: 2,
            WritingSystem.CYRILLIC: 3,
            WritingSystem.ARABIC: 4,
            WritingSystem.HEBREW: 5
        }.get(target_system, 1)
        
        for i in range(size):
            matrix[i, i] += phonetic_shift
            if i < size - 1:
                matrix[i, i + 1] = 1
        
        return matrix
    
    def _create_radical_matrix(self, unicode_vectors: np.ndarray, target_system: WritingSystem) -> np.ndarray:
        """創建部首變換矩陣 (Create radical transformation matrix)"""
        size = len(unicode_vectors)
        matrix = np.eye(size)
        
        # Radical-based transformation for Chinese characters
        if target_system == WritingSystem.CJK:
            for i in range(size):
                # Add radical offset
                matrix[i, i] += (unicode_vectors[i] % 214) + 1  # 214 traditional radicals
                
        return matrix
    
    def transform_to_unicode_range(self, 
                                 transformed_values: np.ndarray, 
                                 target_system: WritingSystem) -> List[str]:
        """
        變換到Unicode範圍 (Transform to Unicode range)
        """
        if target_system not in self.target_ranges:
            # Fallback to symbol range
            start, end = 0x2000, 0x2BFF
        else:
            range_info = self.target_ranges[target_system]
            start, end = range_info['start'], range_info['end']
        
        range_size = end - start + 1
        transformed_chars = []
        
        for val in transformed_values:
            # Map to target range
            mapped_val = (val % range_size) + start
            
            # Ensure it's a valid Unicode character
            try:
                char = chr(mapped_val)
                # Verify it's not a control character or invalid
                if ord(char) >= 32:  # Printable characters
                    transformed_chars.append(char)
                else:
                    # Fallback to safe range
                    fallback_val = (val % 26) + ord('A')
                    transformed_chars.append(chr(fallback_val))
            except ValueError:
                # Invalid Unicode point, use fallback
                fallback_val = (val % 26) + ord('A')
                transformed_chars.append(chr(fallback_val))
        
        return transformed_chars
    
    def transform_text(self, 
                      text: str, 
                      target_system: WritingSystem,
                      method: str = 'random') -> TransformationResult:
        """
        變換文本 (Transform text)
        """
        # Filter out whitespace for transformation
        chars = [c for c in text if not c.isspace()]
        spaces_positions = [i for i, c in enumerate(text) if c.isspace()]
        
        if not chars:
            return TransformationResult(
                original_text=text,
                transformed_text=text,
                transformation_matrix=np.eye(1),
                character_mappings=[],
                target_writing_system=target_system
            )
        
        # Convert to Unicode vectors
        unicode_vectors = np.array([ord(char) for char in chars])
        
        # Create transformation matrix
        transformation_matrix = self.create_transformation_matrix(text, target_system, method)
        
        # Apply transformation
        if transformation_matrix.shape[0] == len(unicode_vectors):
            transformed_vectors = np.dot(transformation_matrix, unicode_vectors)
        else:
            # Fallback for size mismatch
            transformed_vectors = unicode_vectors * 2
        
        # Convert back to characters
        transformed_chars = self.transform_to_unicode_range(transformed_vectors, target_system)
        
        # Rebuild text with spaces
        result_chars = []
        char_idx = 0
        for i, original_char in enumerate(text):
            if original_char.isspace():
                result_chars.append(original_char)
            else:
                if char_idx < len(transformed_chars):
                    result_chars.append(transformed_chars[char_idx])
                    char_idx += 1
                else:
                    result_chars.append(original_char)
        
        transformed_text = ''.join(result_chars)
        
        # Create character mappings
        character_mappings = []
        for i, (orig_char, trans_char) in enumerate(zip(chars, transformed_chars)):
            orig_info = self.analyzer.get_character_info(orig_char)
            trans_info = self.analyzer.get_character_info(trans_char)
            
            character_mappings.append({
                'index': i,
                'original_char': orig_char,
                'original_unicode': f"U+{ord(orig_char):04X}",
                'original_name': orig_info.unicode_name,
                'transformed_char': trans_char,
                'transformed_unicode': f"U+{ord(trans_char):04X}",
                'transformed_name': trans_info.unicode_name,
                'transformation_value': int(transformed_vectors[i]) if i < len(transformed_vectors) else 0
            })
        
        return TransformationResult(
            original_text=text,
            transformed_text=transformed_text,
            transformation_matrix=transformation_matrix,
            character_mappings=character_mappings,
            target_writing_system=target_system
        )
    
    def batch_transform(self, 
                       texts: List[str], 
                       target_system: WritingSystem,
                       method: str = 'random') -> List[TransformationResult]:
        """
        批量變換 (Batch transform)
        """
        return [self.transform_text(text, target_system, method) for text in texts]
    
    def create_transformation_table(self, result: TransformationResult) -> pd.DataFrame:
        """
        創建變換表格 (Create transformation table)
        """
        df_data = []
        for mapping in result.character_mappings:
            df_data.append({
                'Index': mapping['index'] + 1,
                'Original': mapping['original_char'],
                'Original_Unicode': mapping['original_unicode'],
                'Original_Name': mapping['original_name'][:30] + '...' if len(mapping['original_name']) > 30 else mapping['original_name'],
                'Transformed': mapping['transformed_char'],
                'Transformed_Unicode': mapping['transformed_unicode'],
                'Transformed_Name': mapping['transformed_name'][:30] + '...' if len(mapping['transformed_name']) > 30 else mapping['transformed_name'],
                'Transform_Value': mapping['transformation_value']
            })
        
        return pd.DataFrame(df_data)
    
    def analyze_transformation_quality(self, result: TransformationResult) -> Dict[str, Any]:
        """
        分析變換質量 (Analyze transformation quality)
        """
        original_analysis = self.analyzer.analyze_text(result.original_text, "original")
        transformed_analysis = self.analyzer.analyze_text(result.transformed_text, "transformed")
        
        # Calculate diversity metrics
        original_ws = set(original_analysis['writing_systems'].keys())
        transformed_ws = set(transformed_analysis['writing_systems'].keys())
        
        writing_system_diversity = len(transformed_ws) / max(len(original_ws), 1)
        character_diversity = transformed_analysis['unique_characters'] / max(original_analysis['unique_characters'], 1)
        
        # Check if target writing system was achieved
        target_achieved = result.target_writing_system in transformed_ws
        
        return {
            'writing_system_diversity': writing_system_diversity,
            'character_diversity': character_diversity,
            'target_achieved': target_achieved,
            'original_writing_systems': list(original_ws),
            'transformed_writing_systems': list(transformed_ws),
            'original_unique_chars': original_analysis['unique_characters'],
            'transformed_unique_chars': transformed_analysis['unique_characters']
        }


def demo_character_transformation():
    """演示字符變換 (Demo character transformation)"""
    transformer = CharacterTransformer()
    
    # Test different transformations
    test_texts = [
        "Hello World!",
        "我的字符",
        "Здравствуй мир",
        "مرحبا بالعالم",
        "Programming Expression Evaluation"
    ]
    
    target_systems = [
        WritingSystem.CJK,
        WritingSystem.CYRILLIC,
        WritingSystem.ARABIC,
        WritingSystem.LATIN
    ]
    
    methods = ['random', 'linear_shift', 'phonetic']
    
    print("🔄 字符變換演示 (Character Transformation Demo)")
    print("=" * 60)
    
    for text in test_texts:
        print(f"\n📝 原文: {text}")
        
        for target in target_systems[:2]:  # Limit for demo
            for method in methods[:2]:  # Limit for demo
                try:
                    result = transformer.transform_text(text, target, method)
                    quality = transformer.analyze_transformation_quality(result)
                    
                    print(f"  {method} → {target.value}: {result.transformed_text}")
                    print(f"    目標達成: {'✓' if quality['target_achieved'] else '✗'}")
                    print(f"    字符多樣性: {quality['character_diversity']:.2f}")
                    
                except Exception as e:
                    print(f"    錯誤: {e}")
        
        print("-" * 40)
    
    # Detailed example
    print("\n🔍 詳細範例 (Detailed Example)")
    print("=" * 60)
    
    example_text = "My Characters"
    result = transformer.transform_text(example_text, WritingSystem.CJK, 'random')
    
    print(f"原文: {result.original_text}")
    print(f"變換: {result.transformed_text}")
    print(f"目標書寫系統: {result.target_writing_system.value}")
    
    # Show transformation table
    df = transformer.create_transformation_table(result)
    print("\n變換表格:")
    print(df.to_string(index=False))
    
    # Quality analysis
    quality = transformer.analyze_transformation_quality(result)
    print(f"\n質量分析:")
    print(f"- 目標達成: {'✓' if quality['target_achieved'] else '✗'}")
    print(f"- 字符多樣性: {quality['character_diversity']:.2f}")
    print(f"- 書寫系統多樣性: {quality['writing_system_diversity']:.2f}")


if __name__ == "__main__":
    demo_character_transformation()