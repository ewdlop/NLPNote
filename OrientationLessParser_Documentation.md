# OrientationLessParser Documentation
# 無方向性解析器文檔

## Overview 概述

The **OrientationLessParser** is a text parsing system designed to handle text content regardless of its orientation, direction, or cultural layout. It provides orientation-agnostic processing capabilities for multilingual and bidirectional text.

**無方向性解析器** 是一個文本解析系統，旨在處理任意方向、方向或文化布局的文本內容。它為多語言和雙向文本提供無方向性的處理能力。

## Key Features 主要特性

### 1. Orientation Independence 方向無關性
- **Left-to-Right (LTR)**: English, Spanish, French, etc.
- **Right-to-Left (RTL)**: Arabic, Hebrew, etc.
- **Bidirectional**: Mixed LTR and RTL content
- **Neutral**: Numbers, punctuation, symbols

### 2. Script Support 文字支持
- **Latin**: English, European languages
- **Arabic**: Arabic script languages
- **Hebrew**: Hebrew script
- **CJK**: Chinese, Japanese, Korean
- **Devanagari**: Hindi, Sanskrit
- **Cyrillic**: Russian, Bulgarian, etc.
- **Thai**: Thai script
- **Mixed**: Multiple scripts in one text

### 3. Advanced Features 高級功能
- Unicode normalization
- Directional marker removal
- Logical text reordering
- Bidirectional algorithm implementation
- Statistical analysis

## Installation 安裝

The OrientationLessParser requires only Python standard library components. No external dependencies are needed for basic functionality.

無方向性解析器只需要Python標準庫組件。基本功能不需要外部依賴。

```python
# Simply import the module
from OrientationLessParser import OrientationLessParser, TextDirection, ScriptType
```

## Basic Usage 基本用法

### Simple Parsing 簡單解析

```python
from OrientationLessParser import OrientationLessParser

# Create parser instance
parser = OrientationLessParser()

# Parse text
text = "Hello مرحبا 你好"
result = parser.parse(text)

# Access results
print(f"Direction: {result.dominant_direction}")
print(f"Script: {result.dominant_script}")
print(f"Mixed: {result.has_mixed_directions}")
```

### Extract Content 提取內容

```python
# Extract clean text content in logical order
logical_text = parser.extract_text_content(result)
print(f"Logical order: {logical_text}")

# Extract without punctuation
clean_text = parser.extract_text_content(result, include_punctuation=False)
print(f"No punctuation: {clean_text}")
```

### Get Statistics 獲取統計

```python
# Get parsing statistics
stats = parser.get_parsing_statistics(result)
print(f"Word tokens: {stats['word_tokens']}")
print(f"Direction distribution: {stats['direction_distribution']}")
print(f"Script distribution: {stats['script_distribution']}")
```

## API Reference API參考

### Classes 類

#### OrientationLessParser

Main parser class that provides orientation-agnostic text processing.

主要解析器類，提供無方向性文本處理。

**Methods 方法:**

- `parse(text: str) -> ParseResult`: Parse text and return comprehensive results
- `normalize_text(text: str) -> str`: Normalize text by removing directional bias
- `detect_script_type(text: str) -> ScriptType`: Detect the script type of text
- `detect_text_direction(text: str) -> TextDirection`: Detect text direction
- `tokenize_orientation_agnostic(text: str) -> List[ParsedToken]`: Tokenize text
- `extract_text_content(result: ParseResult, **kwargs) -> str`: Extract clean content
- `get_parsing_statistics(result: ParseResult) -> Dict`: Get parsing statistics

#### TextDirection (Enum)

Enumeration of text directions.

文本方向枚舉。

- `LTR`: Left-to-right (從左到右)
- `RTL`: Right-to-left (從右到左)
- `BIDI`: Bidirectional (雙向)
- `NEUTRAL`: Neutral (中性)
- `MIXED`: Mixed directions (混合方向)

#### ScriptType (Enum)

Enumeration of script types.

文字類型枚舉。

- `LATIN`: Latin script (拉丁文字)
- `ARABIC`: Arabic script (阿拉伯文字)
- `HEBREW`: Hebrew script (希伯來文字)
- `CJK`: Chinese/Japanese/Korean (中日韓文字)
- `DEVANAGARI`: Devanagari script (天城文字)
- `CYRILLIC`: Cyrillic script (西里爾文字)
- `THAI`: Thai script (泰文)
- `MIXED`: Mixed scripts (混合文字)
- `UNKNOWN`: Unknown script (未知文字)

#### ParsedToken

Represents a parsed token with orientation information.

表示帶有方向信息的解析標記。

**Attributes 屬性:**

- `text: str`: Token text
- `original_position: int`: Position in original text
- `logical_position: int`: Position in logical order
- `direction: TextDirection`: Token direction
- `script_type: ScriptType`: Token script type
- `is_punctuation: bool`: Whether token is punctuation
- `is_number: bool`: Whether token is number
- `is_whitespace: bool`: Whether token is whitespace

#### ParseResult

Contains comprehensive parsing results.

包含綜合解析結果。

**Attributes 屬性:**

- `original_text: str`: Original input text
- `tokens: List[ParsedToken]`: Parsed tokens in original order
- `dominant_direction: TextDirection`: Dominant text direction
- `dominant_script: ScriptType`: Dominant script type
- `has_mixed_directions: bool`: Whether text has mixed directions
- `normalized_text: str`: Normalized text
- `logical_order: List[ParsedToken]`: Tokens in logical reading order

## Advanced Usage 高級用法

### Handling Complex Text 處理復雜文本

```python
# Complex bidirectional text
complex_text = """
The Arabic phrase مرحبا بك في عالم البرمجة means 
"welcome to the programming world" in English.
Hebrew שלום עולם means "hello world".
Chinese 你好世界 also means "hello world".
"""

result = parser.parse(complex_text)

# Analyze complexity
if result.has_mixed_directions:
    print("⚠️  Complex bidirectional text detected")
    print("🔄 Logical reordering applied")

# Get detailed token information
for token in result.tokens:
    if not token.is_whitespace:
        print(f"'{token.text}' -> {token.direction.value} ({token.script_type.value})")
```

### Custom Processing 自定義處理

```python
# Filter tokens by script type
arabic_tokens = [t for t in result.tokens if t.script_type == ScriptType.ARABIC]
latin_tokens = [t for t in result.tokens if t.script_type == ScriptType.LATIN]

# Process by direction
rtl_content = [t.text for t in result.tokens if t.direction == TextDirection.RTL]
ltr_content = [t.text for t in result.tokens if t.direction == TextDirection.LTR]

# Custom text extraction
def extract_by_script(result, script_type):
    """Extract text content by script type"""
    tokens = [t for t in result.logical_order 
              if t.script_type == script_type and not t.is_whitespace]
    return ' '.join(t.text for t in tokens)

arabic_content = extract_by_script(result, ScriptType.ARABIC)
print(f"Arabic content: {arabic_content}")
```

## Integration with Existing Framework 與現有框架整合

The OrientationLessParser is designed to integrate seamlessly with the existing NLP framework:

無方向性解析器設計為與現有NLP框架無縫整合：

```python
# Integration example
from OrientationLessParser import OrientationLessParser

# Use with existing analysis tools
def enhanced_analysis(text):
    # Step 1: Orientation-agnostic parsing
    parser = OrientationLessParser()
    orientation_result = parser.parse(text)
    
    # Step 2: Use normalized text for other analyses
    normalized_text = orientation_result.normalized_text
    logical_text = parser.extract_text_content(orientation_result)
    
    # Step 3: Apply existing framework tools to normalized text
    # (HumanExpressionEvaluator, SubtextAnalyzer, etc.)
    
    return {
        'orientation_analysis': orientation_result,
        'normalized_text': normalized_text,
        'logical_text': logical_text
    }
```

## Performance Considerations 性能考慮

### Optimization Tips 優化建議

1. **Reuse Parser Instance**: Create one parser instance and reuse it
   **重用解析器實例**：創建一個解析器實例並重複使用

2. **Batch Processing**: Process multiple texts in sequence
   **批量處理**：按順序處理多個文本

3. **Cache Results**: Cache parsing results for repeated texts
   **緩存結果**：為重複文本緩存解析結果

```python
# Efficient batch processing
parser = OrientationLessParser()
results = []

for text in text_list:
    result = parser.parse(text)
    results.append(result)
```

## Best Practices 最佳實踐

### 1. Text Preprocessing 文本預處理

```python
# Clean input text before parsing
def preprocess_text(text):
    # Remove unnecessary whitespace
    text = ' '.join(text.split())
    
    # Handle common encoding issues
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    
    return text

cleaned_text = preprocess_text(raw_text)
result = parser.parse(cleaned_text)
```

### 2. Error Handling 錯誤處理

```python
def safe_parse(text):
    try:
        parser = OrientationLessParser()
        result = parser.parse(text)
        return result
    except Exception as e:
        print(f"Parsing error: {e}")
        return None
```

### 3. Validation 驗證

```python
def validate_result(result):
    """Validate parsing result"""
    if not result:
        return False
    
    # Check for reasonable token count
    if len(result.tokens) == 0 and len(result.original_text.strip()) > 0:
        return False
    
    # Check logical order consistency
    if len(result.logical_order) != len(result.tokens):
        return False
    
    return True
```

## Testing 測試

Run the comprehensive test suite to verify functionality:

運行綜合測試套件來驗證功能：

```bash
python3 test_orientation_less_parser.py
```

Run quick demonstration:

運行快速演示：

```bash
python3 quick_demo.py
```

## Troubleshooting 故障排除

### Common Issues 常見問題

1. **Unicode Handling**: Ensure text is properly encoded as UTF-8
   **Unicode處理**：確保文本正確編碼為UTF-8

2. **Mixed Content**: Use logical order for display in mixed-direction text
   **混合內容**：在混合方向文本中使用邏輯順序進行顯示

3. **Performance**: For large texts, consider chunking into smaller segments
   **性能**：對於大文本，考慮分割成較小的段落

### Debug Information 調試信息

```python
# Enable detailed analysis
result = parser.parse(text)
stats = parser.get_parsing_statistics(result)

print("Debug Information:")
print(f"Total tokens: {stats['total_tokens']}")
print(f"Direction distribution: {stats['direction_distribution']}")
print(f"Script distribution: {stats['script_distribution']}")
print(f"Bidirectional: {stats['is_bidirectional']}")
```

## Contributing 貢獻

The OrientationLessParser is part of the NLPNote project. Contributions are welcome to improve:

無方向性解析器是NLPNote項目的一部分。歡迎貢獻以改進：

- Additional script support
- Performance optimizations
- Better bidirectional algorithms
- Enhanced Unicode handling
- More comprehensive testing

## License 許可證

This project follows the same license as the parent NLPNote repository.

本項目遵循父NLPNote存儲庫的相同許可證。

---

*This documentation covers the core functionality of OrientationLessParser. For the latest updates and examples, refer to the source code and test files.*

*本文檔涵蓋了無方向性解析器的核心功能。有關最新更新和示例，請參考源代碼和測試文件。*