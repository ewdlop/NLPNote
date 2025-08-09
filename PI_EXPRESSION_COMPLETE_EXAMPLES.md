# π表達式評估完整示例 (Complete Pi Expression Evaluation Examples)

## 概述 (Overview)

本文檔提供了π表達式評估的完整示例，展示了如何將數學表達式評估與自然語言處理相結合，類似於程式語言中的表達式評估機制。

This document provides complete examples of π expression evaluation, demonstrating how mathematical expression evaluation can be combined with natural language processing, similar to expression evaluation mechanisms in programming languages.

## 使用方式 (Usage Instructions)

### 基本π計算 (Basic Pi Calculations)

```python
from pi_evaluator import PiExpressionEvaluator

# 初始化評估器
evaluator = PiExpressionEvaluator(precision=20)

# 萊布尼茲級數
result = evaluator.leibniz_series(10000)
print(f"π ≈ {result['pi_approximation']}")
print(f"誤差: {result['error']:.2e}")

# AGM演算法
result = evaluator.agm_algorithm(5)
print(f"π ≈ {result['pi_approximation']}")
print(f"準確位數: {result['decimal_accuracy']}")
```

### 整合表達式評估 (Integrated Expression Evaluation)

```python
from pi_expression_integration import PiExpressionIntegratedEvaluator, ExpressionContext

# 初始化整合評估器
evaluator = PiExpressionIntegratedEvaluator()

# 設定語境
context = ExpressionContext(
    situation='educational',
    formality_level='formal',
    speaker='teacher',
    listener='student'
)

# 評估包含π的自然語言表達
expression = "請用萊布尼茲級數計算π的近似值"
result = evaluator.evaluate_as_human_expression(expression, context)

print(f"認知負荷: {result['cognitive_evaluation']['cognitive_load_score']:.2f}")
print(f"教育價值: {result['educational_evaluation']['educational_score']:.2f}")
```

## 多語言示例 (Multilingual Examples)

### 中文表達 (Chinese Expressions)

1. **基本概念**: "π是圓周率，約等於3.14159"
2. **計算請求**: "請用蒙特卡羅方法計算π的值"
3. **學術討論**: "萊布尼茲級數收斂很慢，但概念很重要"
4. **教學解釋**: "π/4 = 1 - 1/3 + 1/5 - 1/7 + ..."

### English Expressions

1. **Basic concept**: "π is the ratio of circumference to diameter"
2. **Calculation request**: "Calculate π using the Machin formula"
3. **Academic discussion**: "The BBP formula allows digit extraction"
4. **Educational explanation**: "π = 4 × Σ(k=0→∞)[(-1)^k/(2k+1)]"

### 日本語表現 (Japanese Expressions)

1. **基本概念**: "πは円周率で、約3.14159です"
2. **計算依頼**: "ライプニッツ級数を使ってπを計算してください"
3. **学術議論**: "AGMアルゴリズムは高速に収束します"
4. **教育説明**: "π = 4 × arctan(1) = 4 × (1 - 1/3 + 1/5 - ...)"

## 評估維度分析 (Evaluation Dimension Analysis)

### 1. 認知複雜度 (Cognitive Complexity)

| 表達式類型 | 認知負荷 | 理解難度 | 適用對象 |
|------------|----------|----------|----------|
| π ≈ 3.14 | 低 (0.2) | 容易 | 一般大眾 |
| 萊布尼茲級數 | 中 (0.5) | 中等 | 高中以上 |
| 拉馬努金公式 | 高 (0.8) | 困難 | 大學以上 |
| AGM演算法 | 高 (0.9) | 很困難 | 數學專業 |

### 2. 社會適當性 (Social Appropriateness)

| 語境 | π常數 | π計算 | π級數 | π公式 |
|------|-------|-------|-------|-------|
| 日常對話 | 高 | 中 | 低 | 低 |
| 教育場景 | 高 | 高 | 高 | 中 |
| 學術討論 | 中 | 高 | 高 | 高 |
| 正式演講 | 中 | 高 | 高 | 高 |

### 3. 教育價值 (Educational Value)

```
π表達式的教育價值層次：

第一層 - 基本認知
├── π ≈ 3.14159... (數值概念)
├── π = C/d (幾何定義)
└── π的無理性 (數論性質)

第二層 - 計算方法
├── 幾何逼近 (阿基米德方法)
├── 無窮級數 (萊布尼茲、尼爾森)
└── 蒙特卡羅方法 (機率統計)

第三層 - 高級理論
├── 快速收斂算法 (AGM、Borwein)
├── 數位提取 (BBP公式)
└── 解析理論 (拉馬努金公式)
```

## 表達式評估流程 (Expression Evaluation Process)

### 1. 詞法分析 (Lexical Analysis)

```
輸入: "請計算π到10位小數"
詞元: [請, 計算, π, 到, 10, 位, 小數]
分類: [動詞, 動詞, 數學常數, 介詞, 數字, 量詞, 名詞]
```

### 2. 語法分析 (Syntactic Analysis)

```
語法樹:
    請求句
    ├── 動作: 計算
    ├── 對象: π
    └── 條件: 精度要求(10位小數)
```

### 3. 語義分析 (Semantic Analysis)

```
語義解釋:
- 動作類型: 數值計算
- 目標對象: 數學常數π
- 精度需求: 小數點後10位
- 計算意圖: 高精度近似
```

### 4. 語用評估 (Pragmatic Evaluation)

```
語用分析:
- 說話者意圖: 獲取π的數值
- 適當回應: 提供計算結果和方法說明
- 教育機會: 解釋π的重要性和計算方法
```

## 實際應用場景 (Practical Application Scenarios)

### 1. 教育輔助系統 (Educational Assistant System)

**輸入**: "為什麼π是無理數？"
**系統回應**:
- 檢測到π相關問題
- 認知複雜度評估: 中等
- 提供分層解答:
  1. 基本概念: π無法表示為分數
  2. 歷史背景: 蘭伯特證明(1761)
  3. 證明概要: 連分數展開的性質

### 2. 數學問答系統 (Math Q&A System)

**輸入**: "用什麼方法能最快計算π？"
**系統回應**:
- 表達式分析: 方法比較查詢
- 知識檢索: 現代高效算法
- 結果呈現:
  1. AGM算法: 二次收斂
  2. 錢德拉塞卡蘭算法: 每項14位精度
  3. 並行BBP: 可並行計算特定位數

### 3. 跨文化數學交流 (Cross-cultural Math Communication)

**場景**: 國際數學會議
**挑戰**: 多語言π表達式理解
**解決方案**:
- 統一數學記號識別
- 文化背景適應調整
- 自動翻譯和解釋

## 性能基準測試 (Performance Benchmarks)

### 計算精度對比 (Calculation Precision Comparison)

```python
# 測試不同方法達到相同精度所需的時間
methods_benchmark = {
    '萊布尼茲級數': {'時間': '慢', '複雜度': 'O(n)', '精度/項': '~0.3位'},
    '馬欽公式': {'時間': '中', '複雜度': 'O(n)', '精度/項': '~1.4位'},
    'AGM算法': {'時間': '快', '複雜度': 'O(log²n)', '精度/迭代': '位數翻倍'},
    '錢德拉塞卡蘭': {'時間': '很快', '複雜度': 'O(n)', '精度/項': '~14位'}
}
```

### 記憶體使用分析 (Memory Usage Analysis)

| 算法 | 空間複雜度 | 精度要求 | 記憶體效率 |
|------|------------|----------|------------|
| 萊布尼茲 | O(1) | 低 | 極高 |
| AGM | O(1) | 高 | 高 |
| 拉馬努金 | O(n) | 極高 | 中 |
| BBP | O(log n) | 中 | 高 |

## 錯誤處理和邊界情況 (Error Handling and Edge Cases)

### 1. 輸入驗證 (Input Validation)

```python
def validate_pi_expression(expression: str) -> Dict[str, Any]:
    """驗證π表達式輸入"""
    errors = []
    
    # 檢查空輸入
    if not expression.strip():
        errors.append("空輸入 (Empty input)")
    
    # 檢查過長輸入
    if len(expression) > 1000:
        errors.append("輸入過長 (Input too long)")
    
    # 檢查是否包含π相關內容
    if not any(char in expression for char in ['π', 'pi', 'Pi', 'PI']):
        errors.append("未檢測到π相關內容 (No π-related content detected)")
    
    return {'valid': len(errors) == 0, 'errors': errors}
```

### 2. 數值穩定性 (Numerical Stability)

```python
def handle_numerical_precision(value: float, target_precision: int) -> str:
    """處理數值精度問題"""
    try:
        from decimal import Decimal, getcontext
        getcontext().prec = target_precision + 5  # 額外精度緩衝
        
        decimal_value = Decimal(str(value))
        return str(decimal_value)[:target_precision + 2]  # +2 for "3."
    except Exception as e:
        return f"精度處理錯誤: {e}"
```

### 3. 文化適應性 (Cultural Adaptability)

```python
def adapt_to_cultural_context(expression: str, culture: str) -> str:
    """根據文化背景調整表達"""
    adaptations = {
        'chinese': {
            'decimal_separator': '.',
            'pi_symbol': 'π',
            'explanation_style': '詳細解釋',
        },
        'western': {
            'decimal_separator': '.',
            'pi_symbol': 'π',
            'explanation_style': 'concise',
        },
        'indian': {
            'decimal_separator': '.',
            'pi_symbol': 'π',
            'explanation_style': 'mathematical rigor',
        }
    }
    
    config = adaptations.get(culture, adaptations['western'])
    # 應用文化特定的調整...
    return expression  # 簡化示例
```

## 未來發展方向 (Future Development Directions)

### 1. 人工智慧增強 (AI Enhancement)

- **自然語言理解**: 更深層的數學語言理解
- **上下文推理**: 基於對話歷史的智能回應
- **個性化學習**: 根據用戶水平調整解釋深度

### 2. 多模態交互 (Multimodal Interaction)

- **視覺化展示**: 動態圖表顯示收斂過程
- **語音交互**: 語音問答π相關問題
- **手寫識別**: 識別手寫數學公式

### 3. 協作學習平台 (Collaborative Learning Platform)

- **同儕討論**: π計算方法的討論論壇
- **專家指導**: 數學專家在線答疑
- **進度追蹤**: 學習π相關概念的進度

## 結論 (Conclusion)

π表達式評估系統成功地將數學計算與自然語言處理相結合，提供了一個全面的框架來理解、評估和回應涉及π的各種表達。這個系統不僅具有實用價值，還為數學教育和跨文化交流提供了新的可能性。

The π expression evaluation system successfully combines mathematical computation with natural language processing, providing a comprehensive framework for understanding, evaluating, and responding to various expressions involving π. This system not only has practical value but also opens new possibilities for mathematics education and cross-cultural communication.

---

*本文檔展示了π表達式評估的完整生態系統，從基礎概念到高級應用，體現了數學與語言技術融合的巨大潛力。*

*This document demonstrates the complete ecosystem of π expression evaluation, from basic concepts to advanced applications, reflecting the great potential of integrating mathematics with language technology.*