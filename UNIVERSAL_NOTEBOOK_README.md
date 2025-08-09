# Universal Planetary Notebook 全行星通用笔记本

A lightweight, cross-platform notebook interface that integrates with NLP analysis tools without requiring Jupyter dependencies.

## Features 特性

- 🌍 **Cross-platform compatibility** 跨平台兼容性
- 🗣️ **Multilingual support** 多语言支持 (中文/English)
- 🧠 **NLP tools integration** NLP工具集成
- 🚀 **No external dependencies** 无外部依赖
- 💻 **Python code execution** Python代码执行
- 📝 **Markdown rendering** Markdown渲染
- 🌐 **Web-based interface** 基于Web的界面
- 💾 **Save/Export functionality** 保存/导出功能

## Quick Start 快速开始

### Option 1: Python Server Mode (推荐)

```bash
# Launch with demo content
python3 launch_notebook.py --demo

# Launch empty notebook
python3 launch_notebook.py

# Launch on specific port
python3 launch_notebook.py --port 9999

# Load existing notebook file
python3 launch_notebook.py --file my_notebook.upnb
```

### Option 2: Standalone Web Interface

```bash
# Open interactive web interface (no server needed)
python3 launch_notebook.py --interface
```

Or simply open `notebook_interface.html` in your web browser.

## Available NLP Tools 可用的NLP工具

The notebook automatically integrates with existing NLP tools in the repository:

- ✅ **Human Expression Evaluator** 人类表达评估器
  - Evaluates human expressions with cognitive and social dimensions
  - Located in `HumanExpressionEvaluator.py`
  
- ✅ **Subtext Analyzer** 潜文本分析器  
  - Analyzes hidden meanings and implications in text
  - Located in `SubtextAnalyzer.py`
  
- ✅ **A* NLP** 
  - Advanced pathfinding algorithms for natural language processing
  - Located in `AStarNLP.py`

## Usage Examples 使用示例

### Code Cell Example 代码单元格示例

```python
# Basic Python execution
print("Hello from Universal Notebook! 你好！")

# Use NLP tools directly
if 'expression_evaluator' in nlp_tools:
    from HumanExpressionEvaluator import ExpressionContext
    evaluator = nlp_tools['expression_evaluator']
    
    context = ExpressionContext(
        speaker='user',
        listener='ai', 
        situation='demo'
    )
    
    result = evaluator.comprehensive_evaluation("这很有趣！", context)
    print(f"Evaluation score: {result}")
```

### Markdown Cell Example Markdown单元格示例

```markdown
# My Analysis 我的分析

This notebook supports **rich text formatting** including:

- **Bold text** 粗体文本
- *Italic text* 斜体文本
- Headers 标题
- Lists 列表

## Mathematical Expressions 数学表达式

You can include mathematical concepts and explain NLP algorithms.
```

## File Formats 文件格式

### Universal Notebook Format (.upnb)

```json
{
  "title": "My Notebook",
  "metadata": {
    "created": "2024-08-09T01:00:00Z",
    "language": "python",
    "kernel": "universal",
    "version": "1.0"
  },
  "cells": [
    {
      "cell_type": "code",
      "content": "print('Hello World')",
      "output": "Hello World",
      "execution_count": 1
    }
  ]
}
```

### Export Options 导出选项

- **HTML Export**: Full standalone HTML with embedded CSS
- **JSON Export**: Native `.upnb` format for sharing
- **Python Script**: Extract code cells as executable Python

## Integration with Existing Tools 与现有工具的集成

The Universal Notebook automatically detects and integrates with:

1. **HumanExpressionEvaluator.py** - For analyzing human expressions
2. **SubtextAnalyzer.py** - For subtext analysis  
3. **AStarNLP.py** - For advanced NLP algorithms
4. **expression_evaluation_examples.py** - For example demonstrations

### Using in Code Cells

```python
# All NLP tools are available through the nlp_tools dictionary
print("Available tools:", list(nlp_tools.keys()))

# Direct access to tools
evaluator = nlp_tools.get('expression_evaluator')
analyzer = nlp_tools.get('subtext_analyzer')

# Or import modules directly if available
from HumanExpressionEvaluator import ExpressionContext
```

## Architecture 架构

```
Universal Planetary Notebook
├── universal_notebook.py      # Core notebook implementation
├── notebook_interface.html    # Standalone web interface  
├── launch_notebook.py         # Launcher script
└── Integration with:
    ├── HumanExpressionEvaluator.py
    ├── SubtextAnalyzer.py
    ├── AStarNLP.py
    └── expression_evaluation_examples.py
```

## System Requirements 系统要求

- **Python 3.7+** 
- **Web browser** for interface 网页浏览器
- **No external dependencies** required 无需外部依赖

Optional NLP tools require their specific dependencies (see individual tool documentation).

## Keyboard Shortcuts 键盘快捷键

- `Ctrl+S` / `Cmd+S`: Save notebook
- `Shift+Enter`: Execute current cell (in web interface)
- `Ctrl+Enter`: Add new cell below

## Comparison with Jupyter 与Jupyter的比较

| Feature | Universal Notebook | Jupyter |
|---------|-------------------|---------|
| **Installation** | No dependencies | Complex setup |
| **File Size** | Lightweight | Heavy |
| **NLP Integration** | Built-in | Manual setup |
| **Multilingual** | Native support | Plugin required |
| **Offline Usage** | Full support | Limited |
| **Customization** | Easy to modify | Complex |

## Contributing 贡献

The Universal Planetary Notebook is designed to be:

1. **Easy to extend** - Add new cell types or tools
2. **Multilingual** - Support for multiple languages
3. **Platform agnostic** - Works anywhere Python runs
4. **Integration friendly** - Easy to connect with existing tools

## Examples in Other Languages 其他语言示例

### 中文示例 Chinese Example

```python
# 中文自然语言处理示例
text = "今天天气真不错，适合出去走走。"

if 'subtext_analyzer' in nlp_tools:
    analyzer = nlp_tools['subtext_analyzer']
    result = analyzer.analyze_subtext(text)
    print(f"潜文本分析结果: {result}")
```

### 日本語の例 Japanese Example

```python
# 日本語テキスト解析
japanese_text = "今日はとても良い天気ですね。"
print(f"分析対象: {japanese_text}")
```

## License

This Universal Planetary Notebook is part of the NLPNote project. See the main repository for license information.

---

**🪐 Universal Planetary Notebook - Bringing NLP analysis to everyone, everywhere**
**全行星通用笔记本 - 将NLP分析带给每个人，随时随地**