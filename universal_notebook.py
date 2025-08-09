#!/usr/bin/env python3
"""
Universal Planetary Notebook - 全行星通用的 Notebook

A lightweight, web-based notebook interface that works without Jupyter dependencies
and integrates with the existing NLP analysis tools in this repository.

Features:
- Cross-platform compatibility 跨平台兼容性
- Multilingual support 多语言支持  
- Integration with existing NLP tools 集成现有NLP工具
- Code execution environment 代码执行环境
- Markdown rendering Markdown渲染
"""

import os
import sys
import json
import html
import base64
import traceback
import subprocess
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path
import http.server
import socketserver
import webbrowser
import threading
import time

# Try to import existing NLP modules
try:
    from HumanExpressionEvaluator import HumanExpressionEvaluator, ExpressionContext
    HAS_EXPRESSION_EVALUATOR = True
    print("✅ Human Expression Evaluator loaded")
except ImportError as e:
    HAS_EXPRESSION_EVALUATOR = False
    print(f"⚠️  Human Expression Evaluator not available: {e}")

try:
    from SubtextAnalyzer import SubtextAnalyzer
    HAS_SUBTEXT_ANALYZER = True
    print("✅ Subtext Analyzer loaded")
except ImportError as e:
    HAS_SUBTEXT_ANALYZER = False
    print(f"⚠️  Subtext Analyzer not available: {e}")

try:
    import AStarNLP
    HAS_ASTAR_NLP = True
    print("✅ A* NLP loaded")
except ImportError as e:
    HAS_ASTAR_NLP = False
    print(f"⚠️  A* NLP not available: {e}")

# Create simplified NLP tools for demo purposes when dependencies are missing
class SimplifiedExpressionEvaluator:
    """Simplified expression evaluator for demo purposes"""
    
    def comprehensive_evaluation(self, text, context=None):
        """Simple mock evaluation"""
        return {
            "overall_score": 0.75,
            "sentiment": "positive" if any(word in text.lower() for word in ["好", "棒", "great", "awesome", "nice"]) else "neutral",
            "complexity": len(text.split()) / 10.0,
            "language": "chinese" if any('\u4e00' <= char <= '\u9fff' for char in text) else "english"
        }

class SimplifiedSubtextAnalyzer:
    """Simplified subtext analyzer for demo purposes"""
    
    def analyze_subtext(self, text):
        """Simple mock analysis"""
        return {
            "subtext_probability": 0.6 if "?" in text or "..." in text else 0.3,
            "emotional_undertone": "questioning" if "?" in text else "neutral",
            "hidden_meaning_detected": len(text) > 50
        }


class Cell:
    """Represents a notebook cell 笔记本单元格"""
    
    def __init__(self, cell_type: str = "code", content: str = "", metadata: Dict = None):
        self.cell_type = cell_type  # "code", "markdown", "text"
        self.content = content
        self.metadata = metadata or {}
        self.output = ""
        self.execution_count = 0
        self.timestamp = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert cell to dictionary format"""
        return {
            "cell_type": self.cell_type,
            "content": self.content,
            "metadata": self.metadata,
            "output": self.output,
            "execution_count": self.execution_count,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Cell':
        """Create cell from dictionary format"""
        cell = cls(
            cell_type=data.get("cell_type", "code"),
            content=data.get("content", ""),
            metadata=data.get("metadata", {})
        )
        cell.output = data.get("output", "")
        cell.execution_count = data.get("execution_count", 0)
        cell.timestamp = data.get("timestamp")
        return cell


class UniversalNotebook:
    """Universal Planetary Notebook implementation"""
    
    def __init__(self, title: str = "Universal Notebook"):
        self.title = title
        self.cells: List[Cell] = []
        self.metadata = {
            "created": datetime.now().isoformat(),
            "language": "python",
            "kernel": "universal",
            "version": "1.0"
        }
        self.execution_count = 0
        
        # Initialize NLP tools if available
        self.nlp_tools = {}
        if HAS_EXPRESSION_EVALUATOR:
            self.nlp_tools['expression_evaluator'] = HumanExpressionEvaluator()
        else:
            # Provide simplified version for demo
            self.nlp_tools['expression_evaluator'] = SimplifiedExpressionEvaluator()
            
        if HAS_SUBTEXT_ANALYZER:
            self.nlp_tools['subtext_analyzer'] = SubtextAnalyzer()
        else:
            # Provide simplified version for demo  
            self.nlp_tools['subtext_analyzer'] = SimplifiedSubtextAnalyzer()
    
    def add_cell(self, cell_type: str = "code", content: str = "") -> Cell:
        """Add a new cell to the notebook"""
        cell = Cell(cell_type, content)
        self.cells.append(cell)
        return cell
    
    def execute_cell(self, cell_index: int) -> Dict[str, Any]:
        """Execute a specific cell"""
        if cell_index >= len(self.cells):
            return {"error": "Cell index out of range"}
        
        cell = self.cells[cell_index]
        
        if cell.cell_type == "code":
            return self._execute_code_cell(cell)
        elif cell.cell_type == "markdown":
            return self._render_markdown_cell(cell)
        else:
            return {"output": cell.content, "type": "text"}
    
    def _execute_code_cell(self, cell: Cell) -> Dict[str, Any]:
        """Execute a code cell"""
        try:
            self.execution_count += 1
            cell.execution_count = self.execution_count
            cell.timestamp = datetime.now().isoformat()
            
            # Create execution environment with NLP tools
            exec_globals = {
                '__builtins__': __builtins__,
                'print': self._capture_print,
                'nlp_tools': self.nlp_tools,
                **self.nlp_tools  # Make tools directly available
            }
            
            # Capture output
            self._output_buffer = []
            
            # Execute the code
            exec(cell.content, exec_globals)
            
            # Get captured output
            output = '\n'.join(self._output_buffer)
            cell.output = output
            
            return {
                "output": output,
                "type": "code_result",
                "execution_count": cell.execution_count
            }
            
        except Exception as e:
            error_msg = f"Error: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            cell.output = error_msg
            return {
                "output": error_msg,
                "type": "error",
                "execution_count": cell.execution_count
            }
    
    def _capture_print(self, *args, **kwargs):
        """Capture print output"""
        if not hasattr(self, '_output_buffer'):
            self._output_buffer = []
        output = ' '.join(str(arg) for arg in args)
        self._output_buffer.append(output)
    
    def _render_markdown_cell(self, cell: Cell) -> Dict[str, Any]:
        """Render markdown cell (basic implementation)"""
        # Simple markdown-to-HTML conversion
        content = cell.content
        
        # Basic markdown patterns
        content = content.replace('**', '<b>').replace('**', '</b>')
        content = content.replace('*', '<i>').replace('*', '</i>')
        content = content.replace('\n', '<br>')
        
        # Headers
        for i in range(6, 0, -1):
            header_pattern = '#' * i + ' '
            if content.startswith(header_pattern):
                content = f'<h{i}>{content[len(header_pattern):]}</h{i}>'
                break
        
        cell.output = content
        return {
            "output": content,
            "type": "markdown_html"
        }
    
    def save(self, filename: str):
        """Save notebook to file"""
        notebook_data = {
            "title": self.title,
            "metadata": self.metadata,
            "cells": [cell.to_dict() for cell in self.cells]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(notebook_data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filename: str) -> 'UniversalNotebook':
        """Load notebook from file"""
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        notebook = cls(data.get("title", "Universal Notebook"))
        notebook.metadata = data.get("metadata", {})
        
        for cell_data in data.get("cells", []):
            cell = Cell.from_dict(cell_data)
            notebook.cells.append(cell)
        
        return notebook
    
    def to_html(self) -> str:
        """Convert notebook to HTML format"""
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Universal Planetary Notebook</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .notebook-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }}
        .cell {{
            background: white;
            margin: 15px 0;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .cell-input {{
            background: #f8f9fa;
            padding: 15px;
            border-bottom: 1px solid #e9ecef;
            font-family: 'Monaco', 'Consolas', monospace;
        }}
        .cell-output {{
            padding: 15px;
        }}
        .code-cell .cell-input {{
            background: #2d3748;
            color: #e2e8f0;
        }}
        .markdown-cell .cell-input {{
            background: #e8f5e8;
        }}
        .error {{
            background: #fed7d7;
            color: #c53030;
        }}
        .execution-info {{
            font-size: 0.8em;
            color: #666;
            padding: 5px 15px;
            background: #f1f3f4;
        }}
        .nlp-tools-info {{
            background: #e3f2fd;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid #2196f3;
        }}
    </style>
</head>
<body>
    <div class="notebook-header">
        <h1>{title}</h1>
        <p>全行星通用的 Notebook - Universal Planetary Notebook</p>
        <p>Created: {created} | Language: {language}</p>
    </div>
    
    <div class="nlp-tools-info">
        <strong>Available NLP Tools 可用的NLP工具:</strong><br>
        {nlp_tools_status}
    </div>
    
    {cells_html}
</body>
</html>
        """
        
        # Generate cells HTML
        cells_html = ""
        for i, cell in enumerate(self.cells):
            cell_html = f"""
    <div class="cell {cell.cell_type}-cell">
        <div class="cell-input">
            <pre>{html.escape(cell.content)}</pre>
        </div>
        {f'<div class="execution-info">Execution {cell.execution_count} - {cell.timestamp}</div>' if cell.execution_count else ''}
        <div class="cell-output {'error' if 'Error:' in cell.output else ''}">
            <pre>{html.escape(cell.output) if cell.cell_type == 'code' else cell.output}</pre>
        </div>
    </div>
            """
            cells_html += cell_html
        
        # NLP tools status
        nlp_status = []
        if HAS_EXPRESSION_EVALUATOR:
            nlp_status.append("✅ Human Expression Evaluator 人类表达评估器")
        if HAS_SUBTEXT_ANALYZER:
            nlp_status.append("✅ Subtext Analyzer 潜文本分析器")
        if HAS_ASTAR_NLP:
            nlp_status.append("✅ A* NLP")
        
        if not nlp_status:
            nlp_status.append("❌ No NLP tools available")
        
        return html_template.format(
            title=html.escape(self.title),
            created=self.metadata.get('created', 'Unknown'),
            language=self.metadata.get('language', 'python'),
            nlp_tools_status='<br>'.join(nlp_status),
            cells_html=cells_html
        )


class NotebookServer:
    """Simple HTTP server for the notebook interface"""
    
    def __init__(self, notebook: UniversalNotebook, port: int = 8888):
        self.notebook = notebook
        self.port = port
        self.server = None
    
    def start(self):
        """Start the notebook server"""
        handler = self._create_handler()
        self.server = socketserver.TCPServer(("", self.port), handler)
        
        print(f"🚀 Universal Planetary Notebook running at http://localhost:{self.port}")
        print(f"📝 Notebook: {self.notebook.title}")
        print(f"🛠️  Available NLP tools: {len(self.notebook.nlp_tools)}")
        print("Press Ctrl+C to stop the server")
        
        # Open browser automatically
        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{self.port}")).start()
        
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped")
            self.server.shutdown()
    
    def _create_handler(self):
        notebook = self.notebook
        
        class NotebookHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/' or self.path == '/notebook':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html; charset=utf-8')
                    self.end_headers()
                    html_content = notebook.to_html()
                    self.wfile.write(html_content.encode('utf-8'))
                else:
                    super().do_GET()
        
        return NotebookHandler


def create_demo_notebook() -> UniversalNotebook:
    """Create a demo notebook with sample content"""
    notebook = UniversalNotebook("Universal Planetary Notebook Demo")
    
    # Welcome cell
    cell1 = notebook.add_cell("markdown", """# Welcome to Universal Planetary Notebook
## 欢迎使用全行星通用笔记本

This is a lightweight, cross-platform notebook interface that integrates with NLP analysis tools.

### Features 特性:
- **Cross-platform compatibility** 跨平台兼容性
- **Multilingual support** 多语言支持
- **NLP tools integration** NLP工具集成
- **No Jupyter dependency** 无Jupyter依赖""")
    
    # Python code example
    cell2 = notebook.add_cell("code", """# Basic Python execution
print("Hello from Universal Notebook! 你好，来自全宇宙笔记本！")
print(f"Python version: {sys.version}")
print(f"Available NLP tools: {list(nlp_tools.keys())}")""")
    
    # NLP tools demonstration
    cell3 = notebook.add_cell("code", """# NLP Analysis Example - Works with or without full dependencies
print("Available NLP tools:", list(nlp_tools.keys()))
print()

# Test expression evaluation (simplified or full version)
if 'expression_evaluator' in nlp_tools:
    evaluator = nlp_tools['expression_evaluator']
    
    # Test with Chinese text
    result1 = evaluator.comprehensive_evaluation("这个笔记本很有趣！")
    print("Expression Analysis for Chinese text '这个笔记本很有趣！':")
    for key, value in result1.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    print()
    
    # Test with English text
    result2 = evaluator.comprehensive_evaluation("This Universal Notebook is awesome!")
    print("Expression Analysis for English text 'This Universal Notebook is awesome!':")
    for key, value in result2.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

print()

# Test subtext analysis
if 'subtext_analyzer' in nlp_tools:
    analyzer = nlp_tools['subtext_analyzer']
    
    texts = [
        "The sunset was beautiful tonight.",
        "Are you sure about that...?",
        "Everything is fine, absolutely fine."
    ]
    
    print("Subtext Analysis Results:")
    for text in texts:
        analysis = analyzer.analyze_subtext(text)
        print(f"\\nText: '{text}'")
        for key, value in analysis.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")""")
    
    # Mathematical operations
    cell4 = notebook.add_cell("code", """# Mathematical operations and visualization
import math
import random

# Generate some data
data = [random.gauss(0, 1) for _ in range(20)]
mean = sum(data) / len(data)
variance = sum((x - mean) ** 2 for x in data) / len(data)

print(f"Sample data: {[round(x, 2) for x in data[:10]]}...")
print(f"Mean: {mean:.3f}")
print(f"Variance: {variance:.3f}")
print(f"Standard deviation: {math.sqrt(variance):.3f}")""")
    
    # Markdown documentation
    cell5 = notebook.add_cell("markdown", """## How to Use 使用方法

1. **Add cells** by modifying the notebook programmatically
2. **Execute cells** to see results
3. **Mix markdown and code** for documentation
4. **Use NLP tools** for text analysis

### Supported Cell Types 支持的单元格类型:
- `code`: Python code execution
- `markdown`: Documentation and formatting  
- `text`: Plain text

### Integration with Existing Tools 与现有工具的集成:
This notebook integrates with the existing NLP tools in the repository.""")
    
    return notebook


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Universal Planetary Notebook")
    parser.add_argument("--port", type=int, default=8888, help="Server port")
    parser.add_argument("--file", type=str, help="Load notebook from file")
    parser.add_argument("--demo", action="store_true", help="Create demo notebook")
    args = parser.parse_args()
    
    if args.file and os.path.exists(args.file):
        notebook = UniversalNotebook.load(args.file)
        print(f"📖 Loaded notebook from {args.file}")
    elif args.demo:
        notebook = create_demo_notebook()
        print("🎭 Created demo notebook")
    else:
        notebook = UniversalNotebook("New Universal Notebook")
        print("📝 Created new notebook")
    
    # Start server
    server = NotebookServer(notebook, args.port)
    server.start()


if __name__ == "__main__":
    main()