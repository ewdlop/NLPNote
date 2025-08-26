# 語義網知識庫 (Semantic Web Knowledge Base)

## 概述 (Overview)

本專案為 NLP Note 儲存庫添加了完整的語義網知識庫功能，支援 RDF/OWL 本體建模、SPARQL 查詢，以及與 Protégé 本體編輯器的完全相容性。

This project adds comprehensive semantic web knowledge base capabilities to the NLP Note repository, supporting RDF/OWL ontology modeling, SPARQL querying, and full compatibility with the Protégé ontology editor.

## 主要功能 (Key Features)

### 🔹 RDF/OWL 支援 (RDF/OWL Support)
- **三元組管理 (Triple Management)**: 完整的 RDF 三元組增加、查詢、管理功能
- **OWL 本體 (OWL Ontologies)**: 支援複雜的 OWL 本體結構，包括類別層次、屬性定義
- **推理機制 (Reasoning)**: 整合 OWL 推理器，自動推導隱含知識

### 🔹 SPARQL 查詢 (SPARQL Querying)
- **標準查詢 (Standard Queries)**: 支援完整的 SPARQL 1.1 查詢語法
- **複雜查詢 (Complex Queries)**: 支援聯合查詢、可選模式、過濾條件
- **結果處理 (Result Processing)**: 自動將查詢結果轉換為易於處理的格式

### 🔹 Protégé 整合 (Protégé Integration)
- **檔案匯出 (File Export)**: 匯出與 Protégé 完全相容的 OWL 檔案
- **多格式支援 (Multiple Formats)**: 支援 XML、Turtle、N3 等多種格式
- **本體驗證 (Ontology Validation)**: 確保匯出的本體符合 OWL 標準

### 🔹 NLP 領域建模 (NLP Domain Modeling)
- **語言資源 (Language Resources)**: 多語言支援，語言家族建模
- **文本分析 (Text Analysis)**: 語義標註、實體識別、關係抽取
- **概念關係 (Concept Relations)**: 同義詞、上下位詞、語義關係建模

## 快速開始 (Quick Start)

### 安裝依賴 (Installation)

```bash
pip install rdflib owlrl SPARQLWrapper
```

### 基本使用 (Basic Usage)

```python
from SemanticKnowledgeBase import SemanticKnowledgeBase, create_sample_nlp_knowledge_base

# 創建知識庫 (Create knowledge base)
kb = SemanticKnowledgeBase()

# 添加語言 (Add languages)
kb.add_language('en', 'English', 'Indo-European')
kb.add_language('zh', 'Chinese', 'Sino-Tibetan')

# 添加文本文檔 (Add text documents)
kb.add_text_document('doc1', 'Natural language processing is fascinating.', 'en',
                     {'title': 'NLP Introduction', 'creator': 'AI Researcher'})

# 添加概念關係 (Add concept relations)
kb.add_concept_relation('natural language processing', 'synonymOf', 'NLP', 0.95)

# SPARQL 查詢 (SPARQL queries)
query = """
PREFIX nlp: <http://example.org/nlp/>
SELECT ?lang ?name WHERE {
    ?lang a nlp:Language .
    ?lang rdfs:label ?name .
}
"""
results = kb.sparql_query(query)

# 匯出到 Protégé (Export to Protégé)
kb.export_to_owl('my_ontology.owl')
```

### 創建示例知識庫 (Create Sample Knowledge Base)

```python
# 使用預建的示例知識庫 (Use pre-built sample knowledge base)
kb = create_sample_nlp_knowledge_base()

# 查看統計資訊 (View statistics)
stats = kb.get_statistics()
print(f"總三元組數: {stats['total_triples']}")
print(f"語言數量: {stats['languages']}")
print(f"文檔數量: {stats['documents']}")
```

## 詳細範例 (Detailed Examples)

### 範例 1: 基礎本體創建 (Basic Ontology Creation)

```python
def create_linguistic_hierarchy():
    kb = SemanticKnowledgeBase()
    
    # 建立語言學層次結構 (Create linguistic hierarchy)
    kb.add_concept_relation('phoneme', 'isPartOf', 'syllable')
    kb.add_concept_relation('syllable', 'isPartOf', 'word')
    kb.add_concept_relation('word', 'isPartOf', 'phrase')
    kb.add_concept_relation('phrase', 'isPartOf', 'sentence')
    
    return kb
```

### 範例 2: 多語言內容表示 (Multi-lingual Content)

```python
def add_multilingual_greetings():
    kb = SemanticKnowledgeBase()
    
    # 添加多語言問候語 (Add multi-lingual greetings)
    greetings = [
        ('en', 'hello'),
        ('zh', '你好'),
        ('ja', 'こんにちは'),
        ('ar', 'مرحبا')
    ]
    
    for lang, greeting in greetings:
        kb.add_language(lang, f"Language_{lang}")
        concept_uri = kb.nlp_ns[f"greeting_{lang}"]
        kb.graph.add((concept_uri, kb.nlp_ns.lexicalForm, Literal(greeting)))
        kb.graph.add((concept_uri, kb.nlp_ns.hasLanguage, kb.nlp_ns[f"language_{lang}"]))
    
    return kb
```

### 範例 3: 語義文本分析 (Semantic Text Analysis)

```python
def analyze_text_semantically():
    kb = SemanticKnowledgeBase()
    
    # 添加文檔並進行語義標註 (Add document with semantic annotation)
    doc_text = "Natural language processing enables machines to understand human language."
    doc_uri = kb.add_text_document('nlp_intro', doc_text, 'en')
    
    # 添加實體識別結果 (Add named entity recognition results)
    entities = [
        ('natural language processing', 'TECHNOLOGY'),
        ('machines', 'OBJECT'),
        ('human language', 'CONCEPT')
    ]
    
    for entity_text, entity_type in entities:
        entity_uri = kb.nlp_ns[f"entity_{entity_text.replace(' ', '_')}"]
        kb.graph.add((entity_uri, kb.nlp_ns.hasType, kb.nlp_ns[entity_type]))
        kb.graph.add((entity_uri, kb.nlp_ns.textValue, Literal(entity_text)))
        kb.graph.add((doc_uri, kb.nlp_ns.containsEntity, entity_uri))
    
    return kb
```

## SPARQL 查詢範例 (SPARQL Query Examples)

### 查找所有語言 (Find All Languages)

```sparql
PREFIX nlp: <http://example.org/nlp/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?language ?name ?family WHERE {
    ?language a nlp:Language .
    ?language rdfs:label ?name .
    OPTIONAL { 
        ?language nlp:belongsToFamily ?family_uri .
        ?family_uri rdfs:label ?family 
    }
}
```

### 查找概念關係 (Find Concept Relations)

```sparql
PREFIX nlp: <http://example.org/nlp/>

SELECT ?concept1 ?relation ?concept2 ?confidence WHERE {
    ?rel a nlp:SemanticRelation .
    ?rel nlp:hasSubject ?concept1 .
    ?rel nlp:hasObject ?concept2 .
    ?rel nlp:relationType ?relation .
    ?rel nlp:confidence ?confidence .
    FILTER (?confidence > 0.8)
}
```

### 查找文檔實體 (Find Document Entities)

```sparql
PREFIX nlp: <http://example.org/nlp/>

SELECT ?document ?entity ?type ?text WHERE {
    ?document a nlp:Document .
    ?document nlp:containsEntity ?entity .
    ?entity nlp:hasType ?type .
    ?entity nlp:textValue ?text .
}
```

## Protégé 使用指南 (Protégé Usage Guide)

### 匯出到 Protégé (Export to Protégé)

```python
# 匯出不同格式 (Export different formats)
kb.export_to_owl('nlp_ontology.owl', 'xml')      # XML/RDF 格式
kb.export_to_owl('nlp_ontology.ttl', 'turtle')   # Turtle 格式
kb.export_to_owl('nlp_ontology.n3', 'n3')        # N3 格式
```

### 在 Protégé 中開啟 (Open in Protégé)

1. 下載並安裝 Protégé: https://protege.stanford.edu/
2. 開啟 Protégé
3. File → Open → 選擇匯出的 .owl 檔案
4. 查看類別層次、屬性、實例等

### Protégé 功能對應 (Protégé Feature Mapping)

| SemanticKnowledgeBase 功能 | Protégé 對應 |
|---------------------------|-------------|
| `nlp_ns.Language` 類別 | Classes tab → Language |
| `add_concept_relation()` | Object Properties |
| 實例數據 | Individuals tab |
| SPARQL 查詢 | SPARQL Query tab (需安裝插件) |

## 進階功能 (Advanced Features)

### 推理機制 (Reasoning)

```python
# 應用推理 (Apply reasoning)
kb.apply_reasoning()

# 查看推理結果 (View inferred triples)
original_count = len(kb.graph)
kb.apply_reasoning()
inferred_count = len(kb.graph) - original_count
print(f"推理得出 {inferred_count} 個新三元組")
```

### 知識圖譜視覺化 (Knowledge Graph Visualization)

```python
# 導出圖數據 (Export graph data)
query = """
SELECT ?subject ?predicate ?object WHERE {
    ?subject ?predicate ?object .
    ?subject a nlp:Concept .
    ?object a nlp:Concept .
}
"""
results = kb.sparql_query(query)

# 可用於 D3.js, Cytoscape.js 等視覺化工具
# (Can be used with D3.js, Cytoscape.js, etc.)
```

### 本體驗證 (Ontology Validation)

```python
# 檢查本體一致性 (Check ontology consistency)
from owlrl import CombinedClosure

try:
    CombinedClosure.RDFS_OWLRL_Closure(kb.graph)
    print("本體一致性檢查通過")
except Exception as e:
    print(f"本體一致性錯誤: {e}")
```

## API 參考 (API Reference)

### SemanticKnowledgeBase 類別

#### 構造函數 (Constructor)
- `__init__(base_uri: str)`: 初始化知識庫

#### 核心方法 (Core Methods)
- `add_triple(subject, predicate, object)`: 添加三元組
- `add_language(code, name, family)`: 添加語言
- `add_text_document(id, content, lang, metadata)`: 添加文檔
- `add_concept_relation(concept1, relation, concept2, confidence)`: 添加概念關係

#### 查詢方法 (Query Methods)
- `sparql_query(query)`: 執行 SPARQL 查詢
- `search_concepts(term, limit)`: 搜索概念
- `get_related_concepts(concept, depth)`: 獲取相關概念
- `get_concept_hierarchy(root)`: 獲取概念層次

#### 導入導出 (Import/Export)
- `export_to_owl(filename, format)`: 導出到 OWL 檔案
- `import_from_owl(filename)`: 從 OWL 檔案導入

#### 工具方法 (Utility Methods)
- `apply_reasoning()`: 應用推理
- `get_statistics()`: 獲取統計信息

## 檔案結構 (File Structure)

```
NLPNote/
├── SemanticKnowledgeBase.py          # 主要知識庫類別
├── semantic_web_examples.py          # 使用範例
├── semantic_web_knowledge_base.md    # 本文檔
├── nlp_ontology.owl                  # 匯出的 OWL 本體 (XML 格式)
├── nlp_ontology.ttl                  # 匯出的 OWL 本體 (Turtle 格式)
├── nlp_ontology.n3                   # 匯出的 OWL 本體 (N3 格式)
└── nlp_knowledge_graph.json          # 知識圖譜數據
```

## 使用案例 (Use Cases)

### 1. 多語言 NLP 應用 (Multi-lingual NLP Applications)
- 跨語言概念對齊
- 翻譯質量評估
- 多語言知識圖譜構建

### 2. 文本語義分析 (Text Semantic Analysis)
- 實體關係抽取
- 概念語義網絡構建
- 文本相似度計算

### 3. 本體工程 (Ontology Engineering)
- 領域本體設計
- 本體對齊與合併
- 本體質量評估

### 4. 知識圖譜應用 (Knowledge Graph Applications)
- 智能問答系統
- 推薦系統
- 知識推理

## 故障排除 (Troubleshooting)

### 常見問題 (Common Issues)

#### 1. 導入錯誤 (Import Errors)
```bash
# 確保安裝了所有依賴 (Ensure all dependencies are installed)
pip install rdflib owlrl SPARQLWrapper
```

#### 2. SPARQL 查詢失敗 (SPARQL Query Failures)
```python
# 檢查命名空間前綴 (Check namespace prefixes)
query = """
PREFIX nlp: <http://example.org/nlp/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT * WHERE { ?s ?p ?o } LIMIT 10
"""
```

#### 3. OWL 檔案格式問題 (OWL File Format Issues)
```python
# 使用 XML 格式確保 Protégé 相容性 (Use XML format for Protégé compatibility)
kb.export_to_owl('ontology.owl', 'xml')
```

## 貢獻指南 (Contributing)

歡迎貢獻！請遵循以下步驟：

1. Fork 此專案
2. 創建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 開啟 Pull Request

## 授權條款 (License)

本專案採用 MIT 授權條款 - 詳見 LICENSE 檔案。

## 聯絡資訊 (Contact)

如有問題或建議，請提交 Issue 或聯絡專案維護者。

---

**注意**: 這是一個語義網知識庫的完整實現，旨在為 NLP 應用提供強大的語義建模和查詢能力。通過 RDF/OWL 標準和 Protégé 整合，您可以構建複雜的語言學本體和知識圖譜。