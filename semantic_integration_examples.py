"""
Integration Examples: Semantic Knowledge Base with NLP Tools
============================================================

This module demonstrates how to integrate the semantic knowledge base
with existing NLP tools and libraries in the repository.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from SemanticKnowledgeBase import SemanticKnowledgeBase
from rdflib import Literal
import json


def example_semantic_text_analysis():
    """
    Example: Enhanced text analysis with semantic annotations
    Demonstrates integration with text processing and semantic modeling
    """
    print("=" * 60)
    print("Example: Semantic Text Analysis Integration")
    print("=" * 60)
    
    kb = SemanticKnowledgeBase()
    
    # Sample texts for analysis
    texts = [
        {
            'id': 'ai_overview',
            'content': 'Artificial intelligence includes machine learning and natural language processing.',
            'language': 'en',
            'metadata': {'topic': 'AI', 'difficulty': 'beginner'}
        },
        {
            'id': 'nlp_chinese',
            'content': '自然语言处理是人工智能的重要分支。',
            'language': 'zh',
            'metadata': {'topic': 'NLP', 'difficulty': 'intermediate'}
        }
    ]
    
    # Add languages
    kb.add_language('en', 'English', 'Indo-European')
    kb.add_language('zh', 'Chinese', 'Sino-Tibetan')
    
    # Process each text
    for text_info in texts:
        # Add document to knowledge base
        doc_uri = kb.add_text_document(
            text_info['id'],
            text_info['content'],
            text_info['language'],
            text_info['metadata']
        )
        
        # Simulate NLP processing (in real scenario, use spaCy, NLTK, etc.)
        entities = []
        if 'artificial intelligence' in text_info['content'].lower():
            entities.append(('artificial intelligence', 'TECHNOLOGY', 'AI'))
        if 'machine learning' in text_info['content'].lower():
            entities.append(('machine learning', 'TECHNOLOGY', 'ML'))
        if 'natural language processing' in text_info['content'].lower():
            entities.append(('natural language processing', 'TECHNOLOGY', 'NLP'))
        if '自然语言处理' in text_info['content']:
            entities.append(('自然语言处理', 'TECHNOLOGY', 'NLP_ZH'))
        if '人工智能' in text_info['content']:
            entities.append(('人工智能', 'TECHNOLOGY', 'AI_ZH'))
        
        # Add entities to knowledge base
        for entity_text, entity_type, entity_id in entities:
            entity_uri = kb.nlp_ns[f"entity_{entity_id}"]
            kb.graph.add((entity_uri, kb.nlp_ns.hasType, kb.nlp_ns[entity_type]))
            kb.graph.add((entity_uri, kb.nlp_ns.textValue, Literal(entity_text)))
            kb.graph.add((entity_uri, kb.nlp_ns.language, kb.nlp_ns[f"language_{text_info['language']}"]))
            kb.graph.add((doc_uri, kb.nlp_ns.containsEntity, entity_uri))
    
    # Add cross-lingual concept mappings
    kb.add_concept_relation('artificial intelligence', 'translationOf', '人工智能', 1.0)
    kb.add_concept_relation('natural language processing', 'translationOf', '自然语言处理', 1.0)
    
    # Query for cross-lingual analysis
    query = """
    PREFIX nlp: <http://example.org/nlp/>
    SELECT ?doc ?entity_en ?entity_zh WHERE {
        ?doc nlp:containsEntity ?entity1 .
        ?doc nlp:containsEntity ?entity2 .
        ?entity1 nlp:language nlp:language_en .
        ?entity2 nlp:language nlp:language_zh .
        ?entity1 nlp:textValue ?entity_en .
        ?entity2 nlp:textValue ?entity_zh .
        ?concept1 nlp:translationOf ?concept2 .
    }
    """
    
    results = kb.sparql_query(query)
    print("\nCross-lingual entity mappings:")
    for result in results:
        print(f"  EN: {result['entity_en']} ↔ ZH: {result['entity_zh']}")
    
    return kb


def example_knowledge_enhanced_similarity():
    """
    Example: Knowledge-enhanced text similarity using semantic relations
    """
    print("\n" + "=" * 60)
    print("Example: Knowledge-Enhanced Similarity")
    print("=" * 60)
    
    kb = SemanticKnowledgeBase()
    
    # Build a small domain knowledge base
    concepts = [
        ('AI', 'artificial intelligence', 'synonymOf', 1.0),
        ('ML', 'machine learning', 'synonymOf', 1.0),
        ('DL', 'deep learning', 'synonymOf', 1.0),
        ('NLP', 'natural language processing', 'synonymOf', 1.0),
        ('machine learning', 'artificial intelligence', 'partOf', 0.9),
        ('deep learning', 'machine learning', 'partOf', 0.9),
        ('natural language processing', 'artificial intelligence', 'partOf', 0.8),
        ('neural networks', 'deep learning', 'relatedTo', 0.8),
        ('transformers', 'natural language processing', 'relatedTo', 0.9)
    ]
    
    for concept1, concept2, relation, confidence in concepts:
        kb.add_concept_relation(concept1, relation, concept2, confidence)
    
    def get_semantic_similarity(term1, term2):
        """Calculate semantic similarity using knowledge base"""
        # Query for direct relations
        query = f"""
        PREFIX nlp: <http://example.org/nlp/>
        SELECT ?relation ?confidence WHERE {{
            ?c1 rdfs:label "{term1}" .
            ?c2 rdfs:label "{term2}" .
            ?rel nlp:hasSubject ?c1 .
            ?rel nlp:hasObject ?c2 .
            ?rel nlp:relationType ?relation .
            ?rel nlp:confidence ?confidence .
        }}
        """
        results = kb.sparql_query(query)
        
        if results:
            # Return highest confidence score
            confidences = [float(r['confidence']) for r in results]
            return max(confidences)
        
        # Check for indirect relations (synonyms)
        synonym_query = f"""
        PREFIX nlp: <http://example.org/nlp/>
        SELECT ?confidence WHERE {{
            ?c1 rdfs:label "{term1}" .
            ?c2 rdfs:label "{term2}" .
            ?c1 nlp:synonymOf ?common .
            ?c2 nlp:synonymOf ?common .
        }}
        """
        synonym_results = kb.sparql_query(synonym_query)
        if synonym_results:
            return 0.95  # High similarity for synonyms
        
        return 0.0  # No semantic relation found
    
    # Test similarity calculations
    test_pairs = [
        ('AI', 'artificial intelligence'),
        ('machine learning', 'deep learning'),
        ('NLP', 'transformers'),
        ('neural networks', 'deep learning'),
        ('AI', 'natural language processing')
    ]
    
    print("\nSemantic similarity scores:")
    for term1, term2 in test_pairs:
        similarity = get_semantic_similarity(term1, term2)
        print(f"  {term1} ↔ {term2}: {similarity:.2f}")
    
    return kb


def example_ontology_driven_classification():
    """
    Example: Ontology-driven text classification
    """
    print("\n" + "=" * 60)
    print("Example: Ontology-Driven Classification")
    print("=" * 60)
    
    kb = SemanticKnowledgeBase()
    
    # Define classification taxonomy
    taxonomy = [
        ('Technology', None, 'Root technology category'),
        ('AI', 'Technology', 'Artificial Intelligence'),
        ('ML', 'AI', 'Machine Learning'),
        ('DL', 'ML', 'Deep Learning'),
        ('NLP', 'AI', 'Natural Language Processing'),
        ('CV', 'AI', 'Computer Vision'),
        ('Robotics', 'AI', 'Robotics'),
        ('DataScience', 'Technology', 'Data Science'),
        ('BigData', 'DataScience', 'Big Data'),
        ('Analytics', 'DataScience', 'Analytics')
    ]
    
    # Build taxonomy in knowledge base
    for category, parent, description in taxonomy:
        cat_uri = kb.nlp_ns[f"category_{category}"]
        kb.graph.add((cat_uri, kb.nlp_ns.hasLabel, Literal(category)))
        kb.graph.add((cat_uri, kb.nlp_ns.hasDescription, Literal(description)))
        
        if parent:
            parent_uri = kb.nlp_ns[f"category_{parent}"]
            kb.graph.add((cat_uri, kb.nlp_ns.subCategoryOf, parent_uri))
    
    # Define classification rules (keywords → categories)
    classification_rules = {
        'AI': ['artificial intelligence', 'AI', 'intelligent systems'],
        'ML': ['machine learning', 'ML', 'supervised learning', 'unsupervised learning'],
        'DL': ['deep learning', 'neural networks', 'CNN', 'RNN', 'transformer'],
        'NLP': ['natural language processing', 'NLP', 'text mining', 'sentiment analysis'],
        'CV': ['computer vision', 'image recognition', 'object detection'],
        'DataScience': ['data science', 'data analysis', 'statistics'],
        'BigData': ['big data', 'hadoop', 'spark', 'distributed computing']
    }
    
    # Sample documents for classification
    documents = [
        "Deep learning models using transformer architectures achieve state-of-the-art results in NLP.",
        "Machine learning algorithms can be trained on large datasets to recognize patterns.",
        "Computer vision systems use convolutional neural networks for image classification.",
        "Big data analytics platforms process massive datasets using distributed computing."
    ]
    
    def classify_document(text):
        """Classify document using ontology-based rules"""
        text_lower = text.lower()
        detected_categories = []
        
        for category, keywords in classification_rules.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    detected_categories.append(category)
                    break
        
        return detected_categories
    
    print("\nDocument classifications:")
    for i, doc in enumerate(documents):
        categories = classify_document(doc)
        print(f"\nDocument {i+1}: {doc[:50]}...")
        print(f"  Categories: {', '.join(categories) if categories else 'None detected'}")
        
        # Get category hierarchy for each detected category
        for category in categories:
            query = f"""
            PREFIX nlp: <http://example.org/nlp/>
            SELECT ?parent WHERE {{
                nlp:category_{category} nlp:subCategoryOf+ ?parent .
                ?parent nlp:hasLabel ?parent_label .
            }}
            """
            results = kb.sparql_query(query)
            if results:
                parents = [str(r['parent']).split('_')[-1] for r in results]
                print(f"    {category} hierarchy: {' → '.join(reversed(parents + [category]))}")
    
    return kb


def example_multilingual_knowledge_graph():
    """
    Example: Multilingual knowledge graph construction
    """
    print("\n" + "=" * 60)
    print("Example: Multilingual Knowledge Graph")
    print("=" * 60)
    
    kb = SemanticKnowledgeBase()
    
    # Add multiple languages
    languages = [
        ('en', 'English'),
        ('zh', 'Chinese'),
        ('ja', 'Japanese'),
        ('ko', 'Korean'),
        ('ar', 'Arabic')
    ]
    
    for code, name in languages:
        kb.add_language(code, name)
    
    # Multilingual concept definitions
    multilingual_concepts = {
        'AI': {
            'en': 'artificial intelligence',
            'zh': '人工智能',
            'ja': '人工知能',
            'ko': '인공지능',
            'ar': 'الذكاء الاصطناعي'
        },
        'ML': {
            'en': 'machine learning',
            'zh': '机器学习',
            'ja': '機械学習',
            'ko': '기계학습',
            'ar': 'تعلم الآلة'
        },
        'NLP': {
            'en': 'natural language processing',
            'zh': '自然语言处理',
            'ja': '自然言語処理',
            'ko': '자연어처리',
            'ar': 'معالجة اللغة الطبيعية'
        }
    }
    
    # Add multilingual concepts to knowledge base
    for concept_id, translations in multilingual_concepts.items():
        # Create base concept
        base_concept_uri = kb.nlp_ns[f"concept_{concept_id}"]
        
        # Add translations
        for lang_code, translation in translations.items():
            translation_uri = kb.nlp_ns[f"concept_{concept_id}_{lang_code}"]
            kb.graph.add((translation_uri, kb.nlp_ns.baseForm, base_concept_uri))
            kb.graph.add((translation_uri, kb.nlp_ns.language, kb.nlp_ns[f"language_{lang_code}"]))
            kb.graph.add((translation_uri, kb.nlp_ns.textValue, Literal(translation)))
            
            # Link translations to each other
            for other_lang, other_translation in translations.items():
                if other_lang != lang_code:
                    other_uri = kb.nlp_ns[f"concept_{concept_id}_{other_lang}"]
                    kb.graph.add((translation_uri, kb.nlp_ns.translationOf, other_uri))
    
    # Query for multilingual concept mappings
    query = """
    PREFIX nlp: <http://example.org/nlp/>
    SELECT ?concept ?lang ?text WHERE {
        ?concept_instance nlp:baseForm ?concept .
        ?concept_instance nlp:language ?language .
        ?concept_instance nlp:textValue ?text .
        ?language rdfs:label ?lang .
    }
    ORDER BY ?concept ?lang
    """
    
    results = kb.sparql_query(query)
    
    # Group results by concept
    concept_groups = {}
    for result in results:
        concept = str(result['concept']).split('_')[-1]
        lang = str(result['lang'])
        text = str(result['text'])
        
        if concept not in concept_groups:
            concept_groups[concept] = {}
        concept_groups[concept][lang] = text
    
    print("\nMultilingual concept mappings:")
    for concept, translations in concept_groups.items():
        print(f"\n{concept}:")
        for lang, text in translations.items():
            print(f"  {lang}: {text}")
    
    # Export multilingual graph data
    graph_data = {
        'concepts': concept_groups,
        'languages': [lang for code, lang in languages],
        'metadata': {
            'total_concepts': len(concept_groups),
            'total_languages': len(languages),
            'total_translations': sum(len(translations) for translations in concept_groups.values())
        }
    }
    
    with open('multilingual_knowledge_graph.json', 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nMultilingual knowledge graph exported to multilingual_knowledge_graph.json")
    print(f"Total concepts: {graph_data['metadata']['total_concepts']}")
    print(f"Total languages: {graph_data['metadata']['total_languages']}")
    print(f"Total translations: {graph_data['metadata']['total_translations']}")
    
    return kb


if __name__ == "__main__":
    print("Semantic Knowledge Base Integration Examples")
    print("=" * 60)
    
    examples = [
        example_semantic_text_analysis,
        example_knowledge_enhanced_similarity,
        example_ontology_driven_classification,
        example_multilingual_knowledge_graph
    ]
    
    for example_func in examples:
        try:
            kb = example_func()
            print(f"\n✓ {example_func.__name__} completed successfully")
        except Exception as e:
            print(f"\n✗ {example_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Integration examples completed!")
    print("\nGenerated files:")
    print("  - multilingual_knowledge_graph.json (Multilingual concept mappings)")
    print("\nThese examples demonstrate how to:")
    print("  1. Enhance text analysis with semantic annotations")
    print("  2. Calculate semantic similarity using knowledge graphs")
    print("  3. Perform ontology-driven text classification")
    print("  4. Build multilingual knowledge representations")