#!/usr/bin/env python3
"""
Quick Demo: Semantic Web Knowledge Base for NLP
===============================================

A simple demonstration script showing the key features of the semantic web
knowledge base implementation. Run this to see the system in action!

Usage:
    python semantic_demo.py

Requirements:
    pip install rdflib owlrl SPARQLWrapper
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from SemanticKnowledgeBase import SemanticKnowledgeBase, create_sample_nlp_knowledge_base
    from rdflib import Literal
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please install required packages:")
    print("pip install rdflib owlrl SPARQLWrapper")
    sys.exit(1)


def demo_header(title):
    """Print a formatted demo section header"""
    print("\n" + "🔹" * 60)
    print(f"🔹 {title}")
    print("🔹" * 60)


def demo_basic_functionality():
    """Demonstrate basic semantic knowledge base functionality"""
    demo_header("Basic Semantic Knowledge Base Demo")
    
    print("Creating a new semantic knowledge base...")
    kb = SemanticKnowledgeBase()
    
    print("\n✅ Adding languages to the knowledge base:")
    languages = [
        ('en', 'English', 'Indo-European'),
        ('zh', 'Chinese', 'Sino-Tibetan'),
        ('ja', 'Japanese', 'Japonic'),
        ('ar', 'Arabic', 'Afro-Asiatic')
    ]
    
    for code, name, family in languages:
        kb.add_language(code, name, family)
        print(f"   • {name} ({code}) - {family} family")
    
    print("\n✅ Adding text documents:")
    documents = [
        ('doc_en', 'Natural language processing enables machines to understand text.', 'en'),
        ('doc_zh', '自然语言处理帮助机器理解文本。', 'zh'),
        ('doc_ja', '自然言語処理により、機械がテキストを理解できるようになる。', 'ja')
    ]
    
    for doc_id, content, lang in documents:
        kb.add_text_document(doc_id, content, lang, {'topic': 'NLP'})
        print(f"   • {lang.upper()}: {content[:40]}...")
    
    print("\n✅ Adding concept relationships:")
    relationships = [
        ('artificial intelligence', 'includes', 'machine learning'),
        ('machine learning', 'includes', 'deep learning'),
        ('artificial intelligence', 'includes', 'natural language processing'),
        ('NLP', 'synonymOf', 'natural language processing'),
        ('AI', 'synonymOf', 'artificial intelligence')
    ]
    
    for concept1, relation, concept2 in relationships:
        kb.add_concept_relation(concept1, relation, concept2, 0.9)
        print(f"   • {concept1} --{relation}--> {concept2}")
    
    print(f"\n📊 Knowledge Base Statistics:")
    stats = kb.get_statistics()
    for key, value in stats.items():
        print(f"   • {key.replace('_', ' ').title()}: {value}")
    
    return kb


def demo_sparql_queries(kb):
    """Demonstrate SPARQL querying capabilities"""
    demo_header("SPARQL Query Demonstrations")
    
    queries = [
        ("🔍 Find all languages", """
            PREFIX nlp: <http://example.org/nlp/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?language ?family WHERE {
                ?lang a nlp:Language .
                ?lang rdfs:label ?language .
                OPTIONAL { 
                    ?lang nlp:belongsToFamily ?fam .
                    ?fam rdfs:label ?family 
                }
            }
        """),
        
        ("🔍 Find documents by language", """
            PREFIX nlp: <http://example.org/nlp/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?content ?language WHERE {
                ?doc a nlp:Document .
                ?doc nlp:hasContent ?content .
                ?doc nlp:inLanguage ?lang .
                ?lang rdfs:label ?language .
            }
        """),
        
        ("🔍 Find concept relationships", """
            PREFIX nlp: <http://example.org/nlp/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?concept1 ?relation ?concept2 WHERE {
                ?c1 ?rel ?c2 .
                ?c1 a nlp:Concept .
                ?c2 a nlp:Concept .
                ?c1 rdfs:label ?concept1 .
                ?c2 rdfs:label ?concept2 .
                FILTER(?rel != <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>)
                FILTER(?rel != <http://www.w3.org/2000/01/rdf-schema#label>)
                BIND(STRAFTER(STR(?rel), "#") AS ?relation)
            }
        """)
    ]
    
    for query_name, query in queries:
        print(f"\n{query_name}:")
        results = kb.sparql_query(query)
        
        if results:
            for i, result in enumerate(results[:3]):  # Show first 3 results
                result_str = ", ".join([f"{k}: {v}" for k, v in result.items()])
                print(f"   {i+1}. {result_str}")
            
            if len(results) > 3:
                print(f"   ... and {len(results) - 3} more results")
        else:
            print("   No results found")


def demo_protege_export(kb):
    """Demonstrate Protégé export functionality"""
    demo_header("Protégé Export Demonstration")
    
    print("🏗️ Exporting knowledge base to Protégé-compatible formats:")
    
    formats = [
        ('xml', 'demo_nlp_ontology.owl', 'OWL/XML (Protégé default)'),
        ('turtle', 'demo_nlp_ontology.ttl', 'Turtle (human-readable)'),
        ('n3', 'demo_nlp_ontology.n3', 'N3 (Notation3)')
    ]
    
    for format_name, filename, description in formats:
        try:
            kb.export_to_owl(filename, format_name)
            file_size = os.path.getsize(filename) if os.path.exists(filename) else 0
            print(f"   ✅ {filename} ({description}) - {file_size} bytes")
        except Exception as e:
            print(f"   ❌ Failed to export {filename}: {e}")
    
    print(f"\n📋 To use with Protégé:")
    print(f"   1. Download Protégé from: https://protege.stanford.edu/")
    print(f"   2. Open Protégé")
    print(f"   3. File → Open → Select demo_nlp_ontology.owl")
    print(f"   4. Explore the Classes, Object Properties, and Individuals tabs")


def demo_multilingual_features():
    """Demonstrate multilingual semantic features"""
    demo_header("Multilingual Semantic Features")
    
    print("🌍 Creating multilingual concept mappings:")
    kb = SemanticKnowledgeBase()
    
    # Add languages
    languages = [('en', 'English'), ('zh', 'Chinese'), ('ja', 'Japanese'), ('ko', 'Korean')]
    for code, name in languages:
        kb.add_language(code, name)
        print(f"   • Added {name} ({code})")
    
    print(f"\n🔗 Adding multilingual translations:")
    translations = [
        ('AI', [('en', 'artificial intelligence'), ('zh', '人工智能'), ('ja', '人工知能'), ('ko', '인공지능')]),
        ('ML', [('en', 'machine learning'), ('zh', '机器学习'), ('ja', '機械学習'), ('ko', '기계학습')])
    ]
    
    for concept_id, lang_terms in translations:
        print(f"\n   {concept_id}:")
        base_uri = kb.nlp_ns[f"concept_{concept_id}"]
        
        # Create concept instances for each language
        for lang_code, term in lang_terms:
            term_uri = kb.nlp_ns[f"concept_{concept_id}_{lang_code}"]
            kb.graph.add((term_uri, kb.nlp_ns.baseForm, base_uri))
            kb.graph.add((term_uri, kb.nlp_ns.language, kb.nlp_ns[f"language_{lang_code}"]))
            kb.graph.add((term_uri, kb.nlp_ns.textValue, Literal(term)))
            print(f"     • {lang_code}: {term}")
            
            # Link as translations
            for other_lang, other_term in lang_terms:
                if other_lang != lang_code:
                    other_uri = kb.nlp_ns[f"concept_{concept_id}_{other_lang}"]
                    kb.graph.add((term_uri, kb.nlp_ns.translationOf, other_uri))
    
    # Query for cross-lingual mappings
    print(f"\n🔍 Querying cross-lingual concept mappings:")
    query = """
    PREFIX nlp: <http://example.org/nlp/>
    SELECT ?concept ?lang ?text WHERE {
        ?instance nlp:baseForm ?concept .
        ?instance nlp:language ?language .
        ?instance nlp:textValue ?text .
        ?language rdfs:label ?lang .
    } ORDER BY ?concept ?lang
    """
    
    results = kb.sparql_query(query)
    current_concept = None
    for result in results:
        concept = str(result['concept']).split('_')[-1]
        if concept != current_concept:
            print(f"\n   {concept}:")
            current_concept = concept
        print(f"     • {result['lang']}: {result['text']}")
    
    return kb


def demo_knowledge_search():
    """Demonstrate knowledge search and discovery features"""
    demo_header("Knowledge Search and Discovery")
    
    print("🔍 Creating searchable knowledge base...")
    kb = create_sample_nlp_knowledge_base()
    
    # Add more searchable content
    search_concepts = [
        ('neural networks', 'deep learning', 'A type of machine learning model inspired by biological neural networks'),
        ('transformers', 'natural language processing', 'A neural network architecture especially effective for sequential data'),
        ('BERT', 'natural language processing', 'Bidirectional Encoder Representations from Transformers'),
        ('GPT', 'natural language processing', 'Generative Pre-trained Transformer'),
        ('computer vision', 'artificial intelligence', 'Teaching computers to interpret and understand visual information')
    ]
    
    for concept, related_to, description in search_concepts:
        concept_uri = kb.nlp_ns[f"concept_{concept.replace(' ', '_')}"]
        kb.graph.add((concept_uri, kb.nlp_ns.hasDescription, Literal(description)))
        kb.add_concept_relation(concept, 'relatedTo', related_to, 0.8)
    
    print("   ✅ Added advanced NLP and AI concepts")
    
    # Demonstrate search functionality
    search_terms = ['neural', 'language', 'learning', 'transformer']
    
    print(f"\n🔍 Searching knowledge base:")
    for term in search_terms:
        print(f"\n   Searching for '{term}':")
        results = kb.search_concepts(term, limit=3)
        
        if results:
            for result in results:
                label = result.get('label', 'N/A')
                comment = result.get('comment', 'No description')
                print(f"     • {label}: {comment}")
        else:
            print(f"     No results found for '{term}'")
    
    # Demonstrate related concept discovery
    print(f"\n🔗 Finding related concepts:")
    test_concepts = ['natural language processing', 'machine learning']
    
    for concept in test_concepts:
        print(f"\n   Related to '{concept}':")
        related = kb.get_related_concepts(concept, max_depth=2)
        
        for i, relation in enumerate(related[:3]):
            related_concept = str(relation.get('related', '')).split('/')[-1].replace('_', ' ')
            relation_type = str(relation.get('relation', '')).split('/')[-1]
            print(f"     • {related_concept} ({relation_type})")
        
        if len(related) > 3:
            print(f"     ... and {len(related) - 3} more related concepts")


def main():
    """Run the complete semantic web knowledge base demo"""
    print("🚀 Semantic Web Knowledge Base for NLP - Interactive Demo")
    print("=" * 60)
    print("This demo showcases the key features of the semantic knowledge base:")
    print("• RDF/OWL ontology modeling")
    print("• SPARQL querying")
    print("• Protégé integration")
    print("• Multilingual support")
    print("• Knowledge search and discovery")
    
    try:
        # Run demo sections
        kb1 = demo_basic_functionality()
        demo_sparql_queries(kb1)
        demo_protege_export(kb1)
        demo_multilingual_features()
        demo_knowledge_search()
        
        # Final summary
        demo_header("Demo Complete! 🎉")
        print("✅ Successfully demonstrated all major features!")
        print(f"\n📁 Generated files:")
        generated_files = [
            'demo_nlp_ontology.owl',
            'demo_nlp_ontology.ttl', 
            'demo_nlp_ontology.n3'
        ]
        
        for filename in generated_files:
            if os.path.exists(filename):
                size = os.path.getsize(filename)
                print(f"   • {filename} ({size} bytes)")
        
        print(f"\n🎯 Next steps:")
        print(f"   • Explore the generated OWL files in Protégé")
        print(f"   • Run semantic_web_examples.py for more detailed examples")
        print(f"   • Check semantic_integration_examples.py for NLP tool integration")
        print(f"   • Read semantic_web_knowledge_base.md for comprehensive documentation")
        
        print(f"\n📚 Learn more:")
        print(f"   • Protégé: https://protege.stanford.edu/")
        print(f"   • RDF: https://www.w3.org/RDF/")
        print(f"   • OWL: https://www.w3.org/OWL/")
        print(f"   • SPARQL: https://www.w3.org/TR/sparql11-query/")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)