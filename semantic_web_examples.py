"""
Semantic Web Examples for NLP Knowledge Base
============================================

This module demonstrates various semantic web capabilities for NLP applications,
including ontology creation, SPARQL querying, and Protégé integration examples.

Examples covered:
- Creating linguistic ontologies
- Text analysis with semantic annotations
- Multi-lingual knowledge representation
- NLP tool integration with semantic data
"""

from SemanticKnowledgeBase import SemanticKnowledgeBase, create_sample_nlp_knowledge_base
from rdflib import Literal, URIRef
import json


def example_1_basic_ontology():
    """Example 1: Creating a basic NLP ontology"""
    print("=" * 50)
    print("Example 1: Basic NLP Ontology Creation")
    print("=" * 50)
    
    kb = SemanticKnowledgeBase()
    
    # Add some linguistic concepts
    kb.add_concept_relation('phoneme', 'isPartOf', 'syllable')
    kb.add_concept_relation('syllable', 'isPartOf', 'word')
    kb.add_concept_relation('word', 'isPartOf', 'phrase')
    kb.add_concept_relation('phrase', 'isPartOf', 'sentence')
    kb.add_concept_relation('sentence', 'isPartOf', 'paragraph')
    
    # Add POS tag relations
    pos_tags = ['noun', 'verb', 'adjective', 'adverb', 'preposition', 'conjunction']
    for pos in pos_tags:
        kb.add_concept_relation(pos, 'isA', 'part_of_speech')
    
    # Query the hierarchy
    query = """
    PREFIX nlp: <http://example.org/nlp/>
    SELECT ?part ?whole WHERE {
        ?concept_part nlp:isPartOf ?concept_whole .
        ?concept_part rdfs:label ?part .
        ?concept_whole rdfs:label ?whole .
    }
    """
    
    results = kb.sparql_query(query)
    print("\nLinguistic hierarchy:")
    for result in results:
        print(f"  {result['part']} -> {result['whole']}")
    
    return kb


def example_2_multilingual_content():
    """Example 2: Multi-lingual content representation"""
    print("\n" + "=" * 50)
    print("Example 2: Multi-lingual Content")
    print("=" * 50)
    
    kb = SemanticKnowledgeBase()
    
    # Add multiple languages
    languages = [
        ('en', 'English', 'Indo-European'),
        ('zh', 'Chinese', 'Sino-Tibetan'),
        ('ja', 'Japanese', 'Japonic'),
        ('ar', 'Arabic', 'Afro-Asiatic'),
        ('es', 'Spanish', 'Indo-European'),
        ('fr', 'French', 'Indo-European')
    ]
    
    for code, name, family in languages:
        kb.add_language(code, name, family)
    
    # Add same concept in different languages
    translations = [
        ('en', 'hello', 'A greeting'),
        ('zh', '你好', 'A greeting'),
        ('ja', 'こんにちは', 'A greeting'),
        ('ar', 'مرحبا', 'A greeting'),
        ('es', 'hola', 'A greeting'),
        ('fr', 'bonjour', 'A greeting')
    ]
    
    for lang_code, word, description in translations:
        concept_uri = kb.nlp_ns[f"greeting_{lang_code}"]
        kb.graph.add((concept_uri, kb.nlp_ns.hasLanguage, kb.nlp_ns[f"language_{lang_code}"]))
        kb.graph.add((concept_uri, kb.nlp_ns.lexicalForm, Literal(word)))
        kb.graph.add((concept_uri, kb.nlp_ns.hasDescription, Literal(description)))
        
        # Link all greetings as translations of each other
        for other_lang, _, _ in translations:
            if other_lang != lang_code:
                other_uri = kb.nlp_ns[f"greeting_{other_lang}"]
                kb.graph.add((concept_uri, kb.nlp_ns.translationOf, other_uri))
    
    # Query translations
    query = """
    PREFIX nlp: <http://example.org/nlp/>
    SELECT ?word ?lang WHERE {
        ?concept nlp:lexicalForm ?word .
        ?concept nlp:hasLanguage ?language .
        ?language rdfs:label ?lang .
        ?concept nlp:translationOf* ?base_concept .
        ?base_concept nlp:hasLanguage nlp:language_en .
    }
    """
    
    results = kb.sparql_query(query)
    print("\nGreeting translations:")
    for result in results:
        print(f"  {result['lang']}: {result['word']}")
    
    return kb


def example_3_text_analysis_with_semantics():
    """Example 3: Semantic text analysis and annotation"""
    print("\n" + "=" * 50)
    print("Example 3: Semantic Text Analysis")
    print("=" * 50)
    
    kb = SemanticKnowledgeBase()
    
    # Add a document with semantic annotations
    doc_text = "Natural language processing enables machines to understand human language."
    doc_uri = kb.add_text_document('nlp_intro', doc_text, 'en', {
        'title': 'Introduction to NLP',
        'creator': 'AI Researcher',
        'topic': 'artificial intelligence'
    })
    
    # Add named entities found in the text
    entities = [
        ('natural language processing', 'TECHNOLOGY', 0, 26),
        ('machines', 'OBJECT', 35, 43),
        ('human language', 'CONCEPT', 58, 72)
    ]
    
    for entity_text, entity_type, start_pos, end_pos in entities:
        entity_uri = kb.nlp_ns[f"entity_{entity_text.replace(' ', '_')}"]
        kb.graph.add((entity_uri, kb.nlp_ns.hasType, kb.nlp_ns[entity_type]))
        kb.graph.add((entity_uri, kb.nlp_ns.textValue, Literal(entity_text)))
        kb.graph.add((entity_uri, kb.nlp_ns.startPosition, Literal(start_pos)))
        kb.graph.add((entity_uri, kb.nlp_ns.endPosition, Literal(end_pos)))
        kb.graph.add((doc_uri, kb.nlp_ns.containsEntity, entity_uri))
    
    # Add semantic relationships
    kb.add_concept_relation('natural language processing', 'enablesUnderstandingOf', 'human language')
    kb.add_concept_relation('machines', 'usesTechnology', 'natural language processing')
    
    # Query document entities
    query = f"""
    PREFIX nlp: <http://example.org/nlp/>
    SELECT ?entity ?type ?text WHERE {{
        <{doc_uri}> nlp:containsEntity ?entity .
        ?entity nlp:hasType ?type .
        ?entity nlp:textValue ?text .
    }}
    """
    
    results = kb.sparql_query(query)
    print(f"\nEntities in document '{doc_text}':")
    for result in results:
        entity_type = str(result['type']).split('/')[-1]
        print(f"  {result['text']} ({entity_type})")
    
    return kb


def example_4_protege_integration():
    """Example 4: Creating Protégé-compatible ontology"""
    print("\n" + "=" * 50)
    print("Example 4: Protégé Integration")
    print("=" * 50)
    
    kb = create_sample_nlp_knowledge_base()
    
    # Add more complex OWL constructs for Protégé
    from rdflib import OWL, RDF, RDFS
    
    # Define a more complex class hierarchy
    classes_hierarchy = [
        ('LinguisticUnit', None, 'Any unit of linguistic analysis'),
        ('Phoneme', 'LinguisticUnit', 'Smallest unit of sound'),
        ('Morpheme', 'LinguisticUnit', 'Smallest unit of meaning'),
        ('Word', 'LinguisticUnit', 'Basic unit of language'),
        ('Phrase', 'LinguisticUnit', 'Group of words functioning as a unit'),
        ('Clause', 'LinguisticUnit', 'Group of words containing subject and predicate'),
        ('Sentence', 'LinguisticUnit', 'Complete thought in language')
    ]
    
    for class_name, parent, description in classes_hierarchy:
        class_uri = kb.nlp_ns[class_name]
        kb.graph.add((class_uri, RDF.type, OWL.Class))
        kb.graph.add((class_uri, RDFS.label, Literal(class_name)))
        kb.graph.add((class_uri, RDFS.comment, Literal(description)))
        
        if parent:
            parent_uri = kb.nlp_ns[parent]
            kb.graph.add((class_uri, RDFS.subClassOf, parent_uri))
    
    # Add object properties
    properties = [
        ('hasPhoneme', 'Word', 'Phoneme', 'Associates a word with its phonemes'),
        ('hasMorpheme', 'Word', 'Morpheme', 'Associates a word with its morphemes'),
        ('consistsOf', 'LinguisticUnit', 'LinguisticUnit', 'General composition relation'),
        ('precedes', 'LinguisticUnit', 'LinguisticUnit', 'Temporal ordering relation')
    ]
    
    for prop_name, domain, range_class, description in properties:
        prop_uri = kb.nlp_ns[prop_name]
        kb.graph.add((prop_uri, RDF.type, OWL.ObjectProperty))
        kb.graph.add((prop_uri, RDFS.label, Literal(prop_name)))
        kb.graph.add((prop_uri, RDFS.comment, Literal(description)))
        kb.graph.add((prop_uri, RDFS.domain, kb.nlp_ns[domain]))
        kb.graph.add((prop_uri, RDFS.range, kb.nlp_ns[range_class]))
    
    # Export to multiple formats for Protégé
    formats = [
        ('xml', 'nlp_ontology.owl'),
        ('turtle', 'nlp_ontology.ttl'),
        ('n3', 'nlp_ontology.n3')
    ]
    
    for format_name, filename in formats:
        kb.export_to_owl(filename, format_name)
        print(f"  Exported to {filename} ({format_name} format)")
    
    print(f"\nOntology statistics:")
    stats = kb.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return kb


def example_5_advanced_queries():
    """Example 5: Advanced SPARQL queries"""
    print("\n" + "=" * 50)
    print("Example 5: Advanced SPARQL Queries")
    print("=" * 50)
    
    kb = create_sample_nlp_knowledge_base()
    
    # Add more data for complex queries
    kb.add_concept_relation('tokenization', 'isStepIn', 'text preprocessing')
    kb.add_concept_relation('stemming', 'isStepIn', 'text preprocessing')
    kb.add_concept_relation('lemmatization', 'isStepIn', 'text preprocessing')
    kb.add_concept_relation('text preprocessing', 'isPartOf', 'natural language processing')
    
    queries = [
        ("Find all NLP techniques", """
            PREFIX nlp: <http://example.org/nlp/>
            SELECT DISTINCT ?technique ?label WHERE {
                ?technique nlp:isStepIn|nlp:isPartOf ?process .
                ?process nlp:relatedTo|nlp:synonymOf nlp:concept_natural_language_processing .
                OPTIONAL { ?technique rdfs:label ?label }
            }
        """),
        
        ("Find concept relationships with confidence > 0.8", """
            PREFIX nlp: <http://example.org/nlp/>
            SELECT ?subject ?object ?confidence WHERE {
                ?relation a nlp:SemanticRelation .
                ?relation nlp:hasSubject ?subject .
                ?relation nlp:hasObject ?object .
                ?relation nlp:confidence ?confidence .
                FILTER (?confidence > 0.8)
            }
        """),
        
        ("Find documents and their metadata", """
            PREFIX nlp: <http://example.org/nlp/>
            PREFIX dcterms: <http://purl.org/dc/terms/>
            SELECT ?doc ?title ?creator ?language WHERE {
                ?doc a nlp:Document .
                OPTIONAL { ?doc dcterms:title ?title }
                OPTIONAL { ?doc dcterms:creator ?creator }
                OPTIONAL { 
                    ?doc nlp:inLanguage ?lang .
                    ?lang rdfs:label ?language 
                }
            }
        """)
    ]
    
    for query_name, query in queries:
        print(f"\n{query_name}:")
        results = kb.sparql_query(query)
        if results:
            for result in results[:3]:  # Show first 3 results
                print(f"  {result}")
        else:
            print("  No results found")
    
    return kb


def example_6_knowledge_graph_visualization():
    """Example 6: Prepare data for knowledge graph visualization"""
    print("\n" + "=" * 50)
    print("Example 6: Knowledge Graph Data")
    print("=" * 50)
    
    kb = create_sample_nlp_knowledge_base()
    
    # Extract graph data for visualization
    query = """
    PREFIX nlp: <http://example.org/nlp/>
    SELECT ?subject ?predicate ?object ?subject_label ?object_label WHERE {
        ?subject ?predicate ?object .
        ?subject a nlp:Concept .
        ?object a nlp:Concept .
        OPTIONAL { ?subject rdfs:label ?subject_label }
        OPTIONAL { ?object rdfs:label ?object_label }
    }
    """
    
    results = kb.sparql_query(query)
    
    # Create a simple graph representation
    nodes = set()
    edges = []
    
    for result in results:
        subject = result.get('subject_label', str(result['subject']).split('/')[-1])
        obj = result.get('object_label', str(result['object']).split('/')[-1])
        predicate = str(result['predicate']).split('/')[-1]
        
        nodes.add(subject)
        nodes.add(obj)
        edges.append({
            'source': subject,
            'target': obj,
            'relation': predicate
        })
    
    graph_data = {
        'nodes': [{'id': node} for node in nodes],
        'edges': edges
    }
    
    # Save as JSON for visualization tools
    with open('nlp_knowledge_graph.json', 'w') as f:
        json.dump(graph_data, f, indent=2)
    
    print(f"Knowledge graph data exported to nlp_knowledge_graph.json")
    print(f"Nodes: {len(nodes)}, Edges: {len(edges)}")
    print(f"Sample edges:")
    for edge in edges[:3]:
        print(f"  {edge['source']} --{edge['relation']}--> {edge['target']}")
    
    return kb


if __name__ == "__main__":
    print("Semantic Web Examples for NLP Knowledge Base")
    print("=" * 50)
    
    # Run all examples
    examples = [
        example_1_basic_ontology,
        example_2_multilingual_content,
        example_3_text_analysis_with_semantics,
        example_4_protege_integration,
        example_5_advanced_queries,
        example_6_knowledge_graph_visualization
    ]
    
    for example_func in examples:
        try:
            kb = example_func()
            print(f"\n✓ {example_func.__name__} completed successfully")
        except Exception as e:
            print(f"\n✗ {example_func.__name__} failed: {e}")
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("Generated files:")
    print("  - nlp_ontology.owl (OWL/XML format for Protégé)")
    print("  - nlp_ontology.ttl (Turtle format)")
    print("  - nlp_ontology.n3 (N3 format)")
    print("  - nlp_knowledge_graph.json (Graph data for visualization)")