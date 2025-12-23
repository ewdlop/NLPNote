"""
Semantic Web Knowledge Base for NLP
====================================

This module provides semantic web capabilities for natural language processing,
including RDF/OWL ontology management, SPARQL querying, and Protégé integration.

Author: NLP Note Project
License: MIT
"""

from typing import List, Dict, Any, Tuple, Optional, Union
import rdflib
from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef, Literal, BNode
from rdflib.namespace import FOAF, DCTERMS, SKOS
import owlrl
from SPARQLWrapper import SPARQLWrapper, JSON
import json
import os
from pathlib import Path


class SemanticKnowledgeBase:
    """
    A comprehensive semantic knowledge base for NLP applications using RDF/OWL.
    
    Features:
    - RDF triple management
    - OWL ontology support
    - SPARQL querying
    - Protégé compatibility
    - NLP domain modeling
    """
    
    def __init__(self, base_uri: str = "http://example.org/nlp/"):
        """
        Initialize the semantic knowledge base.
        
        Args:
            base_uri: Base URI for the knowledge base namespace
        """
        self.graph = Graph()
        self.base_uri = base_uri
        
        # Define namespaces
        self.nlp_ns = Namespace(base_uri)
        self.graph.bind("nlp", self.nlp_ns)
        self.graph.bind("foaf", FOAF)
        self.graph.bind("dcterms", DCTERMS)
        self.graph.bind("skos", SKOS)
        self.graph.bind("owl", OWL)
        self.graph.bind("rdfs", RDFS)
        
        # Initialize reasoner
        self.reasoner = owlrl.DeductiveClosure
        
        # Create basic NLP ontology structure
        self._create_base_ontology()
    
    def _create_base_ontology(self):
        """Create the foundational NLP ontology structure."""
        # Define main classes
        classes = [
            ("Language", "A natural or artificial language"),
            ("Text", "A piece of written or spoken text"),
            ("Token", "A linguistic unit such as a word or punctuation"),
            ("Sentence", "A grammatical unit of text"),
            ("Document", "A collection of text with semantic meaning"),
            ("Concept", "An abstract or concrete idea represented in text"),
            ("PartOfSpeech", "Grammatical category of a word"),
            ("NamedEntity", "A real-world object denoted by a proper name"),
            ("SemanticRelation", "A relationship between concepts or entities")
        ]
        
        for class_name, description in classes:
            class_uri = self.nlp_ns[class_name]
            self.graph.add((class_uri, RDF.type, OWL.Class))
            self.graph.add((class_uri, RDFS.label, Literal(class_name)))
            self.graph.add((class_uri, RDFS.comment, Literal(description)))
    
    def add_triple(self, subject: Union[str, URIRef], 
                   predicate: Union[str, URIRef], 
                   obj: Union[str, URIRef, Literal]) -> None:
        """
        Add a triple to the knowledge base.
        
        Args:
            subject: Subject of the triple
            predicate: Predicate of the triple  
            obj: Object of the triple
        """
        # Convert strings to URIRefs if needed
        if isinstance(subject, str):
            subject = self.nlp_ns[subject] if not subject.startswith('http') else URIRef(subject)
        if isinstance(predicate, str):
            predicate = self.nlp_ns[predicate] if not predicate.startswith('http') else URIRef(predicate)
        if isinstance(obj, str) and not obj.startswith('http'):
            obj = self.nlp_ns[obj]
        elif isinstance(obj, str):
            obj = URIRef(obj)
            
        self.graph.add((subject, predicate, obj))
    
    def add_language(self, language_code: str, language_name: str, 
                     family: Optional[str] = None) -> URIRef:
        """
        Add a language to the knowledge base.
        
        Args:
            language_code: ISO language code (e.g., 'en', 'zh')
            language_name: Full name of the language
            family: Language family (optional)
            
        Returns:
            URIRef of the created language resource
        """
        lang_uri = self.nlp_ns[f"language_{language_code}"]
        
        self.graph.add((lang_uri, RDF.type, self.nlp_ns.Language))
        self.graph.add((lang_uri, RDFS.label, Literal(language_name)))
        self.graph.add((lang_uri, self.nlp_ns.languageCode, Literal(language_code)))
        
        if family:
            family_uri = self.nlp_ns[f"family_{family.replace(' ', '_')}"]
            self.graph.add((family_uri, RDF.type, self.nlp_ns.LanguageFamily))
            self.graph.add((family_uri, RDFS.label, Literal(family)))
            self.graph.add((lang_uri, self.nlp_ns.belongsToFamily, family_uri))
        
        return lang_uri
    
    def add_text_document(self, doc_id: str, content: str, 
                         language: str, metadata: Optional[Dict] = None) -> URIRef:
        """
        Add a text document to the knowledge base.
        
        Args:
            doc_id: Unique identifier for the document
            content: Text content
            language: Language code
            metadata: Additional metadata
            
        Returns:
            URIRef of the created document resource
        """
        doc_uri = self.nlp_ns[f"document_{doc_id}"]
        lang_uri = self.nlp_ns[f"language_{language}"]
        
        self.graph.add((doc_uri, RDF.type, self.nlp_ns.Document))
        self.graph.add((doc_uri, self.nlp_ns.hasContent, Literal(content)))
        self.graph.add((doc_uri, self.nlp_ns.inLanguage, lang_uri))
        self.graph.add((doc_uri, DCTERMS.identifier, Literal(doc_id)))
        
        if metadata:
            for key, value in metadata.items():
                predicate = self.nlp_ns[key] if key not in ['title', 'creator'] else DCTERMS[key]
                self.graph.add((doc_uri, predicate, Literal(value)))
        
        return doc_uri
    
    def add_concept_relation(self, concept1: str, relation_type: str, 
                           concept2: str, confidence: float = 1.0) -> None:
        """
        Add a semantic relation between concepts.
        
        Args:
            concept1: First concept
            relation_type: Type of relation (e.g., 'synonymOf', 'hyponymOf')
            concept2: Second concept
            confidence: Confidence score (0.0 to 1.0)
        """
        concept1_uri = self.nlp_ns[f"concept_{concept1.replace(' ', '_')}"]
        concept2_uri = self.nlp_ns[f"concept_{concept2.replace(' ', '_')}"]
        relation_uri = self.nlp_ns[relation_type]
        
        # Ensure concepts exist
        self.graph.add((concept1_uri, RDF.type, self.nlp_ns.Concept))
        self.graph.add((concept1_uri, RDFS.label, Literal(concept1)))
        self.graph.add((concept2_uri, RDF.type, self.nlp_ns.Concept))
        self.graph.add((concept2_uri, RDFS.label, Literal(concept2)))
        
        # Add relation
        self.graph.add((concept1_uri, relation_uri, concept2_uri))
        
        # Add confidence if not 1.0
        if confidence != 1.0:
            relation_instance = BNode()
            self.graph.add((relation_instance, RDF.type, self.nlp_ns.SemanticRelation))
            self.graph.add((relation_instance, self.nlp_ns.hasSubject, concept1_uri))
            self.graph.add((relation_instance, self.nlp_ns.hasObject, concept2_uri))
            self.graph.add((relation_instance, self.nlp_ns.relationType, relation_uri))
            self.graph.add((relation_instance, self.nlp_ns.confidence, Literal(confidence)))
    
    def sparql_query(self, query: str) -> List[Dict]:
        """
        Execute a SPARQL query on the knowledge base.
        
        Args:
            query: SPARQL query string
            
        Returns:
            List of result bindings
        """
        try:
            results = self.graph.query(query)
            return [dict(row.asdict()) for row in results]
        except Exception as e:
            print(f"SPARQL query error: {e}")
            return []
    
    def apply_reasoning(self) -> None:
        """Apply OWL/RDFS reasoning to infer new triples."""
        owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(self.graph)
    
    def get_concept_hierarchy(self, root_concept: Optional[str] = None) -> Dict:
        """
        Get the concept hierarchy starting from a root concept.
        
        Args:
            root_concept: Root concept to start from (if None, gets all hierarchies)
            
        Returns:
            Dictionary representing the concept hierarchy
        """
        query = """
        PREFIX nlp: <%s>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?concept ?parent ?label WHERE {
            ?concept a nlp:Concept .
            OPTIONAL { ?concept rdfs:subClassOf ?parent . }
            OPTIONAL { ?concept rdfs:label ?label . }
        }
        """ % self.base_uri
        
        results = self.sparql_query(query)
        
        # Build hierarchy dictionary
        hierarchy = {}
        for result in results:
            concept = str(result.get('concept', ''))
            parent = str(result.get('parent', '')) if result.get('parent') else None
            label = str(result.get('label', '')) if result.get('label') else concept
            
            if concept not in hierarchy:
                hierarchy[concept] = {'label': label, 'children': [], 'parent': parent}
            
            if parent and parent not in hierarchy:
                hierarchy[parent] = {'label': parent, 'children': [], 'parent': None}
            
            if parent:
                hierarchy[parent]['children'].append(concept)
        
        return hierarchy
    
    def export_to_owl(self, filename: str, format: str = 'xml') -> None:
        """
        Export the knowledge base to an OWL file (Protégé compatible).
        
        Args:
            filename: Output filename
            format: Output format ('xml', 'turtle', 'n3')
        """
        # Add OWL ontology metadata
        ontology_uri = URIRef(self.base_uri + "ontology")
        self.graph.add((ontology_uri, RDF.type, OWL.Ontology))
        self.graph.add((ontology_uri, RDFS.label, Literal("NLP Semantic Knowledge Base")))
        self.graph.add((ontology_uri, RDFS.comment, 
                       Literal("A semantic knowledge base for natural language processing applications")))
        
        # Serialize to file
        self.graph.serialize(destination=filename, format=format)
        print(f"Knowledge base exported to {filename} in {format} format")
    
    def import_from_owl(self, filename: str) -> None:
        """
        Import an OWL file into the knowledge base.
        
        Args:
            filename: Path to the OWL file
        """
        if os.path.exists(filename):
            self.graph.parse(filename)
            print(f"Imported knowledge from {filename}")
        else:
            print(f"File {filename} not found")
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Dictionary with counts of different elements
        """
        stats = {}
        
        # Count triples
        stats['total_triples'] = len(self.graph)
        
        # Count by type
        type_queries = {
            'languages': f"SELECT (COUNT(?s) as ?count) WHERE {{ ?s a <{self.nlp_ns.Language}> }}",
            'documents': f"SELECT (COUNT(?s) as ?count) WHERE {{ ?s a <{self.nlp_ns.Document}> }}",
            'concepts': f"SELECT (COUNT(?s) as ?count) WHERE {{ ?s a <{self.nlp_ns.Concept}> }}",
            'classes': f"SELECT (COUNT(?s) as ?count) WHERE {{ ?s a <{OWL.Class}> }}"
        }
        
        for stat_name, query in type_queries.items():
            results = self.sparql_query(query)
            if results:
                stats[stat_name] = int(results[0]['count'])
            else:
                stats[stat_name] = 0
        
        return stats
    
    def search_concepts(self, search_term: str, limit: int = 10) -> List[Dict]:
        """
        Search for concepts by label or description.
        
        Args:
            search_term: Term to search for
            limit: Maximum number of results
            
        Returns:
            List of matching concepts
        """
        query = f"""
        PREFIX nlp: <{self.base_uri}>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?concept ?label ?comment WHERE {{
            ?concept a nlp:Concept .
            OPTIONAL {{ ?concept rdfs:label ?label . }}
            OPTIONAL {{ ?concept rdfs:comment ?comment . }}
            FILTER (
                CONTAINS(LCASE(STR(?label)), LCASE("{search_term}")) ||
                CONTAINS(LCASE(STR(?comment)), LCASE("{search_term}")) ||
                CONTAINS(LCASE(STR(?concept)), LCASE("{search_term}"))
            )
        }} LIMIT {limit}
        """
        
        return self.sparql_query(query)
    
    def get_related_concepts(self, concept: str, max_depth: int = 2) -> List[Dict]:
        """
        Find concepts related to a given concept within specified depth.
        
        Args:
            concept: Starting concept
            max_depth: Maximum relationship depth to explore
            
        Returns:
            List of related concepts with relationship information
        """
        concept_uri = self.nlp_ns[f"concept_{concept.replace(' ', '_')}"]
        
        query = f"""
        PREFIX nlp: <{self.base_uri}>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?related ?relation ?label WHERE {{
            {{
                <{concept_uri}> ?relation ?related .
                ?related a nlp:Concept .
            }} UNION {{
                ?related ?relation <{concept_uri}> .
                ?related a nlp:Concept .
            }}
            OPTIONAL {{ ?related rdfs:label ?label . }}
        }}
        """
        
        return self.sparql_query(query)


def create_sample_nlp_knowledge_base() -> SemanticKnowledgeBase:
    """
    Create a sample NLP knowledge base with example data.
    
    Returns:
        Populated SemanticKnowledgeBase instance
    """
    kb = SemanticKnowledgeBase()
    
    # Add languages
    kb.add_language('en', 'English', 'Indo-European')
    kb.add_language('zh', 'Chinese', 'Sino-Tibetan')
    kb.add_language('es', 'Spanish', 'Indo-European')
    kb.add_language('ar', 'Arabic', 'Afro-Asiatic')
    
    # Add sample documents
    kb.add_text_document('doc1', 'Natural language processing is fascinating.', 'en',
                        {'title': 'NLP Introduction', 'creator': 'AI Researcher'})
    kb.add_text_document('doc2', '自然语言处理很有趣。', 'zh',
                        {'title': 'NLP 介绍', 'creator': 'AI 研究员'})
    
    # Add concept relations
    kb.add_concept_relation('natural language processing', 'synonymOf', 'NLP', 0.95)
    kb.add_concept_relation('machine learning', 'relatedTo', 'natural language processing', 0.8)
    kb.add_concept_relation('linguistics', 'relatedTo', 'natural language processing', 0.9)
    kb.add_concept_relation('artificial intelligence', 'subsumes', 'natural language processing', 0.9)
    
    return kb


if __name__ == "__main__":
    # Demo usage
    print("Creating sample NLP knowledge base...")
    kb = create_sample_nlp_knowledge_base()
    
    print("\nKnowledge base statistics:")
    stats = kb.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nSearching for 'language' concepts:")
    results = kb.search_concepts('language')
    for result in results[:3]:
        print(f"  - {result.get('label', 'N/A')}: {result.get('concept', 'N/A')}")
    
    print("\nExporting to OWL format...")
    kb.export_to_owl('nlp_knowledge_base.owl')
    
    print("\nSample SPARQL query - finding all languages:")
    query = """
    PREFIX nlp: <http://example.org/nlp/>
    SELECT ?lang ?name WHERE {
        ?lang a nlp:Language .
        ?lang rdfs:label ?name .
    }
    """
    results = kb.sparql_query(query)
    for result in results:
        print(f"  - {result.get('name', 'N/A')}")