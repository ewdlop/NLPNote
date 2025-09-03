"""
Tests for Semantic Web Knowledge Base
====================================

Basic unit tests for the semantic web functionality to ensure proper operation.
"""

import unittest
import os
import tempfile
from SemanticKnowledgeBase import SemanticKnowledgeBase, create_sample_nlp_knowledge_base
from rdflib import Literal


class TestSemanticKnowledgeBase(unittest.TestCase):
    """Test cases for SemanticKnowledgeBase class"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.kb = SemanticKnowledgeBase()
    
    def test_initialization(self):
        """Test knowledge base initialization"""
        self.assertIsNotNone(self.kb.graph)
        self.assertEqual(self.kb.base_uri, "http://example.org/nlp/")
        
        # Check that base ontology classes were created
        stats = self.kb.get_statistics()
        self.assertGreater(stats['classes'], 0)
    
    def test_add_language(self):
        """Test adding languages to the knowledge base"""
        lang_uri = self.kb.add_language('en', 'English', 'Indo-European')
        
        # Verify language was added
        query = """
        PREFIX nlp: <http://example.org/nlp/>
        SELECT ?lang WHERE {
            ?lang a nlp:Language .
            ?lang rdfs:label "English" .
        }
        """
        results = self.kb.sparql_query(query)
        self.assertEqual(len(results), 1)
    
    def test_add_text_document(self):
        """Test adding text documents"""
        doc_uri = self.kb.add_text_document(
            'test_doc', 
            'This is a test document.', 
            'en',
            {'title': 'Test Document'}
        )
        
        # Verify document was added
        query = """
        PREFIX nlp: <http://example.org/nlp/>
        SELECT ?doc WHERE {
            ?doc a nlp:Document .
            ?doc nlp:hasContent "This is a test document." .
        }
        """
        results = self.kb.sparql_query(query)
        self.assertEqual(len(results), 1)
    
    def test_add_concept_relation(self):
        """Test adding concept relations"""
        self.kb.add_concept_relation('AI', 'subsumes', 'NLP', 0.9)
        
        # Verify relation was added
        query = """
        PREFIX nlp: <http://example.org/nlp/>
        SELECT ?rel WHERE {
            ?rel a nlp:SemanticRelation .
            ?rel nlp:confidence "0.9"^^<http://www.w3.org/2001/XMLSchema#double> .
        }
        """
        results = self.kb.sparql_query(query)
        self.assertEqual(len(results), 1)
    
    def test_sparql_query(self):
        """Test SPARQL querying functionality"""
        # Add some test data
        self.kb.add_language('en', 'English')
        
        # Test basic query
        query = """
        PREFIX nlp: <http://example.org/nlp/>
        SELECT ?lang WHERE {
            ?lang a nlp:Language .
        }
        """
        results = self.kb.sparql_query(query)
        self.assertGreater(len(results), 0)
        
        # Test invalid query
        invalid_query = "INVALID SPARQL QUERY"
        results = self.kb.sparql_query(invalid_query)
        self.assertEqual(len(results), 0)
    
    def test_search_concepts(self):
        """Test concept search functionality"""
        # Add test concepts
        self.kb.add_concept_relation('machine learning', 'relatedTo', 'data science')
        
        # Search for concepts
        results = self.kb.search_concepts('machine')
        self.assertGreater(len(results), 0)
    
    def test_export_import_owl(self):
        """Test OWL export and import functionality"""
        # Add some test data
        self.kb.add_language('en', 'English')
        self.kb.add_concept_relation('AI', 'includes', 'ML')
        
        # Test export
        with tempfile.NamedTemporaryFile(suffix='.owl', delete=False) as tmp_file:
            tmp_filename = tmp_file.name
        
        try:
            self.kb.export_to_owl(tmp_filename)
            self.assertTrue(os.path.exists(tmp_filename))
            
            # Test import
            new_kb = SemanticKnowledgeBase()
            original_triples = len(new_kb.graph)
            new_kb.import_from_owl(tmp_filename)
            
            # Should have more triples after import
            self.assertGreater(len(new_kb.graph), original_triples)
            
        finally:
            # Clean up
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)
    
    def test_get_statistics(self):
        """Test statistics retrieval"""
        stats = self.kb.get_statistics()
        
        # Check required statistics fields
        required_fields = ['total_triples', 'languages', 'documents', 'concepts', 'classes']
        for field in required_fields:
            self.assertIn(field, stats)
            self.assertIsInstance(stats[field], int)
            self.assertGreaterEqual(stats[field], 0)
    
    def test_get_related_concepts(self):
        """Test finding related concepts"""
        # Add test relations
        self.kb.add_concept_relation('NLP', 'relatedTo', 'linguistics')
        self.kb.add_concept_relation('NLP', 'partOf', 'AI')
        
        # Find related concepts
        related = self.kb.get_related_concepts('NLP')
        self.assertGreater(len(related), 0)


class TestSampleKnowledgeBase(unittest.TestCase):
    """Test cases for the sample knowledge base"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.kb = create_sample_nlp_knowledge_base()
    
    def test_sample_kb_creation(self):
        """Test that sample knowledge base is created correctly"""
        stats = self.kb.get_statistics()
        
        # Should have multiple languages
        self.assertGreater(stats['languages'], 0)
        
        # Should have documents
        self.assertGreater(stats['documents'], 0)
        
        # Should have concepts
        self.assertGreater(stats['concepts'], 0)
        
        # Should have reasonable number of triples
        self.assertGreater(stats['total_triples'], 50)
    
    def test_sample_languages(self):
        """Test that sample languages are properly added"""
        query = """
        PREFIX nlp: <http://example.org/nlp/>
        SELECT ?name WHERE {
            ?lang a nlp:Language .
            ?lang rdfs:label ?name .
        }
        """
        results = self.kb.sparql_query(query)
        
        # Should have at least English, Chinese, Spanish, Arabic
        language_names = [str(result['name']) for result in results]
        expected_languages = ['English', 'Chinese', 'Spanish', 'Arabic']
        
        for expected_lang in expected_languages:
            self.assertIn(expected_lang, language_names)
    
    def test_sample_documents(self):
        """Test that sample documents are properly added"""
        query = """
        PREFIX nlp: <http://example.org/nlp/>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        SELECT ?doc ?title WHERE {
            ?doc a nlp:Document .
            ?doc dcterms:title ?title .
        }
        """
        results = self.kb.sparql_query(query)
        
        # Should have at least 2 documents
        self.assertGreaterEqual(len(results), 2)
        
        # Check for expected titles
        titles = [str(result['title']) for result in results]
        self.assertIn('NLP Introduction', titles)
    
    def test_sample_concept_relations(self):
        """Test that sample concept relations are properly added"""
        query = """
        PREFIX nlp: <http://example.org/nlp/>
        SELECT ?rel WHERE {
            ?rel a nlp:SemanticRelation .
        }
        """
        results = self.kb.sparql_query(query)
        
        # Should have multiple semantic relations
        self.assertGreater(len(results), 0)


def run_semantic_tests():
    """Run all semantic web tests and return results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestSemanticKnowledgeBase))
    suite.addTests(loader.loadTestsFromTestCase(TestSampleKnowledgeBase))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    print("Running Semantic Web Knowledge Base Tests")
    print("=" * 50)
    
    result = run_semantic_tests()
    
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✓ All tests passed successfully!")
    else:
        print(f"✗ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback}")
    
    print(f"\nRan {result.testsRun} tests total")