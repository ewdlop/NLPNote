#!/usr/bin/env python3
"""
Impossible Query Analysis Example
Demonstrates how the NLP framework handles impossible or nonsensical queries
using the "What is an angel's defecation called?" case study.
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import the human expression evaluator framework
try:
    from HumanExpressionEvaluator import HumanExpressionEvaluator, ExpressionContext
    EVALUATOR_AVAILABLE = True
except ImportError:
    print("Human Expression Evaluator not available. Running with simplified analysis.")
    EVALUATOR_AVAILABLE = False

class ImpossibleQueryAnalyzer:
    """Analyzes queries that contain semantic impossibilities or contradictions."""
    
    def __init__(self):
        if EVALUATOR_AVAILABLE:
            self.expression_evaluator = HumanExpressionEvaluator()
        else:
            self.expression_evaluator = None
    
    def detect_impossibility_type(self, query):
        """Detects the type of impossibility in a query."""
        query_lower = query.lower()
        
        impossibility_types = []
        
        # Category mismatch detection
        spiritual_terms = ['angel', 'spirit', 'soul', 'ghost', 'divine', 'heavenly']
        physical_terms = ['defecation', 'excretion', 'bodily', 'physical', 'biological']
        
        has_spiritual = any(term in query_lower for term in spiritual_terms)
        has_physical = any(term in query_lower for term in physical_terms)
        
        if has_spiritual and has_physical:
            impossibility_types.append("category_mismatch")
        
        # Contradictory premises
        if 'angel' in query_lower and any(term in query_lower for term in ['defecation', 'excrete', 'waste']):
            impossibility_types.append("contradictory_premises")
        
        # Non-existent referent
        if query_lower.startswith('what is') and 'angel' in query_lower and 'defecation' in query_lower:
            impossibility_types.append("non_existent_referent")
        
        return impossibility_types if impossibility_types else ["unknown"]
    
    def provide_educational_context(self, query):
        """Provides educational context for impossible queries."""
        if 'angel' in query.lower() and 'defecation' in query.lower():
            return {
                "theological_context": "Angels are traditionally conceived as spiritual beings without physical bodies in most religious traditions.",
                "biological_context": "Defecation is a biological process that requires physical digestive organs.",
                "philosophical_issue": "The query assumes angels have biological functions, which contradicts their spiritual nature.",
                "cultural_note": "Different cultures may have varying concepts of spiritual beings, but bodily functions are typically not attributed to pure spirits."
            }
        return {"general": "This query appears to contain contradictory or impossible elements."}
    
    def suggest_alternative_questions(self, query):
        """Suggests more meaningful alternative questions."""
        if 'angel' in query.lower():
            return [
                "What are the characteristics of angels in different religious traditions?",
                "How are angels depicted in various mythologies?",
                "What functions do angels serve in theological systems?",
                "How do different cultures conceptualize spiritual beings?",
                "What is the difference between spiritual and physical existence in theology?"
            ]
        return ["Could you rephrase your question in a different way?"]
    
    def analyze_query_context(self, query):
        """Analyzes the likely context and intent of the query."""
        possible_contexts = []
        
        query_lower = query.lower()
        
        # Test query indicators
        if any(pattern in query_lower for pattern in ['what is', 'what are', 'how']):
            if self.detect_impossibility_type(query) != ["unknown"]:
                possible_contexts.append("test_query")
        
        # Humorous intent indicators
        if 'defecation' in query_lower and 'angel' in query_lower:
            possible_contexts.append("humorous_intent")
        
        # Academic/philosophical context
        if len(query.split()) > 3:
            possible_contexts.append("academic_inquiry")
        
        # Naive question
        possible_contexts.append("naive_question")
        
        return possible_contexts
    
    def generate_response(self, query, context_type="general"):
        """Generates an appropriate response based on context."""
        impossibility_types = self.detect_impossibility_type(query)
        educational_context = self.provide_educational_context(query)
        alternatives = self.suggest_alternative_questions(query)
        
        if context_type == "academic":
            response = f"""This query presents an interesting semantic impossibility. {educational_context.get('philosophical_issue', 'The query contains contradictory elements.')}

From a theological perspective: {educational_context.get('theological_context', 'Traditional concepts would not support this premise.')}

From a biological perspective: {educational_context.get('biological_context', 'The biological functions mentioned require physical form.')}

This type of query is useful for testing NLP systems' ability to handle semantic contradictions and impossible scenarios."""

        elif context_type == "humorous":
            response = """While this is an amusing question, it highlights an interesting philosophical point about the nature of spiritual beings. Angels, being typically conceived as pure spirits without physical bodies, wouldn't have biological functions like defecation. 

Perhaps we could call it 'divine excretion' or 'celestial waste management' in a purely hypothetical sense! 😇"""

        elif context_type == "test_query":
            response = """This appears to be a test query designed to probe system capabilities. The query contains a semantic impossibility: it assumes angels (spiritual beings) have biological functions (defecation).

System analysis:
- Impossibility type: Category mismatch between spiritual and physical domains
- Response strategy: Acknowledge impossibility while providing educational context
- Handling: Explain the contradictory premises rather than attempting direct answer"""

        else:  # general/naive
            response = f"""Angels are generally understood to be spiritual beings without physical bodies, so they wouldn't have biological functions like defecation. {educational_context.get('theological_context', '')}

If you're interested in learning more about angels, you might want to explore: {', '.join(alternatives[:2])}"""
        
        return response
    
    def full_analysis(self, query):
        """Performs a complete analysis of the impossible query."""
        print("="*60)
        print("IMPOSSIBLE QUERY ANALYSIS")
        print("="*60)
        print(f"Query: '{query}'")
        print()
        
        # Basic analysis
        impossibility_types = self.detect_impossibility_type(query)
        print(f"Impossibility Types: {', '.join(impossibility_types)}")
        
        # Context analysis
        contexts = self.analyze_query_context(query)
        print(f"Possible Contexts: {', '.join(contexts)}")
        print()
        
        # Educational context
        educational_context = self.provide_educational_context(query)
        print("Educational Context:")
        for key, value in educational_context.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        print()
        
        # Alternative questions
        alternatives = self.suggest_alternative_questions(query)
        print("Suggested Alternative Questions:")
        for i, alt in enumerate(alternatives, 1):
            print(f"  {i}. {alt}")
        print()
        
        # Generate responses for different contexts
        print("Response Examples:")
        print()
        
        for context in ["academic", "general", "humorous", "test_query"]:
            print(f"--- {context.upper()} CONTEXT ---")
            response = self.generate_response(query, context)
            print(response)
            print()
        
        # Use expression evaluator if available
        if EVALUATOR_AVAILABLE:
            print("--- HUMAN EXPRESSION EVALUATION FRAMEWORK ANALYSIS ---")
            try:
                context_obj = ExpressionContext(
                    situation="impossible_query",
                    formality_level="neutral",
                    cultural_background="universal"
                )
                result = self.expression_evaluator.comprehensive_evaluation(query, context_obj)
                print("Framework Analysis Results:")
                for key, value in result.items():
                    if isinstance(value, dict):
                        print(f"  {key}:")
                        for subkey, subvalue in value.items():
                            print(f"    {subkey}: {subvalue}")
                    else:
                        print(f"  {key}: {value}")
            except Exception as e:
                print(f"Expression evaluation failed: {e}")
        
        print("="*60)

def main():
    """Main function to demonstrate impossible query analysis."""
    analyzer = ImpossibleQueryAnalyzer()
    
    # The main query from the issue
    main_query = "What is an angel's defecation called?"
    
    # Perform full analysis
    analyzer.full_analysis(main_query)
    
    # Additional test cases
    additional_queries = [
        "How much does an angel weigh?",
        "What color are angel wings?",
        "Do spirits need to eat?",
        "Where do angels sleep?"
    ]
    
    print("\nADDITIONAL TEST CASES:")
    print("="*60)
    
    for query in additional_queries:
        print(f"\nQuery: '{query}'")
        impossibility_types = analyzer.detect_impossibility_type(query)
        contexts = analyzer.analyze_query_context(query)
        
        print(f"Impossibility Types: {', '.join(impossibility_types)}")
        print(f"Possible Contexts: {', '.join(contexts)}")
        
        response = analyzer.generate_response(query, "general")
        print("Response:", response[:100] + "..." if len(response) > 100 else response)

if __name__ == "__main__":
    main()