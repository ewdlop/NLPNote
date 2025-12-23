"""
Integration example showing how SupernaturalNLP extends existing NLP approaches.

This module demonstrates connecting the supernatural framework with traditional
NLP techniques available in the repository.
"""

from SupernaturalNLP import SupernaturalNLP
import re
import math
from typing import List, Dict, Tuple, Any

class IntegratedNLPAnalyzer:
    """
    Integrates Supernatural NLP with traditional approaches for enhanced analysis.
    """
    
    def __init__(self):
        self.super_nlp = SupernaturalNLP(dimension=100)
        
    def enhanced_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """
        Combine traditional sentiment with supersymmetric analysis.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Enhanced sentiment analysis results
        """
        # Traditional sentiment (simplified)
        traditional_sentiment = self._simple_sentiment(text)
        
        # Supernatural analysis
        super_results = self.super_nlp.supersymmetric_transform(text)
        
        # Enhanced analysis using both approaches
        words = super_results['original_words']
        quantum_sentiment = 0.0
        
        for word in words:
            # Get superpartner for sentiment duality
            superpartner = super_results['superpartners'][word]
            
            # Calculate quantum sentiment contribution
            base_sentiment = self._word_sentiment(word)
            partner_sentiment = self._word_sentiment(superpartner)
            
            # Quantum superposition of sentiments
            quantum_contribution = (base_sentiment + partner_sentiment) / 2
            quantum_sentiment += quantum_contribution
        
        if words:
            quantum_sentiment /= len(words)
        
        # Supersymmetry breaking affects sentiment certainty
        breaking = self.super_nlp.detect_supersymmetry_breaking(text)
        certainty = breaking['symmetry_score']  # Higher symmetry = higher certainty
        
        return {
            'traditional_sentiment': traditional_sentiment,
            'quantum_sentiment': quantum_sentiment,
            'sentiment_certainty': certainty,
            'sentiment_superposition': abs(traditional_sentiment - quantum_sentiment),
            'supersymmetric_analysis': super_results,
            'breaking_analysis': breaking
        }
    
    def _simple_sentiment(self, text: str) -> float:
        """Simple rule-based sentiment analysis."""
        positive_words = {'good', 'great', 'excellent', 'wonderful', 'amazing', 'love', 'happy', 'joy'}
        negative_words = {'bad', 'terrible', 'awful', 'horrible', 'hate', 'sad', 'angry', 'pain'}
        
        words = re.findall(r'\b\w+\b', text.lower())
        
        score = 0
        for word in words:
            if word in positive_words:
                score += 1
            elif word in negative_words:
                score -= 1
        
        return score / len(words) if words else 0
    
    def _word_sentiment(self, word: str) -> float:
        """Get sentiment value for a single word."""
        positive_indicators = ['good', 'great', 'love', 'happy', 'wonderful', 'excellent', 'amazing', 'beautiful']
        negative_indicators = ['bad', 'hate', 'sad', 'terrible', 'awful', 'horrible', 'ugly', 'angry']
        
        word_lower = word.lower()
        
        # Check direct matches
        if word_lower in positive_indicators:
            return 1.0
        elif word_lower in negative_indicators:
            return -1.0
        
        # Check partial matches
        for pos in positive_indicators:
            if pos in word_lower or word_lower in pos:
                return 0.5
        
        for neg in negative_indicators:
            if neg in word_lower or word_lower in neg:
                return -0.5
        
        return 0.0
    
    def supernatural_topic_modeling(self, texts: List[str]) -> Dict[str, Any]:
        """
        Topic modeling using supersymmetric principles.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Supernatural topic model results
        """
        all_words = []
        text_analyses = []
        
        # Analyze each text
        for text in texts:
            super_results = self.super_nlp.supersymmetric_transform(text)
            text_analyses.append(super_results)
            all_words.extend(super_results['original_words'])
        
        # Find most entangled word pairs across all texts
        word_entanglements = {}
        unique_words = list(set(all_words))
        
        for i, word1 in enumerate(unique_words):
            for word2 in unique_words[i+1:]:
                entanglement = self.super_nlp.quantum_semantic_entanglement(word1, word2)
                if entanglement > 0.3:  # Threshold for significant entanglement
                    word_entanglements[(word1, word2)] = entanglement
        
        # Create "supernatural topics" based on entanglement clusters
        topics = self._extract_entanglement_clusters(word_entanglements)
        
        return {
            'texts_analyzed': len(texts),
            'total_unique_words': len(unique_words),
            'significant_entanglements': len(word_entanglements),
            'supernatural_topics': topics,
            'entanglement_network': word_entanglements
        }
    
    def _extract_entanglement_clusters(self, entanglements: Dict[Tuple[str, str], float]) -> List[Dict[str, Any]]:
        """Extract topic clusters from word entanglements."""
        # Simple clustering based on entanglement strength
        clusters = []
        used_words = set()
        
        # Sort entanglements by strength
        sorted_entanglements = sorted(entanglements.items(), key=lambda x: x[1], reverse=True)
        
        for (word1, word2), strength in sorted_entanglements[:10]:  # Top 10 entanglements
            if word1 not in used_words and word2 not in used_words:
                cluster = {
                    'core_words': [word1, word2],
                    'entanglement_strength': strength,
                    'topic_id': len(clusters),
                    'superpartners': [
                        self.super_nlp.find_superpartner(word1),
                        self.super_nlp.find_superpartner(word2)
                    ]
                }
                clusters.append(cluster)
                used_words.add(word1)
                used_words.add(word2)
        
        return clusters
    
    def compare_traditional_vs_supernatural(self, text: str) -> Dict[str, Any]:
        """
        Direct comparison between traditional and supernatural approaches.
        
        Args:
            text: Text to analyze
            
        Returns:
            Comparison results
        """
        # Traditional analysis (simplified)
        words = re.findall(r'\b\w+\b', text.lower())
        word_count = len(words)
        unique_words = len(set(words))
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
        
        # Supernatural analysis
        super_results = self.super_nlp.supersymmetric_transform(text)
        holographic = self.super_nlp.holographic_language_encoding(text)
        breaking = self.super_nlp.detect_supersymmetry_breaking(text)
        
        # Calculate enhanced metrics
        total_entanglement = 0
        entanglement_count = 0
        
        for i, word1 in enumerate(words):
            for word2 in words[i+1:]:
                entanglement = self.super_nlp.quantum_semantic_entanglement(word1, word2)
                total_entanglement += entanglement
                entanglement_count += 1
        
        avg_entanglement = total_entanglement / entanglement_count if entanglement_count > 0 else 0
        
        return {
            'traditional_metrics': {
                'word_count': word_count,
                'unique_words': unique_words,
                'avg_word_length': avg_word_length,
                'vocabulary_richness': unique_words / word_count if word_count > 0 else 0
            },
            'supernatural_metrics': {
                'symmetry_score': breaking['symmetry_score'],
                'breaking_strength': breaking['breaking_strength'],
                'avg_quantum_entanglement': avg_entanglement,
                'boundary_entropy': holographic['boundary']['boundary_entropy'],
                'dimension_reduction': holographic['holographic_duality']['dimension_reduction'],
                'superpartner_diversity': len(set(super_results['superpartners'].values()))
            },
            'enhanced_insights': {
                'quantum_complexity': avg_entanglement * breaking['breaking_strength'],
                'holographic_efficiency': (
                    holographic['boundary']['boundary_entropy'] / 
                    holographic['bulk']['bulk_dimensions']
                ),
                'supernatural_richness': (
                    avg_entanglement + breaking['symmetry_score'] + 
                    holographic['boundary']['boundary_entropy']
                ) / 3
            }
        }

def demonstrate_integration():
    """Demonstrate the integrated analysis capabilities."""
    print("=== Integrated Traditional + Supernatural NLP Demo ===\n")
    
    analyzer = IntegratedNLPAnalyzer()
    
    # Test texts
    texts = [
        "I love this wonderful day",
        "The quantum mechanics of consciousness reveal hidden patterns",
        "Traditional algorithms process data efficiently",
        "Poetry dances between meaning and mystery"
    ]
    
    print("1. Enhanced Sentiment Analysis:")
    for text in texts:
        sentiment = analyzer.enhanced_sentiment_analysis(text)
        print(f"Text: '{text}'")
        print(f"  Traditional sentiment: {sentiment['traditional_sentiment']:.3f}")
        print(f"  Quantum sentiment: {sentiment['quantum_sentiment']:.3f}")
        print(f"  Sentiment certainty: {sentiment['sentiment_certainty']:.3f}")
        print(f"  Superposition strength: {sentiment['sentiment_superposition']:.3f}")
        print()
    
    print("\n2. Supernatural Topic Modeling:")
    topic_results = analyzer.supernatural_topic_modeling(texts)
    print(f"Analyzed {topic_results['texts_analyzed']} texts")
    print(f"Found {topic_results['significant_entanglements']} significant entanglements")
    print("Top supernatural topics:")
    for topic in topic_results['supernatural_topics'][:3]:
        print(f"  Topic {topic['topic_id']}: {topic['core_words']} "
              f"(strength: {topic['entanglement_strength']:.3f})")
        print(f"    Superpartners: {topic['superpartners']}")
    
    print("\n3. Traditional vs Supernatural Comparison:")
    comparison_text = "The beautiful quantum cat walks through mysterious gardens of consciousness"
    comparison = analyzer.compare_traditional_vs_supernatural(comparison_text)
    
    print(f"Text: '{comparison_text}'")
    print("\nTraditional Metrics:")
    for key, value in comparison['traditional_metrics'].items():
        print(f"  {key}: {value:.3f}")
    
    print("\nSupernatural Metrics:")
    for key, value in comparison['supernatural_metrics'].items():
        print(f"  {key}: {value:.3f}")
    
    print("\nEnhanced Insights:")
    for key, value in comparison['enhanced_insights'].items():
        print(f"  {key}: {value:.3f}")

if __name__ == "__main__":
    demonstrate_integration()