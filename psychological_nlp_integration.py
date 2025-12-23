"""
Integration of Ego-Id-Superego Neural Network with Human Expression Evaluator

This module demonstrates how to integrate the psychological neural network
with the existing NLP framework for enhanced human expression analysis.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

# Import existing modules
try:
    from HumanExpressionEvaluator import HumanExpressionEvaluator, ExpressionContext
    HUMAN_EXPR_AVAILABLE = True
except ImportError:
    HUMAN_EXPR_AVAILABLE = False
    print("Warning: HumanExpressionEvaluator not available. Some features will be limited.")

# Import our new neural network
from ego_id_superego_nn import (
    EgoIdSuperegoNeuralNetwork,
    PsycheComponent,
    IntegratedResponse
)

logger = logging.getLogger(__name__)


class TextEmbeddingGenerator:
    """Simple text embedding generator for demonstration purposes"""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Generate simple embeddings from text"""
        # Simple hash-based embedding for demonstration
        # In a real implementation, you'd use BERT, GPT, or similar
        text_hash = hash(text.lower()) % (2**31)
        
        # Convert to normalized embedding
        np.random.seed(text_hash)
        embedding = np.random.randn(self.embedding_dim)
        embedding = embedding / np.linalg.norm(embedding)
        
        return torch.from_numpy(embedding).float().unsqueeze(0)


class PsychologicalNLPAnalyzer:
    """
    Advanced NLP analyzer that combines psychological neural networks
    with traditional human expression evaluation
    """
    
    def __init__(self, embedding_dim: int = 384, hidden_dim: int = 256, output_dim: int = 128):
        """
        Initialize the psychological NLP analyzer
        
        Args:
            embedding_dim: Dimension of text embeddings
            hidden_dim: Hidden dimension for neural networks
            output_dim: Output dimension for psychological components
        """
        self.embedding_dim = embedding_dim
        
        # Initialize components
        self.text_encoder = TextEmbeddingGenerator(embedding_dim)
        self.psych_network = EgoIdSuperegoNeuralNetwork(embedding_dim, hidden_dim, output_dim)
        
        # Initialize human expression evaluator if available
        if HUMAN_EXPR_AVAILABLE:
            self.human_evaluator = HumanExpressionEvaluator()
        else:
            self.human_evaluator = None
        
        logger.info(f"PsychologicalNLPAnalyzer initialized with embedding_dim={embedding_dim}")
    
    def analyze_text_comprehensively(self, text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform comprehensive psychological and linguistic analysis of text
        
        Args:
            text: Input text to analyze
            context: Optional context information
            
        Returns:
            Comprehensive analysis results
        """
        logger.info(f"Analyzing text: '{text[:50]}...'")
        
        # Encode text to embeddings
        text_embedding = self.text_encoder.encode_text(text)
        
        # Psychological analysis
        psych_analysis = self.psych_network.analyze_text_psychologically(text_embedding)
        
        # Traditional human expression evaluation (if available)
        human_analysis = None
        if self.human_evaluator and context:
            try:
                expr_context = ExpressionContext(**context) if isinstance(context, dict) else context
                human_analysis = self.human_evaluator.comprehensive_evaluation(text, expr_context)
            except Exception as e:
                logger.warning(f"Human expression evaluation failed: {e}")
        
        # Combine analyses
        result = {
            'text': text,
            'psychological_analysis': psych_analysis,
            'human_expression_analysis': human_analysis,
            'integrated_insights': self._generate_integrated_insights(text, psych_analysis, human_analysis),
            'recommendations': self._generate_recommendations(psych_analysis, human_analysis)
        }
        
        return result
    
    def _generate_integrated_insights(self, text: str, psych_analysis: Dict, human_analysis: Optional[Dict]) -> Dict[str, Any]:
        """Generate integrated insights from both analyses"""
        insights = {
            'psychological_profile': {},
            'communication_style': {},
            'emotional_indicators': {},
            'decision_making_pattern': {}
        }
        
        # Extract psychological insights
        if 'psychological_profile' in psych_analysis:
            profile = psych_analysis['psychological_profile']
            
            # Psychological profile
            insights['psychological_profile'] = {
                'dominant_drive': self._identify_dominant_drive(profile),
                'emotional_regulation': profile['rational_response']['logical_consistency'],
                'moral_orientation': profile['moral_response']['ethical_certainty'],
                'internal_coherence': profile['psychological_dynamics']['component_harmony']
            }
            
            # Communication style
            integrated = psych_analysis['integrated_response']
            insights['communication_style'] = {
                'impulsiveness': profile['instinctual_response']['impulse_level'],
                'deliberation': profile['rational_response']['planning_depth'],
                'moral_consideration': profile['moral_response']['moral_strength'],
                'overall_approach': integrated.decision_rationale
            }
            
            # Decision making pattern
            insights['decision_making_pattern'] = {
                'conflict_level': integrated.conflict_level,
                'decision_clarity': profile['psychological_dynamics']['decision_clarity'],
                'primary_influence': max(integrated.decision_rationale.items(), 
                                       key=lambda x: x[1] if 'weight' in x[0] else 0)[0]
            }
        
        # Integrate with human expression analysis if available
        if human_analysis:
            try:
                # Handle different types of human analysis results
                formal_score = 0
                cognitive_score = 0
                social_score = 0
                
                if isinstance(human_analysis, dict):
                    formal_score = human_analysis.get('formal_semantic', 0)
                    cognitive_score = human_analysis.get('cognitive', 0)
                    social_score = human_analysis.get('social', 0)
                    
                    # Handle if scores are objects with score attributes
                    if hasattr(formal_score, 'score'):
                        formal_score = formal_score.score
                    if hasattr(cognitive_score, 'score'):
                        cognitive_score = cognitive_score.score
                    if hasattr(social_score, 'score'):
                        social_score = social_score.score
                
                insights['expression_quality'] = {
                    'formal_appropriateness': formal_score if isinstance(formal_score, (int, float)) else 0,
                    'cognitive_accessibility': cognitive_score if isinstance(cognitive_score, (int, float)) else 0,
                    'social_appropriateness': social_score if isinstance(social_score, (int, float)) else 0
                }
            except Exception as e:
                logger.debug(f"Could not process human analysis results: {e}")
                insights['expression_quality'] = {
                    'formal_appropriateness': 0,
                    'cognitive_accessibility': 0,
                    'social_appropriateness': 0
                }
        
        return insights
    
    def _identify_dominant_drive(self, profile: Dict) -> str:
        """Identify the dominant psychological drive"""
        instinctual = profile['instinctual_response']['strength']
        rational = profile['rational_response']['logical_consistency']
        moral = profile['moral_response']['ethical_certainty']
        
        scores = {'instinctual': instinctual, 'rational': rational, 'moral': moral}
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _generate_recommendations(self, psych_analysis: Dict, human_analysis: Optional[Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if 'psychological_profile' in psych_analysis:
            profile = psych_analysis['psychological_profile']
            dynamics = profile['psychological_dynamics']
            
            # Conflict-based recommendations
            if dynamics['internal_conflict'] > 0.6:
                recommendations.append("High internal conflict detected. Consider clarifying your core message and values.")
            
            # Component balance recommendations
            rationale = psych_analysis['integrated_response'].decision_rationale
            dominant_weight = max(rationale[k] for k in rationale if 'weight' in k)
            
            if dominant_weight > 0.7:
                if rationale['id_weight'] > 0.7:
                    recommendations.append("Expression appears highly impulsive. Consider adding rational reflection.")
                elif rationale['superego_weight'] > 0.7:
                    recommendations.append("Expression is heavily moralized. Consider practical implications.")
                elif rationale['ego_weight'] > 0.7:
                    recommendations.append("Expression is well-balanced and rational.")
            
            # Harmony recommendations
            if dynamics['component_harmony'] < 0.3:
                recommendations.append("Low psychological harmony. Work on integrating emotional, rational, and moral aspects.")
        
        # Human expression recommendations - handle different return types
        if human_analysis:
            try:
                # Try to extract numeric scores from human analysis
                social_score = None
                cognitive_score = None
                
                if isinstance(human_analysis, dict):
                    social_score = human_analysis.get('social', 0)
                    cognitive_score = human_analysis.get('cognitive', 0)
                    
                    # Handle if scores are objects with score attributes
                    if hasattr(social_score, 'score'):
                        social_score = social_score.score
                    if hasattr(cognitive_score, 'score'):
                        cognitive_score = cognitive_score.score
                
                if social_score is not None and isinstance(social_score, (int, float)) and social_score < 0.5:
                    recommendations.append("Social appropriateness could be improved. Consider your audience and context.")
                if cognitive_score is not None and isinstance(cognitive_score, (int, float)) and cognitive_score < 0.5:
                    recommendations.append("Cognitive accessibility is low. Consider simplifying your expression.")
            except Exception as e:
                logger.debug(f"Could not process human analysis recommendations: {e}")
        
        if not recommendations:
            recommendations.append("Expression analysis shows good balance across psychological dimensions.")
        
        return recommendations
    
    def compare_expressions(self, texts: List[str], context: Optional[Dict] = None) -> Dict[str, Any]:
        """Compare multiple expressions psychologically"""
        analyses = []
        
        for text in texts:
            analysis = self.analyze_text_comprehensively(text, context)
            analyses.append(analysis)
        
        # Generate comparison
        comparison = {
            'individual_analyses': analyses,
            'comparative_insights': self._generate_comparative_insights(analyses),
            'ranking': self._rank_expressions(analyses)
        }
        
        return comparison
    
    def _generate_comparative_insights(self, analyses: List[Dict]) -> Dict[str, Any]:
        """Generate insights from comparing multiple expressions"""
        if len(analyses) < 2:
            return {"message": "Need at least 2 expressions for comparison"}
        
        # Extract psychological profiles
        profiles = []
        for analysis in analyses:
            if 'psychological_analysis' in analysis and 'psychological_profile' in analysis['psychological_analysis']:
                profiles.append(analysis['psychological_analysis']['psychological_profile'])
        
        if not profiles:
            return {"message": "No psychological profiles available for comparison"}
        
        # Compare conflict levels
        conflicts = [p['psychological_dynamics']['internal_conflict'] for p in profiles]
        
        # Compare component harmony
        harmonies = [p['psychological_dynamics']['component_harmony'] for p in profiles]
        
        # Compare dominant drives
        drives = [self._identify_dominant_drive(p) for p in profiles]
        
        return {
            'conflict_analysis': {
                'range': [min(conflicts), max(conflicts)],
                'average': np.mean(conflicts),
                'most_conflicted_index': np.argmax(conflicts),
                'least_conflicted_index': np.argmin(conflicts)
            },
            'harmony_analysis': {
                'range': [min(harmonies), max(harmonies)],
                'average': np.mean(harmonies),
                'most_harmonious_index': np.argmax(harmonies),
                'least_harmonious_index': np.argmin(harmonies)
            },
            'drive_distribution': {drive: drives.count(drive) for drive in set(drives)}
        }
    
    def _rank_expressions(self, analyses: List[Dict]) -> List[Dict[str, Any]]:
        """Rank expressions by psychological balance and clarity"""
        rankings = []
        
        for i, analysis in enumerate(analyses):
            score = 0
            factors = {}
            
            if 'psychological_analysis' in analysis and 'psychological_profile' in analysis['psychological_analysis']:
                profile = analysis['psychological_analysis']['psychological_profile']
                dynamics = profile['psychological_dynamics']
                
                # Factors for ranking
                factors['decision_clarity'] = dynamics['decision_clarity']
                factors['component_harmony'] = dynamics['component_harmony']
                factors['low_conflict'] = 1.0 - dynamics['internal_conflict']
                
                # Calculate composite score
                score = np.mean(list(factors.values()))
            
            rankings.append({
                'index': i,
                'text': analysis['text'][:100] + '...' if len(analysis['text']) > 100 else analysis['text'],
                'score': score,
                'factors': factors
            })
        
        # Sort by score descending
        rankings.sort(key=lambda x: x['score'], reverse=True)
        
        return rankings


def demonstrate_psychological_nlp_analysis():
    """Demonstrate the psychological NLP analyzer"""
    print("=" * 70)
    print("PSYCHOLOGICAL NLP ANALYSIS DEMONSTRATION")
    print("=" * 70)
    
    # Create analyzer
    analyzer = PsychologicalNLPAnalyzer()
    
    # Sample texts with different psychological characteristics
    sample_texts = [
        "I absolutely must have this right now! It's driving me crazy!",  # High id
        "Let me carefully consider all the options and their consequences before deciding.",  # High ego
        "We should do what's morally right, regardless of personal cost.",  # High superego
        "I want it, but I know I should think about whether it's the right thing to do.",  # Balanced
        "This is terrible! But maybe we can find a logical solution that's also ethical."  # Mixed
    ]
    
    # Context for analysis
    context = {
        'speaker': 'user',
        'situation': 'decision_making',
        'formality_level': 'informal',
        'cultural_background': 'universal'
    }
    
    print("\n1. INDIVIDUAL TEXT ANALYSIS")
    print("-" * 40)
    
    for i, text in enumerate(sample_texts[:3]):  # Analyze first 3
        print(f"\nText {i+1}: \"{text}\"")
        analysis = analyzer.analyze_text_comprehensively(text, context)
        
        # Display key insights
        insights = analysis['integrated_insights']
        psych_profile = insights['psychological_profile']
        
        print(f"  Dominant Drive: {psych_profile['dominant_drive']}")
        print(f"  Emotional Regulation: {psych_profile['emotional_regulation']:.3f}")
        print(f"  Moral Orientation: {psych_profile['moral_orientation']:.3f}")
        print(f"  Internal Coherence: {psych_profile['internal_coherence']:.3f}")
        
        # Display recommendations
        print("  Recommendations:")
        for rec in analysis['recommendations'][:2]:  # Show first 2
            print(f"    - {rec}")
    
    print("\n2. COMPARATIVE ANALYSIS")
    print("-" * 40)
    
    comparison = analyzer.compare_expressions(sample_texts, context)
    
    # Display comparative insights
    comp_insights = comparison['comparative_insights']
    print(f"Conflict Analysis:")
    print(f"  Average conflict level: {comp_insights['conflict_analysis']['average']:.3f}")
    print(f"  Most conflicted: Text {comp_insights['conflict_analysis']['most_conflicted_index'] + 1}")
    print(f"  Least conflicted: Text {comp_insights['conflict_analysis']['least_conflicted_index'] + 1}")
    
    print(f"\nDrive Distribution:")
    for drive, count in comp_insights['drive_distribution'].items():
        print(f"  {drive.capitalize()}: {count} texts")
    
    print(f"\n3. EXPRESSION RANKING")
    print("-" * 40)
    
    rankings = comparison['ranking']
    print("Ranked by psychological balance and clarity:")
    for i, rank in enumerate(rankings[:3]):  # Show top 3
        print(f"{i+1}. Text {rank['index']+1} (Score: {rank['score']:.3f})")
        print(f"   \"{rank['text']}\"")
        print(f"   Key factors: Clarity={rank['factors'].get('decision_clarity', 0):.3f}, "
              f"Harmony={rank['factors'].get('component_harmony', 0):.3f}")
    
    print(f"\n4. PSYCHOLOGICAL COMPONENT ANALYSIS")
    print("-" * 40)
    
    # Analyze the psychological components for one text
    test_text = sample_texts[3]  # The balanced one
    analysis = analyzer.analyze_text_comprehensively(test_text, context)
    
    if 'psychological_analysis' in analysis:
        psych = analysis['psychological_analysis']
        if 'integrated_response' in psych:
            rationale = psych['integrated_response'].decision_rationale
            print(f"Text: \"{test_text}\"")
            print(f"Component weights:")
            print(f"  Id (instinctual): {rationale.get('id_weight', 0):.3f}")
            print(f"  Ego (rational): {rationale.get('ego_weight', 0):.3f}")
            print(f"  Superego (moral): {rationale.get('superego_weight', 0):.3f}")
            
            conflict = psych['integrated_response'].conflict_level
            print(f"Internal conflict level: {conflict:.3f}")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run demonstration
    demonstrate_psychological_nlp_analysis()