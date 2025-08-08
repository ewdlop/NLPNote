import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet
from nltk.tag import pos_tag
from collections import defaultdict
from typing import Dict, List, Any  # Add typing imports
import re

# Try to import spacy, handle gracefully if not available
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

# Import the enhanced human expression evaluation framework
try:
    from HumanExpressionEvaluator import HumanExpressionEvaluator, ExpressionContext
    EXPRESSION_EVALUATOR_AVAILABLE = True
except ImportError:
    # Fallback if module not available
    HumanExpressionEvaluator = None
    ExpressionContext = None
    EXPRESSION_EVALUATOR_AVAILABLE = False

# Import the new crackpot evaluation capabilities
try:
    from CrackpotEvaluator import CrackpotEvaluator, CrackpotGenerator
    CRACKPOT_AVAILABLE = True
except ImportError:
    CrackpotEvaluator = None
    CrackpotGenerator = None
    CRACKPOT_AVAILABLE = False

class SubtextAnalyzer:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            nltk.download('averaged_perceptron_tagger')
            nltk.download('wordnet')
        
        # Load spaCy model for advanced NLP
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except:
                print("spaCy model 'en_core_web_sm' not found. Some advanced features will be limited.")
                self.nlp = None
        else:
            print("spaCy not available. Some advanced features will be limited.")
            self.nlp = None
        
        # Initialize human expression evaluator if available
        if EXPRESSION_EVALUATOR_AVAILABLE:
            self.expression_evaluator = HumanExpressionEvaluator()
        else:
            self.expression_evaluator = None
        
        # Initialize crackpot evaluator if available
        if CRACKPOT_AVAILABLE:
            self.crackpot_evaluator = CrackpotEvaluator()
            self.crackpot_generator = CrackpotGenerator()
        else:
            self.crackpot_evaluator = None
            self.crackpot_generator = None
    
    def calculate_lexical_density(self, text):
        """Calculate the lexical density (ratio of content words to total words)"""
        tokens = word_tokenize(text.lower())
        pos_tags = pos_tag(tokens)
        
        content_words = [word for word, pos in pos_tags 
                        if pos.startswith(('NN', 'VB', 'JJ', 'RB'))]
        
        return len(content_words) / len(tokens) if tokens else 0

    def analyze_ambiguity(self, text):
        """Analyze word ambiguity using WordNet"""
        tokens = word_tokenize(text.lower())
        ambiguity_scores = []
        
        for word in tokens:
            synsets = wordnet.synsets(word)
            if synsets:
                # Score based on number of different meanings
                ambiguity_scores.append(len(synsets))
        
        return np.mean(ambiguity_scores) if ambiguity_scores else 0

    def analyze_symbolism(self, text):
        """Analyze potential symbolic content"""
        if self.nlp:
            doc = self.nlp(text)
            word_count = len(doc)
        else:
            # Fallback to simple tokenization
            words = text.lower().split()
            word_count = len(words)
        
        symbolic_score = 0
        
        # Common symbolic elements
        symbolic_categories = {
            'nature': ['sun', 'moon', 'star', 'tree', 'river', 'mountain', 'ocean'],
            'colors': ['red', 'blue', 'white', 'black', 'green', 'gold'],
            'animals': ['lion', 'eagle', 'snake', 'wolf', 'dove', 'raven'],
            'elements': ['fire', 'water', 'earth', 'air', 'wind'],
            'time': ['dawn', 'dusk', 'night', 'day', 'twilight', 'sunrise', 'sunset']
        }
        
        # Count symbolic references
        symbol_count = 0
        
        if self.nlp:
            for token in doc:
                word = token.text.lower()
                for category in symbolic_categories.values():
                    if word in category:
                        symbol_count += 1
        else:
            # Fallback approach
            for word in text.lower().split():
                for category in symbolic_categories.values():
                    if word in category:
                        symbol_count += 1
        
        return symbol_count / word_count if word_count else 0

    def analyze_emotion_depth(self, text):
        """Analyze emotional complexity and depth"""
        if self.nlp:
            doc = self.nlp(text)
            total_words = len(doc)
        else:
            words = text.lower().split()
            total_words = len(words)
        
        # Emotional indicators
        emotional_words = {
            'basic': ['happy', 'sad', 'angry', 'scared', 'surprised'],
            'complex': ['melancholic', 'euphoric', 'ambivalent', 'nostalgic', 'contemplative',
                       'bittersweet', 'wistful', 'yearning', 'transcendent']
        }
        
        basic_count = 0
        complex_count = 0
        
        if self.nlp:
            for token in doc:
                word = token.text.lower()
                if word in emotional_words['basic']:
                    basic_count += 1
                if word in emotional_words['complex']:
                    complex_count += 2  # Weight complex emotions more heavily
        else:
            # Fallback approach
            for word in text.lower().split():
                if word in emotional_words['basic']:
                    basic_count += 1
                if word in emotional_words['complex']:
                    complex_count += 2
        
        emotion_score = (basic_count + complex_count) / total_words if total_words else 0
        
        return emotion_score

    def analyze_metaphorical_content(self, text):
        """Analyze potential metaphorical content"""
        if self.nlp:
            doc = self.nlp(text)
            sentences = list(doc.sents)
        else:
            # Fallback to simple sentence splitting
            sentences = re.split(r'[.!?]', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        # Indicators of metaphorical language
        metaphor_markers = ['like', 'as', 'seems', 'appears', 'represents']
        abstract_concepts = ['love', 'time', 'life', 'death', 'freedom', 'hope', 'fear']
        
        metaphor_score = 0
        
        if self.nlp and sentences:
            for sent in sentences:
                # Check for metaphor markers
                has_marker = any(token.text.lower() in metaphor_markers for token in sent)
                has_abstract = any(token.text.lower() in abstract_concepts for token in sent)
                
                if has_marker and has_abstract:
                    metaphor_score += 1
        else:
            # Fallback approach
            for sent in sentences:
                sent_lower = sent.lower()
                has_marker = any(marker in sent_lower for marker in metaphor_markers)
                has_abstract = any(concept in sent_lower for concept in abstract_concepts)
                
                if has_marker and has_abstract:
                    metaphor_score += 1
        
        return metaphor_score / len(sentences) if sentences else 0

    def calculate_subtext_probability(self, text):
        """Calculate overall probability of deeper meaning"""
        # Calculate individual metrics
        lexical_density = self.calculate_lexical_density(text)
        ambiguity = self.analyze_ambiguity(text)
        symbolism = self.analyze_symbolism(text)
        emotion_depth = self.analyze_emotion_depth(text)
        metaphorical = self.analyze_metaphorical_content(text)
        
        # Weight the features
        weights = {
            'lexical_density': 0.15,
            'ambiguity': 0.20,
            'symbolism': 0.25,
            'emotion_depth': 0.20,
            'metaphorical': 0.20
        }
        
        # Calculate weighted score
        score = (
            lexical_density * weights['lexical_density'] +
            ambiguity * weights['ambiguity'] +
            symbolism * weights['symbolism'] +
            emotion_depth * weights['emotion_depth'] +
            metaphorical * weights['metaphorical']
        )
        
        # Normalize to probability between 0 and 1
        probability = min(max(score, 0), 1)
        
        return {
            'probability': probability,
            'components': {
                'lexical_density': lexical_density,
                'ambiguity': ambiguity,
                'symbolism': symbolism,
                'emotion_depth': emotion_depth,
                'metaphorical': metaphorical
            }
        }

    def analyze_expression_evaluation(self, text, context=None):
        """
        分析人類表達的評估過程 (Analyze human expression evaluation process)
        整合新的人類表達評估框架 (Integrate the new human expression evaluation framework)
        """
        if not self.expression_evaluator:
            return {
                'error': 'Human Expression Evaluator not available',
                'traditional_analysis': self.calculate_subtext_probability(text)
            }
        
        # Create context if not provided
        if context is None and ExpressionContext:
            context = ExpressionContext(
                situation='literary_analysis',
                formality_level='neutral'
            )
        
        # Get comprehensive evaluation
        evaluation_results = self.expression_evaluator.comprehensive_evaluation(text, context)
        
        # Combine with traditional subtext analysis
        traditional_analysis = self.calculate_subtext_probability(text)
        
        # Integrate results
        integrated_analysis = {
            'expression_evaluation': evaluation_results,
            'subtext_analysis': traditional_analysis,
            'comparison': self._compare_analyses(evaluation_results, traditional_analysis),
            'interpretation': self._generate_integrated_interpretation(evaluation_results, traditional_analysis)
        }
        
        return integrated_analysis
    
    def _compare_analyses(self, expression_eval, subtext_eval):
        """比較兩種分析方法的結果 (Compare results from both analysis methods)"""
        comparison = {}
        
        # Compare overall scores
        expr_score = expression_eval['integrated']['overall_score']
        subtext_score = subtext_eval['probability']
        
        comparison['score_correlation'] = abs(expr_score - subtext_score)
        comparison['agreement_level'] = 'high' if comparison['score_correlation'] < 0.2 else 'medium' if comparison['score_correlation'] < 0.4 else 'low'
        
        # Compare specific dimensions
        if 'formal_semantic' in expression_eval:
            formal_score = expression_eval['formal_semantic'].score
            comparison['semantic_vs_subtext'] = {
                'formal_semantic': formal_score,
                'subtext_probability': subtext_score,
                'difference': abs(formal_score - subtext_score)
            }
        
        return comparison
    
    def _generate_integrated_interpretation(self, expression_eval, subtext_eval):
        """生成整合解釋 (Generate integrated interpretation)"""
        interpretation = []
        
        # Expression evaluation insights
        expr_score = expression_eval['integrated']['overall_score']
        expr_characteristics = expression_eval['integrated']['characteristics']
        
        interpretation.append(f"表達評估分數: {expr_score:.2f}")
        interpretation.append(f"表達特徵: {expr_characteristics}")
        
        # Subtext analysis insights
        subtext_score = subtext_eval['probability']
        subtext_components = subtext_eval['components']
        
        interpretation.append(f"潛文本分析分數: {subtext_score:.2f}")
        interpretation.append(f"主要潛文本指標: 象徵性 {subtext_components['symbolism']:.2f}, 情感深度 {subtext_components['emotion_depth']:.2f}")
        
        # Combined insight
        if expr_score > 0.7 and subtext_score > 0.7:
            interpretation.append("這是一個高質量的表達，具有豐富的潛在含義和良好的表達形式。")
        elif expr_score > 0.5 or subtext_score > 0.5:
            interpretation.append("這個表達在某些維度表現較好，可能具有一定的深層含義。")
        else:
            interpretation.append("這是一個相對直接的表達，潛在含義較少。")
        
        return "\n".join(interpretation)
    
    def make_text_more_crackpot(self, text: str, intensity: float = 0.6) -> Dict[str, Any]:
        """
        讓文本更加crackpot (Make text more crackpot)
        Enhanced subtext analysis with crackpot transformation
        """
        if not self.crackpot_generator:
            return {
                'error': 'Crackpot generator not available',
                'suggestion': 'Install CrackpotEvaluator for enhanced creativity features'
            }
        
        # Original analysis
        original_analysis = self.calculate_subtext_probability(text)
        
        # Crackpot enhancement
        enhanced_text = self.crackpot_generator.enhance_text_crackpotness(text, intensity)
        
        # Enhanced analysis
        enhanced_analysis = self.calculate_subtext_probability(enhanced_text)
        
        # Crackpot evaluation
        crackpot_results = self.crackpot_evaluator.evaluate_crackpot_level(enhanced_text)
        avg_crackpot_score = sum(result.score for result in crackpot_results.values()) / len(crackpot_results)
        
        return {
            'original_text': text,
            'enhanced_text': enhanced_text,
            'original_subtext_score': original_analysis['probability'],
            'enhanced_subtext_score': enhanced_analysis['probability'],
            'crackpot_score': avg_crackpot_score,
            'enhancement_intensity': intensity,
            'improvement_factor': enhanced_analysis['probability'] / max(original_analysis['probability'], 0.01),
            'analysis': {
                'original': original_analysis,
                'enhanced': enhanced_analysis,
                'crackpot_breakdown': crackpot_results
            },
            'recommendation': self._get_crackpot_recommendation(avg_crackpot_score)
        }
    
    def _get_crackpot_recommendation(self, score: float) -> str:
        """Get recommendation based on crackpot score"""
        if score > 0.7:
            return "🎉 Excellent! Your text is now beautifully unconventional and creative!"
        elif score > 0.4:
            return "🌟 Good progress! Consider adding more wild concepts or conspiracy elements."
        elif score > 0.2:
            return "💡 Getting there! Try including more pseudoscientific terms or extreme language."
        else:
            return "⚡ Still quite conventional. Consider dramatic enhancement with quantum mysticism!"
    
    def generate_crackpot_interpretation(self, text: str) -> str:
        """
        生成text的crackpot解釋 (Generate crackpot interpretation of text)
        """
        if not self.crackpot_generator:
            return "Crackpot interpretation requires CrackpotEvaluator installation."
        
        # Generate multiple wild interpretations
        interpretations = []
        
        # Extract key words from the text
        words = text.lower().split()
        key_words = [word for word in words if len(word) > 4][:3]  # Get up to 3 meaningful words
        
        for word in key_words:
            associations = self.crackpot_generator.generate_random_associations(word, 2)
            interpretations.extend(associations)
        
        # Generate a wild theory about the text's "true meaning"
        wild_theory = self.crackpot_generator.generate_crackpot_theory("this text")
        
        interpretation = f"""
🔮 CRACKPOT INTERPRETATION 🔮
=============================

The text appears normal on the surface, but deeper analysis reveals:

🌟 Hidden Meanings:
{chr(10).join(f"  • {interp}" for interp in interpretations[:4])}

🚀 Ultimate Truth:
{wild_theory}

⚡ Conclusion: This text operates on multiple dimensional frequencies that most conventional analysis cannot detect!
"""
        return interpretation
    def generate_analysis_report(self, text):
        """Generate a detailed analysis report"""
        analysis = self.calculate_subtext_probability(text)
        
        report = "Text Subtext Analysis Report\n"
        report += "==========================\n\n"
        
        report += f"Overall Probability of Deeper Meaning: {analysis['probability']:.2%}\n\n"
        
        report += "Component Analysis:\n"
        report += "-----------------\n"
        for component, score in analysis['components'].items():
            report += f"{component.replace('_', ' ').title()}: {score:.2%}\n"
        
        # Add interpretation
        report += "\nInterpretation:\n"
        report += "-------------\n"
        if analysis['probability'] > 0.7:
            report += "High likelihood of deeper meaning. Text shows significant complexity and layered meaning.\n"
        elif analysis['probability'] > 0.4:
            report += "Moderate likelihood of deeper meaning. Some subtle layers present.\n"
        else:
            report += "Lower likelihood of deeper meaning. Text appears more straightforward.\n"
        
        # Add human expression evaluation if available
        if self.expression_evaluator:
            report += "\n" + "="*50 + "\n"
            report += "HUMAN EXPRESSION EVALUATION ANALYSIS\n"
            report += "="*50 + "\n"
            
            expr_analysis = self.analyze_expression_evaluation(text)
            if 'error' not in expr_analysis:
                report += "\n人類表達評估結果 (Human Expression Evaluation Results):\n"
                report += "-" * 50 + "\n"
                report += expr_analysis['interpretation']
                
                report += "\n\n分析方法比較 (Analysis Method Comparison):\n"
                report += "-" * 50 + "\n"
                comparison = expr_analysis['comparison']
                report += f"分析一致性: {comparison['agreement_level']}\n"
                report += f"分數差異: {comparison['score_correlation']:.2f}\n"
        
        return report

def main():
    analyzer = SubtextAnalyzer()
    
    # Example texts - Enhanced with crackpot demonstrations!
    texts = [
        """The old man watched the sunset, his weathered hands gripping the wooden rail. 
        Golden light stretched across the water like molten dreams, each wave carrying 
        memories of youth and forgotten promises.""",
        
        """The store closes at 5 PM. Please make sure to complete your shopping before then. 
        The parking lot will be locked afterward.""",
        
        """Artificial intelligence is transforming how we work and communicate."""
    ]
    
    print("🌟 ENHANCED SUBTEXT ANALYZER 🌟")
    print("Now with Crackpot Enhancement Capabilities!")
    print("=" * 60)
    
    for i, text in enumerate(texts, 1):
        print(f"\n🔍 Analyzing Text {i}:")
        print("-" * 50)
        print(text)
        print("\n" + analyzer.generate_analysis_report(text))
        
        # NEW: Crackpot enhancement demo
        if analyzer.crackpot_generator:
            print("\n" + "🚀 CRACKPOT ENHANCEMENT DEMO:")
            print("-" * 30)
            
            crackpot_result = analyzer.make_text_more_crackpot(text, 0.7)
            if 'error' not in crackpot_result:
                print(f"Original: {crackpot_result['original_text']}")
                print(f"Enhanced: {crackpot_result['enhanced_text']}")
                print(f"Subtext Score: {crackpot_result['original_subtext_score']:.2f} → {crackpot_result['enhanced_subtext_score']:.2f}")
                print(f"Crackpot Score: {crackpot_result['crackpot_score']:.2f}")
                print(f"Improvement Factor: {crackpot_result['improvement_factor']:.1f}x")
                print(f"Recommendation: {crackpot_result['recommendation']}")
            
            # Generate wild interpretation
            print(f"\n{analyzer.generate_crackpot_interpretation(text)}")
        
        print("\n" + "="*60)
    
    # Bonus: Pure crackpot generation
    if analyzer.crackpot_generator:
        print("\n🎉 BONUS: PURE CRACKPOT THEORY GENERATION 🎉")
        topics = ["natural language processing", "text analysis", "machine learning"]
        for topic in topics:
            theory = analyzer.crackpot_generator.generate_crackpot_theory(topic)
            print(f"💫 {topic.title()}: {theory}")

if __name__ == "__main__":
    main()
