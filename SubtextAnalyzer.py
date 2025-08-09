import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet
from nltk.tag import pos_tag
from collections import defaultdict
import re

# Try to import spacy, handle gracefully if not available
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

# Import the new human expression evaluation framework
try:
    from HumanExpressionEvaluator import HumanExpressionEvaluator, ExpressionContext
    EXPRESSION_EVALUATOR_AVAILABLE = True
except ImportError:
    # Fallback if module not available
    HumanExpressionEvaluator = None
    ExpressionContext = None
    EXPRESSION_EVALUATOR_AVAILABLE = False

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
        
        # Add neural firing analysis simulation for LLM-based text analysis
        neural_firing_analysis = self._analyze_neural_firing_for_text(text)
        
        # Integrate results
        integrated_analysis = {
            'expression_evaluation': evaluation_results,
            'subtext_analysis': traditional_analysis,
            'neural_firing_analysis': neural_firing_analysis,
            'comparison': self._compare_analyses(evaluation_results, traditional_analysis),
            'interpretation': self._generate_integrated_interpretation(evaluation_results, traditional_analysis, neural_firing_analysis)
        }
        
        return integrated_analysis
    
    def _analyze_neural_firing_for_text(self, text):
        """
        Simulate neural firing analysis for text processing
        模擬文本處理的神經元激發分析
        """
        try:
            from NeuralFiringAnalyzer import NeuralFiringAnalyzer
            
            # Create neural firing analyzer
            firing_analyzer = NeuralFiringAnalyzer()
            
            # Simulate neural activations based on text characteristics
            text_length = len(text.split())
            text_complexity = len(set(text.lower().split())) / len(text.split()) if text.split() else 0
            
            # Create simulated activation patterns based on text properties
            # This simulates how an LLM might process this text
            activation_patterns = []
            
            # Simulate different processing stages
            for stage in ['tokenization', 'embedding', 'attention', 'processing', 'output']:
                if stage == 'tokenization':
                    # High activation for complex texts
                    activation = np.random.normal(0.8, 0.2, (text_length, 64))
                elif stage == 'embedding':
                    # Activation based on vocabulary diversity
                    activation = np.random.normal(text_complexity, 0.1, (text_length, 128))
                elif stage == 'attention':
                    # Simulate attention patterns - some missing firing for long texts
                    base_activation = 0.7 if text_length < 50 else 0.4
                    activation = np.random.normal(base_activation, 0.3, (text_length, 256))
                    # Simulate missing firing in attention heads
                    if text_length > 100:
                        missing_mask = np.random.random(activation.shape) < 0.3
                        activation[missing_mask] = 0
                elif stage == 'processing':
                    # Middle layer processing - potential vanishing gradients
                    activation = np.random.normal(0.5, 0.4, (text_length, 512))
                    if text_complexity < 0.3:  # Simple texts might cause weak firing
                        activation *= 0.1
                elif stage == 'output':
                    # Output layer - final decision making
                    activation = np.random.normal(0.6, 0.2, (text_length, 64))
                
                # Apply tanh activation function
                activation = np.tanh(activation)
                
                # Analyze this stage
                stage_analysis = firing_analyzer.analyze_activation_tensor(activation, f"text_{stage}")
                activation_patterns.append({
                    'stage': stage,
                    'analysis': stage_analysis
                })
            
            # Calculate overall neural firing health for text processing
            overall_firing_rate = np.mean([pattern['analysis'].activation_rate for pattern in activation_patterns])
            total_issues = sum(len(pattern['analysis'].issues) for pattern in activation_patterns)
            
            neural_health_score = max(0.0, 1.0 - (total_issues * 0.1))
            
            return {
                'overall_firing_rate': overall_firing_rate,
                'neural_health_score': neural_health_score,
                'total_issues': total_issues,
                'stage_analyses': activation_patterns,
                'interpretation': self._interpret_neural_firing_for_text(overall_firing_rate, neural_health_score, total_issues)
            }
            
        except ImportError:
            return {
                'error': 'Neural Firing Analyzer not available',
                'interpretation': 'Neural firing analysis requires NeuralFiringAnalyzer module'
            }
    
    def _interpret_neural_firing_for_text(self, firing_rate, health_score, total_issues):
        """Interpret neural firing results for text processing"""
        if health_score > 0.8 and firing_rate > 0.7:
            return "文本處理的神經激發模式良好，LLM應能有效處理此文本 (Good neural firing patterns for text processing, LLM should handle this text effectively)"
        elif health_score > 0.6 and firing_rate > 0.5:
            return "文本處理存在一些神經激發問題，可能影響LLM性能 (Some neural firing issues in text processing, may affect LLM performance)"
        elif total_issues > 10:
            return "檢測到嚴重的神經激發問題，LLM可能難以正確處理此文本 (Serious neural firing issues detected, LLM may struggle to process this text correctly)"
        else:
            return "神經激發模式需要改善以獲得更好的文本處理效果 (Neural firing patterns need improvement for better text processing)"
    
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
    
    def _generate_integrated_interpretation(self, expression_eval, subtext_eval, neural_firing_eval=None):
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
        
        # Neural firing analysis insights (if available)
        if neural_firing_eval and 'error' not in neural_firing_eval:
            neural_score = neural_firing_eval['neural_health_score']
            firing_rate = neural_firing_eval['overall_firing_rate']
            interpretation.append(f"神經激發健康分數: {neural_score:.2f}")
            interpretation.append(f"整體激發率: {firing_rate:.2f}")
            interpretation.append(f"神經激發分析: {neural_firing_eval['interpretation']}")
        
        # Combined insight
        if expr_score > 0.7 and subtext_score > 0.7:
            interpretation.append("這是一個高質量的表達，具有豐富的潛在含義和良好的表達形式。")
        elif expr_score > 0.5 or subtext_score > 0.5:
            interpretation.append("這個表達在某些維度表現較好，可能具有一定的深層含義。")
        else:
            interpretation.append("這是一個相對直接的表達，潛在含義較少。")
        
        # Add neural firing insights to overall interpretation
        if neural_firing_eval and 'neural_health_score' in neural_firing_eval:
            neural_score = neural_firing_eval['neural_health_score']
            if neural_score < 0.5:
                interpretation.append("警告: 檢測到神經激發問題，可能影響LLM對此文本的處理能力。")
            elif neural_score > 0.8:
                interpretation.append("神經激發模式良好，LLM應能有效處理此文本。")
        
        return "\n".join(interpretation)
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
    
    # Example texts
    texts = [
        """The old man watched the sunset, his weathered hands gripping the wooden rail. 
        Golden light stretched across the water like molten dreams, each wave carrying 
        memories of youth and forgotten promises.""",
        
        """The store closes at 5 PM. Please make sure to complete your shopping before then. 
        The parking lot will be locked afterward."""
    ]
    
    for i, text in enumerate(texts, 1):
        print(f"\nAnalyzing Text {i}:")
        print("-" * 50)
        print(text)
        print("\n" + analyzer.generate_analysis_report(text))

if __name__ == "__main__":
    main()
