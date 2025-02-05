import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet
from nltk.tag import pos_tag
from collections import defaultdict
import re
import spacy

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
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            print("Please install spaCy and the English model: python -m spacy download en_core_web_sm")
    
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
        doc = self.nlp(text)
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
        word_count = len(doc)
        symbol_count = 0
        
        for token in doc:
            word = token.text.lower()
            for category in symbolic_categories.values():
                if word in category:
                    symbol_count += 1
        
        return symbol_count / word_count if word_count else 0

    def analyze_emotion_depth(self, text):
        """Analyze emotional complexity and depth"""
        doc = self.nlp(text)
        
        # Emotional indicators
        emotional_words = {
            'basic': ['happy', 'sad', 'angry', 'scared', 'surprised'],
            'complex': ['melancholic', 'euphoric', 'ambivalent', 'nostalgic', 'contemplative',
                       'bittersweet', 'wistful', 'yearning', 'transcendent']
        }
        
        basic_count = 0
        complex_count = 0
        
        for token in doc:
            word = token.text.lower()
            if word in emotional_words['basic']:
                basic_count += 1
            if word in emotional_words['complex']:
                complex_count += 2  # Weight complex emotions more heavily
        
        total_words = len(doc)
        emotion_score = (basic_count + complex_count) / total_words if total_words else 0
        
        return emotion_score

    def analyze_metaphorical_content(self, text):
        """Analyze potential metaphorical content"""
        doc = self.nlp(text)
        
        # Indicators of metaphorical language
        metaphor_markers = ['like', 'as', 'seems', 'appears', 'represents']
        abstract_concepts = ['love', 'time', 'life', 'death', 'freedom', 'hope', 'fear']
        
        metaphor_score = 0
        sentences = list(doc.sents)
        
        for sent in sentences:
            # Check for metaphor markers
            has_marker = any(token.text.lower() in metaphor_markers for token in sent)
            has_abstract = any(token.text.lower() in abstract_concepts for token in sent)
            
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
