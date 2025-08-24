#!/usr/bin/env python3
"""
Extraterrestrial Plea Analyzer - A computational analysis of cosmic linguistic patterns
Related to: Universe Spare We - Linguistic Analysis of Extraterrestrial Plea.md
"""

import re
import random
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import Counter

@dataclass
class LinguisticAnalysis:
    """Data structure for linguistic analysis results"""
    grammatical_deviation: List[Tuple[str, str]]
    emotional_intensity: float
    cosmic_scope: float
    agency_confusion: float
    case_errors: List[str]
    suggestions: List[str]

class ExtraterrestrialPleaAnalyzer:
    """Analyzes linguistic patterns in cosmic appeals and extraterrestrial pleas"""
    
    def __init__(self):
        self.grammatical_patterns = {
            'standard': 'could you spare us',
            'observed': 'could you spare we',
            'archaic': 'couldst thou spare we',
            'desperate': 'please spare we somehow'
        }
        
        self.cosmic_entities = [
            'universe', 'cosmos', 'space', 'galaxy', 'stars',
            'aliens', 'extraterrestrials', 'beings', 'intelligence'
        ]
        
        self.emotional_markers = [
            'please', 'help', 'save', 'spare', 'rescue', 'mercy',
            'desperate', 'urgent', 'critical', 'emergency'
        ]
        
        self.case_error_patterns = [
            (r'\bspare we\b', 'spare us'),
            (r'\bhelp we\b', 'help us'),
            (r'\bsave we\b', 'save us'),
            (r'\bforgive we\b', 'forgive us'),
            (r'\brescue we\b', 'rescue us')
        ]
    
    def analyze_plea(self, text: str) -> LinguisticAnalysis:
        """Comprehensive analysis of cosmic plea text"""
        text_lower = text.lower()
        
        analysis = LinguisticAnalysis(
            grammatical_deviation=self.detect_case_errors(text_lower),
            emotional_intensity=self.measure_desperation(text_lower),
            cosmic_scope=self.detect_universal_address(text_lower),
            agency_confusion=self.analyze_pronoun_usage(text_lower),
            case_errors=self.find_case_errors(text_lower),
            suggestions=self.generate_corrections(text)
        )
        
        return analysis
    
    def detect_case_errors(self, text: str) -> List[Tuple[str, str]]:
        """Detect nominative/accusative case confusion"""
        errors = []
        for pattern, error_type in [
            ('spare we', 'case_mismatch'),
            ('help we', 'case_mismatch'),
            ('save we', 'case_mismatch'),
            ('forgive we', 'case_mismatch')
        ]:
            if pattern in text:
                errors.append((pattern, error_type))
        return errors
    
    def measure_desperation(self, text: str) -> float:
        """Measure emotional intensity/desperation level (0-1)"""
        words = text.split()
        emotional_words = sum(1 for word in words if word in self.emotional_markers)
        
        # Count exclamation marks, question marks, and repeated letters
        punctuation_intensity = text.count('!') + text.count('?') * 0.5
        repeated_letters = len(re.findall(r'(.)\1{2,}', text)) * 0.1
        
        # Normalize to 0-1 scale
        intensity = min(1.0, (emotional_words / len(words) * 2) + 
                             (punctuation_intensity / 10) + repeated_letters)
        return round(intensity, 3)
    
    def detect_universal_address(self, text: str) -> float:
        """Detect how much the text addresses cosmic/universal entities"""
        cosmic_words = sum(1 for entity in self.cosmic_entities if entity in text)
        words = text.split()
        
        # Check for direct address patterns
        direct_address = 1.0 if any(pattern in text for pattern in [
            'universe,', 'cosmos,', 'space,', 'aliens,'
        ]) else 0.0
        
        scope = min(1.0, (cosmic_words / len(words) * 5) + direct_address * 0.5)
        return round(scope, 3)
    
    def analyze_pronoun_usage(self, text: str) -> float:
        """Analyze pronoun usage for agency confusion"""
        we_count = len(re.findall(r'\bwe\b', text))
        us_count = len(re.findall(r'\bus\b', text))
        
        # Calculate confusion score based on unexpected "we" usage
        confusion_patterns = [
            'spare we', 'help we', 'save we', 'forgive we'
        ]
        confusion_count = sum(1 for pattern in confusion_patterns if pattern in text)
        
        # Normalize confusion score
        total_pronouns = we_count + us_count
        if total_pronouns == 0:
            return 0.0
        
        confusion_score = min(1.0, confusion_count / total_pronouns)
        return round(confusion_score, 3)
    
    def find_case_errors(self, text: str) -> List[str]:
        """Find specific case errors in the text"""
        errors = []
        for pattern, correction in self.case_error_patterns:
            matches = re.findall(pattern, text)
            errors.extend(matches)
        return errors
    
    def generate_corrections(self, text: str) -> List[str]:
        """Generate corrected versions of the text"""
        corrections = []
        corrected_text = text
        
        for pattern, correction in self.case_error_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                corrected_text = re.sub(pattern, correction, corrected_text, flags=re.IGNORECASE)
        
        if corrected_text != text:
            corrections.append(f"Grammatically corrected: {corrected_text}")
        
        return corrections
    
    def generate_cosmic_appeal(self, formality: str = 'desperate') -> str:
        """Generate variations of cosmic appeals"""
        templates = {
            'formal': "Universe, could you spare us by sending assistance?",
            'desperate': "Universe, could you spare we by sending aliens to us?",
            'poetic': "O Cosmos, spare we mortals through celestial intervention",
            'archaic': "Great Universe, we beseech thee, spare we thy children",
            'modern': "Hey universe, can you help us out by sending some aliens?",
            'scientific': "To any cosmic intelligence: requesting intervention via extraterrestrial contact"
        }
        return templates.get(formality, templates['desperate'])
    
    def probability_analysis(self, text: str) -> Dict[str, float]:
        """Analyze probability of different linguistic patterns"""
        # Simplified probability model
        standard_grammar_prob = 0.95
        case_error_prob = 0.05
        
        has_case_errors = bool(self.find_case_errors(text.lower()))
        cosmic_context = self.detect_universal_address(text.lower()) > 0.3
        emotional_stress = self.measure_desperation(text.lower()) > 0.5
        
        # Adjust probabilities based on context
        if cosmic_context and emotional_stress:
            # Grammar becomes less reliable under cosmic stress
            standard_grammar_prob = 0.3
            case_error_prob = 0.7
        
        return {
            'standard_grammar_probability': standard_grammar_prob,
            'case_error_probability': case_error_prob,
            'context_adjustment': cosmic_context and emotional_stress
        }

def demonstrate_analysis():
    """Demonstrate the extraterrestrial plea analyzer"""
    analyzer = ExtraterrestrialPleaAnalyzer()
    
    # Test phrases
    test_phrases = [
        "Universe, could you spare we by sending aliens to us?",
        "Universe, could you spare us by sending aliens to us?",
        "Please help we! The aliens must come save us!",
        "O cosmos, spare we mortals from this earthly suffering",
        "Hello aliens, can you please help us contact the universe?",
        "We need help - could the universe spare we somehow?"
    ]
    
    print("Extraterrestrial Plea Analysis Results")
    print("=" * 50)
    
    for i, phrase in enumerate(test_phrases, 1):
        print(f"\n{i}. Analyzing: \"{phrase}\"")
        analysis = analyzer.analyze_plea(phrase)
        
        print(f"   Grammatical deviations: {analysis.grammatical_deviation}")
        print(f"   Emotional intensity: {analysis.emotional_intensity}")
        print(f"   Cosmic scope: {analysis.cosmic_scope}")
        print(f"   Agency confusion: {analysis.agency_confusion}")
        
        if analysis.case_errors:
            print(f"   Case errors found: {analysis.case_errors}")
        
        if analysis.suggestions:
            print(f"   Suggestions: {analysis.suggestions}")
        
        # Probability analysis
        prob_analysis = analyzer.probability_analysis(phrase)
        print(f"   Grammar probability: {prob_analysis['standard_grammar_probability']:.2f}")
        
        print("-" * 40)
    
    print("\nGenerated Cosmic Appeals:")
    print("=" * 30)
    
    formality_levels = ['formal', 'desperate', 'poetic', 'archaic', 'modern', 'scientific']
    for level in formality_levels:
        appeal = analyzer.generate_cosmic_appeal(level)
        print(f"{level.capitalize()}: {appeal}")

def linguistic_pattern_statistics():
    """Generate statistics about linguistic patterns in cosmic appeals"""
    analyzer = ExtraterrestrialPleaAnalyzer()
    
    # Simulate analysis of various cosmic appeals
    sample_appeals = [
        "Universe, spare we from destruction",
        "Aliens, please help we humans",
        "Cosmos, could you save we?",
        "Space beings, rescue we from Earth",
        "Universe, could you spare us graciously?",
        "Please aliens, help us contact the cosmos",
        "O universe, we beseech your mercy"
    ]
    
    print("\nLinguistic Pattern Statistics")
    print("=" * 40)
    
    case_error_count = 0
    total_appeals = len(sample_appeals)
    emotional_scores = []
    cosmic_scores = []
    
    for appeal in sample_appeals:
        analysis = analyzer.analyze_plea(appeal)
        if analysis.case_errors:
            case_error_count += 1
        emotional_scores.append(analysis.emotional_intensity)
        cosmic_scores.append(analysis.cosmic_scope)
    
    print(f"Total appeals analyzed: {total_appeals}")
    print(f"Appeals with case errors: {case_error_count} ({case_error_count/total_appeals*100:.1f}%)")
    print(f"Average emotional intensity: {sum(emotional_scores)/len(emotional_scores):.3f}")
    print(f"Average cosmic scope: {sum(cosmic_scores)/len(cosmic_scores):.3f}")
    
    print(f"\nCase error rate in cosmic contexts: {case_error_count/total_appeals*100:.1f}%")
    print("This suggests increased grammatical instability when addressing cosmic entities.")

if __name__ == "__main__":
    print("Extraterrestrial Plea Analyzer")
    print("Computational analysis of cosmic linguistic patterns")
    print("Related to 'Universe Spare We' linguistic analysis\n")
    
    # Run demonstrations
    demonstrate_analysis()
    linguistic_pattern_statistics()
    
    print("\n" + "="*60)
    print("Analysis complete. The grammatical deviation 'spare we' instead of")
    print("'spare us' appears to be a pattern that emerges under cosmic stress,")
    print("where traditional grammatical rules break down when humans attempt")
    print("to communicate with universal or extraterrestrial entities.")
    print("="*60)