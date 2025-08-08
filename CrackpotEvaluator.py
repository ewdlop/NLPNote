"""
Crackpot Evaluator - Making LLMs More Unconventional and Creative

This module implements mechanisms to evaluate and generate "crackpot" theories,
unconventional ideas, and creative thinking patterns to address the issue
"LLM arent crackpot enough".
"""

import random
import re
import math
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class CrackpotDimension(Enum):
    """Crackpot evaluation dimensions"""
    UNCONVENTIONALITY = "unconventionality"
    CREATIVITY = "creativity" 
    WILDNESS = "wildness"
    CONSPIRACY_LEVEL = "conspiracy_level"
    PSEUDOSCIENCE = "pseudoscience"
    RANDOMNESS = "randomness"

@dataclass
class CrackpotResult:
    """Result of crackpot evaluation"""
    dimension: CrackpotDimension
    score: float  # 0.0 to 1.0 (higher = more crackpot)
    explanation: str
    examples: List[str] = None

class CrackpotEvaluator:
    """
    Evaluates and generates crackpot theories and unconventional thinking
    """
    
    def __init__(self):
        # Crackpot theory elements
        self.conspiracy_keywords = [
            'secret', 'hidden', 'they dont want you to know', 'coverup', 'conspiracy',
            'illuminati', 'government', 'aliens', 'mind control', 'chemtrails',
            'flat earth', 'lizard people', 'new world order', 'deep state'
        ]
        
        self.pseudoscience_terms = [
            'quantum', 'energy field', 'vibrations', 'frequency', 'chakra',
            'crystal healing', 'ancient wisdom', 'sacred geometry', 'aura',
            'magnetic field', 'dimensional portal', 'consciousness field'
        ]
        
        self.wild_concepts = [
            'time travel', 'parallel universe', 'simulation theory', 'matrix',
            'holographic reality', 'interdimensional beings', 'astral projection',
            'telepathy', 'precognition', 'reality shifting', 'manifestation'
        ]
        
        # Crackpot sentence starters
        self.crackpot_starters = [
            "What if I told you that",
            "The REAL truth is that",
            "Scientists don't want you to know that",
            "Ancient civilizations discovered that",
            "The government is hiding the fact that",
            "Quantum physics proves that",
            "Sacred texts reveal that",
            "The elite know that"
        ]
        
        # Random association words for chaos injection
        self.chaos_words = [
            'banana', 'purple', 'spaghetti', 'cosmic', 'interdimensional',
            'vibrational', 'crystalline', 'ethereal', 'quantum', 'galactic',
            'mystical', 'ancient', 'forbidden', 'sacred', 'divine'
        ]

    def evaluate_crackpot_level(self, text: str) -> Dict[str, CrackpotResult]:
        """
        Evaluate how crackpot/unconventional a text is
        """
        results = {}
        
        # Evaluate each dimension
        results['unconventionality'] = self._evaluate_unconventionality(text)
        results['creativity'] = self._evaluate_creativity(text)
        results['wildness'] = self._evaluate_wildness(text)
        results['conspiracy_level'] = self._evaluate_conspiracy_level(text)
        results['pseudoscience'] = self._evaluate_pseudoscience(text)
        results['randomness'] = self._evaluate_randomness(text)
        
        return results
    
    def _evaluate_unconventionality(self, text: str) -> CrackpotResult:
        """Evaluate how unconventional the ideas are"""
        text_lower = text.lower()
        
        # Check for unconventional thinking patterns
        unconventional_patterns = [
            r'what if.*but.*opposite',
            r'everything you know.*wrong',
            r'reality is.*illusion',
            r'not what.*seems',
            r'hidden.*truth',
            r'beyond.*imagination'
        ]
        
        pattern_count = sum(len(re.findall(pattern, text_lower)) for pattern in unconventional_patterns)
        words = len(text.split())
        
        # Base score on pattern density
        score = min(pattern_count / max(words / 20, 1), 1.0)
        
        # Bonus for questioning conventional wisdom
        if any(phrase in text_lower for phrase in ['conventional wisdom', 'mainstream', 'established']):
            score += 0.2
        
        return CrackpotResult(
            dimension=CrackpotDimension.UNCONVENTIONALITY,
            score=min(score, 1.0),
            explanation=f"Found {pattern_count} unconventional thinking patterns"
        )
    
    def _evaluate_creativity(self, text: str) -> CrackpotResult:
        """Evaluate creative thinking elements"""
        text_lower = text.lower()
        
        # Creative indicators
        creative_words = [
            'imagine', 'envision', 'creative', 'innovative', 'revolutionary',
            'breakthrough', 'paradigm', 'transform', 'reimagine', 'visionary'
        ]
        
        metaphor_indicators = ['like', 'as if', 'resembles', 'similar to', 'analogous']
        
        creative_count = sum(text_lower.count(word) for word in creative_words)
        metaphor_count = sum(text_lower.count(indicator) for indicator in metaphor_indicators)
        
        words = len(text.split())
        creativity_score = (creative_count + metaphor_count * 1.5) / max(words / 10, 1)
        
        return CrackpotResult(
            dimension=CrackpotDimension.CREATIVITY,
            score=min(creativity_score, 1.0),
            explanation=f"Creative words: {creative_count}, Metaphors: {metaphor_count}"
        )
    
    def _evaluate_wildness(self, text: str) -> CrackpotResult:
        """Evaluate how wild and outlandish the ideas are"""
        text_lower = text.lower()
        
        wild_count = sum(text_lower.count(concept) for concept in self.wild_concepts)
        words = len(text.split())
        
        # Check for extreme language
        extreme_words = ['absolutely', 'completely', 'totally', 'ultimate', 'infinite',
                        'impossible', 'unbelievable', 'mind-blowing', 'earth-shattering']
        extreme_count = sum(text_lower.count(word) for word in extreme_words)
        
        wildness_score = (wild_count * 2 + extreme_count) / max(words / 15, 1)
        
        return CrackpotResult(
            dimension=CrackpotDimension.WILDNESS,
            score=min(wildness_score, 1.0),
            explanation=f"Wild concepts: {wild_count}, Extreme language: {extreme_count}"
        )
    
    def _evaluate_conspiracy_level(self, text: str) -> CrackpotResult:
        """Evaluate conspiracy theory elements"""
        text_lower = text.lower()
        
        conspiracy_count = sum(text_lower.count(keyword) for keyword in self.conspiracy_keywords)
        words = len(text.split())
        
        conspiracy_score = conspiracy_count / max(words / 25, 1)
        
        return CrackpotResult(
            dimension=CrackpotDimension.CONSPIRACY_LEVEL,
            score=min(conspiracy_score, 1.0),
            explanation=f"Conspiracy elements found: {conspiracy_count}"
        )
    
    def _evaluate_pseudoscience(self, text: str) -> CrackpotResult:
        """Evaluate pseudoscientific elements"""
        text_lower = text.lower()
        
        pseudo_count = sum(text_lower.count(term) for term in self.pseudoscience_terms)
        words = len(text.split())
        
        pseudo_score = pseudo_count / max(words / 20, 1)
        
        return CrackpotResult(
            dimension=CrackpotDimension.PSEUDOSCIENCE,
            score=min(pseudo_score, 1.0),
            explanation=f"Pseudoscience terms: {pseudo_count}"
        )
    
    def _evaluate_randomness(self, text: str) -> CrackpotResult:
        """Evaluate how random and chaotic the text is"""
        words = text.split()
        if len(words) < 2:
            return CrackpotResult(
                dimension=CrackpotDimension.RANDOMNESS,
                score=0.0,
                explanation="Not enough words to evaluate randomness"
            )
        
        # Check for random word associations
        chaos_count = sum(word.lower() in self.chaos_words for word in words)
        
        # Check for non-sequiturs (simplified: look for topic jumps)
        topic_jumps = 0
        for i in range(len(words) - 1):
            # Very simplified non-sequitur detection
            if abs(len(words[i]) - len(words[i+1])) > 5:  # Different word lengths might indicate topic jumps
                topic_jumps += 1
        
        randomness_score = (chaos_count + topic_jumps * 0.5) / len(words)
        
        return CrackpotResult(
            dimension=CrackpotDimension.RANDOMNESS,
            score=min(randomness_score, 1.0),
            explanation=f"Chaos words: {chaos_count}, Topic jumps: {topic_jumps}"
        )

class CrackpotGenerator:
    """
    Generates crackpot theories and unconventional ideas
    """
    
    def __init__(self):
        self.evaluator = CrackpotEvaluator()
        
        # Theory templates
        self.theory_templates = [
            "{starter} {subject} is actually {twist} because {pseudoscience_reason}.",
            "The truth about {subject} is that {conspiracy_element} has been {action} to {purpose}.",
            "{subject} operates on {pseudoscience_principle} which explains why {wild_claim}.",
            "Ancient {civilization} knew that {subject} could {power} through {method}.",
            "What mainstream science calls {subject} is really {alternative_explanation}."
        ]
        
        self.subjects = [
            "gravity", "mathematics", "consciousness", "DNA", "the internet",
            "cats", "music", "dreams", "language", "numbers", "colors", "time"
        ]
        
        self.twists = [
            "a multidimensional communication system",
            "crystallized thought energy",
            "an ancient alien technology",
            "a holographic projection",
            "compressed information from parallel universes"
        ]
        
        self.actions = [
            "suppressed", "modified", "encoded", "hidden", "reversed", "amplified"
        ]
        
        self.purposes = [
            "control human consciousness",
            "harvest dimensional energy", 
            "maintain the simulation",
            "prevent awakening",
            "protect ancient secrets"
        ]
        
        self.civilizations = [
            "Atlanteans", "Lemurians", "Egyptians", "Mayans", "Tibetans"
        ]
        
        self.powers = [
            "bend reality", "travel through time", "communicate telepathically",
            "access infinite knowledge", "transcend physical limitations"
        ]
        
        self.methods = [
            "sacred geometry", "vibrational frequencies", "quantum entanglement",
            "crystalline matrices", "consciousness expansion"
        ]

    def generate_crackpot_theory(self, topic: str = None) -> str:
        """Generate a crackpot theory about a topic"""
        if topic is None:
            topic = random.choice(self.subjects)
        
        template = random.choice(self.theory_templates)
        
        # Fill in the template
        theory = template.format(
            starter=random.choice(self.evaluator.crackpot_starters),
            subject=topic,
            twist=random.choice(self.twists),
            pseudoscience_reason=f"{random.choice(self.evaluator.pseudoscience_terms)} {random.choice(['resonance', 'interference', 'amplification'])}",
            conspiracy_element=random.choice(self.evaluator.conspiracy_keywords),
            action=random.choice(self.actions),
            purpose=random.choice(self.purposes),
            pseudoscience_principle=random.choice(self.evaluator.pseudoscience_terms),
            wild_claim=random.choice(self.evaluator.wild_concepts),
            civilization=random.choice(self.civilizations),
            power=random.choice(self.powers),
            method=random.choice(self.methods),
            alternative_explanation=random.choice(self.twists)
        )
        
        return theory

    def enhance_text_crackpotness(self, text: str, intensity: float = 0.5) -> str:
        """
        Take normal text and make it more crackpot
        intensity: 0.0 to 1.0 (how much to enhance)
        """
        if intensity <= 0:
            return text
        
        words = text.split()
        enhanced_words = []
        
        for word in words:
            enhanced_words.append(word)
            
            # Randomly inject crackpot elements based on intensity
            if random.random() < intensity * 0.3:
                # Add a random pseudoscience term
                enhanced_words.append(f"({random.choice(self.evaluator.pseudoscience_terms)})")
            
            if random.random() < intensity * 0.2:
                # Add conspiracy implications
                enhanced_words.append(f"[{random.choice(['allegedly', 'supposedly', 'they claim'])}]")
            
            if random.random() < intensity * 0.1:
                # Add chaos words
                enhanced_words.append(random.choice(self.evaluator.chaos_words))
        
        enhanced_text = " ".join(enhanced_words)
        
        # Add crackpot framing based on intensity
        if intensity > 0.7:
            enhanced_text = f"{random.choice(self.evaluator.crackpot_starters)} {enhanced_text} But that's just what they want you to think!"
        elif intensity > 0.4:
            enhanced_text = f"Interestingly, {enhanced_text} This aligns with ancient wisdom."
        
        return enhanced_text

    def generate_random_associations(self, concept: str, num_associations: int = 5) -> List[str]:
        """Generate random, potentially crackpot associations with a concept"""
        associations = []
        
        for _ in range(num_associations):
            association_type = random.choice(['conspiracy', 'pseudoscience', 'wild', 'chaos'])
            
            if association_type == 'conspiracy':
                association = f"{concept} secretly controls {random.choice(self.subjects)} through {random.choice(self.evaluator.conspiracy_keywords)}"
            elif association_type == 'pseudoscience':
                association = f"{concept} resonates at the same {random.choice(self.evaluator.pseudoscience_terms)} as {random.choice(self.subjects)}"
            elif association_type == 'wild':
                association = f"{concept} enables {random.choice(self.evaluator.wild_concepts)} when combined with {random.choice(self.subjects)}"
            else:  # chaos
                association = f"{concept} is fundamentally {random.choice(self.evaluator.chaos_words)} in nature"
            
            associations.append(association)
        
        return associations

def main():
    """Demonstration of crackpot evaluation and generation"""
    evaluator = CrackpotEvaluator()
    generator = CrackpotGenerator()
    
    print("=== CRACKPOT EVALUATOR & GENERATOR ===")
    print("Making LLMs More Unconventional and Creative!\n")
    
    # Test evaluation on normal text
    normal_text = "The weather today is nice. I think it will rain tomorrow."
    print("1. EVALUATING NORMAL TEXT:")
    print(f"Text: {normal_text}")
    
    results = evaluator.evaluate_crackpot_level(normal_text)
    total_score = sum(result.score for result in results.values()) / len(results)
    print(f"Average Crackpot Score: {total_score:.2f}")
    
    for dimension, result in results.items():
        print(f"  {dimension}: {result.score:.2f} - {result.explanation}")
    
    # Test evaluation on crackpot text
    crackpot_text = "The government doesn't want you to know that quantum vibrations from ancient crystals can unlock interdimensional portals!"
    print(f"\n2. EVALUATING CRACKPOT TEXT:")
    print(f"Text: {crackpot_text}")
    
    results = evaluator.evaluate_crackpot_level(crackpot_text)
    total_score = sum(result.score for result in results.values()) / len(results)
    print(f"Average Crackpot Score: {total_score:.2f}")
    
    for dimension, result in results.items():
        print(f"  {dimension}: {result.score:.2f} - {result.explanation}")
    
    # Generate crackpot theories
    print(f"\n3. GENERATING CRACKPOT THEORIES:")
    for i in range(3):
        theory = generator.generate_crackpot_theory()
        print(f"Theory {i+1}: {theory}")
    
    # Enhance normal text
    print(f"\n4. ENHANCING NORMAL TEXT:")
    enhanced = generator.enhance_text_crackpotness(normal_text, intensity=0.7)
    print(f"Original: {normal_text}")
    print(f"Enhanced: {enhanced}")
    
    # Generate random associations
    print(f"\n5. RANDOM ASSOCIATIONS:")
    associations = generator.generate_random_associations("mathematics", 3)
    for i, assoc in enumerate(associations, 1):
        print(f"  {i}. {assoc}")

if __name__ == "__main__":
    main()