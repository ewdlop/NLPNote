#!/usr/bin/env python3
"""
English Language Patcher (英文語言修補器)

This module provides comprehensive tools for patching, correcting, and enhancing
English text, addressing common issues like typos, grammar errors, and stylistic
improvements.

Features:
- Spelling correction
- Grammar checking and fixing
- Text normalization
- Common typo detection and correction
- Style improvement suggestions
- Context-aware corrections
"""

import re
import string
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum


class PatchType(Enum):
    """Types of patches that can be applied to English text"""
    SPELLING = "spelling"
    GRAMMAR = "grammar"
    PUNCTUATION = "punctuation"
    CAPITALIZATION = "capitalization"
    SPACING = "spacing"
    STYLE = "style"
    TYPO = "typo"


@dataclass
class Patch:
    """Represents a single text patch"""
    original: str
    corrected: str
    position: int
    patch_type: PatchType
    confidence: float
    explanation: str


@dataclass
class PatchResult:
    """Result of patching operation"""
    original_text: str
    patched_text: str
    patches: List[Patch]
    success_rate: float


class EnglishPatcher:
    """Main English language patching class"""
    
    def __init__(self):
        """Initialize the English patcher with correction rules and dictionaries"""
        self.common_typos = self._load_common_typos()
        self.grammar_rules = self._load_grammar_rules()
        self.punctuation_rules = self._load_punctuation_rules()
        self.capitalization_rules = self._load_capitalization_rules()
        self.spacing_rules = self._load_spacing_rules()
    
    def patch_text(self, text: str, aggressive: bool = False) -> PatchResult:
        """
        Apply comprehensive patches to English text
        
        Args:
            text: Input text to patch
            aggressive: If True, applies more aggressive corrections
            
        Returns:
            PatchResult with original, patched text, and applied patches
        """
        all_patches = []
        current_text = text
        
        # Apply patches in order of priority, ensuring each step uses the current text
        current_text, spelling_patches = self._apply_spelling_patches(current_text)
        all_patches.extend(spelling_patches)
        
        current_text, grammar_patches = self._apply_grammar_patches(current_text, aggressive)
        all_patches.extend(grammar_patches)
        
        current_text, punct_patches = self._apply_punctuation_patches(current_text)
        all_patches.extend(punct_patches)
        
        current_text, cap_patches = self._apply_capitalization_patches(current_text)
        all_patches.extend(cap_patches)
        
        current_text, space_patches = self._apply_spacing_patches(current_text)
        all_patches.extend(space_patches)
        
        if aggressive:
            current_text, style_patches = self._apply_style_patches(current_text)
            all_patches.extend(style_patches)
        
        success_rate = len(all_patches) / max(1, len(text.split()))
        
        return PatchResult(
            original_text=text,
            patched_text=current_text,
            patches=all_patches,
            success_rate=min(1.0, success_rate)
        )
    
    def _apply_spelling_patches(self, text: str) -> Tuple[str, List[Patch]]:
        """Apply spelling corrections"""
        patches = []
        words = text.split()
        corrected_words = []
        
        for i, word in enumerate(words):
            clean_word = self._clean_word(word)
            correction = self._get_spelling_correction(clean_word)
            
            if correction and correction != clean_word:
                # Preserve original punctuation and capitalization
                corrected_word = self._preserve_formatting(word, correction)
                patches.append(Patch(
                    original=word,
                    corrected=corrected_word,
                    position=i,
                    patch_type=PatchType.SPELLING,
                    confidence=0.8,
                    explanation=f"Spelling correction: '{word}' → '{corrected_word}'"
                ))
                corrected_words.append(corrected_word)
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words), patches
    
    def _apply_grammar_patches(self, text: str, aggressive: bool) -> Tuple[str, List[Patch]]:
        """Apply grammar corrections"""
        patches = []
        corrected_text = text
        
        for rule in self.grammar_rules:
            pattern, replacement, explanation = rule
            matches = list(re.finditer(pattern, corrected_text, re.IGNORECASE))
            
            for match in reversed(matches):  # Process from end to preserve positions
                if aggressive or self._is_confident_grammar_fix(match, pattern):
                    original = match.group(0)
                    corrected = re.sub(pattern, replacement, original, flags=re.IGNORECASE)
                    
                    patches.append(Patch(
                        original=original,
                        corrected=corrected,
                        position=match.start(),
                        patch_type=PatchType.GRAMMAR,
                        confidence=0.7 if aggressive else 0.9,
                        explanation=explanation
                    ))
                    
                    corrected_text = (corrected_text[:match.start()] + 
                                    corrected + 
                                    corrected_text[match.end():])
        
        return corrected_text, patches
    
    def _apply_punctuation_patches(self, text: str) -> Tuple[str, List[Patch]]:
        """Apply punctuation corrections"""
        patches = []
        corrected_text = text
        
        # Fix multiple spaces before punctuation
        pattern = r'\s+([.!?,:;])'
        matches = list(re.finditer(pattern, corrected_text))
        for match in reversed(matches):
            original = match.group(0)
            corrected = match.group(1)
            patches.append(Patch(
                original=original,
                corrected=corrected,
                position=match.start(),
                patch_type=PatchType.PUNCTUATION,
                confidence=0.95,
                explanation="Removed extra space before punctuation"
            ))
            corrected_text = (corrected_text[:match.start()] + 
                            corrected + 
                            corrected_text[match.end():])
        
        # Add space after punctuation if missing
        pattern = r'([.!?,:;])([A-Za-z])'
        matches = list(re.finditer(pattern, corrected_text))
        for match in reversed(matches):
            original = match.group(0)
            corrected = match.group(1) + ' ' + match.group(2)
            patches.append(Patch(
                original=original,
                corrected=corrected,
                position=match.start(),
                patch_type=PatchType.PUNCTUATION,
                confidence=0.9,
                explanation="Added space after punctuation"
            ))
            corrected_text = (corrected_text[:match.start()] + 
                            corrected + 
                            corrected_text[match.end():])
        
        return corrected_text, patches
    
    def _apply_capitalization_patches(self, text: str) -> Tuple[str, List[Patch]]:
        """Apply capitalization corrections"""
        patches = []
        corrected_text = text
        
        # Capitalize first letter of text
        if corrected_text and corrected_text[0].islower():
            patches.append(Patch(
                original=corrected_text[0],
                corrected=corrected_text[0].upper(),
                position=0,
                patch_type=PatchType.CAPITALIZATION,
                confidence=0.95,
                explanation="Capitalized first letter of text"
            ))
            corrected_text = corrected_text[0].upper() + corrected_text[1:]
        
        # Capitalize after sentence endings
        pattern = r'([.!?]\s+)([a-z])'
        matches = list(re.finditer(pattern, corrected_text))
        for match in reversed(matches):
            original = match.group(0)
            corrected = match.group(1) + match.group(2).upper()
            patches.append(Patch(
                original=original,
                corrected=corrected,
                position=match.start(),
                patch_type=PatchType.CAPITALIZATION,
                confidence=0.95,
                explanation="Capitalized first letter after sentence ending"
            ))
            corrected_text = (corrected_text[:match.start()] + 
                            corrected + 
                            corrected_text[match.end():])
        
        return corrected_text, patches
    
    def _apply_spacing_patches(self, text: str) -> Tuple[str, List[Patch]]:
        """Apply spacing corrections"""
        patches = []
        corrected_text = text
        
        # Fix multiple spaces
        pattern = r'\s{2,}'
        matches = list(re.finditer(pattern, corrected_text))
        for match in reversed(matches):
            original = match.group(0)
            corrected = ' '
            patches.append(Patch(
                original=original,
                corrected=corrected,
                position=match.start(),
                patch_type=PatchType.SPACING,
                confidence=0.95,
                explanation="Reduced multiple spaces to single space"
            ))
            corrected_text = (corrected_text[:match.start()] + 
                            corrected + 
                            corrected_text[match.end():])
        
        return corrected_text, patches
    
    def _apply_style_patches(self, text: str) -> Tuple[str, List[Patch]]:
        """Apply style improvements (aggressive mode)"""
        patches = []
        corrected_text = text
        
        # Replace contractions with full forms in formal text
        contractions = {
            r"\bcan't\b": "cannot",
            r"\bwon't\b": "will not",
            r"\bdon't\b": "do not",
            r"\bdoesn't\b": "does not",
            r"\bdidn't\b": "did not",
            r"\bisn't\b": "is not",
            r"\baren't\b": "are not",
            r"\bwasn't\b": "was not",
            r"\bweren't\b": "were not",
            r"\bhadn't\b": "had not",
            r"\bhasn't\b": "has not",
            r"\bhaven't\b": "have not",
        }
        
        for pattern, replacement in contractions.items():
            matches = list(re.finditer(pattern, corrected_text, re.IGNORECASE))
            for match in reversed(matches):
                original = match.group(0)
                patches.append(Patch(
                    original=original,
                    corrected=replacement,
                    position=match.start(),
                    patch_type=PatchType.STYLE,
                    confidence=0.6,
                    explanation=f"Expanded contraction: '{original}' → '{replacement}'"
                ))
                corrected_text = (corrected_text[:match.start()] + 
                                replacement + 
                                corrected_text[match.end():])
        
        return corrected_text, patches
    
    def _load_common_typos(self) -> Dict[str, str]:
        """Load common typo corrections"""
        return {
            # Common letter swaps
            'teh': 'the',
            'adn': 'and',
            'recieve': 'receive',
            'seperate': 'separate',
            'definately': 'definitely',
            'occured': 'occurred',
            'accomodate': 'accommodate',
            'neccessary': 'necessary',
            'existance': 'existence',
            'consistant': 'consistent',
            'independant': 'independent',
            'arguement': 'argument',
            'maintainance': 'maintenance',
            'appearence': 'appearance',
            'beleive': 'believe',
            'acheive': 'achieve',
            'wierd': 'weird',
            'freind': 'friend',
            'calender': 'calendar',
            'grammer': 'grammar',
            # Number/letter substitutions
            '2': 'to',
            '4': 'for',
            'u': 'you',
            'ur': 'your',
            'thru': 'through',
            # Spacing issues
            'alot': 'a lot',
            'infront': 'in front',
            'ontop': 'on top',
            'aswell': 'as well',
            # Homophones and common confusions
            'wandering': 'wondering',  # Context dependent but often confused
            'there': 'their',  # Context dependent
            'they\'re': 'their',  # Context dependent  
            'its': 'it\'s',   # Context dependent
            'your': 'you\'re', # Context dependent
            'loose': 'lose',   # Context dependent
            'effect': 'affect', # Context dependent
        }
    
    def _load_grammar_rules(self) -> List[Tuple[str, str, str]]:
        """Load grammar correction rules"""
        return [
            # Subject-verb agreement  
            (r'\bI\s+is\b', 'I am', "Subject-verb agreement: I am"),
            (r'\byou\s+is\b', 'you are', "Subject-verb agreement: you are"),
            (r'\bwe\s+is\b', 'we are', "Subject-verb agreement: we are"),
            (r'\bthey\s+is\b', 'they are', "Subject-verb agreement: they are"),
            (r'\b(he|she|it)\s+are\b', r'\1 is', "Subject-verb agreement: singular subject with 'is'"),
            
            # Article usage
            (r'\ba\s+([aeiouAEIOU])', r'an \1', "Article correction: 'an' before vowel sound"),
            (r'\ban\s+([bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ])', r'a \1', "Article correction: 'a' before consonant sound"),
            
            # Double negatives
            (r"\bdon't\s+have\s+no\b", "don't have any", "Fixed double negative"),
            (r"\bcan't\s+get\s+no\b", "can't get any", "Fixed double negative"),
            
            # Common preposition errors
            (r'\bdifferent\s+than\b', 'different from', "Preposition correction: 'different from'"),
            (r'\bon\s+accident\b', 'by accident', "Preposition correction: 'by accident'"),
        ]
    
    def _load_punctuation_rules(self) -> List[Tuple[str, str, str]]:
        """Load punctuation correction rules"""
        return []  # Handled in _apply_punctuation_patches
    
    def _load_capitalization_rules(self) -> List[Tuple[str, str, str]]:
        """Load capitalization correction rules"""
        return []  # Handled in _apply_capitalization_patches
    
    def _load_spacing_rules(self) -> List[Tuple[str, str, str]]:
        """Load spacing correction rules"""
        return []  # Handled in _apply_spacing_patches
    
    def _clean_word(self, word: str) -> str:
        """Remove punctuation from word for spell checking"""
        return word.strip(string.punctuation).lower()
    
    def _get_spelling_correction(self, word: str) -> Optional[str]:
        """Get spelling correction for a word"""
        return self.common_typos.get(word.lower())
    
    def _preserve_formatting(self, original: str, correction: str) -> str:
        """Preserve capitalization and punctuation from original word"""
        if not original or not correction:
            return correction
        
        # Preserve capitalization pattern
        if original[0].isupper():
            correction = correction[0].upper() + correction[1:]
        
        if original.isupper():
            correction = correction.upper()
        
        # Preserve trailing punctuation
        trailing_punct = ''
        for char in reversed(original):
            if char in string.punctuation:
                trailing_punct = char + trailing_punct
            else:
                break
        
        return correction + trailing_punct
    
    def _is_confident_grammar_fix(self, match, pattern: str) -> bool:
        """Determine if we're confident about a grammar fix"""
        # Simple heuristic: be more confident about common patterns
        confident_patterns = [
            r'\bI\s+is\b',
            r'\byou\s+is\b', 
            r'\bwe\s+is\b',
            r'\bthey\s+is\b',
            r'\b(he|she|it)\s+are\b',
            r'\ba\s+([aeiouAEIOU])',
            r'\ban\s+([bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ])',
            r"\bdon't\s+have\s+no\b",
            r"\bcan't\s+get\s+no\b",
        ]
        return pattern in confident_patterns
    
    def get_patch_summary(self, result: PatchResult) -> str:
        """Generate a human-readable summary of patches applied"""
        if not result.patches:
            return "No patches were applied. The text appears to be correct."
        
        summary = f"Applied {len(result.patches)} patches:\n"
        
        # Group patches by type
        by_type = {}
        for patch in result.patches:
            patch_type = patch.patch_type.value
            if patch_type not in by_type:
                by_type[patch_type] = []
            by_type[patch_type].append(patch)
        
        for patch_type, patches in by_type.items():
            summary += f"\n{patch_type.capitalize()} ({len(patches)} fixes):\n"
            for patch in patches[:3]:  # Show first 3 examples
                summary += f"  • {patch.explanation}\n"
            if len(patches) > 3:
                summary += f"  • ... and {len(patches) - 3} more\n"
        
        return summary


def main():
    """Demo of the English patcher"""
    patcher = EnglishPatcher()
    
    test_texts = [
        "teh quick brown fox jumps over the lazy dog.can you beleive it?",
        "I is going to the store and she are coming with me.",
        "This is a example of an text that need some corrections.",
        "recieve the package tommorrow.  seperate the items carefully.",
        "don't have no money left in my account",
        "Its been a long day and I cant wait to get home.",
    ]
    
    print("English Language Patcher Demo")
    print("=" * 50)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}:")
        print(f"Original: {text}")
        
        result = patcher.patch_text(text, aggressive=True)
        print(f"Patched:  {result.patched_text}")
        print(f"Applied {len(result.patches)} patches")
        
        if result.patches:
            print("Patches:")
            for patch in result.patches:
                print(f"  • {patch.explanation}")


if __name__ == "__main__":
    main()