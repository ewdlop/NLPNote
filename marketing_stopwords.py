"""
Marketing Stopwords Module

This module provides utilities for filtering marketing and promotional language
from text, based on government and expert writing guidelines that recommend
avoiding vague promotional terms in favor of specific, measurable claims.

Sources:
- GOV.UK Style Guide
- UK Office for National Statistics (ONS) "Words to Avoid"  
- Microsoft Style Guide
- Nielsen Norman Group research (removing promotional language improves usability by 27%)
- Plain English guidelines

Usage:
    from marketing_stopwords import MarketingStopwords
    
    filter = MarketingStopwords()
    clean_text = filter.filter_text("Our best-in-class solution delivers optimal performance")
    # Result: "Our solution delivers performance"
"""

import json
import re
import os
from typing import List, Set, Dict, Optional, Tuple
from pathlib import Path

class MarketingStopwords:
    """
    Marketing and promotional language filter based on expert writing guidelines.
    
    Filters out vague promotional terms that lack specificity, as recommended by
    government style guides and usability research.
    """
    
    def __init__(self, data_file: Optional[str] = None):
        """
        Initialize the marketing stopwords filter.
        
        Args:
            data_file: Path to JSON file with stopwords data. If None, uses bundled data.
        """
        self.data_file = data_file or self._get_default_data_file()
        self.stopwords: Set[str] = set()
        self.whitelist: Set[str] = set()
        self.regex_patterns: List[re.Pattern] = []
        self._load_data()
    
    def _get_default_data_file(self) -> str:
        """Get path to the default marketing stopwords JSON file."""
        current_dir = Path(__file__).parent
        return str(current_dir / "marketing_stopwords.json")
    
    def _load_data(self):
        """Load stopwords and whitelist from JSON file."""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load stopwords (convert to lowercase for case-insensitive matching)
            self.stopwords = {word.lower() for word in data.get('stopwords', [])}
            
            # Load whitelist terms that should be preserved
            whitelist_data = data.get('whitelist', {})
            self.whitelist = {term.lower() for term in whitelist_data.get('terms', [])}
            
            # Generate regex patterns for different categories
            self._generate_regex_patterns()
            
        except FileNotFoundError:
            print(f"Warning: Marketing stopwords file not found: {self.data_file}")
            self._load_default_words()
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON in stopwords file: {self.data_file}")
            self._load_default_words()
    
    def _load_default_words(self):
        """Fallback method to load basic stopwords if file is unavailable."""
        basic_stopwords = [
            "best", "fastest", "optimal", "ultimate", "revolutionary",
            "cutting-edge", "state-of-the-art", "world-class", "leading",
            "top", "premier", "award-winning", "innovative", "superior"
        ]
        self.stopwords = {word.lower() for word in basic_stopwords}
        self.whitelist = {"first aid", "optimal transport", "fast fourier transform"}
    
    def _generate_regex_patterns(self):
        """Generate regex patterns for capturing word variations."""
        patterns = [
            # Ranking/superiority terms
            r'\b(?:no\.?\s*1|#\s*1|number[-\s]?one|best([-\s]?in[-\s]?class)?|top([-\s]?rated)?|leading|market[-\s]?leading|industry[-\s]?leading|world[-\s]?class|best[-\s]?of[-\s]?breed)\b',
            
            # Technology/quality hype  
            r'\b(?:cutting[-\s]?edge|state[-\s]?of[-\s]?the[-\s]?art|next[-\s]?generation|revolutionary|groundbreaking|innovative|optimal|optimized?|superior|mission[-\s]?critical|industry[-\s]?standard|game[-\s]?changing|bleeding[-\s]?edge)\b',
            
            # Speed/ease claims
            r'\b(?:fast(est)?|quick(est)?|rapid|instant(aneous)?|blazing|lightning[-\s]?fast|easy|easily|simple(st)?|seamless|frictionless|intuitive|user[-\s]?friendly|effortless|hassle[-\s]?free)\b',
            
            # Vague business jargon
            r'\b(?:deliver|deploy|enable|empower|facilitate|leverage|drive|transform|revolutionize|optimi[sz]e|maximize|accelerate|streamline|synergy|synergize|key)\b',
            
            # Comprehensiveness claims
            r'\b(?:end[-\s]?to[-\s]?end|turnkey|holistic|comprehensive|all[-\s]?in[-\s]?one|full[-\s]?service|one[-\s]?stop\s+shop|360[-\s]?degree)\b'
        ]
        
        self.regex_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def is_marketing_term(self, word: str) -> bool:
        """
        Check if a word is a marketing/promotional term.
        
        Args:
            word: Word or phrase to check
            
        Returns:
            True if word is a marketing term and not whitelisted
        """
        word_lower = word.lower().strip()
        
        # Check whitelist first - preserve legitimate technical terms
        if word_lower in self.whitelist:
            return False
            
        return word_lower in self.stopwords
    
    def filter_text(self, text: str, replacement: str = "") -> str:
        """
        Remove marketing terms from text.
        
        Args:
            text: Input text to filter
            replacement: String to replace marketing terms with (default: empty)
            
        Returns:
            Filtered text with marketing terms removed
        """
        if not text:
            return text
        
        # First check for exact phrase matches in whitelist
        for whitelisted in self.whitelist:
            if whitelisted in text.lower():
                # Temporarily replace whitelisted terms to protect them
                text = re.sub(re.escape(whitelisted), f"__WHITELIST_{hash(whitelisted)}__", text, flags=re.IGNORECASE)
        
        # Apply regex patterns to remove marketing terms
        for pattern in self.regex_patterns:
            text = pattern.sub(replacement, text)
        
        # Remove individual stopwords while preserving punctuation
        words = re.findall(r'\S+', text)  # Split on whitespace but keep punctuation
        filtered_words = []
        
        for word in words:
            # Clean word of punctuation for checking, but preserve original
            clean_word = re.sub(r'[^\w\s-]', '', word).lower()
            if clean_word and not self.is_marketing_term(clean_word):
                filtered_words.append(word)
            elif replacement and clean_word:
                # Keep punctuation from original word
                punct_match = re.search(r'[^\w\s-]+$', word)
                punct = punct_match.group() if punct_match else ''
                filtered_words.append(replacement + punct)
        
        filtered_text = ' '.join(filtered_words)
        
        # Restore whitelisted terms
        for whitelisted in self.whitelist:
            placeholder = f"__WHITELIST_{hash(whitelisted)}__"
            filtered_text = filtered_text.replace(placeholder, whitelisted.title())
        
        # Clean up extra whitespace
        filtered_text = re.sub(r'\s+', ' ', filtered_text).strip()
        
        return filtered_text
    
    def get_marketing_terms_in_text(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Find all marketing terms in text with their positions.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of tuples (term, start_pos, end_pos) for each marketing term found
        """
        found_terms = []
        
        # Check regex patterns
        for pattern in self.regex_patterns:
            for match in pattern.finditer(text):
                term = match.group()
                # Don't report if it's a whitelisted term
                if not any(whitelist in term.lower() for whitelist in self.whitelist):
                    found_terms.append((term, match.start(), match.end()))
        
        # Check individual words
        words = re.finditer(r'\b\w+(?:[-\s]\w+)*\b', text)
        for match in words:
            word = match.group().lower()
            if self.is_marketing_term(word):
                found_terms.append((match.group(), match.start(), match.end()))
        
        # Sort by position and remove duplicates
        found_terms = sorted(set(found_terms), key=lambda x: x[1])
        
        return found_terms
    
    def get_stopwords_list(self) -> List[str]:
        """Get list of all marketing stopwords."""
        return sorted(list(self.stopwords))
    
    def get_whitelist(self) -> List[str]:
        """Get list of whitelisted terms."""
        return sorted(list(self.whitelist))
    
    def add_stopword(self, word: str):
        """Add a custom marketing stopword."""
        self.stopwords.add(word.lower())
    
    def add_whitelist_term(self, term: str):
        """Add a term to the whitelist (will not be filtered)."""
        self.whitelist.add(term.lower())


# Convenience functions for quick usage
_default_filter = None

def get_marketing_filter() -> MarketingStopwords:
    """Get singleton instance of marketing filter."""
    global _default_filter
    if _default_filter is None:
        _default_filter = MarketingStopwords()
    return _default_filter

def filter_marketing_terms(text: str, replacement: str = "") -> str:
    """
    Quick function to filter marketing terms from text.
    
    Args:
        text: Text to filter
        replacement: Replacement string for marketing terms
        
    Returns:
        Filtered text
    """
    return get_marketing_filter().filter_text(text, replacement)

def find_marketing_terms(text: str) -> List[Tuple[str, int, int]]:
    """
    Quick function to find marketing terms in text.
    
    Args:
        text: Text to analyze
        
    Returns:
        List of (term, start_pos, end_pos) tuples
    """
    return get_marketing_filter().get_marketing_terms_in_text(text)


# Example usage and testing
if __name__ == "__main__":
    # Demo the marketing stopwords filter
    filter = MarketingStopwords()
    
    test_texts = [
        "Our best-in-class solution delivers optimal performance and cutting-edge technology.",
        "The fastest, most innovative platform for streamlined workflows.",  
        "Revolutionary AI that empowers teams with world-class results.",
        "First Aid training uses Fast Fourier Transform algorithms for optimal processing.",
        "We provide comprehensive, end-to-end solutions that leverage synergy."
    ]
    
    print("Marketing Stopwords Filter Demo")
    print("=" * 50)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}:")
        print(f"Original: {text}")
        print(f"Filtered: {filter.filter_text(text)}")
        
        found_terms = filter.get_marketing_terms_in_text(text)
        if found_terms:
            print(f"Marketing terms found: {[term for term, _, _ in found_terms]}")
        else:
            print("No marketing terms found")
    
    print(f"\nTotal stopwords loaded: {len(filter.get_stopwords_list())}")
    print(f"Whitelisted terms: {len(filter.get_whitelist())}")