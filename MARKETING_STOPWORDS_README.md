# Marketing Stopwords Filter

A comprehensive filter for removing marketing and promotional language from text, based on government and expert writing guidelines.

## Overview

This module implements recommendations from:
- **GOV.UK Style Guide** - advocates for specific, verifiable language over promotional terms
- **UK Office for National Statistics (ONS)** - "Words not to use" guidelines  
- **Microsoft Style Guide** - recommendations to avoid business jargon
- **Nielsen Norman Group** - research showing 27% usability improvement when removing promotional language
- **Plain English guidelines** - promoting clarity over marketing speak

## Features

- **Comprehensive stopwords list**: 117+ marketing terms and their variations
- **Regex patterns**: Catches hyphenated variants and common promotional phrases
- **Whitelist protection**: Preserves legitimate technical terms (e.g., "First Aid", "Fast Fourier Transform")
- **Multiple formats**: JSON data, Python module, and plain text list
- **Case-insensitive matching**: Works regardless of capitalization
- **Punctuation preservation**: Maintains text formatting while filtering content

## Quick Start

### Basic Usage

```python
from marketing_stopwords import filter_marketing_terms

# Filter promotional language
text = "Our best-in-class solution delivers optimal performance"
filtered = filter_marketing_terms(text)
print(filtered)  # "Our solution delivers performance"
```

### Advanced Usage

```python
from marketing_stopwords import MarketingStopwords

# Create filter instance
filter = MarketingStopwords()

# Filter text with custom replacement
filtered = filter.filter_text(
    "Revolutionary AI platform with cutting-edge algorithms", 
    replacement="[PROMOTIONAL]"
)
print(filtered)  # "[PROMOTIONAL] AI platform with [PROMOTIONAL] algorithms"

# Find marketing terms in text
text = "Industry-leading performance with state-of-the-art technology"
terms = filter.get_marketing_terms_in_text(text)
for term, start, end in terms:
    print(f"Found '{term}' at position {start}-{end}")
```

### Integration with Existing NLP Tools

```python
# Example integration with NLTK or other NLP processing
from marketing_stopwords import MarketingStopwords
from nltk.tokenize import sent_tokenize

def clean_sentences(text):
    filter = MarketingStopwords()
    sentences = sent_tokenize(text)
    return [filter.filter_text(sent) for sent in sentences]
```

## Files Included

- `marketing_stopwords.json` - Complete stopwords data with metadata and whitelist
- `marketing_stopwords.py` - Main Python module with filtering functionality  
- `marketing_stopwords.txt` - Simple text list of terms for reference
- `test_marketing_stopwords.py` - Test suite demonstrating functionality

## Stopwords Categories

The filter targets these types of promotional language:

### Ranking/Superiority Terms
- best, top, leading, premier, ultimate
- best-in-class, world-class, industry-leading
- #1, number one, award-winning

### Speed/Performance Claims  
- fastest, quickest, rapid, instant
- lightning-fast, high-performance
- blazing, instantaneous

### Technology Hype
- cutting-edge, state-of-the-art, revolutionary
- next-generation, groundbreaking, innovative
- bleeding-edge, game-changing

### Ease/Simplicity Claims
- easy, simple, seamless, effortless
- user-friendly, hassle-free, intuitive
- frictionless

### Vague Business Jargon
- deliver, deploy, enable, empower
- leverage, drive, transform, streamline
- synergy, optimize, maximize

### Comprehensiveness Claims
- comprehensive, holistic, end-to-end
- all-in-one, full-service, turnkey
- one-stop shop, 360-degree

## Whitelist Protection

The filter preserves legitimate technical terms that might otherwise be caught:

- **First Aid** (medical term)
- **Fast Fourier Transform** (mathematical algorithm)
- **Mission Critical Linux** (specific software)
- **Optimal Transport** (mathematical field)
- **Quick Sort** (algorithm name)

Add custom whitelist terms:

```python
filter = MarketingStopwords()
filter.add_whitelist_term("Rapid Prototyping")  # Preserve technical term
filter.add_stopword("amazing")  # Add custom marketing term
```

## Regex Patterns

For advanced users, the module includes pre-built regex patterns:

```python
# Ranking terms: \b(?:no\.?\s*1|#\s*1|number[-\s]?one|best([-\s]?in[-\s]?class)?|...)\b
# Technology hype: \b(?:cutting[-\s]?edge|state[-\s]?of[-\s]?the[-\s]?art|...)\b  
# Speed claims: \b(?:fast(est)?|quick(est)?|rapid|instant(aneous)?|...)\b
```

## Testing

Run the test suite to verify functionality:

```bash
python3 test_marketing_stopwords.py
```

Expected output shows filtering of promotional terms while preserving technical language and maintaining text structure.

## Use Cases

### Content Editing
- Clean marketing copy for government or academic publications
- Improve technical documentation clarity
- Remove promotional language from user-generated content

### Text Analysis
- Preprocess text before sentiment analysis
- Normalize content for fair comparison
- Identify marketing-heavy content automatically

### Writing Tools
- Real-time editing suggestions
- Style guide enforcement
- Content quality scoring

## Research Foundation

This implementation is based on:

1. **Government Guidelines**: GOV.UK and ONS explicitly list many of these terms as "words to avoid" because they're vague and unverifiable.

2. **Usability Research**: Nielsen Norman Group found 27% improvement in task completion when promotional language was replaced with objective descriptions.

3. **Style Guide Best Practices**: Microsoft, Google, and other tech companies recommend avoiding business jargon in favor of specific, measurable claims.

4. **Plain English Movement**: Advocates for clear, direct communication over promotional rhetoric.

## Limitations

- **Context sensitivity**: Some terms may be appropriate in certain technical contexts
- **Language support**: Primarily designed for English text
- **Cultural variations**: Marketing language varies across cultures and regions
- **False positives**: May occasionally filter legitimate usage

## Recommendations

1. **Review filtered content**: Always review automated filtering results
2. **Domain-specific tuning**: Adjust whitelist for your specific field
3. **Combine with human judgment**: Use as a starting point, not final arbiter
4. **Update regularly**: Marketing language evolves over time

## Contributing

To add new terms or improve the filtering:

1. Update `marketing_stopwords.json` with new terms
2. Add legitimate technical terms to the whitelist
3. Test changes with `test_marketing_stopwords.py`
4. Consider cultural and domain-specific implications

## License

This module is designed for educational and practical use in improving text clarity. The stopwords list is compiled from publicly available style guides and research.