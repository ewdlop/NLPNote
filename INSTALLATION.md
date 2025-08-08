# Installation Guide

## Quick Start (No Dependencies)

The core functionality of this NLP repository works out of the box with Python 3.6+ and no additional dependencies required. You can run:

```bash
python3 AStarNLP.py
python3 HumanExpressionEvaluator.py
python3 SubtextAnalyzer.py
python3 expression_evaluation_examples.py
```

## Enhanced Features (With Dependencies)

For full functionality including advanced NLP features, install the optional dependencies:

### Install Core Dependencies
```bash
pip install -r requirements.txt
```

### Install Advanced Features (Optional)
```bash
# For advanced spaCy features
pip install spacy
python -m spacy download en_core_web_sm

# For numerical operations (if needed)
pip install numpy
```

## Dependencies Explained

- **nltk**: Used for tokenization, POS tagging, and WordNet access in SubtextAnalyzer
- **spacy** (optional): Advanced NLP features like named entity recognition
- **numpy** (optional): Previously used but now removed - basic math operations use built-in Python

## Graceful Degradation

All modules are designed to work with graceful degradation:

- If `nltk` is not available, SubtextAnalyzer uses simple fallback implementations
- If `spacy` is not available, advanced features are disabled with informative messages
- Core functionality always works without any external dependencies

## Testing Installation

```bash
# Test core functionality (no dependencies required)
python3 -c "import AStarNLP; import HumanExpressionEvaluator; print('✓ Core modules work')"

# Test with basic dependencies
python3 -c "import SubtextAnalyzer; print('✓ All modules work')"
```