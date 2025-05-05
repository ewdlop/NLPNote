# profanity

Looking at profanity detection in NLP, there are several common approaches:

**Traditional Methods:**
- Wordlist matching (check text against known profanity dictionaries)
- Regular expressions for variations/leetspeak
- Context-aware rules for common euphemisms

**Modern ML Approaches:**
- Toxicity models (like Perspective API)
- BERT/transformer models fine-tuned on labeled data
- Multi-class classification (toxicity, obscenity, threats, etc.)

**Key Challenges:**
- Context matters ("damn the river" vs "damn you")
- Multilingual support
- Handling misspellings and creative substitutions
- Cultural/regional variations
- False positives/negatives

Popular libraries include:
- `better-profanity` (Python)
- `profanity-filter` (Python)
- `bad-words` (JavaScript)
- Perspective API (comprehensive toxicity detection)

Would you like me to walk through a specific implementation approach?
