# GitHub README Marketing Bot

A Python bot that crawls GitHub repositories and identifies/removes marketing content from README files using an evidence-based marketing stopwords filtering system.

## Why Remove Marketing Language from Open Source?

While marketing language might seem counterintuitive in open-source projects, it actually appears frequently in README files and can harm usability:

### Common Marketing Problems in Open Source READMEs:
- **Competitive positioning**: "best", "fastest", "leading solution"
- **Vague claims**: "revolutionary", "cutting-edge", "optimal" 
- **Business jargon**: Terms that obscure actual functionality
- **Unverifiable statements**: Claims that can't be objectively assessed

### Why This Matters:
- **Developer Evaluation**: Helps developers quickly understand what a tool actually does
- **Objective Documentation**: Focuses on functionality rather than promotional language  
- **Usability Research**: Nielsen Norman Group found 27% usability improvement when removing promotional language
- **Professional Standards**: Aligns with government style guides (GOV.UK, ONS) that recommend specific, verifiable language

## Features

🔍 **Repository Analysis**
- Scan individual repositories or entire user/org accounts
- Identify marketing terms with position tracking
- Calculate marketing density metrics
- Generate comprehensive reports

🤖 **Automated Filtering**
- Remove promotional language while preserving technical terms
- Whitelist protection for legitimate technical terminology
- Maintain proper formatting and punctuation

📊 **Detailed Reporting**
- Marketing density analysis
- Most common promotional terms
- Before/after comparisons
- Export filtered READMEs

## Quick Start

### Installation

```bash
# Clone the repository (if using local marketing_stopwords module)
git clone <repository-url>
cd NLPNote

# Install dependencies
pip install -r requirements_bot.txt

# Set GitHub token (optional but recommended for higher rate limits)
export GITHUB_TOKEN=your_github_token_here
```

### Basic Usage

```bash
# Analyze a specific repository
python github_readme_marketing_bot.py --repo microsoft/vscode

# Scan repositories for a user/organization
python github_readme_marketing_bot.py --scan-user tensorflow --limit 5

# Run demo analysis on sample repositories
python github_readme_marketing_bot.py --demo

# Analysis only (don't save filtered files)
python github_readme_marketing_bot.py --analyze-only --repo facebook/react
```

### Python API Usage

```python
from github_readme_marketing_bot import GitHubReadmeBot

# Initialize bot
bot = GitHubReadmeBot(github_token="your_token")

# Analyze a repository
result = bot.analyze_readme("tensorflow/tensorflow")

if result and result.needs_filtering:
    print(f"Marketing density: {result.marketing_density*100:.1f}%")
    print(f"Terms found: {[term for term, _, _ in result.marketing_terms_found]}")
    
    # Save filtered README
    bot.save_filtered_readme(result)
```

## Command Line Options

```
usage: github_readme_marketing_bot.py [-h] [--repo REPO] [--scan-user SCAN_USER] 
                                      [--limit LIMIT] [--analyze-only] 
                                      [--output-dir OUTPUT_DIR] [--token TOKEN] 
                                      [--demo] [--verbose]

options:
  --repo REPO           Analyze specific repository (owner/repo format)
  --scan-user SCAN_USER Scan repositories for a user/organization  
  --limit LIMIT         Limit number of repos to scan (default: 10)
  --analyze-only        Only analyze, don't save filtered files
  --output-dir DIR      Output directory for filtered READMEs
  --token TOKEN         GitHub API token (or set GITHUB_TOKEN env var)
  --demo                Run demo analysis on sample repositories
  --verbose, -v         Enable verbose logging
```

## Example Output

```
GitHub README Marketing Content Analysis Report
============================================================
Total repositories analyzed: 4
Repositories with significant marketing content: 2
Overall marketing detection rate: 50.0%

🚨 REPOSITORIES NEEDING ATTENTION:
----------------------------------------
📍 example/ml-framework
   Marketing density: 8.2%
   Marketing terms found: 12
   Terms: ['best-in-class', 'cutting-edge', 'revolutionary', 'optimal', 'leading']

📍 example/web-toolkit  
   Marketing density: 4.1%
   Marketing terms found: 6
   Terms: ['fastest', 'seamless', 'innovative', 'superior']

📊 SUMMARY STATISTICS:
-------------------------
Average marketing density: 3.12%
Total marketing terms found: 18
Most common terms: ['best', 'optimal', 'cutting-edge', 'revolutionary', 'leading']
```

## Architecture

The bot leverages the existing `marketing_stopwords` module which provides:

- **117 promotional terms** based on government style guides
- **Regex patterns** for catching variations and hyphenated terms
- **Whitelist protection** to preserve legitimate technical terms
- **Evidence-based filtering** from GOV.UK, ONS, Microsoft, and Nielsen Norman Group research

## GitHub API Rate Limits

- **Without authentication**: 60 requests/hour
- **With GitHub token**: 5,000 requests/hour  
- The bot includes automatic rate limiting with configurable delays

## Use Cases

### For Project Maintainers
- Clean up promotional language in your project's README
- Ensure documentation focuses on functionality
- Improve accessibility for non-native English speakers

### For Organizations
- Audit repositories for promotional language
- Maintain consistent, objective documentation standards
- Identify repositories that may need documentation review

### For Researchers
- Analyze marketing language patterns in open-source projects
- Study the evolution of promotional language in technical documentation
- Gather data on documentation quality trends

## Configuration

Create a `.env` file for easy configuration:

```bash
GITHUB_TOKEN=your_github_token_here
RATE_LIMIT_DELAY=1.0
DEFAULT_OUTPUT_DIR=filtered_readmes
```

## Integration with CI/CD

You can integrate the bot into your CI/CD pipeline to automatically check for promotional language:

```yaml
# GitHub Actions example
name: README Marketing Check
on: [push, pull_request]

jobs:
  check-marketing:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    - name: Install dependencies
      run: pip install -r requirements_bot.txt
    - name: Check README for marketing content
      run: python github_readme_marketing_bot.py --repo ${{ github.repository }} --analyze-only
```

## Contributing

The bot is designed to work with the existing marketing stopwords system. To contribute:

1. Enhance the `marketing_stopwords.py` filtering engine
2. Add new promotional term patterns
3. Improve whitelist protection for technical terms
4. Add support for additional file formats beyond README

## License

Same as parent project. See LICENSE file for details.