#!/usr/bin/env python3
"""
GitHub README Marketing Content Removal Bot

This bot crawls through GitHub repositories and identifies/removes marketing content 
from README files using the marketing stopwords filtering system.

While marketing language might seem counterintuitive in open-source, it actually appears 
frequently in README files through:
- Competitive positioning ("best", "fastest", "leading")
- Promotional claims ("revolutionary", "cutting-edge", "optimal") 
- Business jargon that obscures actual functionality
- Vague statements that don't help developers evaluate tools

The bot helps maintain objective, technical documentation that focuses on 
functionality rather than promotional language.

Usage:
    python github_readme_marketing_bot.py --repo owner/repo
    python github_readme_marketing_bot.py --scan-user username --limit 10
    python github_readme_marketing_bot.py --analyze-only --repo owner/repo

Requirements:
    pip install requests PyGithub marketing_stopwords
"""

import os
import sys
import argparse
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse

# Third-party imports
try:
    import requests
    from github import Github, GithubException
    from github.Repository import Repository
    from github.ContentFile import ContentFile
except ImportError:
    print("Missing required packages. Install with:")
    print("pip install requests PyGithub")
    sys.exit(1)

# Local imports
try:
    from marketing_stopwords import MarketingStopwords, filter_marketing_terms
except ImportError:
    print("Marketing stopwords module not found. Ensure marketing_stopwords.py is in the same directory.")
    sys.exit(1)


@dataclass
class AnalysisResult:
    """Results from analyzing a README file."""
    repo_name: str
    readme_path: str
    original_content: str
    filtered_content: str
    marketing_terms_found: List[Tuple[str, int, int]]
    marketing_density: float
    word_count: int
    marketing_word_count: int
    needs_filtering: bool


class GitHubReadmeBot:
    """
    Bot for analyzing and filtering marketing content from GitHub README files.
    """
    
    def __init__(self, github_token: Optional[str] = None, rate_limit_delay: float = 1.0):
        """
        Initialize the GitHub README bot.
        
        Args:
            github_token: GitHub API token. If None, uses environment variable GITHUB_TOKEN
            rate_limit_delay: Delay between API calls to respect rate limits
        """
        self.token = github_token or os.getenv('GITHUB_TOKEN')
        if not self.token:
            logging.warning("No GitHub token provided. API rate limits will be severely restricted.")
        
        self.github = Github(self.token) if self.token else Github()
        self.marketing_filter = MarketingStopwords()
        self.rate_limit_delay = rate_limit_delay
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def _respect_rate_limit(self):
        """Add delay to respect GitHub API rate limits."""
        time.sleep(self.rate_limit_delay)
    
    def _get_readme_content(self, repo: Repository) -> Optional[Tuple[str, str]]:
        """
        Get README content from repository.
        
        Args:
            repo: GitHub repository object
            
        Returns:
            Tuple of (content, file_path) or None if no README found
        """
        readme_names = ['README.md', 'README.rst', 'README.txt', 'README', 'readme.md', 'Readme.md']
        
        for readme_name in readme_names:
            try:
                readme_file = repo.get_contents(readme_name)
                if isinstance(readme_file, ContentFile):
                    content = readme_file.decoded_content.decode('utf-8')
                    return content, readme_name
            except GithubException:
                continue
        
        return None
    
    def analyze_readme(self, repo_name: str) -> Optional[AnalysisResult]:
        """
        Analyze marketing content in a repository's README.
        
        Args:
            repo_name: Repository name in format 'owner/repo'
            
        Returns:
            AnalysisResult object or None if analysis failed
        """
        try:
            repo = self.github.get_repo(repo_name)
            self._respect_rate_limit()
            
            readme_data = self._get_readme_content(repo)
            if not readme_data:
                self.logger.warning(f"No README found in {repo_name}")
                return None
            
            content, file_path = readme_data
            
            # Analyze marketing content
            marketing_terms = self.marketing_filter.get_marketing_terms_in_text(content)
            filtered_content = self.marketing_filter.filter_text(content)
            
            # Calculate metrics
            words = content.split()
            word_count = len(words)
            marketing_word_count = len(marketing_terms)
            marketing_density = marketing_word_count / word_count if word_count > 0 else 0
            
            # Determine if filtering is needed (threshold: 2% marketing density or 3+ terms)
            needs_filtering = marketing_density > 0.02 or marketing_word_count >= 3
            
            return AnalysisResult(
                repo_name=repo_name,
                readme_path=file_path,
                original_content=content,
                filtered_content=filtered_content,
                marketing_terms_found=marketing_terms,
                marketing_density=marketing_density,
                word_count=word_count,
                marketing_word_count=marketing_word_count,
                needs_filtering=needs_filtering
            )
            
        except GithubException as e:
            self.logger.error(f"GitHub API error for {repo_name}: {str(e)[:200]}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error analyzing {repo_name}: {e}")
            return None
    
    def scan_user_repositories(self, username: str, limit: int = 20) -> List[AnalysisResult]:
        """
        Scan repositories for a user/organization.
        
        Args:
            username: GitHub username or organization
            limit: Maximum number of repositories to scan
            
        Returns:
            List of AnalysisResult objects
        """
        results = []
        
        try:
            user = self.github.get_user(username)
            repos = user.get_repos(type='public', sort='updated')
            
            self.logger.info(f"Scanning up to {limit} repositories for user: {username}")
            
            count = 0
            for repo in repos:
                if count >= limit:
                    break
                
                self.logger.info(f"Analyzing {repo.full_name}...")
                result = self.analyze_readme(repo.full_name)
                if result:
                    results.append(result)
                
                count += 1
                self._respect_rate_limit()
            
        except GithubException as e:
            self.logger.error(f"Error scanning user {username}: {e}")
        
        return results
    
    def generate_report(self, results: List[AnalysisResult]) -> str:
        """
        Generate a comprehensive report of the analysis.
        
        Args:
            results: List of analysis results
            
        Returns:
            Formatted report string
        """
        if not results:
            return "No repositories analyzed."
        
        # Filter results that need attention
        needs_filtering = [r for r in results if r.needs_filtering]
        
        report = ["GitHub README Marketing Content Analysis Report"]
        report.append("=" * 60)
        report.append(f"Total repositories analyzed: {len(results)}")
        report.append(f"Repositories with significant marketing content: {len(needs_filtering)}")
        report.append(f"Overall marketing detection rate: {len(needs_filtering)/len(results)*100:.1f}%")
        report.append("")
        
        if needs_filtering:
            report.append("🚨 REPOSITORIES NEEDING ATTENTION:")
            report.append("-" * 40)
            
            for result in sorted(needs_filtering, key=lambda x: x.marketing_density, reverse=True):
                report.append(f"📍 {result.repo_name}")
                report.append(f"   Marketing density: {result.marketing_density*100:.1f}%")
                report.append(f"   Marketing terms found: {result.marketing_word_count}")
                report.append(f"   Terms: {[term for term, _, _ in result.marketing_terms_found[:5]]}")
                if len(result.marketing_terms_found) > 5:
                    report.append(f"   ... and {len(result.marketing_terms_found)-5} more")
                report.append("")
        
        # Summary statistics
        if results:
            avg_density = sum(r.marketing_density for r in results) / len(results)
            total_terms = sum(r.marketing_word_count for r in results)
            
            report.append("📊 SUMMARY STATISTICS:")
            report.append("-" * 25)
            report.append(f"Average marketing density: {avg_density*100:.2f}%")
            report.append(f"Total marketing terms found: {total_terms}")
            report.append(f"Most common terms: {self._get_most_common_terms(results)}")
        
        return "\n".join(report)
    
    def _get_most_common_terms(self, results: List[AnalysisResult]) -> List[str]:
        """Get most commonly found marketing terms across all repositories."""
        term_counts = {}
        for result in results:
            for term, _, _ in result.marketing_terms_found:
                term_lower = term.lower()
                term_counts[term_lower] = term_counts.get(term_lower, 0) + 1
        
        # Return top 5 most common terms
        sorted_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)
        return [term for term, count in sorted_terms[:5]]
    
    def save_filtered_readme(self, result: AnalysisResult, output_dir: str = "filtered_readmes"):
        """
        Save filtered README content to a file.
        
        Args:
            result: Analysis result containing filtered content
            output_dir: Directory to save filtered READMEs
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create safe filename
        safe_repo_name = result.repo_name.replace('/', '_')
        filename = f"{safe_repo_name}_{result.readme_path}"
        
        file_path = output_path / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"# Filtered README for {result.repo_name}\n\n")
            f.write(f"Original marketing density: {result.marketing_density*100:.1f}%\n")
            f.write(f"Marketing terms removed: {result.marketing_word_count}\n\n")
            f.write("---\n\n")
            f.write(result.filtered_content)
        
        self.logger.info(f"Saved filtered README to: {file_path}")


def create_sample_analysis():
    """Create a sample analysis for demonstration purposes."""
    bot = GitHubReadmeBot()
    
    # Analyze a few well-known repositories
    sample_repos = [
        "microsoft/vscode",
        "facebook/react", 
        "tensorflow/tensorflow",
        "kubernetes/kubernetes"
    ]
    
    results = []
    for repo in sample_repos:
        print(f"Analyzing {repo}...")
        result = bot.analyze_readme(repo)
        if result:
            results.append(result)
    
    return results


def main():
    """Main CLI interface for the GitHub README marketing bot."""
    parser = argparse.ArgumentParser(
        description="GitHub README Marketing Content Removal Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python github_readme_marketing_bot.py --repo microsoft/vscode
  python github_readme_marketing_bot.py --scan-user tensorflow --limit 5
  python github_readme_marketing_bot.py --analyze-only --repo facebook/react
  python github_readme_marketing_bot.py --demo
        """
    )
    
    parser.add_argument('--repo', help='Analyze specific repository (owner/repo format)')
    parser.add_argument('--scan-user', help='Scan repositories for a user/organization')
    parser.add_argument('--limit', type=int, default=10, help='Limit number of repos to scan (default: 10)')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze, don\'t save filtered files')
    parser.add_argument('--output-dir', default='filtered_readmes', help='Output directory for filtered READMEs')
    parser.add_argument('--token', help='GitHub API token (or set GITHUB_TOKEN env var)')
    parser.add_argument('--demo', action='store_true', help='Run demo analysis on sample repositories')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize bot
    bot = GitHubReadmeBot(github_token=args.token)
    
    results = []
    
    if args.demo:
        print("🤖 Running demo analysis on sample repositories...")
        results = create_sample_analysis()
    
    elif args.repo:
        print(f"🔍 Analyzing repository: {args.repo}")
        result = bot.analyze_readme(args.repo)
        if result:
            results = [result]
    
    elif args.scan_user:
        print(f"🔍 Scanning user/organization: {args.scan_user}")
        results = bot.scan_user_repositories(args.scan_user, args.limit)
    
    else:
        parser.print_help()
        return
    
    # Generate and display report
    if results:
        report = bot.generate_report(results)
        print("\n" + report)
        
        # Save filtered READMEs if requested
        if not args.analyze_only:
            print(f"\n💾 Saving filtered READMEs to: {args.output_dir}")
            for result in results:
                if result.needs_filtering:
                    bot.save_filtered_readme(result, args.output_dir)
        
        # Show sample before/after for most problematic repo
        problematic = [r for r in results if r.needs_filtering]
        if problematic:
            worst = max(problematic, key=lambda x: x.marketing_density)
            print(f"\n📝 SAMPLE: Before/After for {worst.repo_name}")
            print("=" * 60)
            print("BEFORE (first 200 chars):")
            print(worst.original_content[:200] + "...")
            print("\nAFTER (first 200 chars):")
            print(worst.filtered_content[:200] + "...")
    
    else:
        print("No repositories were successfully analyzed.")


if __name__ == "__main__":
    main()