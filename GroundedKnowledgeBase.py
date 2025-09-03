"""
Grounded Knowledge Base for NLP - Non-Brain-in-a-Vat Approach

This module implements a practical, grounded knowledge base system that focuses on
real-world linguistic applications and empirical language data rather than purely
theoretical constructs. It organizes and provides access to practical NLP knowledge.
"""

import os
import re
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from pathlib import Path
import logging

# Try to import optional dependencies gracefully
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from HumanExpressionEvaluator import HumanExpressionEvaluator, ExpressionContext
    EXPRESSION_EVALUATOR_AVAILABLE = True
except ImportError:
    EXPRESSION_EVALUATOR_AVAILABLE = False

try:
    from SubtextAnalyzer import SubtextAnalyzer
    SUBTEXT_ANALYZER_AVAILABLE = True
except ImportError:
    SUBTEXT_ANALYZER_AVAILABLE = False


@dataclass
class KnowledgeEntry:
    """Represents a single entry in the grounded knowledge base."""
    id: str
    title: str
    content: str
    file_path: str
    language: str
    topics: List[str]
    examples: List[str]
    practical_applications: List[str]
    related_entries: List[str]
    confidence_score: float
    last_updated: str
    metadata: Dict[str, Any]


@dataclass
class QueryResult:
    """Result of a knowledge base query."""
    entries: List[KnowledgeEntry]
    query: str
    relevance_scores: List[float]
    total_found: int
    query_time: float
    suggestions: List[str]


class GroundedKnowledgeBase:
    """
    A practical, grounded knowledge base system that organizes real-world
    linguistic knowledge and NLP applications.
    
    This system emphasizes:
    - Practical, empirical examples over theoretical abstractions
    - Real-world applications and use cases
    - Multilingual support with actual language data
    - Integration with existing NLP tools and frameworks
    """
    
    def __init__(self, repository_path: str = "."):
        self.repository_path = Path(repository_path)
        self.knowledge_entries: Dict[str, KnowledgeEntry] = {}
        self.topic_index: Dict[str, List[str]] = defaultdict(list)
        self.language_index: Dict[str, List[str]] = defaultdict(list)
        self.content_index: Dict[str, List[str]] = defaultdict(list)
        
        # Initialize optional components
        self.expression_evaluator = None
        self.subtext_analyzer = None
        
        if EXPRESSION_EVALUATOR_AVAILABLE:
            self.expression_evaluator = HumanExpressionEvaluator()
            
        if SUBTEXT_ANALYZER_AVAILABLE:
            self.subtext_analyzer = SubtextAnalyzer()
            
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize the knowledge base
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Initialize the knowledge base by scanning and indexing repository content."""
        self.logger.info("Initializing grounded knowledge base...")
        
        # Scan markdown files for linguistic content
        self._scan_markdown_files()
        
        # Scan Python files for practical implementations
        self._scan_python_files()
        
        # Build cross-references and relationships
        self._build_relationships()
        
        self.logger.info(f"Knowledge base initialized with {len(self.knowledge_entries)} entries")
    
    def _scan_markdown_files(self):
        """Scan markdown files to extract practical linguistic knowledge."""
        markdown_files = list(self.repository_path.glob("**/*.md"))
        
        for file_path in markdown_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract practical information from markdown
                entry = self._extract_knowledge_from_markdown(file_path, content)
                if entry:
                    self.knowledge_entries[entry.id] = entry
                    self._update_indices(entry)
                    
            except Exception as e:
                self.logger.warning(f"Could not process {file_path}: {e}")
    
    def _scan_python_files(self):
        """Scan Python files to extract practical implementations."""
        python_files = list(self.repository_path.glob("**/*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract practical information from Python code
                entry = self._extract_knowledge_from_python(file_path, content)
                if entry:
                    self.knowledge_entries[entry.id] = entry
                    self._update_indices(entry)
                    
            except Exception as e:
                self.logger.warning(f"Could not process {file_path}: {e}")
    
    def _extract_knowledge_from_markdown(self, file_path: Path, content: str) -> Optional[KnowledgeEntry]:
        """Extract practical knowledge from markdown files."""
        # Generate unique ID
        entry_id = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
        
        # Extract title
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else file_path.stem
        
        # Detect language based on content and filename
        language = self._detect_language(content, file_path.name)
        
        # Extract examples (code blocks, quoted text, etc.)
        examples = self._extract_examples(content)
        
        # Extract topics/keywords
        topics = self._extract_topics(content, title)
        
        # Extract practical applications
        applications = self._extract_applications(content)
        
        # Calculate confidence based on content quality
        confidence = self._calculate_confidence(content, examples, applications)
        
        # Skip if not substantial enough
        if confidence < 0.3 or len(content.strip()) < 100:
            return None
        
        return KnowledgeEntry(
            id=entry_id,
            title=title,
            content=content,
            file_path=str(file_path),
            language=language,
            topics=topics,
            examples=examples,
            practical_applications=applications,
            related_entries=[],  # Will be populated later
            confidence_score=confidence,
            last_updated=str(file_path.stat().st_mtime),
            metadata={
                'file_size': len(content),
                'example_count': len(examples),
                'application_count': len(applications)
            }
        )
    
    def _extract_knowledge_from_python(self, file_path: Path, content: str) -> Optional[KnowledgeEntry]:
        """Extract practical knowledge from Python files."""
        # Generate unique ID
        entry_id = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
        
        # Extract docstring as title/description
        docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
        title = file_path.stem
        
        if docstring_match:
            docstring = docstring_match.group(1).strip()
            title_line = docstring.split('\n')[0].strip()
            if title_line:
                title = title_line
        
        # Extract class and function names as practical examples
        examples = self._extract_code_examples(content)
        
        # Extract topics from comments and docstrings
        topics = self._extract_code_topics(content)
        
        # Extract applications from comments and usage
        applications = self._extract_code_applications(content)
        
        # Calculate confidence
        confidence = self._calculate_code_confidence(content, examples, applications)
        
        # Skip if not substantial enough
        if confidence < 0.4 or len(content.strip()) < 200:
            return None
        
        return KnowledgeEntry(
            id=entry_id,
            title=f"Implementation: {title}",
            content=content,
            file_path=str(file_path),
            language="python",
            topics=topics,
            examples=examples,
            practical_applications=applications,
            related_entries=[],
            confidence_score=confidence,
            last_updated=str(file_path.stat().st_mtime),
            metadata={
                'file_size': len(content),
                'function_count': len(re.findall(r'def\s+\w+', content)),
                'class_count': len(re.findall(r'class\s+\w+', content))
            }
        )
    
    def _detect_language(self, content: str, filename: str) -> str:
        """Detect the primary language of content."""
        # Simple language detection based on character patterns
        if re.search(r'[\u4e00-\u9fff]', content):  # Chinese characters
            if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', content):  # Japanese
                return "japanese"
            return "chinese"
        elif re.search(r'[\u3040-\u309f\u30a0-\u30ff]', content):  # Japanese
            return "japanese"
        elif re.search(r'[\uac00-\ud7af]', content):  # Korean
            return "korean"
        else:
            return "english"
    
    def _extract_examples(self, content: str) -> List[str]:
        """Extract practical examples from content."""
        examples = []
        
        # Code blocks
        code_blocks = re.findall(r'```[\s\S]*?```', content)
        examples.extend([block.strip('`').strip() for block in code_blocks])
        
        # Quoted examples
        quotes = re.findall(r'"([^"]{10,100})"', content)
        examples.extend(quotes)
        
        # Numbered examples
        numbered = re.findall(r'\d+\.\s+([^\n]{10,100})', content)
        examples.extend(numbered)
        
        return examples[:10]  # Limit to most relevant
    
    def _extract_topics(self, content: str, title: str) -> List[str]:
        """Extract key topics and keywords."""
        topics = []
        
        # Common NLP terms
        nlp_terms = [
            'nlp', 'natural language processing', 'tokenization', 'parsing',
            'sentiment', 'classification', 'machine translation', 'evaluation',
            'expression', 'linguistic', 'semantic', 'syntax', 'pragmatic',
            'cognitive', 'social', 'cultural', 'multilingual', 'corpus'
        ]
        
        content_lower = content.lower()
        for term in nlp_terms:
            if term in content_lower or term in title.lower():
                topics.append(term)
        
        # Extract from headings
        headings = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
        for heading in headings[:5]:  # Limit headings
            words = re.findall(r'\w+', heading.lower())
            topics.extend([w for w in words if len(w) > 3])
        
        return list(set(topics))[:15]  # Limit and deduplicate
    
    def _extract_applications(self, content: str) -> List[str]:
        """Extract practical applications mentioned in content."""
        applications = []
        
        # Common application patterns
        app_patterns = [
            r'use(?:d|s)?\s+(?:in|for)\s+([^.]{10,50})',
            r'application(?:s)?\s+(?:in|for|of)\s+([^.]{10,50})',
            r'example(?:s)?\s+(?:in|of)\s+([^.]{10,50})',
            r'practical\s+([^.]{10,50})',
            r'real-world\s+([^.]{10,50})'
        ]
        
        for pattern in app_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            applications.extend([match.strip() for match in matches])
        
        return list(set(applications))[:10]
    
    def _extract_code_examples(self, content: str) -> List[str]:
        """Extract examples from Python code."""
        examples = []
        
        # Function definitions
        functions = re.findall(r'def\s+(\w+)', content)
        examples.extend([f"Function: {func}" for func in functions[:5]])
        
        # Class definitions  
        classes = re.findall(r'class\s+(\w+)', content)
        examples.extend([f"Class: {cls}" for cls in classes[:3]])
        
        # Example usage in comments
        example_comments = re.findall(r'#\s*[Ee]xample:?\s*(.+)', content)
        examples.extend(example_comments[:5])
        
        return examples
    
    def _extract_code_topics(self, content: str) -> List[str]:
        """Extract topics from code structure and comments."""
        topics = []
        
        # From imports
        imports = re.findall(r'from\s+(\w+)|import\s+(\w+)', content)
        for imp in imports:
            module = imp[0] or imp[1]
            if module in ['nltk', 'spacy', 'numpy', 'sklearn']:
                topics.append(module)
        
        # From class/function names
        names = re.findall(r'(?:class|def)\s+(\w+)', content)
        for name in names:
            if any(term in name.lower() for term in ['nlp', 'text', 'language', 'eval']):
                topics.append(name.lower())
        
        return topics[:10]
    
    def _extract_code_applications(self, content: str) -> List[str]:
        """Extract applications from code comments and docstrings."""
        applications = []
        
        # From docstrings
        docstrings = re.findall(r'"""(.*?)"""', content, re.DOTALL)
        for doc in docstrings:
            if 'application' in doc.lower() or 'use' in doc.lower():
                applications.append(doc.strip()[:100])
        
        # From comments
        comments = re.findall(r'#\s*(.+)', content)
        for comment in comments:
            if any(word in comment.lower() for word in ['use', 'application', 'example']):
                applications.append(comment.strip())
        
        return applications[:5]
    
    def _calculate_confidence(self, content: str, examples: List[str], applications: List[str]) -> float:
        """Calculate confidence score based on content quality."""
        score = 0.0
        
        # Base score from content length
        if len(content) > 500:
            score += 0.3
        elif len(content) > 200:
            score += 0.2
        
        # Examples boost confidence
        score += min(len(examples) * 0.1, 0.3)
        
        # Applications boost confidence
        score += min(len(applications) * 0.1, 0.2)
        
        # Structure indicators
        if re.search(r'^#+\s', content, re.MULTILINE):  # Has headings
            score += 0.1
        
        if re.search(r'```', content):  # Has code blocks
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_code_confidence(self, content: str, examples: List[str], applications: List[str]) -> float:
        """Calculate confidence score for code content."""
        score = 0.0
        
        # Has docstrings
        if '"""' in content:
            score += 0.3
        
        # Has functions/classes
        if re.search(r'def\s+\w+', content):
            score += 0.2
        if re.search(r'class\s+\w+', content):
            score += 0.2
        
        # Has imports
        if re.search(r'import\s+\w+', content):
            score += 0.1
        
        # Examples and applications
        score += min(len(examples) * 0.05, 0.2)
        score += min(len(applications) * 0.05, 0.1)
        
        return min(score, 1.0)
    
    def _update_indices(self, entry: KnowledgeEntry):
        """Update search indices with new entry."""
        # Topic index
        for topic in entry.topics:
            self.topic_index[topic.lower()].append(entry.id)
        
        # Language index
        self.language_index[entry.language].append(entry.id)
        
        # Content index (simple word-based)
        words = re.findall(r'\w+', entry.content.lower())
        for word in set(words):
            if len(word) > 2:  # Skip very short words
                self.content_index[word].append(entry.id)
    
    def _build_relationships(self):
        """Build relationships between knowledge entries."""
        for entry_id, entry in self.knowledge_entries.items():
            related = self._find_related_entries(entry)
            entry.related_entries = related[:5]  # Limit to top 5
    
    def _find_related_entries(self, entry: KnowledgeEntry) -> List[str]:
        """Find entries related to the given entry."""
        related_scores = defaultdict(float)
        
        # Topic similarity
        for other_id, other_entry in self.knowledge_entries.items():
            if other_id == entry.id:
                continue
            
            common_topics = set(entry.topics) & set(other_entry.topics)
            if common_topics:
                related_scores[other_id] += len(common_topics) * 0.3
            
            # Language similarity
            if entry.language == other_entry.language:
                related_scores[other_id] += 0.1
            
            # Title similarity
            title_words = set(re.findall(r'\w+', entry.title.lower()))
            other_title_words = set(re.findall(r'\w+', other_entry.title.lower()))
            common_title_words = title_words & other_title_words
            if common_title_words:
                related_scores[other_id] += len(common_title_words) * 0.2
        
        # Sort by score and return IDs
        sorted_related = sorted(related_scores.items(), key=lambda x: x[1], reverse=True)
        return [rel_id for rel_id, score in sorted_related if score > 0.1]
    
    def query(self, query_string: str, language: Optional[str] = None, 
              topic: Optional[str] = None, limit: int = 10) -> QueryResult:
        """
        Query the knowledge base for relevant entries.
        
        Args:
            query_string: The search query
            language: Optional language filter
            topic: Optional topic filter  
            limit: Maximum number of results
        
        Returns:
            QueryResult with matched entries and metadata
        """
        import time
        start_time = time.time()
        
        query_words = set(re.findall(r'\w+', query_string.lower()))
        matches = []
        
        for entry_id, entry in self.knowledge_entries.items():
            score = self._calculate_relevance_score(entry, query_words, language, topic)
            if score > 0.1:  # Minimum relevance threshold
                matches.append((entry, score))
        
        # Sort by relevance score
        matches.sort(key=lambda x: x[1], reverse=True)
        
        # Prepare results
        entries = [match[0] for match in matches[:limit]]
        scores = [match[1] for match in matches[:limit]]
        
        query_time = time.time() - start_time
        
        # Generate suggestions
        suggestions = self._generate_suggestions(query_words)
        
        return QueryResult(
            entries=entries,
            query=query_string,
            relevance_scores=scores,
            total_found=len(matches),
            query_time=query_time,
            suggestions=suggestions
        )
    
    def _calculate_relevance_score(self, entry: KnowledgeEntry, query_words: Set[str],
                                 language: Optional[str], topic: Optional[str]) -> float:
        """Calculate relevance score for an entry given a query."""
        score = 0.0
        
        # Apply filters first
        if language and entry.language != language:
            return 0.0
        
        if topic and topic.lower() not in [t.lower() for t in entry.topics]:
            return 0.0
        
        # Title match
        title_words = set(re.findall(r'\w+', entry.title.lower()))
        title_matches = query_words & title_words
        score += len(title_matches) * 0.5
        
        # Topic match
        topic_words = set([t.lower() for t in entry.topics])
        topic_matches = query_words & topic_words
        score += len(topic_matches) * 0.3
        
        # Content match
        content_words = set(re.findall(r'\w+', entry.content.lower()))
        content_matches = query_words & content_words
        score += len(content_matches) * 0.1
        
        # Confidence boost
        score *= entry.confidence_score
        
        return score
    
    def _generate_suggestions(self, query_words: Set[str]) -> List[str]:
        """Generate query suggestions based on available topics and content."""
        suggestions = []
        
        # Common topic suggestions
        common_topics = Counter()
        for topics in [entry.topics for entry in self.knowledge_entries.values()]:
            common_topics.update(topics)
        
        # Suggest topics that partially match query
        for topic, count in common_topics.most_common(10):
            topic_words = set(re.findall(r'\w+', topic.lower()))
            if topic_words & query_words:
                suggestions.append(f"topic:{topic}")
        
        # Language suggestions
        languages = set(entry.language for entry in self.knowledge_entries.values())
        suggestions.extend([f"language:{lang}" for lang in languages])
        
        return suggestions[:5]
    
    def get_entry_by_id(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Get a specific entry by its ID."""
        return self.knowledge_entries.get(entry_id)
    
    def get_all_topics(self) -> List[Tuple[str, int]]:
        """Get all topics with their frequency counts."""
        topic_counts = Counter()
        for entry in self.knowledge_entries.values():
            topic_counts.update(entry.topics)
        return topic_counts.most_common()
    
    def get_all_languages(self) -> List[Tuple[str, int]]:
        """Get all languages with their entry counts."""
        language_counts = Counter(entry.language for entry in self.knowledge_entries.values())
        return language_counts.most_common()
    
    def evaluate_expression_with_knowledge(self, expression: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Evaluate an expression using the knowledge base and expression evaluator.
        This provides a grounded, practical evaluation based on real linguistic data.
        """
        if not self.expression_evaluator:
            return {"error": "Expression evaluator not available"}
        
        # Query knowledge base for relevant context
        query_result = self.query(expression, limit=5)
        
        # Create enriched context
        if EXPRESSION_EVALUATOR_AVAILABLE:
            eval_context = ExpressionContext(
                cultural_background=context.get('culture', 'universal') if context else 'universal',
                formality_level=context.get('formality', 'neutral') if context else 'neutral',
                situation=context.get('situation', 'general') if context else 'general'
            )
            
            # Evaluate with knowledge-enhanced context
            evaluation = self.expression_evaluator.comprehensive_evaluation(expression, eval_context)
            
            # Add knowledge context
            evaluation['knowledge_context'] = {
                'relevant_entries': len(query_result.entries),
                'knowledge_confidence': sum(query_result.relevance_scores) / len(query_result.relevance_scores) if query_result.relevance_scores else 0,
                'related_topics': list(set(topic for entry in query_result.entries[:3] for topic in entry.topics)),
                'suggested_applications': list(set(app for entry in query_result.entries[:3] for app in entry.practical_applications))
            }
            
            return evaluation
        else:
            return {"error": "Expression evaluator not available"}
    
    def export_knowledge_summary(self) -> Dict[str, Any]:
        """Export a summary of the knowledge base for analysis."""
        return {
            'total_entries': len(self.knowledge_entries),
            'languages': dict(self.get_all_languages()),
            'top_topics': dict(self.get_all_topics()[:20]),
            'average_confidence': sum(entry.confidence_score for entry in self.knowledge_entries.values()) / len(self.knowledge_entries),
            'total_examples': sum(len(entry.examples) for entry in self.knowledge_entries.values()),
            'total_applications': sum(len(entry.practical_applications) for entry in self.knowledge_entries.values()),
            'content_types': {
                'markdown': len([e for e in self.knowledge_entries.values() if e.file_path.endswith('.md')]),
                'python': len([e for e in self.knowledge_entries.values() if e.file_path.endswith('.py')])
            }
        }


def main():
    """Demonstrate the grounded knowledge base functionality."""
    print("🧠 Initializing Grounded Knowledge Base (Non-Brain-in-a-Vat Approach)")
    print("=" * 70)
    
    # Initialize knowledge base
    kb = GroundedKnowledgeBase()
    
    # Show summary
    summary = kb.export_knowledge_summary()
    print(f"\n📊 Knowledge Base Summary:")
    print(f"   Total Entries: {summary['total_entries']}")
    print(f"   Languages: {list(summary['languages'].keys())}")
    print(f"   Top Topics: {list(dict(summary['top_topics']).keys())[:5]}")
    print(f"   Average Confidence: {summary['average_confidence']:.2f}")
    
    # Example queries
    example_queries = [
        "human expression evaluation",
        "natural language processing",
        "中文 Chinese language",
        "practical applications",
        "sentiment analysis"
    ]
    
    print(f"\n🔍 Example Queries:")
    print("-" * 30)
    
    for query in example_queries:
        result = kb.query(query, limit=3)
        print(f"\nQuery: '{query}'")
        print(f"Found: {result.total_found} entries ({result.query_time:.3f}s)")
        
        for i, entry in enumerate(result.entries):
            print(f"  {i+1}. {entry.title} (score: {result.relevance_scores[i]:.2f})")
            print(f"     Language: {entry.language}, Topics: {entry.topics[:3]}")
    
    # Demo expression evaluation with knowledge
    if EXPRESSION_EVALUATOR_AVAILABLE:
        print(f"\n💭 Expression Evaluation with Knowledge Context:")
        print("-" * 50)
        
        test_expressions = [
            "這個想法很有創意",
            "Hello, how are you doing today?",
            "The implementation is quite elegant"
        ]
        
        for expr in test_expressions:
            result = kb.evaluate_expression_with_knowledge(expr)
            if 'error' not in result:
                print(f"\nExpression: '{expr}'")
                print(f"Overall Score: {result.get('integrated', {}).get('overall_score', 'N/A')}")
                print(f"Knowledge Context: {result.get('knowledge_context', {}).get('relevant_entries', 0)} relevant entries")


if __name__ == "__main__":
    main()