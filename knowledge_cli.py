#!/usr/bin/env python3
"""
Interactive CLI for the Grounded Knowledge Base

This script provides a command-line interface for interacting with the
non-brain-in-a-vat knowledge base, allowing users to query, explore,
and evaluate expressions using grounded, practical linguistic knowledge.
"""

import sys
import argparse
from typing import Optional
from GroundedKnowledgeBase import GroundedKnowledgeBase, QueryResult


class KnowledgeBaseCLI:
    """Interactive command-line interface for the grounded knowledge base."""
    
    def __init__(self):
        self.kb = GroundedKnowledgeBase()
        self.running = True
        
    def display_banner(self):
        """Display the application banner."""
        banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                    Grounded Knowledge Base CLI                       ║
║              Non-Brain-in-a-Vat Linguistic Knowledge                 ║
║                                                                      ║
║  🧠 Practical, empirical language knowledge and NLP tools           ║
║  🌍 Multilingual support with real-world examples                   ║
║  🔍 Query linguistic patterns and applications                      ║
║  💭 Evaluate expressions with contextual knowledge                  ║
╚══════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
        
        # Show knowledge base statistics
        summary = self.kb.export_knowledge_summary()
        print(f"📊 Loaded {summary['total_entries']} knowledge entries")
        print(f"🌐 Languages: {', '.join(summary['languages'].keys())}")
        print(f"📚 Topics: {len(summary['top_topics'])} unique topics")
        print()
    
    def display_help(self):
        """Display available commands."""
        help_text = """
Available Commands:
  🔍 search <query>              - Search knowledge base
  📊 stats                       - Show knowledge base statistics  
  💭 eval <expression>           - Evaluate human expression
  🌐 languages                   - List available languages
  📚 topics                      - List available topics
  📖 show <entry_id>             - Show specific entry details
  💡 examples                    - Show example queries
  ❓ help                        - Show this help message
  🚪 quit/exit                   - Exit the application

Search Options:
  search <query> --lang <language>    - Filter by language
  search <query> --topic <topic>      - Filter by topic  
  search <query> --limit <number>     - Limit results
  
Examples:
  search "natural language processing"
  search "中文" --lang chinese
  eval "这个想法很有创意"
  topics
        """
        print(help_text)
    
    def search_knowledge(self, query: str, language: Optional[str] = None, 
                        topic: Optional[str] = None, limit: int = 10):
        """Search the knowledge base and display results."""
        print(f"🔍 Searching for: '{query}'")
        if language:
            print(f"   Language filter: {language}")
        if topic:
            print(f"   Topic filter: {topic}")
        print()
        
        result = self.kb.query(query, language=language, topic=topic, limit=limit)
        
        if not result.entries:
            print("❌ No results found.")
            if result.suggestions:
                print("💡 Suggestions:")
                for suggestion in result.suggestions:
                    print(f"   • {suggestion}")
            return
        
        print(f"✅ Found {result.total_found} entries (showing top {len(result.entries)}) in {result.query_time:.3f}s")
        print("=" * 70)
        
        for i, entry in enumerate(result.entries):
            score = result.relevance_scores[i]
            print(f"\n{i+1}. {entry.title}")
            print(f"   🎯 Relevance: {score:.2f} | 🌐 Language: {entry.language} | 💪 Confidence: {entry.confidence_score:.2f}")
            print(f"   📚 Topics: {', '.join(entry.topics[:5])}")
            print(f"   💡 Examples: {len(entry.examples)} | 🔧 Applications: {len(entry.practical_applications)}")
            print(f"   📁 File: {entry.file_path}")
            print(f"   🔗 ID: {entry.id}")
            
            # Show preview of content
            preview = entry.content[:200].replace('\n', ' ').strip()
            if len(entry.content) > 200:
                preview += "..."
            print(f"   📄 Preview: {preview}")
        
        if result.suggestions:
            print(f"\n💡 Related suggestions:")
            for suggestion in result.suggestions:
                print(f"   • {suggestion}")
    
    def evaluate_expression(self, expression: str):
        """Evaluate a human expression using the knowledge base."""
        print(f"💭 Evaluating expression: '{expression}'")
        print()
        
        result = self.kb.evaluate_expression_with_knowledge(expression)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return
        
        # Display evaluation results
        print("📊 Evaluation Results:")
        print("=" * 50)
        
        # Overall score
        overall = result.get('integrated', {})
        if overall:
            score = overall.get('overall_score', 'N/A')
            confidence = overall.get('confidence', 'N/A')
            if isinstance(score, (int, float)):
                print(f"🎯 Overall Score: {score:.3f}")
            else:
                print(f"🎯 Overall Score: {score}")
            if isinstance(confidence, (int, float)):
                print(f"🔬 Confidence: {confidence:.3f}")
            else:
                print(f"🔬 Confidence: {confidence}")
        
        # Dimension scores
        dimensions = ['formal_semantic', 'cognitive', 'social']
        for dim in dimensions:
            score = result.get(dim, {})
            if isinstance(score, dict):
                print(f"📏 {dim.title()}: {score.get('score', 'N/A'):.3f}")
            elif isinstance(score, (int, float)):
                print(f"📏 {dim.title()}: {score:.3f}")
        
        # Knowledge context
        knowledge_ctx = result.get('knowledge_context', {})
        if knowledge_ctx:
            print(f"\n🧠 Knowledge Context:")
            print(f"   📚 Relevant entries: {knowledge_ctx.get('relevant_entries', 0)}")
            print(f"   🎯 Knowledge confidence: {knowledge_ctx.get('knowledge_confidence', 0):.3f}")
            
            topics = knowledge_ctx.get('related_topics', [])
            if topics:
                print(f"   🏷️  Related topics: {', '.join(topics[:5])}")
            
            apps = knowledge_ctx.get('suggested_applications', [])
            if apps:
                print(f"   🔧 Applications: {', '.join(apps[:3])}")
    
    def show_statistics(self):
        """Display knowledge base statistics."""
        summary = self.kb.export_knowledge_summary()
        
        print("📊 Knowledge Base Statistics")
        print("=" * 50)
        print(f"📁 Total Entries: {summary['total_entries']}")
        print(f"🎯 Average Confidence: {summary['average_confidence']:.3f}")
        print(f"💡 Total Examples: {summary['total_examples']}")
        print(f"🔧 Total Applications: {summary['total_applications']}")
        
        print(f"\n🌐 Languages:")
        for lang, count in summary['languages'].items():
            print(f"   • {lang}: {count} entries")
        
        print(f"\n📚 Top Topics:")
        for topic, count in list(summary['top_topics'].items())[:10]:
            print(f"   • {topic}: {count} entries")
        
        print(f"\n📄 Content Types:")
        for content_type, count in summary['content_types'].items():
            print(f"   • {content_type}: {count} files")
    
    def show_languages(self):
        """Display available languages."""
        languages = self.kb.get_all_languages()
        
        print("🌐 Available Languages")
        print("=" * 30)
        for language, count in languages:
            print(f"• {language}: {count} entries")
    
    def show_topics(self):
        """Display available topics."""
        topics = self.kb.get_all_topics()
        
        print("📚 Available Topics")
        print("=" * 30)
        for topic, count in topics[:20]:  # Show top 20
            print(f"• {topic}: {count} entries")
        
        if len(topics) > 20:
            print(f"... and {len(topics) - 20} more topics")
    
    def show_entry_details(self, entry_id: str):
        """Show detailed information about a specific entry."""
        entry = self.kb.get_entry_by_id(entry_id)
        
        if not entry:
            print(f"❌ Entry with ID '{entry_id}' not found.")
            return
        
        print(f"📖 Entry Details: {entry.title}")
        print("=" * 70)
        print(f"🔗 ID: {entry.id}")
        print(f"📁 File: {entry.file_path}")
        print(f"🌐 Language: {entry.language}")
        print(f"💪 Confidence: {entry.confidence_score:.3f}")
        print(f"🗓️  Last Updated: {entry.last_updated}")
        
        print(f"\n📚 Topics ({len(entry.topics)}):")
        for topic in entry.topics:
            print(f"   • {topic}")
        
        if entry.examples:
            print(f"\n💡 Examples ({len(entry.examples)}):")
            for example in entry.examples[:5]:  # Show first 5
                print(f"   • {example[:100]}{'...' if len(example) > 100 else ''}")
        
        if entry.practical_applications:
            print(f"\n🔧 Applications ({len(entry.practical_applications)}):")
            for app in entry.practical_applications[:5]:  # Show first 5
                print(f"   • {app[:100]}{'...' if len(app) > 100 else ''}")
        
        if entry.related_entries:
            print(f"\n🔗 Related Entries ({len(entry.related_entries)}):")
            for related_id in entry.related_entries:
                related = self.kb.get_entry_by_id(related_id)
                if related:
                    print(f"   • {related.title} ({related_id})")
        
        print(f"\n📄 Content Preview:")
        preview = entry.content[:500].strip()
        if len(entry.content) > 500:
            preview += "\n... (truncated)"
        print(preview)
    
    def show_examples(self):
        """Show example queries and usage."""
        examples = """
💡 Example Queries and Usage

🔍 Search Examples:
  search "natural language processing"
  search "sentiment analysis" --limit 5
  search "中文" --lang chinese
  search "evaluation" --topic nlp

💭 Expression Evaluation Examples:
  eval "這個想法很有創意"
  eval "Hello, how are you doing today?"
  eval "The implementation is quite elegant"

📊 Information Commands:
  stats                    # Show detailed statistics
  languages               # List all available languages
  topics                  # Show most common topics
  show abc123de           # Show details for entry with ID abc123de

🔧 Advanced Usage:
  # Search for multilingual content
  search "語言" --lang chinese
  
  # Find practical applications
  search "practical applications"
  
  # Explore specific topics
  search "machine learning" --topic nlp
        """
        print(examples)
    
    def parse_search_command(self, args: list) -> tuple:
        """Parse search command arguments."""
        query = ""
        language = None
        topic = None
        limit = 10
        
        i = 0
        while i < len(args):
            if args[i] == "--lang" and i + 1 < len(args):
                language = args[i + 1]
                i += 2
            elif args[i] == "--topic" and i + 1 < len(args):
                topic = args[i + 1]
                i += 2
            elif args[i] == "--limit" and i + 1 < len(args):
                try:
                    limit = int(args[i + 1])
                except ValueError:
                    print("⚠️  Invalid limit value, using default (10)")
                i += 2
            else:
                query += args[i] + " "
                i += 1
        
        return query.strip(), language, topic, limit
    
    def process_command(self, command: str):
        """Process a user command."""
        parts = command.strip().split()
        if not parts:
            return
        
        cmd = parts[0].lower()
        args = parts[1:]
        
        try:
            if cmd in ['quit', 'exit', 'q']:
                self.running = False
                print("👋 Goodbye!")
            
            elif cmd in ['help', 'h', '?']:
                self.display_help()
            
            elif cmd == 'search':
                if not args:
                    print("❌ Please provide a search query. Use 'help' for usage information.")
                    return
                query, language, topic, limit = self.parse_search_command(args)
                self.search_knowledge(query, language, topic, limit)
            
            elif cmd == 'eval':
                if not args:
                    print("❌ Please provide an expression to evaluate.")
                    return
                expression = ' '.join(args)
                self.evaluate_expression(expression)
            
            elif cmd == 'stats':
                self.show_statistics()
            
            elif cmd == 'languages':
                self.show_languages()
            
            elif cmd == 'topics':
                self.show_topics()
            
            elif cmd == 'show':
                if not args:
                    print("❌ Please provide an entry ID.")
                    return
                self.show_entry_details(args[0])
            
            elif cmd == 'examples':
                self.show_examples()
            
            else:
                print(f"❌ Unknown command: '{cmd}'. Type 'help' for available commands.")
        
        except Exception as e:
            print(f"❌ Error processing command: {e}")
    
    def run_interactive(self):
        """Run the interactive CLI."""
        self.display_banner()
        print("Type 'help' for available commands or 'quit' to exit.\n")
        
        while self.running:
            try:
                command = input("🔍 > ").strip()
                if command:
                    print()  # Add spacing
                    self.process_command(command)
                    print()  # Add spacing after output
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except EOFError:
                print("\n👋 Goodbye!")
                break
    
    def run_single_command(self, command: str):
        """Run a single command and exit."""
        self.process_command(command)


def main():
    """Main entry point for the CLI application."""
    parser = argparse.ArgumentParser(
        description="Interactive CLI for the Grounded Knowledge Base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python knowledge_cli.py                           # Interactive mode
  python knowledge_cli.py stats                     # Show statistics
  python knowledge_cli.py                          # Then type: search "natural language" --limit 3
        """
    )
    
    parser.add_argument('command', nargs='?', help='Command to execute (optional, starts interactive mode if not provided)')
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = KnowledgeBaseCLI()
    
    if args.command:
        # Single command mode
        cli.run_single_command(args.command)
    else:
        # Interactive mode
        cli.run_interactive()


if __name__ == "__main__":
    main()