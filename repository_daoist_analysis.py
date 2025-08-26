#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Repository Daoist Analysis - Practical Application
倉庫道教分析 - 實際應用

This script analyzes Chinese texts in the NLP repository to demonstrate
how Daoist concepts are encoded throughout the linguistic examples.

Author: NLP Research Team  
Date: 2024-12-22
"""

import os
import re
from DaoistEncodingAnalyzer import DaoistEncodingAnalyzer
from typing import List, Dict, Tuple

class RepositoryDaoistAnalysis:
    """
    Analyzes the entire repository for Daoist encoding patterns
    """
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path
        self.analyzer = DaoistEncodingAnalyzer()
        self.file_analyses = []
    
    def extract_chinese_text(self, content: str) -> List[str]:
        """
        Extract Chinese text segments from mixed-language content
        """
        # Pattern to match Chinese characters
        chinese_pattern = r'[\u4e00-\u9fff]+'
        chinese_segments = re.findall(chinese_pattern, content)
        
        # Filter out short segments and combine nearby ones
        meaningful_segments = []
        for segment in chinese_segments:
            if len(segment) >= 2:  # At least 2 characters
                meaningful_segments.append(segment)
        
        return meaningful_segments
    
    def analyze_file(self, file_path: str) -> Dict:
        """
        Analyze a single file for Daoist encoding
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            chinese_segments = self.extract_chinese_text(content)
            
            if not chinese_segments:
                return None
            
            # Analyze each segment
            segment_analyses = []
            total_score = 0
            total_themes = set()
            
            for segment in chinese_segments:
                if len(segment) >= 3:  # Focus on meaningful segments
                    analysis = self.analyzer.analyze_text(segment)
                    if analysis['overall_score'] > 0:
                        segment_analyses.append({
                            'text': segment,
                            'score': analysis['overall_score'],
                            'themes': analysis['philosophical_themes'],
                            'patterns': analysis['concept_patterns']
                        })
                        total_score += analysis['overall_score']
                        total_themes.update(analysis['philosophical_themes'])
            
            if segment_analyses:
                avg_score = total_score / len(segment_analyses)
                return {
                    'file_path': file_path,
                    'chinese_segments_count': len(chinese_segments),
                    'analyzed_segments': segment_analyses,
                    'average_dao_score': avg_score,
                    'total_themes': list(total_themes),
                    'highest_score_segment': max(segment_analyses, key=lambda x: x['score']) if segment_analyses else None
                }
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None
        
        return None
    
    def scan_repository(self) -> List[Dict]:
        """
        Scan the entire repository for Chinese text with Daoist encoding
        """
        results = []
        
        # File extensions to analyze
        text_extensions = ['.md', '.txt', '.py', '.ipynb']
        
        for root, dirs, files in os.walk(self.repo_path):
            # Skip hidden directories and git
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if any(file.endswith(ext) for ext in text_extensions):
                    file_path = os.path.join(root, file)
                    analysis = self.analyze_file(file_path)
                    if analysis:
                        results.append(analysis)
        
        return results
    
    def generate_repository_report(self) -> str:
        """
        Generate a comprehensive report of Daoist encoding in the repository
        """
        analyses = self.scan_repository()
        
        if not analyses:
            return "No Chinese text with Daoist encoding found in repository."
        
        # Sort by average Daoist score
        analyses.sort(key=lambda x: x['average_dao_score'], reverse=True)
        
        report = []
        report.append("=" * 60)
        report.append("REPOSITORY DAOIST ENCODING ANALYSIS REPORT")
        report.append("倉庫道教編碼分析報告")
        report.append("=" * 60)
        
        # Summary statistics
        total_files = len(analyses)
        total_segments = sum(len(a['analyzed_segments']) for a in analyses)
        avg_repo_score = sum(a['average_dao_score'] for a in analyses) / total_files
        
        report.append(f"\nSUMMARY:")
        report.append(f"Files with Chinese text: {total_files}")
        report.append(f"Total Chinese segments analyzed: {total_segments}")
        report.append(f"Average repository Daoist score: {avg_repo_score:.3f}")
        
        # All themes found
        all_themes = set()
        for analysis in analyses:
            all_themes.update(analysis['total_themes'])
        
        if all_themes:
            report.append(f"Daoist themes found: {', '.join(sorted(all_themes))}")
        
        # Top files by Daoist content
        report.append(f"\nTOP FILES BY DAOIST ENCODING:")
        for i, analysis in enumerate(analyses[:10], 1):  # Top 10
            report.append(f"\n{i}. {analysis['file_path']}")
            report.append(f"   Average score: {analysis['average_dao_score']:.3f}")
            report.append(f"   Chinese segments: {analysis['chinese_segments_count']}")
            
            if analysis['highest_score_segment']:
                best_segment = analysis['highest_score_segment']
                report.append(f"   Best segment: {best_segment['text']} (score: {best_segment['score']:.3f})")
                
                if best_segment['themes']:
                    report.append(f"   Themes: {', '.join(best_segment['themes'])}")
        
        # Detailed analysis of top file
        if analyses:
            top_file = analyses[0]
            report.append(f"\n\nDETAILED ANALYSIS OF TOP FILE:")
            report.append(f"File: {top_file['file_path']}")
            report.append(f"Average Daoist score: {top_file['average_dao_score']:.3f}")
            
            for segment in top_file['analyzed_segments'][:5]:  # Top 5 segments
                report.append(f"\nSegment: {segment['text']}")
                report.append(f"Score: {segment['score']:.3f}")
                if segment['themes']:
                    report.append(f"Themes: {', '.join(segment['themes'])}")
        
        return '\n'.join(report)
    
    def find_daoist_connections(self) -> Dict:
        """
        Find interesting connections and patterns across the repository
        """
        analyses = self.scan_repository()
        
        connections = {
            'files_with_wu_wei': [],
            'files_with_yin_yang': [],
            'files_with_cycles': [],
            'files_with_reversal': [],
            'high_score_files': []
        }
        
        for analysis in analyses:
            file_path = analysis['file_path']
            themes = analysis['total_themes']
            avg_score = analysis['average_dao_score']
            
            # Check for specific concepts
            if any('wu wei' in theme.lower() or 'non-action' in theme.lower() for theme in themes):
                connections['files_with_wu_wei'].append(file_path)
            
            if any('yin' in theme.lower() or 'yang' in theme.lower() or 'duality' in theme.lower() for theme in themes):
                connections['files_with_yin_yang'].append(file_path)
            
            if any('cycle' in theme.lower() or 'cyclical' in theme.lower() for theme in themes):
                connections['files_with_cycles'].append(file_path)
            
            if any('reversal' in theme.lower() or 'extremes' in theme.lower() for theme in themes):
                connections['files_with_reversal'].append(file_path)
            
            if avg_score > 0.7:
                connections['high_score_files'].append((file_path, avg_score))
        
        return connections

def main():
    """Main demonstration function"""
    
    print("Repository Daoist Analysis - Practical Application")
    print("倉庫道教分析 - 實際應用")
    print()
    
    # Analyze the current repository
    repo_analyzer = RepositoryDaoistAnalysis(".")
    
    # Generate and display the report
    report = repo_analyzer.generate_repository_report()
    print(report)
    
    # Find interesting connections
    print("\n" + "="*60)
    print("INTERESTING CONNECTIONS FOUND:")
    print("發現的有趣聯繫:")
    
    connections = repo_analyzer.find_daoist_connections()
    
    for concept, files in connections.items():
        if files:
            print(f"\n{concept.replace('_', ' ').title()}:")
            if concept == 'high_score_files':
                for file_path, score in files:
                    print(f"  {file_path} (score: {score:.3f})")
            else:
                for file_path in files:
                    print(f"  {file_path}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("分析完成")

if __name__ == "__main__":
    main()