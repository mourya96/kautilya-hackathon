"""
Narrative Engine - Builds narratives from news articles
"""

from typing import List, Dict, Any
import numpy as np
from datetime import datetime
from collections import defaultdict
from sklearn.cluster import DBSCAN
import logging

logger = logging.getLogger(__name__)


class NarrativeEngine:
    """Generates narrative structures from news articles"""
    
    def __init__(self, articles: List[Dict[str, Any]], topic: str,
                 llm_narrative: Dict[str, Any] = None):
        """
        Initialize narrative engine

        Args:
            articles: List of relevant articles
            topic: Topic being analyzed
            llm_narrative: Optional grounded narrative from the RAG generator,
                shaped {"summary": str, "timeline": [{"id", "why_it_matters"}]}.
                When provided, the LLM summary and per-article reasoning are used;
                otherwise the heuristic fallbacks below are used.
        """
        self.articles = sorted(articles, key=lambda x: x.get('date', ''))
        self.topic = topic
        self.llm_narrative = llm_narrative or {}
        # Map article id -> LLM-generated "why it matters" reasoning.
        self._llm_reasons = {
            entry.get('id'): entry.get('why_it_matters')
            for entry in self.llm_narrative.get('timeline', [])
            if entry.get('why_it_matters')
        }
        
    def generate_summary(self) -> str:
        """
        Generate 5-10 sentence narrative summary
        
        Returns:
            Narrative summary string
        """
        if not self.articles:
            return f"No relevant articles found for topic: {self.topic}"

        # Prefer the grounded LLM summary when available.
        llm_summary = self.llm_narrative.get('summary')
        if llm_summary:
            return llm_summary

        # Get key statistics
        num_articles = len(self.articles)
        date_range = self._get_date_range()
        sources = set(a['source'] for a in self.articles)
        
        # Extract key themes from headlines
        key_events = self._extract_key_events()
        
        # Build summary
        summary_parts = [
            f"Analysis of {num_articles} articles covering '{self.topic}' from {len(sources)} sources.",
        ]
        
        if date_range:
            summary_parts.append(f"Coverage spans from {date_range[0]} to {date_range[1]}.")
        
        # Add key narrative points
        if key_events:
            summary_parts.append("Key developments include: " + "; ".join(key_events[:5]) + ".")
        
        # Analyze temporal patterns
        temporal_analysis = self._analyze_temporal_patterns()
        if temporal_analysis:
            summary_parts.append(temporal_analysis)
        
        # Source diversity
        top_sources = self._get_top_sources(3)
        if top_sources:
            source_str = ", ".join([f"{s[0]} ({s[1]} articles)" for s in top_sources])
            summary_parts.append(f"Most active sources: {source_str}.")
        
        # Overall narrative arc
        arc = self._determine_narrative_arc()
        summary_parts.append(arc)
        
        return " ".join(summary_parts)
    
    def build_timeline(self) -> List[Dict[str, Any]]:
        """
        Build chronological timeline of events
        
        Returns:
            List of timeline entries
        """
        timeline = []
        
        for article in self.articles:
            # Use the grounded LLM reasoning when present, else the heuristic.
            why = self._llm_reasons.get(article.get('id')) or \
                self._generate_importance_reason(article)
            entry = {
                'date': article.get('date', 'Unknown'),
                'headline': article['headline'],
                'url': article.get('url', ''),
                'source': article.get('source', 'Unknown'),
                'why_it_matters': why
            }
            timeline.append(entry)
        
        return timeline
    
    def create_clusters(self) -> List[Dict[str, Any]]:
        """
        Group semantically similar articles into thematic clusters
        
        Returns:
            List of cluster dictionaries
        """
        if len(self.articles) < 2:
            return [{
                'cluster_id': 0,
                'theme': 'General Coverage',
                'article_count': len(self.articles),
                'articles': [a['id'] for a in self.articles]
            }]
        
        # Use headline similarity for clustering
        headlines = [a['headline'] for a in self.articles]
        
        # Simple clustering based on keyword overlap
        clusters = self._cluster_by_keywords()
        
        # Format clusters
        cluster_list = []
        for cluster_id, cluster_info in enumerate(clusters):
            cluster_list.append({
                'cluster_id': cluster_id,
                'theme': cluster_info['theme'],
                'article_count': len(cluster_info['article_ids']),
                'articles': cluster_info['article_ids'],
                'key_terms': cluster_info.get('key_terms', [])
            })
        
        return cluster_list
    
    def build_narrative_graph(self) -> Dict[str, Any]:
        """
        Build narrative graph showing relationships between articles
        
        Returns:
            Graph structure with nodes and edges
        """
        nodes = []
        edges = []
        
        # Create nodes
        for i, article in enumerate(self.articles):
            nodes.append({
                'id': str(article['id']),
                'headline': article['headline'],
                'date': article.get('date', ''),
                'source': article.get('source', 'Unknown'),
                'index': i
            })
        
        # Create edges based on relationships
        for i in range(len(self.articles)):
            for j in range(i + 1, len(self.articles)):
                relation = self._determine_relationship(
                    self.articles[i],
                    self.articles[j]
                )
                
                if relation:
                    edges.append({
                        'source': str(self.articles[i]['id']),
                        'target': str(self.articles[j]['id']),
                        'relation': relation,
                        'weight': self._calculate_edge_weight(relation)
                    })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'node_count': len(nodes),
            'edge_count': len(edges)
        }
    
    def _get_date_range(self) -> tuple:
        """Get earliest and latest dates"""
        dates = [a['date'] for a in self.articles if a['date']]
        if not dates:
            return None
        return min(dates)[:10], max(dates)[:10]
    
    def _extract_key_events(self) -> List[str]:
        """Extract key events from headlines"""
        # Simple extraction: get diverse headlines
        key_events = []
        seen_words = set()
        
        for article in self.articles:
            headline_words = set(article['headline'].lower().split())
            
            # Check if this headline adds new information
            new_words = headline_words - seen_words
            if len(new_words) > 3:
                key_events.append(article['headline'][:80])
                seen_words.update(headline_words)
            
            if len(key_events) >= 8:
                break
        
        return key_events
    
    def _analyze_temporal_patterns(self) -> str:
        """Analyze temporal patterns in coverage"""
        dates = [a['date'] for a in self.articles if a['date']]
        if not dates:
            return ""
        
        # Count articles per month
        monthly_counts = defaultdict(int)
        for date in dates:
            month = date[:7]  # YYYY-MM
            monthly_counts[month] += 1
        
        if len(monthly_counts) > 1:
            peak_month = max(monthly_counts.items(), key=lambda x: x[1])
            return f"Coverage peaked in {peak_month[0]} with {peak_month[1]} articles."
        
        return ""
    
    def _get_top_sources(self, n: int = 3) -> List[tuple]:
        """Get top N sources by article count"""
        source_counts = defaultdict(int)
        for article in self.articles:
            source_counts[article['source']] += 1
        
        return sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:n]
    
    def _determine_narrative_arc(self) -> str:
        """Determine overall narrative arc"""
        if len(self.articles) < 3:
            return "Coverage is limited but consistent."
        
        # Analyze headline sentiment/intensity changes
        first_third = self.articles[:len(self.articles)//3]
        last_third = self.articles[-len(self.articles)//3:]
        
        # Simple heuristic: look for escalation keywords
        escalation_words = ['crisis', 'urgent', 'breaking', 'critical', 'emergency']
        
        first_intensity = sum(
            1 for a in first_third 
            if any(word in a['headline'].lower() for word in escalation_words)
        )
        last_intensity = sum(
            1 for a in last_third
            if any(word in a['headline'].lower() for word in escalation_words)
        )
        
        if last_intensity > first_intensity:
            return "The narrative shows escalating intensity over time."
        elif last_intensity < first_intensity:
            return "Coverage intensity has decreased over the timeline."
        else:
            return "The story maintains consistent coverage throughout the period."
    
    def _generate_importance_reason(self, article: Dict[str, Any]) -> str:
        """Generate why an article matters"""
        reasons = []
        
        # High similarity to query
        if article.get('similarity_score', 0) > 0.7:
            reasons.append("highly relevant to topic")
        
        # Trusted source
        if article.get('source_rating', 0) > 9:
            reasons.append("from trusted source")
        
        # Recent
        if article.get('date', ''):
            try:
                date_obj = datetime.fromisoformat(article['date'][:10])
                days_old = (datetime.now() - date_obj).days
                if days_old < 7:
                    reasons.append("recent development")
            except:
                pass
        
        # Key event indicators
        key_words = ['announce', 'launch', 'break', 'reveal', 'confirm', 'declare']
        if any(word in article['headline'].lower() for word in key_words):
            reasons.append("marks significant event")
        
        if reasons:
            return "Important because it " + ", ".join(reasons) + "."
        
        return "Provides context for the narrative."
    
    def _cluster_by_keywords(self) -> List[Dict[str, Any]]:
        """Cluster articles by keyword similarity"""
        # Extract keywords from headlines
        from collections import Counter
        
        # Simple keyword extraction
        all_words = []
        for article in self.articles:
            words = [
                w.lower() for w in article['headline'].split()
                if len(w) > 4 and w.isalnum()
            ]
            all_words.extend(words)
        
        # Find common keywords
        word_counts = Counter(all_words)
        common_words = [w for w, c in word_counts.most_common(20)]
        
        # Cluster by shared keywords
        clusters = defaultdict(lambda: {'article_ids': [], 'keywords': set()})
        
        for article in self.articles:
            article_words = set(w.lower() for w in article['headline'].split())
            
            # Find best matching cluster
            best_cluster = None
            best_overlap = 0
            
            for cluster_key, cluster_data in clusters.items():
                overlap = len(article_words & cluster_data['keywords'])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_cluster = cluster_key
            
            # Assign to cluster or create new one
            if best_cluster and best_overlap >= 2:
                clusters[best_cluster]['article_ids'].append(article['id'])
                clusters[best_cluster]['keywords'].update(article_words & set(common_words))
            else:
                # Create new cluster
                cluster_key = len(clusters)
                clusters[cluster_key]['article_ids'].append(article['id'])
                clusters[cluster_key]['keywords'].update(article_words & set(common_words))
        
        # Format clusters with themes
        result = []
        for cluster_data in clusters.values():
            theme = " ".join(list(cluster_data['keywords'])[:3]).title() or "General"
            result.append({
                'theme': theme,
                'article_ids': cluster_data['article_ids'],
                'key_terms': list(cluster_data['keywords'])[:5]
            })
        
        return result
    
    def _determine_relationship(self, article1: Dict, article2: Dict) -> str:
        """Determine relationship type between two articles"""
        # Temporal relationship
        if article1['date'] and article2['date']:
            date1 = article1['date']
            date2 = article2['date']
            
            if date1 < date2:
                # Check if similar topic (builds on)
                common_words = self._common_words(
                    article1['headline'],
                    article2['headline']
                )
                if len(common_words) >= 3:
                    return 'builds_on'
        
        # Check for contradictions
        contradiction_pairs = [
            ('increase', 'decrease'),
            ('rise', 'fall'),
            ('approve', 'reject'),
            ('success', 'failure')
        ]
        
        headline1_lower = article1['headline'].lower()
        headline2_lower = article2['headline'].lower()
        
        for word1, word2 in contradiction_pairs:
            if (word1 in headline1_lower and word2 in headline2_lower) or \
               (word2 in headline1_lower and word1 in headline2_lower):
                return 'contradicts'
        
        # Check for context addition
        common_words = self._common_words(article1['headline'], article2['headline'])
        if 2 <= len(common_words) < 4:
            return 'adds_context'
        
        # Check for escalation
        escalation_words = ['crisis', 'escalate', 'worsen', 'intensify']
        if any(word in headline2_lower for word in escalation_words) and \
           any(word in headline1_lower for word in ['concern', 'issue', 'problem']):
            return 'escalates'
        
        # Strong similarity might indicate related coverage
        if len(common_words) >= 4:
            return 'builds_on'
        
        return None
    
    def _common_words(self, text1: str, text2: str) -> set:
        """Find common significant words between two texts"""
        words1 = set(w.lower() for w in text1.split() if len(w) > 4)
        words2 = set(w.lower() for w in text2.split() if len(w) > 4)
        return words1 & words2
    
    def _calculate_edge_weight(self, relation: str) -> float:
        """Calculate edge weight based on relation type"""
        weights = {
            'builds_on': 0.9,
            'contradicts': 0.7,
            'adds_context': 0.6,
            'escalates': 0.8
        }
        return weights.get(relation, 0.5)
