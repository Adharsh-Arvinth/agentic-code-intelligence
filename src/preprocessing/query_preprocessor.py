import re
from typing import Dict, List, Any

class QueryPreprocessor:
    def __init__(self):
        self.synonyms = {
            'bug': ['error', 'defect', 'issue', 'fault'],
            'fix': ['repair', 'resolve', 'patch', 'correct'],
            'function': ['method', 'routine', 'procedure'],
            'class': ['object', 'type'],
            'fast': ['optimized', 'efficient', 'quick'],
            'string': ['text', 'str'],
            'integer': ['int', 'number']
        }

    def normalize(self, query: str) -> str:
        if not query:
            return ""
        return re.sub(r'\s+', ' ', query).strip()

    def extract_identifiers(self, query: str) -> List[str]:
        if not query:
            return []
        pattern = r'\b(?:[a-z]+[A-Z][a-zA-Z0-9]*|[A-Z][a-zA-Z0-9]*[a-z][a-zA-Z0-9]*|[a-zA-Z_][a-zA-Z0-9_]*_[a-zA-Z0-9_]+)\b'
        identifiers = re.findall(pattern, query)
        return list(set(identifiers))

    def extract_keywords(self, query: str) -> List[str]:
        if not query:
            return []
        keywords_list = {
            'def', 'class', 'import', 'from', 'return', 'yield', 'if', 'else', 
            'elif', 'for', 'while', 'try', 'except', 'finally', 'with', 'as', 
            'async', 'await', 'pass', 'break', 'continue', 'lambda', 'global',
            'nonlocal', 'assert', 'del', 'in', 'is', 'and', 'or', 'not'
        }
        words = set(re.findall(r'\b[a-zA-Z_]\w*\b', query.lower()))
        return list(words.intersection(keywords_list))

    def expand_query(self, query: str) -> str:
        if not query:
            return ""
        words = re.findall(r'\b\w+\b', query)
        expanded_words = list(words)
        for word in words:
            lower_word = word.lower()
            if lower_word in self.synonyms:
                expanded_words.extend(self.synonyms[lower_word])
        return " ".join(set(expanded_words))

    def process(self, query: str) -> Dict[str, Any]:
        return {
            'original': query,
            'normalized': self.normalize(query),
            'identifiers': self.extract_identifiers(query),
            'keywords': self.extract_keywords(query),
            'expanded': self.expand_query(query)
        }
