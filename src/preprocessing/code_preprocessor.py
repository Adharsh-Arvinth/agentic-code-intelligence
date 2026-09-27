import re
from typing import Dict, List, Any

class CodePreprocessor:
    def __init__(self):
        pass

    def extract_identifiers(self, code: str) -> List[str]:
        if not code:
            return []
        words = re.findall(r'\b[a-zA-Z_]\w*\b', code)
        keywords = {
            'def', 'class', 'import', 'from', 'return', 'if', 'else', 'elif',
            'for', 'while', 'try', 'except', 'with', 'as', 'pass', 'break',
            'continue', 'and', 'or', 'not', 'is', 'in'
        }
        identifiers = [w for w in words if w not in keywords]
        return list(set(identifiers))

    def extract_imports(self, code: str) -> List[str]:
        if not code:
            return []
        pattern = r'^\s*(?:import\s+[a-zA-Z0-9_\., \t]+|from\s+[a-zA-Z0-9_\.]+\s+import\s+[a-zA-Z0-9_\., \t\*]+)'
        imports = re.findall(pattern, code, re.MULTILINE)
        return [i.strip() for i in imports]

    def extract_comments(self, code: str) -> List[str]:
        if not code:
            return []
        comments = re.findall(r'#.*', code)
        docstrings = re.findall(r'\"\"\"(.*?)\"\"\"|\'\'\'(.*?)\'\'\'', code, re.DOTALL)
        for doc in docstrings:
            comments.extend([d for d in doc if d])
        return [c.strip() for c in comments if c.strip()]

    def extract_strings(self, code: str) -> List[str]:
        if not code:
            return []
        pattern = r'"([^"\\]*(?:\\.[^"\\]*)*)"|\'([^\'\\]*(?:\\.[^\'\\]*)*)\''
        strings = re.findall(pattern, code)
        result = []
        for s in strings:
            result.extend([x for x in s if x])
        return list(set(result))

    def extract_function_names(self, code: str) -> List[str]:
        if not code:
            return []
        pattern = r'^\s*def\s+([a-zA-Z_]\w*)\s*\('
        return re.findall(pattern, code, re.MULTILINE)

    def extract_class_names(self, code: str) -> List[str]:
        if not code:
            return []
        pattern = r'^\s*class\s+([a-zA-Z_]\w*)\s*(?:\(|:)'
        return re.findall(pattern, code, re.MULTILINE)

    def get_language(self, code: str) -> str:
        if re.search(r'^\s*(def|class|import|from)\s+', code, re.MULTILINE):
            return 'python'
        if re.search(r'^\s*#include\s+<', code, re.MULTILINE):
            return 'cpp'
        if re.search(r'^\s*public\s+class\s+', code, re.MULTILINE):
            return 'java'
        return 'python'

    def build_searchable_text(self, code: str) -> str:
        if not code:
            return ""
        identifiers = self.extract_identifiers(code)
        comments = self.extract_comments(code)
        func_names = self.extract_function_names(code)
        class_names = self.extract_class_names(code)
        
        parts = [code] + comments + identifiers + func_names + class_names
        return " ".join(parts)

    def process(self, code: str) -> Dict[str, Any]:
        return {
            'language': self.get_language(code),
            'identifiers': self.extract_identifiers(code),
            'imports': self.extract_imports(code),
            'comments': self.extract_comments(code),
            'strings': self.extract_strings(code),
            'function_names': self.extract_function_names(code),
            'class_names': self.extract_class_names(code),
            'searchable_text': self.build_searchable_text(code)
        }
