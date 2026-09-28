"""
Document (A-E) and Query Representations + AST Structural & I/O Extractor
for CoIR AppsRetrieval Text-to-Code Retrieval.
"""

import ast
import re
import warnings
from typing import Dict, List, Tuple, Optional


# Common template helper functions in competitive programming that obscure core logic
_TEMPLATE_HELPERS = {
    "input", "iinput", "finput", "tinput", "linput", "rinput",
    "fiinput", "rlinput", "trinput", "srlinput", "NOYES", "YESNO",
    "read", "readline", "read_int", "read_ints", "read_str", "fast_io",
    " bootstrap", "gcd", "lcm"
}


def strip_cp_boilerplate(code: str) -> str:
    """
    Remove dead commented-out template lines and unused CP input wrappers
    so the core algorithm fits within the 512-token embedding window.
    """
    if not code:
        return ""
    lines = code.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip commented-out template assignment lines like #n, k = rinput()
        if stripped.startswith("#") and any(
            pat in stripped for pat in ["rinput(", "iinput(", "input(", "stdin", "===", "---"]
        ):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Remove unused standard CP helper one-liner defs if main() or loop exists
    if "def main(" in text or "for " in text:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(text)
            # Find all called function names in the module
            called_names = {
                node.func.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
            keep_segments = []
            src_lines = text.splitlines(keepends=True)
            skip_ranges = []
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name in _TEMPLATE_HELPERS:
                    if node.name not in called_names and hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                        skip_ranges.append((node.lineno - 1, node.end_lineno))
            if skip_ranges:
                skip_set = set()
                for s, e in skip_ranges:
                    for idx in range(s, e):
                        skip_set.add(idx)
                text = "".join(line for idx, line in enumerate(src_lines) if idx not in skip_set)
        except Exception:
            pass

    # Collapse 3+ consecutive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def analyze_code_structure(code: str) -> Dict[str, any]:
    """Extract rich structural, I/O, and algorithmic features from Python solution code."""
    features = {
        "language": "python",
        "functions": [],
        "classes": [],
        "imports": [],
        "comments": [],
        "identifiers": [],
        "io_tags": [],
        "algo_tags": [],
        "constants": [],
    }
    if not code:
        return features

    # Functions & classes
    features["functions"] = [
        f for f in re.findall(r"\bdef\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code)
        if f not in _TEMPLATE_HELPERS
    ]
    features["classes"] = re.findall(r"\bclass\s+([a-zA-Z_][a-zA-Z0-9_]*)", code)
    features["imports"] = re.findall(r"(?:from\s+(\S+)\s+import\s+(\S+)|import\s+(\S+))", code)
    features["imports"] = [
        " ".join(filter(None, imp)) for imp in features["imports"]
    ]

    # Comments & docstrings
    comments = re.findall(r"#\s*(.+)", code)
    docstrings = re.findall(r'"""(.*?)"""|\'\'\'(.*?)\'\'\'', code, re.DOTALL)
    for d1, d2 in docstrings:
        comments.append((d1 or d2).strip())
    features["comments"] = [
        c.strip() for c in comments
        if c.strip() and not any(p in c for p in ["rinput(", "iinput(", "type ", "rtype"])
    ][:5]

    # I/O pattern detection
    if "class Solution" in code:
        features["io_tags"].append("class_method_solution")
    else:
        features["io_tags"].append("stdin_stdout_script")
        if re.search(r"for\s+\w+\s+in\s+range\s*\(\s*(?:int\s*\(\s*input|iinput|t\b|q\b)", code):
            features["io_tags"].append("multiple_test_cases_loop")
        else:
            features["io_tags"].append("single_test_case")

    if re.search(r'print\s*\(\s*["\']YES["\']\s*\)|NOYES|YESNO', code, re.IGNORECASE):
        features["io_tags"].append("prints_YES_NO")
    if re.search(r"print\s*\(\s*-1\s*\)|ans\s*=\s*-1|return\s+-1", code):
        features["io_tags"].append("outputs_minus_one_impossible")
    if "1000000007" in code or "10**9 + 7" in code or "10**9+7" in code or "10 ** 9 + 7" in code:
        features["io_tags"].append("modulo_1000000007")
    if "998244353" in code:
        features["io_tags"].append("modulo_998244353")

    # Algorithmic concepts detection
    algo_patterns = [
        (r"\bsorted\b|\.sort\(", "sorting greedy order"),
        (r"\bbisect\b|bisect_left|bisect_right", "binary search bisect"),
        (r"\bheapq\b|heappush|heappop", "priority queue heap"),
        (r"\bdeque\b|popleft", "bfs queue deque"),
        (r"\bCounter\b|\bdefaultdict\b", "frequency map hash counting"),
        (r"\bset\s*\(|\{|\badd\(", "hash set distinct elements"),
        (r"\bgcd\b|\bmath\.gcd\b", "greatest common divisor gcd number theory"),
        (r"\bdp\b|memo|lru_cache", "dynamic programming memoization"),
        (r"\[::-1\]|\breversed\(", "string array reversal palindrome"),
        (r"%\s*2\b|&\s*1\b", "parity even odd modulo 2"),
        (r"\^|\b<<\b|\b>>\b|\bbin\(", "bitwise xor binary shift"),
        (r"\bdfs\b|\badj\b|\bgraph\b|\bedges\b", "graph tree traversal dfs"),
        (r"\bstack\b|\.pop\(\)", "stack bracket monotonic"),
        (r"\bcomb\b|\bfactorial\b", "combinatorics binomial factorial"),
        (r"\babs\(|max\(|min\(", "extremum absolute difference"),
    ]
    for pat, label in algo_patterns:
        if re.search(pat, code):
            features["algo_tags"].append(label)

    # Extract meaningful identifiers
    raw_ids = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", code)
    stop_ids = {
        "for", "in", "range", "int", "input", "print", "if", "else", "elif",
        "def", "return", "while", "import", "from", "sys", "stdin", "readline",
        "split", "map", "list", "len", "True", "False", "None", "and", "or", "not",
        "continue", "break", "pass", "self", "class", "Solution", "main"
    } | _TEMPLATE_HELPERS
    seen_ids = []
    for ident in raw_ids:
        if ident not in stop_ids and ident not in seen_ids and len(ident) >= 2:
            seen_ids.append(ident)
            if len(seen_ids) >= 20:
                break
    features["identifiers"] = seen_ids

    # Extract special integer constants
    nums = re.findall(r"\b\d+\b", code)
    special_nums = [
        n for n in nums
        if n not in {"0", "1", "2"} and len(n) <= 10
    ]
    features["constants"] = list(dict.fromkeys(special_nums))[:8]

    return features


# ==============================================================================
# 5 DOCUMENT REPRESENTATIONS (A, B, C, D, E) — Requirement 5
# ==============================================================================

def format_doc_rep_a(code: str, title: str = "") -> str:
    """Representation A: Raw code."""
    return f"{title}\n{code}".strip() if title else code.strip()


def format_doc_rep_b(code: str, title: str = "") -> str:
    """Representation B: Function/class signatures + code."""
    feats = analyze_code_structure(code)
    sigs = []
    if feats["classes"]:
        sigs.append("Classes: " + ", ".join(feats["classes"]))
    if feats["functions"]:
        sigs.append("Functions: " + ", ".join(feats["functions"]))
    header = " | ".join(sigs)
    return f"{header}\n{code}".strip() if header else code.strip()


def format_doc_rep_c(code: str, title: str = "") -> str:
    """Representation C: Comments/docstrings + code."""
    feats = analyze_code_structure(code)
    comments_str = " ; ".join(feats["comments"])
    return f"Docstring/Comments: {comments_str}\n{code}".strip() if comments_str else code.strip()


def format_doc_rep_d(code: str, title: str = "") -> str:
    """Representation D: Identifiers + code."""
    feats = analyze_code_structure(code)
    id_str = " ".join(feats["identifiers"])
    return f"Identifiers: {id_str}\n{code}".strip() if id_str else code.strip()


def format_doc_rep_e(code: str, title: str = "") -> str:
    """
    Representation E: Structured representation combining boilerplate-cleaned
    Python code with a concise structural metadata trailer (language, signatures,
    imports, I/O patterns, algorithmic concepts, constants, identifiers).
    """
    cleaned_code = strip_cp_boilerplate(code)
    feats = analyze_code_structure(cleaned_code)

    parts = ["Language: Python"]
    if feats["functions"] or feats["classes"]:
        parts.append("Definitions: " + ", ".join(feats["classes"] + feats["functions"]))
    if feats["imports"]:
        parts.append("Imports: " + ", ".join(feats["imports"][:5]))
    if feats["io_tags"]:
        parts.append("I/O: " + ", ".join(feats["io_tags"]))
    if feats["algo_tags"]:
        parts.append("Algorithm: " + ", ".join(feats["algo_tags"]))
    if feats["constants"]:
        parts.append("Constants: " + " ".join(feats["constants"]))
    if feats["identifiers"]:
        parts.append("Variables: " + " ".join(feats["identifiers"][:15]))

    header = " | ".join(parts)
    return f"{header}\n{cleaned_code}".strip()


DOCUMENT_REPRESENTATIONS = {
    "A": format_doc_rep_a,
    "B": format_doc_rep_b,
    "C": format_doc_rep_c,
    "D": format_doc_rep_d,
    "E": format_doc_rep_e,
}


# ==============================================================================
# QUERY REPRESENTATIONS & EXAMPLE I/O EXTRACTION — Requirement 6 & 7
# ==============================================================================

def extract_query_sections(query: str) -> Dict[str, str]:
    """Split a competitive programming problem query into narrative, input, output, example."""
    sections = {
        "narrative": query,
        "input_spec": "",
        "output_spec": "",
        "example_spec": "",
    }
    if not query:
        return sections

    inp_match = re.search(r"-+\s*Input\s*-+(.*?)(?=-+\s*Output\s*-+|-+\s*Example\s*-+|$)", query, re.DOTALL | re.IGNORECASE)
    out_match = re.search(r"-+\s*Output\s*-+(.*?)(?=-+\s*Example\s*-+|-+\s*Note\s*-+|$)", query, re.DOTALL | re.IGNORECASE)
    ex_match = re.search(r"-+\s*Example[s]?\s*-+(.*?)(?=-+\s*Note\s*-+|$)", query, re.DOTALL | re.IGNORECASE)

    if inp_match:
        sections["narrative"] = query[:inp_match.start()].strip()
        sections["input_spec"] = inp_match.group(1).strip()
    if out_match:
        sections["output_spec"] = out_match.group(1).strip()
    if ex_match:
        sections["example_spec"] = ex_match.group(1).strip()

    return sections


def extract_query_structural_tags(query: str) -> List[str]:
    """Extract structural I/O and algorithmic tags from a natural-language problem statement."""
    tags = []
    q_lower = query.lower()
    secs = extract_query_sections(query)
    inp_lower = secs["input_spec"].lower()
    out_lower = secs["output_spec"].lower()

    if secs["input_spec"] or secs["output_spec"]:
        tags.append("stdin_stdout_script")
        if any(w in inp_lower for w in ["test cases", "testcases", "number of queries", "each test case", "t lines", "q lines"]):
            tags.append("multiple_test_cases_loop")
        else:
            tags.append("single_test_case")
    else:
        tags.append("class_method_solution")

    if any(w in out_lower for w in ['"yes"', "'yes'", "print yes", "output yes"]):
        tags.append("prints_YES_NO")
    if "-1" in out_lower or "impossible" in q_lower:
        tags.append("outputs_minus_one_impossible")
    if "10^9 + 7" in query or "10^9+7" in query or "1000000007" in query or "10^{9} + 7" in query:
        tags.append("modulo_1000000007")
    if "998244353" in query:
        tags.append("modulo_998244353")

    concept_map = [
        (["palindrome", "reverse", "reversal", "backward"], "string array reversal palindrome"),
        (["even", "odd", "parity", "divisible by 2"], "parity even odd modulo 2"),
        (["gcd", "greatest common divisor", "coprime", "divisible", "multiple"], "greatest common divisor gcd number theory"),
        (["sort", "increasing", "non-decreasing", "lexicographically", "permutation"], "sorting greedy order"),
        (["tree", "connected", "vertices", "edges", "path", "graph", "cycle"], "graph tree traversal dfs"),
        (["subsequence", "dynamic programming", "ways to", "modulo"], "dynamic programming memoization"),
        (["xor", "bitwise", "binary representation", "bit"], "bitwise xor binary shift"),
        (["distinct", "unique", "appear at least", "frequency", "occurrences"], "hash set distinct elements"),
        (["minimum", "maximum", "at most", "at least", "optimal"], "extremum absolute difference"),
    ]
    for keywords, tag in concept_map:
        if any(k in q_lower for k in keywords):
            tags.append(tag)

    return tags


def format_query_smart(query: str, instruction_prefix: str = "") -> str:
    """
    Smart Query Formatting (Requirement 6 & 7):
    1. Prepends model-required query instruction prefix directly before the problem narrative.
    2. Preserves the problem narrative + Input/Output specs within the token window.
    3. Appends normalized algorithmic & I/O tags at the tail.
    """
    if not query:
        return ""
    secs = extract_query_sections(query)
    tags = extract_query_structural_tags(query)
    tag_tail = " ".join(tags)

    narrative = re.sub(r"\s+", " ", secs["narrative"]).strip()
    inp_spec = re.sub(r"\s+", " ", secs["input_spec"]).strip()
    out_spec = re.sub(r"\s+", " ", secs["output_spec"]).strip()

    narr_words = narrative.split()
    if len(narr_words) > 200 and (inp_spec or out_spec):
        narrative_compact = " ".join(narr_words[:170]) + " " + " ".join(narr_words[-30:])
    else:
        narrative_compact = narrative

    structured_q = narrative_compact
    if inp_spec:
        structured_q += f" Input: {inp_spec[:250]}"
    if out_spec:
        structured_q += f" Output: {out_spec[:180]}"
    if tag_tail:
        structured_q += f" Keywords: {tag_tail}"

    if instruction_prefix:
        return f"{instruction_prefix}{structured_q}"
    return structured_q


def extract_example_io(query: str) -> List[Tuple[str, str]]:
    """
    Extract (example_input, example_output) pairs from the query's Example section.
    Supports Codeforces (Input/Output, Входные данные/Выходные данные),
    AtCoder (-----Sample Input-----/-----Sample Output-----), and Input:/Output: blocks.
    """
    if not query:
        return []
    pairs = []
    patterns = [
        re.compile(
            r"Input\s*:?\s*\n(.*?)\n\s*Output\s*:?\s*\n(.*?)(?=\n\s*Input\s*:?\s*\n|\n\s*-+\s*Note|\n\s*Explanation|\Z)",
            re.DOTALL,
        ),
        re.compile(
            r"-+\s*Sample Input\s*\d*\s*-+\s*\n(.*?)\n\s*-+\s*Sample Output\s*\d*\s*-+\s*\n(.*?)(?=\n\s*-+\s*Sample Input|\n\s*-+\s*Note|\n\n[A-Z]|\Z)",
            re.DOTALL,
        ),
        re.compile(
            r"Входные данные\s*\n(.*?)\n\s*Выходные данные\s*\n(.*?)(?=\n\s*Входные данные|\n\s*-+\s*Примечание|\Z)",
            re.DOTALL,
        ),
    ]
    for pat in patterns:
        for match in pat.finditer(query):
            ex_in = match.group(1).strip()
            ex_out = match.group(2).strip()
            # Exclude LeetCode assignment syntax like `nums = [1, 2, 3]` from stdin verifier
            if "=" in ex_in.split("\n")[0] and ("[" in ex_in or '"' in ex_in):
                continue
            if ex_in and ex_out and len(ex_in) < 4000 and len(ex_out) < 4000:
                pairs.append((ex_in + "\n", ex_out))
        if pairs:
            break
    return pairs
