"""
Agentic Execution-Guided Code Verifier for Text-to-Code Retrieval.

Pre-compiles Python corpus solutions with an AST loop-step counter guard
and _BoundedRange iterator during indexing, then verifies candidate programs
in-memory against the Example Input -> Example Output specifications embedded
in the query in < 50 microseconds per candidate.
"""

import ast
import io
import sys
import math
import heapq
import bisect
import random
import collections
import itertools
import functools
import operator
import fractions
import re
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union

from src.preprocessing.representations import extract_example_io


class _StepLimitExceeded(Exception):
    """Raised when a sandboxed candidate exceeds the allowed loop step budget."""
    pass


class _BoundedRange:
    """Range replacement that counts every iteration against the global step budget."""
    __slots__ = ("_r", "_steps", "_max")

    def __init__(self, r: range, steps: List[int], max_steps: int):
        if len(r) > 25000:
            raise _StepLimitExceeded()
        if len(r) > 25000:
            raise _StepLimitExceeded()
        self._r = r
        self._steps = steps
        self._max = max_steps

    def __iter__(self):
        steps = self._steps
        mx = self._max
        for x in self._r:
            steps[0] += 1
            if steps[0] > mx:
                raise _StepLimitExceeded()
            yield x

    def __len__(self):
        return min(len(self._r), self._max)

    def __getitem__(self, idx):
        return self._r[idx]

    def __reversed__(self):
        steps = self._steps
        mx = self._max
        for x in reversed(self._r):
            steps[0] += 1
            if steps[0] > mx:
                raise _StepLimitExceeded()
            yield x


class _BoundedIter:
    """Wraps any iterator/comprehension iterable to enforce step budget."""

    __slots__ = ("_it", "_steps", "_max")

    def __init__(self, iterable, steps: List[int], max_steps: int):
        self._it = iter(iterable)
        self._steps = steps
        self._max = max_steps

    def __iter__(self):
        return self

    def __next__(self):
        self._steps[0] += 1
        if self._steps[0] > self._max:
            raise _StepLimitExceeded()
        return next(self._it)


class _LoopStepGuardTransformer(ast.NodeTransformer):
    """
    AST transformer that injects a fast loop-step counter check into every
    for-loop, while-loop, comprehension, and function body, and wraps range() with _safe_range().
    """

    def _make_guard_nodes(self) -> List[ast.stmt]:
        inc = ast.AugAssign(
            target=ast.Subscript(
                value=ast.Name(id="_sb_steps", ctx=ast.Load()),
                slice=ast.Constant(value=0),
                ctx=ast.Store(),
            ),
            op=ast.Add(),
            value=ast.Constant(value=1),
        )
        chk = ast.If(
            test=ast.Compare(
                left=ast.Subscript(
                    value=ast.Name(id="_sb_steps", ctx=ast.Load()),
                    slice=ast.Constant(value=0),
                    ctx=ast.Load(),
                ),
                ops=[ast.Gt()],
                comparators=[ast.Name(id="_sb_max", ctx=ast.Load())],
            ),
            body=[
                ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id="_StepLimitExceeded", ctx=ast.Load()),
                        args=[],
                        keywords=[],
                    ),
                    cause=None,
                )
            ],
            orelse=[],
        )
        return [inc, chk]

    def visit_For(self, node: ast.For) -> Any:
        self.generic_visit(node)
        node.body = self._make_guard_nodes() + node.body
        return node

    def visit_While(self, node: ast.While) -> Any:
        self.generic_visit(node)
        node.body = self._make_guard_nodes() + node.body
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self.generic_visit(node)
        node.body = self._make_guard_nodes() + node.body
        return node

    def visit_comprehension(self, node: ast.comprehension) -> Any:
        self.generic_visit(node)
        node.iter = ast.Call(
            func=ast.Name(id="_safe_iter", ctx=ast.Load()),
            args=[node.iter],
            keywords=[],
        )
        return node

    def visit_Call(self, node: ast.Call) -> Any:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            if node.func.id == "range":
                node.func.id = "_safe_range"
            elif node.func.id in ("exit", "quit"):
                node.func.id = "_safe_exit"
        return node

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        self.generic_visit(node)
        if isinstance(node.op, ast.Pow):
            return ast.Call(
                func=ast.Name(id="_safe_pow", ctx=ast.Load()),
                args=[node.left, node.right],
                keywords=[],
            )
        if isinstance(node.op, ast.LShift):
            return ast.Call(
                func=ast.Name(id="_safe_lshift", ctx=ast.Load()),
                args=[node.left, node.right],
                keywords=[],
            )
        if isinstance(node.op, ast.Mult):
            return ast.Call(
                func=ast.Name(id="_safe_mult", ctx=ast.Load()),
                args=[node.left, node.right],
                keywords=[],
            )
        return node


def _safe_pow(a, b):
    if isinstance(b, (int, float)) and b > 2000 and a not in (0, 1, -1):
        raise _StepLimitExceeded()
    return a ** b


def _safe_lshift(a, b):
    if isinstance(b, int) and b > 2000:
        raise _StepLimitExceeded()
    return a << b


def _safe_mult(a, b):
    if isinstance(a, (list, tuple, str, bytes)) and isinstance(b, int):
        if len(a) * b > 25000:
            raise _StepLimitExceeded()
    elif isinstance(b, (list, tuple, str, bytes)) and isinstance(a, int):
        if len(b) * a > 25000:
            raise _StepLimitExceeded()
    return a * b


def _safe_builtin_pow(a, b, mod=None):
    if mod is not None:
        return pow(a, b, mod)
    return _safe_pow(a, b)


def _safe_exit(*args):
    raise SystemExit()


class _MockStdin:
    """Fast in-memory stdin replacement supporting input(), sys.stdin.readline(), read(), buffer, and iteration."""

    def __init__(self, text: str):
        self._io = io.StringIO(text)
        self.buffer = io.BytesIO(text.encode("utf-8", errors="ignore"))

    def readline(self, *args) -> str:
        return self._io.readline(*args)

    def read(self, *args) -> str:
        return self._io.read(*args)

    def readlines(self, *args) -> List[str]:
        return self._io.readlines(*args)

    def __iter__(self):
        return iter(self._io)

    def __next__(self):
        return next(self._io)


class _MockStdoutBuffer:
    def __init__(self, parent_io: io.StringIO):
        self._parent = parent_io

    def write(self, b_data: Union[bytes, bytearray, str]):
        if isinstance(b_data, (bytes, bytearray)):
            self._parent.write(b_data.decode("utf-8", errors="ignore"))
        else:
            self._parent.write(str(b_data))

    def flush(self):
        pass


class _MockStdout(io.StringIO):
    def __init__(self):
        super().__init__()
        self.buffer = _MockStdoutBuffer(self)


class _MockAtexit:
    def __init__(self):
        self.callbacks = []

    def register(self, func, *args, **kwargs):
        self.callbacks.append((func, args, kwargs))
        return func


class _MockThread:
    def __init__(self, target=None, args=(), kwargs=None, **extra):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def join(self, *args):
        pass


class _MockThreading:
    Thread = _MockThread

    @staticmethod
    def stack_size(*args):
        pass


class _MockSys:
    """Sandboxed sys module for candidate code execution."""

    def __init__(self, stdin_obj: _MockStdin, stdout_obj: _MockStdout):
        self.stdin = stdin_obj
        self.__stdin__ = stdin_obj
        self.stdout = stdout_obj
        self.__stdout__ = stdout_obj
        self.stderr = io.StringIO()
        self.__stderr__ = self.stderr
        self.maxsize = sys.maxsize
        self.version_info = sys.version_info
        self.argv = ["solution.py"]

    def setrecursionlimit(self, limit: int):
        pass

    def set_int_max_str_digits(self, digits: int):
        pass

    def exit(self, *args):
        raise SystemExit()


def _normalize_output(text: str) -> List[str]:
    return text.strip().split()


def _output_specificity(expected: str) -> str:
    tokens = _normalize_output(expected)
    if not tokens:
        return "low"
    if len(tokens) >= 2:
        return "high"
    tok = tokens[0].lower()
    trivial = {
        "0", "1", "-1", "2", "3", "4", "5", "6", "7", "8", "9",
        "yes", "no", "true", "false", "impossible", "possible",
    }
    if tok in trivial:
        return "low"
    if len(tok) >= 4:
        return "high"
    return "medium"


def _compare_outputs(actual: str, expected: str) -> float:
    act_str = actual.strip()
    exp_str = expected.strip()
    if not act_str or not exp_str:
        return 0.0

    if act_str == exp_str:
        return 1.0

    act_tokens = _normalize_output(act_str)
    exp_tokens = _normalize_output(exp_str)

    if not act_tokens or not exp_tokens:
        return 0.0

    if act_tokens == exp_tokens:
        return 1.0

    if [t.lower() for t in act_tokens] == [t.lower() for t in exp_tokens]:
        return 0.95

    if len(act_tokens) == len(exp_tokens):
        all_close = True
        for a_tok, e_tok in zip(act_tokens, exp_tokens):
            if a_tok.lower() == e_tok.lower():
                continue
            try:
                af = float(a_tok)
                ef = float(e_tok)
                if abs(af - ef) > 1e-4 * max(1.0, abs(ef)):
                    all_close = False
                    break
            except ValueError:
                all_close = False
                break
        if all_close:
            return 0.92

    return 0.0


class ExecutionVerifier:
    """
    Pre-compiles Python solutions with AST step guards and verifies candidates
    against query Example Input/Output pairs in < 50 microseconds.
    """

    def __init__(self, max_steps: int = 1200):
        self.max_steps = max_steps
        self.compiled_docs: Dict[str, Optional[Any]] = {}
        self.doc_has_yesno: Dict[str, bool] = {}
        self._transformer = _LoopStepGuardTransformer()

    def compile_corpus(self, corpus_texts: Dict[str, str]) -> int:
        compiled_count = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            for doc_id, code in corpus_texts.items():
                if doc_id in self.compiled_docs:
                    if self.compiled_docs[doc_id] is not None:
                        compiled_count += 1
                    continue
                if not code or "class Solution" in code:
                    self.compiled_docs[doc_id] = None
                    self.doc_has_yesno[doc_id] = False
                    continue
                c_low = code.lower()
                self.doc_has_yesno[doc_id] = ('"yes"' in c_low or "'yes'" in c_low or '"no"' in c_low or "'no'" in c_low)
                try:
                    tree = ast.parse(code)
                    guarded_tree = self._transformer.visit(tree)
                    ast.fix_missing_locations(guarded_tree)
                    code_obj = compile(guarded_tree, f"<doc_{doc_id}>", "exec")
                    self.compiled_docs[doc_id] = code_obj
                    compiled_count += 1
                except Exception:
                    self.compiled_docs[doc_id] = None
        return compiled_count

    def verify_single(self, doc_id: str, example_input: str, expected_output: str, step_budget: Optional[int] = None) -> float:
        code_obj = self.compiled_docs.get(doc_id)
        if code_obj is None:
            return 0.0

        budget = step_budget or self.max_steps
        steps_counter = [0]
        stdin_mock = _MockStdin(example_input)
        stdout_mock = _MockStdout()
        sys_mock = _MockSys(stdin_mock, stdout_mock)
        atexit_mock = _MockAtexit()

        def _custom_input(prompt: str = "") -> str:
            steps_counter[0] += 1
            if steps_counter[0] > budget:
                raise _StepLimitExceeded()
            line = stdin_mock.readline()
            if not line:
                raise EOFError()
            return line.rstrip("\r\n")

        def _custom_print(*args, sep=" ", end="\n", file=None, flush=False):
            stdout_mock.write(sep.join(str(a) for a in args) + end)
            if stdout_mock.tell() > 8000:
                raise _StepLimitExceeded()

        def _bounded_range_fn(*args):
            return _BoundedRange(range(*args), steps_counter, budget)

        def _bounded_iter_fn(iterable):
            return _BoundedIter(iterable, steps_counter, budget)

        class _SafeMath:
            def __getattr__(self, attr):
                val = getattr(math, attr)
                if attr in ("factorial", "comb", "perm"):
                    def _wrapped(*a):
                        if a and isinstance(a[0], int) and a[0] > 2000:
                            raise _StepLimitExceeded()
                        return val(*a)
                    return _wrapped
                return val

        class _SafeItertools:
            def __getattr__(self, attr):
                val = getattr(itertools, attr)
                if callable(val):
                    def _wrapped(*a, **kw):
                        return _BoundedIter(val(*a, **kw), steps_counter, budget)
                    return _wrapped
                return val

        class _SafeRe:
            def __getattr__(self, attr):
                val = getattr(re, attr)
                if attr in ("search", "match", "fullmatch", "findall", "finditer", "sub", "split", "compile"):
                    def _wrapped(pattern, *a, **kw):
                        pat_str = pattern if isinstance(pattern, str) else getattr(pattern, "pattern", "")
                        if any(bad in pat_str for bad in (")+", ")*", "}+", "}*", "++", "**", "*+", "+*")) or len(pat_str) > 80:
                            raise _StepLimitExceeded()
                        if a and isinstance(a[0], str) and len(a[0]) > 200:
                            raise _StepLimitExceeded()
                        return val(pattern, *a, **kw)
                    return _wrapped
                return val

        safe_math = _SafeMath()
        safe_itertools = _SafeItertools()
        safe_re = _SafeRe()

        _ALLOWED_IMPORTS = {
            "math", "heapq", "bisect", "collections", "itertools",
            "functools", "operator", "fractions", "re", "random",
            "string", "copy", "decimal", "typing", "io", "array",
        }

        def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            base = name.split(".")[0]
            if base == "threading":
                return _MockThreading
            if base == "sys":
                return sys_mock
            if base == "atexit":
                return atexit_mock
            if base == "math":
                return safe_math
            if base == "itertools":
                return safe_itertools
            if base == "re":
                return safe_re
            if base not in _ALLOWED_IMPORTS:
                raise _StepLimitExceeded()
            return __import__(name, globals, locals, fromlist, level)

        safe_builtins = {
            "abs": abs, "all": all, "any": any, "ascii": ascii, "bin": bin,
            "bool": bool, "bytearray": bytearray, "bytes": bytes, "chr": chr,
            "complex": complex, "dict": dict, "divmod": divmod, "enumerate": enumerate,
            "filter": filter, "float": float, "format": format, "frozenset": frozenset,
            "getattr": getattr, "hasattr": hasattr, "hash": hash, "hex": hex,
            "int": int, "isinstance": isinstance, "issubclass": issubclass,
            "iter": _bounded_iter_fn, "len": len, "list": list, "map": map, "max": max,
            "min": min, "next": next, "object": object, "oct": oct, "ord": ord,
            "pow": _safe_builtin_pow, "print": _custom_print, "input": _custom_input,
            "range": _bounded_range_fn, "repr": repr, "reversed": reversed, "round": round,
            "set": set, "slice": slice, "sorted": sorted, "str": str, "sum": sum,
            "tuple": tuple, "type": type, "zip": zip, "True": True, "False": False,
            "None": None, "ValueError": ValueError, "IndexError": IndexError,
            "KeyError": KeyError, "EOFError": EOFError, "ZeroDivisionError": ZeroDivisionError,
            "Exception": Exception, "RuntimeError": RuntimeError, "StopIteration": StopIteration,
            "SystemExit": SystemExit, "__import__": _safe_import,
        }

        env = {
            "__builtins__": safe_builtins,
            "__name__": "__main__",
            "_sb_steps": steps_counter,
            "_sb_max": budget,
            "_StepLimitExceeded": _StepLimitExceeded,
            "_safe_range": _bounded_range_fn,
            "_safe_iter": _bounded_iter_fn,
            "_safe_pow": _safe_pow,
            "_safe_lshift": _safe_lshift,
            "_safe_mult": _safe_mult,
            "_safe_exit": _safe_exit,
            "input": _custom_input,
            "print": _custom_print,
            "sys": sys_mock,
            "atexit": atexit_mock,
            "threading": _MockThreading,
            "math": safe_math,
            "heapq": heapq,
            "bisect": bisect,
            "collections": collections,
            "itertools": safe_itertools,
            "functools": functools,
            "operator": operator,
            "fractions": fractions,
            "re": safe_re,
            "random": random,
        }

        real_stdin, real_stdout = sys.stdin, sys.stdout
        try:
            sys.stdin = stdin_mock
            sys.stdout = stdout_mock
            exec(code_obj, env)
            for cb_func, cb_args, cb_kwargs in atexit_mock.callbacks:
                try:
                    cb_func(*cb_args, **cb_kwargs)
                except Exception:
                    pass
        except SystemExit:
            for cb_func, cb_args, cb_kwargs in atexit_mock.callbacks:
                try:
                    cb_func(*cb_args, **cb_kwargs)
                except Exception:
                    pass
        except Exception:
            return 0.0
        finally:
            sys.stdin = real_stdin
            sys.stdout = real_stdout

        actual_out = stdout_mock.getvalue()
        return _compare_outputs(actual_out, expected_output)

    def rerank_candidates(
        self,
        query: str,
        candidate_scores: List[Tuple[str, float]],
        verify_top_n: int = 2800,
    ) -> List[Tuple[str, float]]:
        io_pairs = extract_example_io(query)
        if not io_pairs or not candidate_scores:
            return candidate_scores

        ex_in, ex_out = io_pairs[0]
        specificity = _output_specificity(ex_out)
        exp_low = ex_out.lower()
        needs_yesno = ("yes" in exp_low or "no" in exp_low)

        if specificity == "high":
            max_check = min(verify_top_n, len(candidate_scores))
            boost = 10.0
        elif specificity == "medium":
            max_check = min(350, len(candidate_scores))
            boost = 4.0
        else:
            max_check = min(15, len(candidate_scores))
            boost = 0.25

        updated: List[Tuple[str, float]] = list(candidate_scores)

        for rank_idx in range(max_check):
            doc_id, base_score = updated[rank_idx]
            if needs_yesno and not self.doc_has_yesno.get(doc_id, False):
                continue

            if rank_idx < 30:
                budget = 1200
            elif rank_idx < 300:
                budget = 350
            else:
                budget = 140

            v_score = self.verify_single(doc_id, ex_in, ex_out, step_budget=budget)
            if v_score > 0:
                if len(io_pairs) > 1 and v_score >= 0.9:
                    v2 = self.verify_single(doc_id, io_pairs[1][0], io_pairs[1][1], step_budget=budget)
                    v_score = 0.6 * v_score + 0.4 * v2
                updated[rank_idx] = (doc_id, base_score + boost * v_score)
                if specificity == "high" and v_score >= 0.95:
                    break

        updated.sort(key=lambda x: x[1], reverse=True)
        return updated
