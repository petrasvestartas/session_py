from __future__ import annotations

import inspect
import json
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, List


@dataclass
class MiniTestResult:
    """Result of a single mini test run.

    Attributes
    ----------
    test_name : str
        Name of the test.
    passed : bool
        Whether the test passed.
    time_ms : float
        Execution time of the non-assertion code in milliseconds.
    line : int
        Line number of the test function definition.
    code : str
        Timed body code of the test function.
    checks : list
        Per-assertion metadata recorded by ``check_equal``.
    failures : list
        Failure records for unexpected exceptions or assertion errors.
    """

    group: str
    test_name: str
    passed: bool
    time_ms: float
    line: int
    code: str
    checks: list
    failures: list


 # Global registry of tests in this process
_REGISTERED_TESTS: List[tuple[str, str, Callable[[], None]]] = []
_CURRENT_CHECKS: list[dict[str, Any]] | None = None
_CURRENT_ASSERTION_TIME: float | None = None


class MiniTestAssertionError(AssertionError):
    """Custom assertion error for the mini framework."""


def _record_check(passed: bool, label: str | None = None) -> None:
    """Record metadata for a single ``check_equal`` call.

    Parameters
    ----------
    passed : bool
        Whether the check passed.
    label : str, optional
        Optional label for the check (used only in error messages).
    """

    global _CURRENT_CHECKS
    if _CURRENT_CHECKS is None:
        return

    frame = inspect.currentframe()
    if frame is None:
        return

    # Walk up the stack until we leave this module, so we record the
    # actual test call site (MINI_CHECK/check_equal) in the test file
    # rather than the helper functions in mini_test.py.
    frame = frame.f_back  # start from the caller of _record_check
    while frame is not None and frame.f_code.co_filename == __file__:
        frame = frame.f_back

    if frame is None:
        return

    info = inspect.getframeinfo(frame)
    source_line = info.code_context[0].rstrip("\n") if info.code_context else ""
    code_line = source_line.lstrip() if source_line else ""

    for name in ("MINI_CHECK(", "check_equal("):
        idx = code_line.find(name)
        if idx != -1:
            args_part = code_line[idx + len(name) :]
            depth = 0
            expr_chars: list[str] = []
            for ch in args_part:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    if depth == 0:
                        break
                    depth -= 1
                expr_chars.append(ch)
            args_str = "".join(expr_chars).strip()
            if args_str:
                # Only split for check_equal(actual, expected) - use top-level comma split
                # For MINI_CHECK(expr), just use the whole expression
                if name == "check_equal(":
                    # Split at top-level comma (not inside parentheses)
                    parts = []
                    current = []
                    paren_depth = 0
                    for ch in args_str:
                        if ch == '(':
                            paren_depth += 1
                        elif ch == ')':
                            paren_depth -= 1
                        elif ch == ',' and paren_depth == 0:
                            parts.append("".join(current).strip())
                            current = []
                            continue
                        current.append(ch)
                    parts.append("".join(current).strip())
                    if len(parts) >= 2:
                        code_line = f"{parts[0]} == {parts[1]}"
                    else:
                        code_line = args_str
                else:
                    code_line = args_str
            break

    _CURRENT_CHECKS.append(
        {
            "line": info.lineno,
            "code_line": code_line or None,
            "passed": passed,
        }
    )


def _extract_timed_code(source_lines: list[str]) -> str:
    """Extract the timed body of a test function, excluding MINI_CHECK calls.

    Parameters
    ----------
    source_lines : list of str
        Source lines of the test function, including decorators.

    Returns
    -------
    str
        Concatenated body lines excluding ``check_equal`` and ``MINI_CHECK`` calls.
    """

    body_lines: list[str] = []
    in_body = False
    in_check = False  # Track if we're inside a multi-line MINI_CHECK
    paren_depth = 0   # Track parenthesis depth for multi-line calls

    for line in source_lines:
        stripped = line.lstrip()

        if not in_body:
            # Skip decorators and the def line
            if stripped.startswith("def "):
                in_body = True
            continue

        # Check if this line starts a MINI_CHECK or check_equal call
        if "check_equal(" in stripped or "MINI_CHECK(" in stripped:
            in_check = True
            paren_depth = 0
            # Count parentheses to track multi-line calls
            for ch in line:
                if ch == '(':
                    paren_depth += 1
                elif ch == ')':
                    paren_depth -= 1
            # If parentheses are balanced, we're done with this check
            if paren_depth <= 0:
                in_check = False
            continue

        # If we're in a multi-line check, skip and track parentheses
        if in_check:
            for ch in line:
                if ch == '(':
                    paren_depth += 1
                elif ch == ')':
                    paren_depth -= 1
            if paren_depth <= 0:
                in_check = False
            continue

        body_lines.append(line)

    # Remove empty if-blocks (entire body was MINI_CHECK calls)
    filtered: list[str] = []
    idx = 0
    while idx < len(body_lines):
        line = body_lines[idx]
        stripped = line.lstrip()
        if stripped.startswith("if ") and stripped.rstrip("\n").endswith(":"):
            indent = len(line) - len(stripped)
            j = idx + 1
            while j < len(body_lines) and body_lines[j].strip() == "":
                j += 1
            if j >= len(body_lines) or (len(body_lines[j]) - len(body_lines[j].lstrip())) <= indent:
                idx += 1
                continue
        filtered.append(line)
        idx += 1
    body_lines = filtered

    # Strip trailing newlines but keep original indentation
    return "".join(body_lines).rstrip("\n")


def check_equal(actual: Any, expected: Any, label: str | None = None) -> None:
    """Assert that two values are equal.

    Parameters
    ----------
    actual : Any
        Actual value.
    expected : Any
        Expected value.
    label : str, optional
        Optional label used in the failure message.

    Raises
    ------
    MiniTestAssertionError
        If ``actual`` and ``expected`` differ.
    """

    global _CURRENT_ASSERTION_TIME
    if _CURRENT_ASSERTION_TIME is None:
        _CURRENT_ASSERTION_TIME = 0.0

    start_t = time.perf_counter()
    try:
        passed = actual == expected
        _record_check(passed, label)

        if not passed:
            if label:
                msg = f"{label}: expected {expected!r}, got {actual!r}"
            else:
                msg = f"expected {expected!r}, got {actual!r}"
            raise MiniTestAssertionError(msg)
    finally:
        if _CURRENT_ASSERTION_TIME is not None:
            _CURRENT_ASSERTION_TIME += time.perf_counter() - start_t


def mini_test(class_name: str, test_name: str) -> Callable[[Callable[[], None]], Callable[[], None]]:
    """Decorator to register a mini test function.

    Parameters
    ----------
    class_name : str
        Logical group name for the test (for example ``"Point"``).
    test_name : str
        Human-readable name of the test (for example ``"constructor"``).

    Returns
    -------
    Callable
        Decorator that registers the wrapped function in the mini test registry.
    """

    def decorator(func: Callable[[], None]) -> Callable[[], None]:
        _REGISTERED_TESTS.append((class_name, test_name, func))
        return func

    return decorator


def _default_output_path(language: str, test_file: Path) -> Path:
    # __file__ is session_py/src/session_py/mini_test.py
    repo_root = Path(__file__).resolve().parents[3]
    if language == "python":
        subdir = "session_py"
    elif language == "cpp":
        subdir = "session_cpp"
    elif language == "rust":
        subdir = "session_rust"
    else:
        subdir = language
    return repo_root / "session_tests" / subdir / f"{test_file.stem}.json"


def run_all(language: str = "python") -> None:
    """Run all registered mini tests and write JSON results.

    Parameters
    ----------
    language : str, optional
        Logical language key used to choose the output subfolder
        (for example ``"python"``, ``"cpp"``, ``"rust"``).

    Notes
    -----
    Tests are grouped by ``class_name`` and written as one JSON file per
    test source file into ``session_tests/<subdir>/<test_file>.json``.
    """

    # Group tests by class
    tests_by_class: dict[str, list[tuple[str, Callable[[], None]]]] = {}
    for class_name, test_name, func in _REGISTERED_TESTS:
        tests_by_class.setdefault(class_name, []).append((test_name, func))

    total_passed = 0
    total_tests = 0
    failed_tests: list[tuple[str, str, list[dict[str, Any]]]] = []

    for class_name, tests in tests_by_class.items():
        results: list[dict[str, Any]] = []
        test_file: Path | None = None

        for test_name, func in tests:
            global _CURRENT_CHECKS, _CURRENT_ASSERTION_TIME
            _CURRENT_CHECKS = []
            _CURRENT_ASSERTION_TIME = 0.0

            start = time.perf_counter()
            passed = True
            failures: list[dict[str, Any]] = []

            try:
                func()
            except AssertionError:
                passed = False
                exc_type, exc, tb = sys.exc_info()
                # Last traceback frame is where the assertion failed
                if tb is not None:
                    last_frame = traceback.extract_tb(tb)[-1]
                    failures.append(
                        {
                            "file": str(last_frame.filename),
                            "line": last_frame.lineno,
                            "code_line": last_frame.line,
                        }
                    )
            except Exception as e:  # Other unexpected errors
                passed = False
                exc_type, exc, tb = sys.exc_info()
                msg = f"{type(e).__name__}: {e}"
                if tb is not None:
                    last_frame = traceback.extract_tb(tb)[-1]
                    failures.append(
                        {
                            "file": str(last_frame.filename),
                            "line": last_frame.lineno,
                            "code_line": last_frame.line,
                            "error": msg,
                        }
                    )
                else:
                    failures.append({"error": msg})
            finally:
                checks = _CURRENT_CHECKS or []
                assertion_time = _CURRENT_ASSERTION_TIME or 0.0
                _CURRENT_CHECKS = None
                _CURRENT_ASSERTION_TIME = None

            elapsed = time.perf_counter() - start
            effective_elapsed = max(0.0, elapsed - assertion_time)

            # Capture source of the test function and keep only the timed body code
            try:
                source_lines, start_line = inspect.getsourcelines(func)
                code = _extract_timed_code(source_lines)
            except OSError:
                code = func.__name__
                start_line = -1

            source_file = inspect.getsourcefile(func) or func.__code__.co_filename
            if test_file is None:
                test_file = Path(source_file)

            total_tests += 1
            if passed:
                total_passed += 1
            else:
                failed_tests.append((class_name, test_name, failures, source_file, start_line))

            result = MiniTestResult(
                group=class_name,
                test_name=test_name,
                passed=passed,
                time_ms=round(effective_elapsed * 1e3, 3),
                line=start_line,
                code=code,
                checks=checks,
                failures=failures,
            )
            results.append(asdict(result))

        if test_file is None:
            continue

        out_path = _default_output_path(language, test_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    if failed_tests:
        print(f"\n[py-minitest] FAILURES:", file=sys.stderr)
        for group, name, fails, src_file, src_line in failed_tests:
            loc = ""
            if fails and "file" in fails[0] and "line" in fails[0]:
                loc = f"  {fails[0]['file']}:{fails[0]['line']}"
            elif src_file and src_line and src_line > 0:
                loc = f"  {src_file}:{src_line}"
            print(f"  FAIL {group}::{name}{loc}", file=sys.stderr)
            for fl in fails:
                err = fl.get("error") or fl.get("code_line") or "assertion failed"
                print(f"       {err}", file=sys.stderr)
        print(f"\n[py-minitest] {total_passed}/{total_tests} passed, {len(failed_tests)} failed", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"[py-minitest] {total_passed}/{total_tests} passed")


def MINI_TEST(class_name: str, test_name: str) -> Callable[[Callable[[], None]], Callable[[], None]]:
    """Uppercase alias for :func:`mini_test` for API parity with C++.

    This allows using ``@MINI_TEST("Point", "Constructor")`` in Python
    test files while reusing the same registration implementation.
    """

    return mini_test(class_name, test_name)


def MINI_CHECK(expr: Any, label: str | None = None) -> None:
    """Assert that an expression is truthy (C++-style ``MINI_CHECK``).

    Parameters
    ----------
    expr : Any
        Boolean expression to check.
    label : str, optional
        Optional label used in the failure message.

    Raises
    ------
    MiniTestAssertionError
        If ``expr`` is falsy.
    """

    global _CURRENT_ASSERTION_TIME
    if _CURRENT_ASSERTION_TIME is None:
        _CURRENT_ASSERTION_TIME = 0.0

    start_t = time.perf_counter()
    try:
        passed = bool(expr)
        _record_check(passed, label)

        if not passed:
            if label:
                msg = f"{label}: expression is not true"
            else:
                msg = "expression is not true"
            raise MiniTestAssertionError(msg)
    finally:
        if _CURRENT_ASSERTION_TIME is not None:
            _CURRENT_ASSERTION_TIME += time.perf_counter() - start_t


__all__ = [
    "mini_test",
    "run_all",
    "MiniTestResult",
    "MiniTestAssertionError",
    "check_equal",
    "MINI_TEST",
    "MINI_CHECK",
]
