# session_py Agent Guide

Python geometry kernel. One of three parallel implementations (`session_cpp`, `session_py`, `session_rust`) sharing protobuf schemas in `session_proto` and a Vue test viewer in `session_tests`.

## Goal

- Keep the Python API identical to C++ and Rust (names, signatures, test logic, line counts).
- Add type hints to public APIs.
- Keep docstrings NumPy-style and accurate.
- Keep every class covered by minitests.

## Scope

These instructions apply to the whole `session_py` repository. Preserve unrelated working-tree changes and keep patches focused on the requested task.

## Cross-Language Parity

- C++ is ground truth. Port to Python without renaming methods, parameters, or test variables.
- A change to a public method here is incomplete until the same change exists in `session_cpp` and `session_rust`. If the task is Python-only, say so explicitly in the summary.
- Serialization names are fixed across languages: `file_json_dump` / `file_json_load` / `file_json_dumps` / `file_json_loads`, `pb_dump` / `pb_load` / `pb_dumps` / `pb_loads`, `to_proto` / `from_proto`.
- JSON fields are written in alphabetical order in all three languages.
- Method order in a class: constructors → accessors → mutators (`*_self`) → operators → utilities → serialization → `__str__` / `__repr__`.

## Typing

- Support Python 3.10 as declared in `pyproject.toml`.
- Use `typing.Union[...]` / `typing.Optional[...]` in annotations, not PEP 604 `X | Y`; the repo is uniformly `Union`/`Optional` today. The `|` notation may appear in docstrings and prose.
- Prefer builtin generics in new or touched code: `list[tuple[float, float]]`, not `List[Tuple[float, float]]`. Do not sweep-convert existing `List[...]` in files the task does not otherwise touch.
- Annotate what the runtime already accepts. Do not add conversions solely to satisfy a narrow annotation.
- Coordinate and component parameters are `float`, never `Union[float, str]`. Passing `int` is valid for a `float` annotation under Python's numeric typing rules; that does not justify widening the type or adding a runtime cast.
- Model fixed-size coordinate data as two or three components where the type system can express it. Do not broaden a fixed-size alias to an arbitrary-length collection to accommodate a lower-level helper.
- `Point`, `Vector`, `Plane`, and friends are iterable and support indexing/unpacking. Pass them directly where an API consumes coordinate iterables; avoid needless `list(...)` allocation.
- Low-level numeric modules (`matrix.py`, `closest.py`, `intersection.py` internals) should be typed against raw tuples/lists and primitives and stay unaware of the geometry classes where they already are.
- Methods returning an instance of their own class use a quoted class name (`-> "Plane"`), because the class is not yet bound while its body executes. `typing.Self` is 3.11+ and the package supports 3.10 without `typing_extensions`, so adopt `Self` only when the floor moves to 3.11.
- Use overloads when a public method's return type is controlled by its arguments, rather than falling back to `Any`.
- Avoid broad `Any`. Prefer fixing the annotation, protocol, or overload over `cast`.
- Do not import heavy or optional dependencies (`numpy`, `numba`) only for typing — use `TYPE_CHECKING`.

## Runtime-Behavior Preservation

- Refactors and typing work must not change valid-input behavior, return types, ordering, orientation, side effects, or error behavior.
- Before changing an implementation, compare old and new control flow and identify any equivalence that depends on data-structure invariants.
- Fix types at API boundaries rather than changing runtime values inside algorithms.
- When a typing or modernization change requires a small, non-obvious change to a function body, add one short comment explaining which contract or invariant it preserves.
- Keep diffs narrow. Do not fold speculative architecture changes into typing or documentation patches.
- Tolerances come from `.tolerance` (`TOLERANCE`, `PI`) — never hardcode epsilon literals.
- Use names that reflect cardinality: a returned collection is `points`, not `point`.
- Keep a short expression on one line when it stays within the formatter's line length and splitting does not improve readability.

## Imports and Style

- One import per line.
- `TOLERANCE` / `PI` are imported from `.tolerance` at the top of the file.
- Production code uses relative imports inside the package (`from .point import Point`); test files use flat imports (`from session_py import Point`).
- Geometry imports in test files go *inside* each test function, not at module top. Exception in production code: `from session_py.intersection import line_line`.
- New classes must be exported from `src/session_py/__init__.py`.
- No print statements in library code. No decorative or restating comments — comment only what is non-obvious.
- Format with `../bash/format.sh --py` (the shared deterministic formatter — one coordinate object per line in arrays); `test.sh` additionally runs `python3 -m black src examples`.

## Tests and Validation

- Run the smallest relevant selection first, then broaden:
  - `../bash/quicktest.sh <class> --py` — one class
  - `../bash/minitest.sh --py --no-web` — all Python minitests
  - `../bash/minitest.sh` — all three languages plus the viewer on `localhost:8769`
- CI imports every `src/session_py/*_test.py` and runs `mini_test.run_all(language="python")` on Ubuntu, macOS (ARM + Intel), and Windows with Python 3.11. Keep tests platform-agnostic — build paths with `pathlib`, never hardcode separators.
- Always run `git diff --check` on changed patches.
- Minitest conventions (identical test names and logic across languages, one test per API method, operators tested inside the constructor test, `file_json_*` and `to_proto`/`from_proto` tests for every class, one object per line in collections) are documented in the parent repo's `/test-rules` command — follow them exactly.
- The constructor test covers: default and parameterized construction, `[]`, `==`, `!=`, `str`, `repr`, in-place and copy operators, and `duplicate()` with a fresh GUID.
- Test artifacts write to `serialization/` and `session_tests/session_py/`; regenerated artifacts are untracked — do not commit them.
- Prefer behavioral assertions over assertions about implementation metadata.

## Documentation

- Public classes, methods, and module-level functions get NumPy-style docstrings with the existing section structure: `Parameters`, `Returns`, `Raises`, `Notes`, `Examples`, `References`, `See Also`.
- Keep parameter types in the `Parameters` section — this repo builds docs with Sphinx (`doc.sh` / `doc.bat`, `sphinx-compas-theme`), not mkdocstrings.
- Write types in `Returns` with canonical Python syntax: `list[float]`, `tuple[float, float]`, `Sequence[float]`. Do not use prose forms such as `list of list` or `[float, float, float]`.
- Document a tuple return as one entry matching the return annotation, describing the elements and their order — not as one entry per element.
- Omit the `Returns` section when the function returns `None`; remove empty `Returns` and `Examples` sections.
- Document public dunder behavior in the class docstring when it is part of the user-facing API, covering ordinary, reflected, and in-place arithmetic variants as applicable.
- Add a short example for each public classmethod that is an alternative constructor. Do not document deserialization hooks in that overview.
- Document non-obvious setter side effects — in particular when setting one axis of a `Plane` or `Xform` normalizes it or recomputes another axis to preserve orthonormality.
- Keep examples valid as doctests where they are meant to execute.

## Public API

- Do not expose implementation-only payload, encoder, or helper types without a clear public use case.
- Preserve existing convenience APIs during refactors unless the task explicitly includes a deprecation or breaking-change plan.
- Removing a parameter or method is part of a task only when explicitly requested; then update implementation, docstring, tests, and the C++/Rust counterparts together rather than leaving hidden aliases.

## Git

- Never add Claude or any AI as git author, contributor, or co-author.
- Push all submodules with `../bash/git_push.sh "message"`.
- Check CI with `gh run list --limit 5`; inspect failures with `gh run view <id> --log-failed`.
