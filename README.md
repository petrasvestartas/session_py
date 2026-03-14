# session_py

Python geometry kernel. Published to [PyPI](https://pypi.org/project/session-py/) automatically on every push to main.

## Install

```bash
pip install session-py
```

## Development Setup

```bash
uv venv uvsession --python 3.11
source uvsession/bin/activate   # macOS/Linux
source uvsession/Scripts/activate  # Windows (Git Bash)

uv pip install -e .
```

## Test

```bash
pytest -v
```
