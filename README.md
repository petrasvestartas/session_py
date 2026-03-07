# session_py

Python NURBS geometry library.

## Setup

```bash
# Create environment
uv venv uvsession --python 3.11
source uvsession/bin/activate

# Install
uv pip install -e .
```

## Test

```bash
source uvsession/bin/activate
pytest -v
```

## Publish

```bash
    Edit pyproject.toml version: "0.1.0" → "0.1.1"
    cd session_py
    git add pyproject.toml .github/
    git commit -m "setup PyPI publish workflow"
    git tag v0.1.0
    git push && git push --tags
```