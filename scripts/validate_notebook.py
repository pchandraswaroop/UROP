"""
validate_notebook.py — Full Structural and Syntax Validator for 02_model_training.ipynb
"""

import ast
import json
import sys
from pathlib import Path

nb_path = Path("notebooks/02_model_training.ipynb")
assert nb_path.exists(), f"Notebook not found: {nb_path}"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Loaded notebook: {nb_path.name}")
print(f"Total cells: {len(nb['cells'])}")
print(f"Format: v{nb['nbformat']}.{nb['nbformat_minor']}")

code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
markdown_cells = [c for c in nb["cells"] if c["cell_type"] == "markdown"]

print(f"Markdown cells: {len(markdown_cells)}")
print(f"Code cells:     {len(code_cells)}")

# 1. Syntax check every code cell with Python's ast parser
syntax_errors = []
for idx, cell in enumerate(code_cells):
    src = "".join(cell["source"])
    try:
        ast.parse(src)
    except SyntaxError as e:
        syntax_errors.append((idx + 1, str(e)))

if syntax_errors:
    print(f"\n[FAILED] Found {len(syntax_errors)} syntax errors:")
    for c_num, err in syntax_errors:
        print(f" - Code Cell {c_num}: {err}")
    sys.exit(1)
else:
    print("\n[SUCCESS] All code cells passed Python AST syntax validation cleanly.")

# 2. Check for common anti-patterns
forbidden_patterns = [
    ("getattr(locals()", "Invalid dict attribute lookup"),
    (".join(parsed)", "Unconverted integer list join"),
]

pattern_errors = []
for idx, cell in enumerate(code_cells):
    src = "".join(cell["source"])
    for pat, desc in forbidden_patterns:
        if pat in src:
            pattern_errors.append((idx + 1, pat, desc))

if pattern_errors:
    print(f"\n[FAILED] Found {len(pattern_errors)} anti-patterns:")
    for c_num, pat, desc in pattern_errors:
        print(f" - Code Cell {c_num}: '{pat}' ({desc})")
    sys.exit(1)
else:
    print("[SUCCESS] No anti-patterns found in any code cell.")

print("\nNotebook validation 100% complete: Valid JSON, valid Python syntax, pristine structure.")
