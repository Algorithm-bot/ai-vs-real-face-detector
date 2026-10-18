import nbformat
import ast

notebook_path = "notebooks/kaggle_train.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Cell index 8 is the training cell (0-indexed)
cell = nb.cells[8]
if cell.cell_type != "code":
    print(f"Cell {8} is not a code cell")
else:
    source = cell.source
    print("Source of cell 8:")
    print(source)
    print("---")
    try:
        tree = ast.parse(source)
        print("AST parse succeeded")
    except SyntaxError as e:
        print(f"AST parse failed: {e}")
        print(f"Line {e.lineno}: {e.text}")