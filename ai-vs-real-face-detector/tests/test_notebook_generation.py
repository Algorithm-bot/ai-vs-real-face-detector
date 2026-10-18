import ast
import nbformat
import os

def test_notebook_generation():
    notebook_path = os.path.join("notebooks", "kaggle_train.ipynb")
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            source = cell.source
            # Check for forbidden strings
            if "api.stat" in source or "list_repo_paths" in source:
                raise AssertionError(f"Cell {i} contains forbidden string: {source[:200]}...")
            # Try to parse the source to ensure it's valid Python
            try:
                ast.parse(source)
            except SyntaxError as e:
                raise AssertionError(f"Cell {i} has a syntax error: {e}") from e

if __name__ == "__main__":
    test_notebook_generation()
    print("Test passed")