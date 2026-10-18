import nbformat
import os
import subprocess
import sys

def test_pyflakes_on_notebook():
    # Path to the generated notebook
    notebook_path = os.path.join("notebooks", "kaggle_train.ipynb")
    if not os.path.exists(notebook_path):
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    # Read the notebook
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Concatenate all code cells
    code_cells = []
    for cell in nb.cells:
        if cell.cell_type == "code":
            code_cells.append(cell.source)
    full_code = "\n".join(code_cells)

    # Try to import pyflakes, install if necessary
    try:
        import pyflakes.api as pyflakes_api
    except ImportError:
        # Install pyflakes
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyflakes"])
        import pyflakes.api as pyflakes_api

    # Check the code
    num_warnings = pyflakes_api.check(full_code, '<string>')
    if num_warnings > 0:
        raise AssertionError(f"Pyflakes found {num_warnings} warning(s)")
    else:
        print("Pyflakes check passed: no warnings.")

if __name__ == "__main__":
    test_pyflakes_on_notebook()