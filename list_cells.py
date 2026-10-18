import nbformat

notebook_path = "notebooks/kaggle_train.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

for i, cell in enumerate(nb.cells):
    print(f"Cell {i}: {cell.cell_type}")
    if cell.cell_type == "code":
        # Print first 200 chars of source
        source = cell.source[:200]
        print(f"  Source: {source}")
    else:
        source = cell.source[:200]
        print(f"  Source: {source}")
    print()