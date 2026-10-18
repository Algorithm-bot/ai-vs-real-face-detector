import nbformat

nb = nbformat.read("notebooks/kaggle_train.ipynb", as_version=4)
scope = {}
exec(nb.cells[1].source, scope)
scope["project_root"] = __import__("pathlib").Path.cwd()
exec(nb.cells[3].source, scope)
