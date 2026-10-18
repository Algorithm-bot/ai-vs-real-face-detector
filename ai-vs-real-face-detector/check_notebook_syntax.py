import ast
import nbformat

# Check the generated notebook for syntax errors
notebook_path = "notebooks/kaggle_train.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

syntax_errors = []
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code':
        try:
            ast.parse(cell.source)
        except SyntaxError as e:
            syntax_errors.append({
                'cell_index': i,
                'cell_id': getattr(cell, 'id', 'unknown'),
                'line': e.lineno,
                'offset': e.offset,
                'message': str(e),
                'source': cell.source[:200] + "..." if len(cell.source) > 200 else cell.source
            })

if syntax_errors:
    print("SYNTAX ERRORS FOUND:")
    for err in syntax_errors:
        print(f"  Cell {err['cell_index']} (ID: {err['cell_id']}): Line {err['line']}")
        print(f"    {err['message']}")
        print(f"    Source: {err['source']}")
else:
    print("NO SYNTAX ERRORS FOUND")