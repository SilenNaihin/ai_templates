import argparse
import nbformat as nbf
from typing import Optional
from aitemplates.cells import get_cell1_content, get_cell2_content, get_cell3_content, get_cell4_content, get_cell5_content


def create_notebook(
    filename: str, db: Optional[bool] = False, asnc: Optional[bool] = False, func: Optional[bool] = False
):
    """Creates a notebook using nbformat and populate the first 3 cells

    Args:
        filename (str): The name of the notebook
        db (bool, optional): Whether to use a vector database. Defaults to False.
        asnc (bool, optional): Whether to use async chat completion. Defaults to False.
        func (bool, optional): Whether to use function completion. Defaults to False.

    Returns:
        None
    """
    

    nb = nbf.v4.new_notebook()

    cell1_content = get_cell1_content(db, asnc, func)
    
    cell1 = nbf.v4.new_code_cell(cell1_content)
    
    cell2_content = get_cell2_content(db, asnc, func)
    
    cell2 = nbf.v4.new_code_cell(cell2_content)
    
    cell3_content = get_cell3_content(db, asnc, func)
    
    cell3 = nbf.v4.new_code_cell(cell3_content)
    
    cell4_content = get_cell4_content(db, asnc, func)
    
    cell4 = nbf.v4.new_code_cell(cell4_content)
    
    cell5_content = get_cell5_content(db, asnc, func)
    
    cell5 = nbf.v4.new_code_cell(cell5_content)

    # Add cells to notebook
    nb.cells = [cell1, cell2, cell3, cell4, cell5]

    with open(filename, "w") as f:
        nbf.write(nb, f)


def main():
    parser = argparse.ArgumentParser(description="Generate a new Jupyter notebook")
    parser.add_argument("filename", help="Name of the notebook to create")
    parser.add_argument(
        "-db", action="store_true", help="With or without a vector database"
    )
    parser.add_argument(
        "-asnc",
        action="store_true",
        help="Async chat completion instead of a regular one",
    )
    parser.add_argument(
        "-func",
        action="store_true",
        help="Include openai function templates in the notebook",
    )
    args = parser.parse_args()
    
    if args.asnc and args.func:
        print("openai async method create does not have a function parameter")
        return

    create_notebook(f"{args.filename}.ipynb", args.db, args.asnc, args.func)


if __name__ == "__main__":
    main()
