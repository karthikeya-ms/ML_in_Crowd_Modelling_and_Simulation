# Final Project: Present and Implement Neural Network Gaussian Processes

To run this project, poetry is necessary. After installing it, run:

```sh 
poetry install
poetry shell
```
These first command installs the needed dependencies for the project. The second activates a virtual environment to run the project in

## Adding Dependencies

To ensure that the `pyproject.toml` file is cross-platform, we start with a minimalist environment and only add the dependencies we need.

To add a package, run the following command:

```sh
poetry add <package>
```

## Deactivate Environment

To deactivate the environment, simmply run:

```sh conda
exit
````

## Cleaning Notebooks for Git

Use the tool `nb-clean` to remove the output and metadata from the notebooks before pushing them to git. The aim is to make the git history easier to read and avoid unnecessary merge conflicts.

Install the utility with:

```sh
pip install nb-clean
````

### Using Bash script

Use the script `clean_notebooks.sh` to clean all notebooks in the repository. Requires `nb-clean` in path.

### Manually

```sh
# run this before committing changes to a notebook
nb-clean clean notebooks/my_notebook.ipynb
```

### Windows

On Windows, the utility `nb-clean` may be installed to an obscure path. To find the installation path, run:

```pwsh
# This will show the path to the nb-clean utility. Run the CLI file under this path
pip show nb-clean
````

An example filepath is `C:\Users\f0r\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\site-packages\nb_clean\`. In that case:

```pwsh
$my_path="C:\Users\f0r\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\site-packages\nb_clean"
python.exe "$my_path\__main__.py" clean \path\to\my_notebook.ipynb
```
Feel free to modify the `clean_notebooks.ps1` script to your needs and use that instead.