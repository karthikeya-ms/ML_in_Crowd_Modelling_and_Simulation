# Exercise 4: Bifurcation Analysis

To run this exercise, anaconda is necessary. After installing it, run:

```sh conda
conda env create -f environment.yml
conda activate MLCMS_GroupI_EX4
```

## Adding Dependencies

To ensure that the `environment.yaml` file is cross-platform, we start with a minimalist environment and only add the dependencies we need.

To add a package, open the file itself and look for the `dependencies` key. Insert your new dependency in here with, at least, a major version specification (the first number of the version).

## Removing Environment

You might want to reinstall the environment from scratch, discoarding your previous environment and replacing it with a fresh environment from `environment.yaml`. To do this, run:

```sh conda
conda deactivate
conda env remove --name MLCMS_GroupI_EX4
````

## Creating a Notebook

Create notebooks in the notebooks folder. To import modules developed by the group, all users must place the following import at the beginning of their notebooks:

```python
import init_notebook
```

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