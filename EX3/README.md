# Exercise 3: Representations of Data

To run this exercise, anaconda is necessary. After installing it, run:

```sh conda
conda env create -f environment.yml
conda activate MLCMS_GroupI
```

## Adding Dependencies

To ensure that the `environment.yaml` file is cross-platform, we start with a minimalist environment and only add the dependencies we need.

To add a package, open the file itself and look for the `dependencies` key. Insert your new dependency in here with, at least, a major version specification (the first number of the version).

## Removing Environment

You might want to reinstall the environment from scratch. To do this, run:

```sh conda
conda deactivate
conda env remove --name MLCMS_GroupI
````

## Creating a Notebook

Create notebooks in the notebooks folder. To import modules developed by the group, all users must place the following import at the beginning of their notebooks:

```python
import init_notebook
```

## Pushing notebooks to Git

Use the tool `nb-clean` to remove the output and metadata from the notebooks before pushing them to git. The aim is to make the git history easier to read and avoid unnecessary merge conflicts.

### Using automated script

Use the script `clean_notebooks.sh` to clean all notebooks in the repository.

```sh
./clean_notebooks.sh
```
### Manually

```sh
# run this to install nb-clean
pip install nb-clean

# run this before committing changes to a notebook
nb-clean notebooks/my_notebook.ipynb
```