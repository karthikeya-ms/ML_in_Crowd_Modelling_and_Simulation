# Exercise 3: Representations of Data

To run this exercise, anaconda is necessary. After installing it, run:

```sh conda
conda env create -f environment.yml
conda activate MLCMS_GroupI
```

## Adding Dependencies

To ensure that the `environment.yaml` file is cross-platform, we start with a minimalist environment and only add the dependencies we need.

To add a package, open the file itself and look for the `dependencies` key. Insert your new dependency in here with, at least, a major version specification (the first number of the version).

## Updating Dependencies

Run this each time the `environment.yml` file is updated, whether by you or by someone else:

```sh conda
conda env update --file environment.yml --prune
```

The optional `--prune` flag removes dependencies that are no longer required, which helps keep the environment clean.

## Creating a Notebook

Create notebooks in the notebooks folder. To import modules developed by the group, all users must place the following import at the beginning of their notebooks:

```python
import init_notebook
```