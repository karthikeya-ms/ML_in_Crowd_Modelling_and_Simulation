# Exercise 3: Representations of Data

To run this exercise anaconda is necessary. After installing it run:

```sh conda
conda env create -f environment.yml
conda activate MLCMS_GroupI
```

## Adding Dependencies

So that the `environment.yaml` file is cross platform new dependencies must be added manually. To do this open the file itself and look for the `dependencies` key. Insert your new dependency in here with, at least, a major version specification (the first number of the version).

## Creating a Notebook

Create notebooks in the notebooks folder. To import modules devloped by the group all users must place the following import at the beginning of their notebooks:

```python
import init_notebook
```