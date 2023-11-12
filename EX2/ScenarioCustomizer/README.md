# Scenario Customizer

This small project is meant to solve task 3 of the exercise sheet. It provides an installable python project, that comes bundled with a corresponding command, to customize scenario files produced by the Vadere simulation software.

## Installation

To install the project start by moving to the root of the this subproject, the `ScenarioCustomizer` directory. Then run the following command to install the project:

 - Statically:
```bash
pip install .
```

 - Or in editable mode, wich reflects the changes made to project files in the instalation live:
```bash
pip install -e .
```

After either of these steps the project has been successfully installed.

## Running

After installing the project you can run it with:
```bash
python path/to/scenario_customizer/main.py
```

However, this depends on your current directory which makes the use of the script a bit awkward with the rest of the repository code and scenarios. To solve this issue the installation previously performed also provides a command version. By running the following command from anywhere, while your virtual environment is active, you can custumize scenarios files through the command line!

```bash
customize-scenario
```

## Uninstall

To remove the package from your system run the following command anywhere:

```bash
pip uninstall scenario-customizer
```

