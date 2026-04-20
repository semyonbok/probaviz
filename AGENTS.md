## Python environment

All Python commands must be executed inside the `probaviz` conda environment.

Never use the system Python or global environment.

### How to run commands

Always use `conda run -n probaviz` when executing Python-related commands, including:

- running scripts:
    conda run -n probaviz python script.py

- running tests:
    conda run -n probaviz pytest

- installing packages:
    conda run -n probaviz pip install <package>

- interactive checks:
    conda run -n probaviz python -c "<code>"

Do not use:
- `python`
- `pip`
- `pytest`

without the `conda run -n probaviz` prefix.