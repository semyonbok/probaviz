## Local environment
All Python-related commands must run inside the `probaviz` Conda environment.
Do not use the system Python or global environment.

### How to run commands
Prefer running commands with:

`conda run -n probaviz <command>`

Examples:

* Run scripts:
  `conda run -n probaviz python script.py`
* Run tests:
  `conda run -n probaviz pytest`
* One-off code execution:
  `conda run -n probaviz python -c "print('hello')"`

### Package installation
* Prefer installing packages with Conda:
  `conda install -n probaviz <package>`

* If a package is not available via Conda, use pip as a fallback:
  `conda run -n probaviz pip install <package>`

### Notes
* Avoid using `python`, `pip`, or `pytest` without the `conda run -n probaviz` prefix.
* For interactive development (e.g. local shells or IDEs), activating the environment with
  `conda activate probaviz` is acceptable.
* Be careful with shell quoting when using `python -c`, especially for complex code.

## Cloud / Codex environments
When running in Codex web or other cloud agents:

* Do **not** assume Conda is available.
* If Conda is available, prefer creating/updating the environment from `environment.yml`.
* If Conda is unavailable, install the runtime dependencies into the active Python environment with pip:

  `python -m pip install numpy pandas scikit-learn==1.8.0 matplotlib protobuf streamlit`

* If tests or notebooks are needed, also install the dev-only dependencies from `environment-dev.yml`:

  `python -m pip install pytest jupyterlab ipywidgets plotly`

* After pip setup, use the active Python environment directly:
  - `python <package>.py`
  - `pytest`
  - `pip install <package>`
