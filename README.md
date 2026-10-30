<p align='left'>
    <img src="assets/analytic-boost.png" style="width: 15%; height: auto;" alt="Analytic Boost Logo">
</p>

# Pokémon VGC Matchup Predictor

This repository contains a modular Python-based machine learning project on predicting Pokémon double battles from starting information (such as players' teams and starting leads). The README will briefly demonstrate how you can clone the project and run the model on your own machine.

All Python libraries utilized in this project are listed in `requirements.txt`. No other external libraries or packages are needed to be installed beforehand. Ensure you have at least Python version *3.10* on your machine for this project.

## Quick Links

- [Setting up the project](#1-clone-the-repository)
- [Running modules](#running-modules)

---

### 1. Clone the repository

Clone the repository onto your local machine however way you choose, into a suitable directory.

### 2. Setting up the virtual environment

To install the required libraries, we need a virtual environment first. In the project's root directory, open a terminal and run:
```
python -m venv .venv
```

This will create a virtual environment with the name `.venv`. Make sure to use either this name or one of `venv`, `env`, or `ENV`, so that the virtual environment is ignored by Git.

Now activate it with the command:
```
.\.venv\Scripts\activate
```

There should be a green `(.venv)` at the beginning of each command line prompt in your terminal now. It is ready to use.

### 3. Installing dependencies & setting up modules

All of the dependencies are stored in `.\requirements.txt`. We will use Pip to install them via the terminal:
```
pip install -r requirements.txt
```

This will take a moment. Once it's done, the last step is to set up the project code itself as a package, so we can import and use its modules (e.g. in Jupyter notebooks and other files outside of `.\src\...`). To do this, run:
```
pip install -e .
```
<span style="color:orange">Make sure to run this command in the project root!</span>

---

### Running modules

Currently, modules must be run through a terminal, or you can write code in a Jupyter notebook/Python source file that imports the modules and runs their functions.

Start with the notebook `.\notebooks\01_milestone2_exploratory_data_analysis.ipynb` to import and process the data files. Once that is done, you can try out the actual models by running the following Python files in `.\src\vgc_matchup_predictor\models\`:

- `baseline_logistic_regression.py`: the baseline Logistic Regression model
- `benchmark.py`: the naive benchmark majority classifier model

---

<small>made by Abbas Syed, Shan Jeofry, and Ethan Jallim</small>