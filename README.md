# Setup the Visayan Text Classifier

Guide to setting up the Visayan Text Classifier.

## Installation

Open the terminal in VSCode and clone this repository `https://github.com/UkiyoC/Visayan-Text-Classifier.git`

## Create a Virtual Environment

### For Windows Users:

```python
python -m venv .venv 
```

### For Mac/Linux Users:

```python
python3 -m venv .venv 
```

## Running the Virtual Environment

### For Windows Users:

```python
.venv/Scripts/Activate 
```

### For Mac/Linux Users:

```python
source .venv/bin/activate 
```

## Install the Requirements

```python
pip install -r requirements.txt
```

## Problems with the Virtual Environment (Execution Policies)

If you encounter any problems with the `.venv use Set-ExecutionPolicy Unrestricted -Scope Process`

## Run the Program

```python
python gui.py
```
