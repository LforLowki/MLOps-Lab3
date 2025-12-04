# Lab0 - Data Preprocessing CLI

This project implements a set of data preprocessing functionalities for numerical, textual, and structural data, accessible via a command-line interface (CLI). It was developed as part of the MLOps course.

---

## Features

The project provides the following preprocessing functionalities:

### **Cleaning**
- **Remove missing values** (`None`, empty strings, `NaN`)  
- **Fill missing values** with a default or user-specified value  
- **Remove duplicate values** from a list  

### **Numerical Processing**
- **Normalization** (Min-Max scaling)  
- **Standardization** (Z-score)  
- **Clipping** numerical values to a range  
- **Conversion to integers** (non-numeric values are ignored)  
- **Logarithmic transformation** (applied to positive values)  

### **Text Processing**
- **Tokenization** into lowercase alphanumeric words  
- **Remove punctuation** (keeping only letters, digits, and spaces)  
- **Remove stopwords** from text  

### **Structural Operations**
- **Flatten** a list of lists  
- **Shuffle** a list with optional seed for reproducibility  

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/LforLowki/Lab0.git
cd Lab0
```

## Usage
```bash
uv run python -m src.cli numeric normalize 1 2 3 --min 0 --max 1
uv run python -m src.cli numeric standardize 10 20 30
uv run python -m src.cli numeric clip --min 0 --max 1 -- -1 0.5 2
uv run python -m src.cli numeric log 1 10 100
uv run python -m src.cli numeric convert-int 1 "2" "a" 3.5 "4.0"

uv run python -m src.cli clean remove-missing 1 "" 2 None 3
uv run python -m src.cli clean fill-missing 1 "" None 3 --fill 0
uv run python -m src.cli clean unique 1 2 2 3 3 3


uv run python -m src.cli text tokenize "Hola mundo! Esto es una prueba."
uv run python -m src.cli text remove-punct "Hola@# mundo!!! 123."
uv run python -m src.cli text remove-stops "hola esto es una prueba" --stopwords esto --stopwords es --stopwords una

uv run python -m src.cli struct flatten "[[1,2],[3,4],[5]]"
uv run python -m src.cli struct shuffle 1 2 3 4 5 --seed 42


uv run python -m pytest -v  --cov=src
```