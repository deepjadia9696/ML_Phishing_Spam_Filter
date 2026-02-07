# Text-based Spam Filter

This repository contains a small, educational text-based spam filtering project implemented in Python. It includes two simple classifier implementations (Naive Bayes and an SVM-based method using LIBLINEAR), a ctypes-based LIBLINEAR Python binding, and utility code for reading and evaluating LIBSVM-style data.

## Repository structure

- `commonutil.py` — utilities for reading LIBSVM data, basic evaluations, and sparse matrix scaling.
- `liblinear.py` — ctypes bindings to the LIBLINEAR C library (loads a local `windows/liblinear.dll` on Windows or the system `liblinear` library on Unix).
- `liblinearutil.py` — Python-friendly wrappers around the `liblinear` binding: `train()`, `predict()`, `load_model()`, `save_model()` and option parsing.
- `naive_bayesian.py` — a minimal, educational Naive Bayes text classifier implemented from raw token counts.
- `svm.py` — an SVM-based classifier that builds a top-N token lexicon and trains/predicts using `liblinearutil`.
- `dataset/` — the email dataset split into `ham/` and `spam/`. See `dataset/Summary.txt` for dataset statistics.
- `windows/` — (included) directory with a Windows `liblinear.dll` used by the binding on Windows systems.

## Overview of implemented algorithms

Naive Bayes (`naive_bayesian.py`)
- Loads all files from `dataset/ham` and `dataset/spam` into memory.
- Randomly shuffles and splits the data roughly 50% train / 50% test.
- Builds token frequency dictionaries for ham and spam from the training set.
- For each test message, computes per-token likelihoods and multiplies them to form posterior scores for ham vs spam.
- Notes: this is a very small, didactic implementation — it uses a tiny epsilon in place of proper smoothing and multiplies many probabilities (risk of underflow); it is not production-ready.

SVM with LIBLINEAR (`svm.py`)
- Loads dataset similarly and splits into train/test.
- Counts token occurrences in each class on the training set and selects the top-N (default N=1000) tokens from ham and spam to create a shared lexicon.
- Converts messages into binary feature vectors (presence/absence of the lexicon tokens) and trains a LIBLINEAR model via `liblinearutil.train()` with default parameters.
- Evaluates using `liblinearutil.predict()` and prints reported accuracy.

LIBLINEAR binding and utilities
- `liblinear.py` provides a low-level ctypes wrapper around the native LIBLINEAR library.
- `liblinearutil.py` exposes `train()` and `predict()` functions that accept plain Python lists/NumPy arrays and provide familiar option flags (e.g. `-s`, `-c`, `-v`).

## Dataset

The repository includes a split dataset under `dataset/` with two folders: `ham/` (legitimate emails) and `spam/` (spam emails). See `dataset/Summary.txt` for counts and time ranges. The code expects the dataset to be present in these two directories.

## Requirements

- Python 3.x (recommended).
- Optional but recommended: `numpy` and `scipy` for faster sparse handling and some utility functions. If `scipy` is available, the bindings will use sparse matrices internally.
- The LIBLINEAR native library. On Windows the project provides `windows/liblinear.dll`; on Unix-like systems install a liblinear shared library (e.g. `liblinear.so`) accessible via `ctypes.util.find_library` or place `liblinear.so.3` adjacent to the Python files.

Typical installs:

```powershell
pip install numpy scipy
# On Unix you might install liblinear via your package manager or build from source.
```

## Usage

Simple, direct usages (the scripts are written as small runnable programs rather than command-line tools):

- Run the Naive Bayes classifier:

```powershell
python naive_bayesian.py
```

- Run the SVM (LIBLINEAR) classifier:

```powershell
python svm.py
```

Notes:
- Both scripts expect the dataset directories to be `dataset/ham` and `dataset/spam` relative to the script.
- The scripts perform an in-memory shuffle-and-split with a 50/50 split and print basic accuracy metrics. They do not currently accept command-line arguments — modify the top-level constants or add an argument parser to change behavior.

## Suggestions & Known limitations

- The Naive Bayes implementation uses raw token splitting on whitespace and multiplies many small probabilities; replace this with log-probabilities and add proper smoothing (Laplace) to improve numeric stability and results.
- The SVM pipeline selects top-N tokens by raw frequency and uses binary presence features. Consider using TF-IDF weighting and a proper train/test split or cross-validation for more reliable evaluation.
- Add command-line options (e.g., dataset path, train/test ratio, N for top tokens, solver / C parameters for LIBLINEAR) to make the scripts reproducible and easier to experiment with.
- Add a small driver or unit tests to make it easy to reproduce results.

## Extending the project

- Swap `svm.py`'s feature creation for scikit-learn's `CountVectorizer` / `TfidfVectorizer` and use `sklearn.svm.LinearSVC` for an alternative to the ctypes binding.
- Implement proper preprocessing (tokenization, stopword removal, stemming) to improve classifier performance.
