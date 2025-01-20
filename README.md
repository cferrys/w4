# BBC News Classification

## Overview
This project classifies BBC news articles into their respective categories using machine learning techniques. It leverages both unsupervised learning (Non-Negative Matrix Factorization - NMF) and supervised learning (Logistic Regression). The workflow includes:
- Data loading and preprocessing
- Exploratory Data Analysis (EDA)
- Feature extraction using TF-IDF
- Model training and evaluation
- Saving predictions to CSV files

## Files
- **`classification.py`**: The main Python script containing the full workflow from data loading to model evaluation.
- **`classification.ipynb`**: Jupyter Notebook version of the project for an interactive walkthrough.
- **`BBC News Train.csv`**: Training dataset.
- **`BBC News Test.csv`**: Test dataset.
- **`cleaned_train_data.csv`**: Preprocessed training data (output).
- **`cleaned_test_data.csv`**: Preprocessed test data (output).
- **`nmf_predictions.csv`**: Predictions from the NMF model.
- **`logreg_predictions.csv`**: Predictions from the Logistic Regression model.

## Prerequisites
- Python 3.x
- Recommended: Use a virtual environment.

### Required Libraries
Install the required libraries using pip:

```bash
pip install -r requirements.txt
```

## How to Run

1. Ensure the `classification.py` file and datasets (`BBC News Train.csv` and `BBC News Test.csv`) are in the same directory.
2. Open a terminal and activate your Python environment.
3. Execute the script:

```bash
   python3 classification.py
```

## Conclusions

- Logistic Regression achieves higher accuracy on labeled data.
- NMF provides a way to uncover latent topics and classify articles without labels.
- Both approaches demonstrate the power of machine learning for text classification tasks.