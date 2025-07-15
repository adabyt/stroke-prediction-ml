# Stroke Prediction Machine Learning Pipeline

## Project Overview

This project demonstrates a complete machine learning pipeline for predicting the likelihood of stroke in patients using clinical and demographic data. It showcases key data science skills including data exploration, preprocessing, handling imbalanced data, model training, evaluation, threshold tuning, and model serialisation — all implemented in Python with best practices.

The primary goal is to develop and evaluate models that can predict stroke cases despite the challenges of a highly imbalanced dataset.

---

## Dataset

The dataset used in this project is sourced from Kaggle:  
[Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset?resource=download)

It contains demographic, lifestyle, and medical information along with a binary label indicating stroke occurrence.

---

## Project Structure

```
stroke-prediction-ml/
├── data/
│   └── stroke_data.csv                     # Raw dataset file
├── models/
│   └── final_model_histgbc.joblib          # Serialised trained model and threshold
├── notebooks/
│   └── stroke_prediction_pipeline.ipynb    # Jupyter notebook with full analysis and modelling
├── requirements.txt                        # Project dependencies
├── README.md                               # This file
└── .gitignore                              # Specifies files/folders to ignore in Git
```

---

## Key Steps and Methods

### 1. Exploratory Data Analysis (EDA)

- Loaded and inspected dataset for missing values and data types.
- Visualised distributions and relationships of features.
- Identified class imbalance in the target variable (`stroke`).

### 2. Data Preprocessing and Feature Engineering

- Handled missing data and categorical variables.
- Created meaningful feature transformations informed by domain knowledge.

### 3. Handling Imbalanced Data

- Investigated imbalanced class distribution (stroke cases are the minority).
- Applied **random oversampling** to augment minority class for training data.
- Discussed alternative sampling techniques (e.g., SMOTE), but faced compatibility constraints.

### 4. Model Training and Evaluation

- Trained multiple models on both original imbalanced and oversampled datasets:

  - Logistic Regression
  - Random Forest
  - HistGradientBoostingClassifier (HGB)
  - LightGBM

- Evaluated models using:

  - Precision, Recall, F1-score per class
  - Confusion matrices
  - ROC AUC score
  - Precision-Recall curves

- Found **HistGradientBoostingClassifier trained on the oversampled data** to achieve the best balance between precision and recall, especially for the minority class.

### 5. Threshold Tuning

- Performed threshold tuning on predicted probabilities to optimise F1-score.
- Selected an optimal threshold of 0.22 instead of the default 0.5 to improve detection of stroke cases.
- Demonstrated the trade-off between false positives and false negatives using classification reports and confusion matrices.

### 6. Model Finalisation and Serialisation

- Retrained the best model on the full oversampled training dataset.
- Saved the model along with the optimised threshold using `joblib` for easy loading and inference.

### 7. Model Loading and Testing

- Demonstrated loading the saved model and threshold.
- Used the model to make predictions on the test set with the custom threshold.
- Recomputed evaluation metrics and visualisations to verify consistent performance.

---

## Technologies and Libraries

- Python 3.10+
- Jupyter Notebook
- NumPy
- pandas
- matplotlib
- seaborn
- scikit-learn
- joblib
- LightGBM
- VSCode for development

---

## How to Run

1. Clone the repository:

   ```bash
   git clone <your-repo-url>
   cd stroke-prediction-ml
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset?resource=download) and place the CSV file inside the data/ folder.

5. Launch Jupyter Notebook and run the pipeline:

   ```bash
   jupyter notebook notebooks/stroke_prediction_pipeline.ipynb
   ```

6. Follow the notebook to explore data, train models, tune thresholds, and save/load models.

---

## Project Insights and Limitations

- The dataset is highly imbalanced with relatively few stroke cases, posing a significant challenge for predictive modelling.
- Despite efforts including oversampling and model tuning, recall and precision for stroke cases remain modest.
- Threshold tuning improved minority class detection but introduced more false positives.
- Future work could explore advanced sampling techniques such as SMOTE or ensemble methods to better balance the classes.
- Additional features or clinical data may be required for substantial improvements in stroke prediction.

---

## License

This project is licensed under the MIT License.
