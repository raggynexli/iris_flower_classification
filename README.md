# Iris Flower Classification

## Internship Details
* **Intern ID:** CITS3192
* **Full Name:** Mayank Kushwah
* **No. of Weeks:** 1
* **Project Name:** Iris Flower Classification
* **Project Scope:** To develop and evaluate a supervised machine learning model (Random Forest Classifier) capable of accurately predicting the species of an Iris flower (Setosa, Versicolor, or Virginica)
               based on its sepal and petal measurements. The project encompasses data preprocessing, model training, performance evaluation, and data visualization.

---

## 📁 Repository Contents (Deliverables)
This project repository includes all required deliverables:
* **Source Code:** `iris.py` contains the complete Machine Learning Python script.
* **README File:** `README.md` (This document) provides project context and instructions.
* **Screenshots:** The `screenshots/` folder contains terminal captures of the dataset loading, training/testing splits, model performance, and prediction outputs.
* **Output Images:** The `graphs/` folder contains the generated Confusion Matrix heatmap.
* **Documentation:** `report.docx` contains the comprehensive, formatted internship project report.

---

## Technologies Used
* **Python**
* **Pandas**
* **Matplotlib & Seaborn**
* **Scikit-learn**

## Dataset
The dataset utilized is the classic Fisher's Iris dataset, which contains:
* Sepal Length
* Sepal Width
* Petal Length
* Petal Width
* Species (Target: Setosa, Versicolor, Virginica)

## Machine Learning Algorithm
* Random Forest Classifier

## Project Workflow
1. **Load Dataset:** Imported the data and exported it to `iris.csv`.
2. **Train-Test Split:** Divided the data into 80% training and 20% testing sets.
3. **Train Model:** Fitted the Random Forest algorithm to the training data.
4. **Evaluate Performance:** Generated accuracy scores and a detailed classification report.
5. **Predict New Data:** Tested the model's capability to predict a brand new, custom flower measurement.
6. **Visualize:** Generated a Confusion Matrix to map out actual versus predicted classifications.

## Results
* **Model Accuracy:** 100.0% on the testing data.
* **Classification Metrics:** Perfect precision, recall, and f1-scores across all three species categories. The Confusion Matrix confirms zero false positives and zero false negatives.

## How to Run Locally
1. Install required libraries:
   `pip install pandas matplotlib seaborn scikit-learn`
2. Run the main script:
   `python iris.py`
