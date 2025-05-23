# Praca-Magisterska
Source code developed for the project carried out as part of a master's thesis.

Language: Python

### Title (PL): Metody przetwarzania niezbalansowanych danych dotyczących ryzyka kredytowego przy użyciu technik uczenia maszynowego
### Title (ENG): Processing unbalanced credit risk data using machine learning techniques

## Description of the project

The aim of this thesis is to investigate and evaluate various methods for mitigating the problem of data imbalance in credit risk datasets within the context of machine learning. The central research hypothesis assumes that modifications applied at the data and algorithm levels can significantly improve classification accuracy on imbalanced credit risk datasets. The thesis focuses on comparing the effectiveness of techniques designed for handling imbalanced data, such as undersampling, oversampling, hybrid methods, and class weight adjustment. These techniques are combined with selected machine learning algorithms to assess their performance.

The study was conducted using five different datasets, varying in size and degree of imbalance. Each file in the project contains the same experiment repeated for a different dataset, allowing a comparative analysis of how each method performs across varying data characteristics. 

The implementation was carried out in Python, using the PyCharm development environment.

The project includes:

- Data preprocessing and cleaning
- Exploratory data analysis
- Application of various resampling techniques to datasets (undersampling, oversampling, hybrid)
- Training and evaluation of several machine learning models, including:
  - K-Nearest Neighbors (KNN)
  - Logistic Regression
  - Decision Trees
  - Support Vector Machines (SVM)
  - Bagging and Boosting (with both Decision Trees and SVM)

Each experiment is evaluated using performance metrics such as accuracy, AUC, recall, F1-score, and Matthews correlation coefficient.

## Dependencies

- Python 3
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- tensorflow / keras


