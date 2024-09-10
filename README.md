# Wine_Quality_Classification

This project applies three different machine learning classification algorithms—K-Nearest Neighbors (KNN), Logistic Regression, and Random Forest—to predict the quality of wine based on various chemical properties of the wine. The dataset used is a well-known wine dataset that includes features such as acidity, alcohol content, and pH level.

Project Overview
Wine quality is determined by several physicochemical tests and sensory data. The goal of this project is to build predictive models using supervised machine learning algorithms to classify the quality of wine into different classes. The three models used are:

K-Nearest Neighbors (KNN)
Logistic Regression
Random Forest Classifier
Dataset
The dataset used in this project is the Wine Quality Dataset, which contains the following features:

Fixed acidity
Volatile acidity
Citric acid
Residual sugar
Chlorides
Free sulfur dioxide
Total sulfur dioxide
Density
pH
Sulphates
Alcohol content
Wine quality (Target/Label)
The dataset is available in the file wine.csv.

Project Structure
wine.csv: The dataset used for training and testing the models.
wine_classification.ipynb: Jupyter notebook with the implementation of the classification algorithms.
scaler.pkl: Pre-trained scaler file for transforming data (if feature scaling is applied).
model_knn.pkl, model_logreg.pkl, model_rf.pkl: Trained model files for each classifier (optional, if included).
README.md: This file that describes the project.
Algorithms Used
## K-Nearest Neighbors (KNN):
A simple, instance-based learning method that classifies a new observation based on its distance to the nearest neighbors in the training data.

## Logistic Regression:
A linear model that estimates the probability of an instance belonging to a certain class, commonly used for binary and multi-class classification problems.

## Random Forest Classifier:
An ensemble learning method that constructs multiple decision trees during training and outputs the class that is the mode of the classes of the individual trees.

## Evaluation Metrics
For each model, the following performance metrics were computed:
1. Accuracy: Proportion of correct predictions.
2. Precision: Ratio of true positives to the total predicted positives.
3. Recall: Ratio of true positives to the total actual positives.
4. F1-Score: Harmonic mean of precision and recall, providing a balance between the two.
5. Confusion Matrix: A matrix that breaks down predictions into true positives, false positives, true negatives, and false negatives.

## How to Use
Clone the repository:
git clone https://github.com/yunus-abdi/wine-quality-classification.git

## Results
Each algorithm was trained and evaluated on the wine quality dataset, and the results were compared in terms of accuracy, precision, recall, and F1-score. The Random Forest Classifier performed the best in terms of overall accuracy, while the Logistic Regression provided competitive results with fewer resources.
