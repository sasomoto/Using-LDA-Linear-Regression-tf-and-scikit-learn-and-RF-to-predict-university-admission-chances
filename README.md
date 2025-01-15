# Using-LDA-Linear-Regression-tf-and-scikit-learn-and-RF-to-predict-university-admission-chances

MACHINE LEARNING ASSIGNMENT 3
TEAM MEMBERS
Dhairya Luthra(2022A7PS1377H)
Shashwat Sharma(2022AAPS0508H)
Animesh Agrahari(2022A7PS1367H)
Assignment Tasks
1. Data Loading and Preparation
Import necessary libraries (os, pandas, numpy, tensorflow).
Load dataset into a Pandas DataFrame (pd.read_csv()).
Handle missing values and outliers
2. Data Scaling
Analyze the distribution of features (df.describe(), seaborn/matplotlib for plots).
Select an appropriate scaling method (StandardScaler, MinMaxScaler, RobustScaler).
Apply the scaler to the features.
3. Dimensionality Reduction with LDA
Convert "Chance of Admit" to categorical (low, medium, high) using pd.cut().
Apply Linear Discriminant Analysis (LDA) on the dataset using sklearn.discriminant_analysis.LinearDiscriminantAnalysis.
Retain appropriate number of LDA components and justify the selection.
4. Linear Regression on LDA Transformed Data
Using TensorFlow:
Convert LDA-transformed data into TensorFlow tensors (tf.convert_to_tensor()).
Define a linear regression model using TensorFlow (initialize weights and biases).
Train the model using Mean Squared Error and Stochastic Gradient Descent (learning rate = 0.01, 1000 epochs).
Visualize actual vs. predicted labels using matplotlib.
Using Scikit-Learn:
Implement linear regression using LinearRegression from scikit-learn.
Train the model and visualize performance.
Compare results with the TensorFlow model.
5. Logistic Regression for Categorized Admission Chances
Convert "Chance of Admit" into low, medium, high categories.
Implement Logistic Regression using TensorFlow (Softmax activation for multi-class classification).
Train the model, tune the learning rate (try values like 0.001, 0.01).
Compare Logistic Regression performance with Random Forest from the previous assignment.
6. Hyperparameter Tuning
Explore different learning rates (e.g., 0.001, 0.01, 0.1).
Use Grid Search or Random Search for hyperparameter tuning (e.g., regularization).
7. Model Evaluation
Apply k-fold cross-validation (using cross_val_score from sklearn.model_selection).
Report metrics: Accuracy, Precision, Recall, F1-Score, and AUC-ROC.
