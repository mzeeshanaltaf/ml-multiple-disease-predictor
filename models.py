import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score)


# This function reads the data from csv file and returns the cleaned after pre-processing
def get_heart_disease_data():
    # Read the data from csv file
    data = pd.read_csv('heart_disease_data.csv')

    return data


# This function reads the data from csv file and returns the cleaned after pre-processing
def get_diabetes_disease_data():
    # Read the data from csv file
    df = pd.read_csv('diabetes_disease_data.csv')
    # Some of the values in Glucose, BloodPressure, SkinThickness, Insulin, BMI are 0 which reflects either missing
    # data or measurement error as these values
    # can't be 0. To maintain the data integrity, replace the 0 values with mean value.

    # Replace 0 values in Glucose column with mean value of the Glucose
    mean_bp = df[df['Glucose'] != 0]['Glucose'].mean()
    df['Glucose'] = df['Glucose'].replace(0, mean_bp)

    # Replace 0 values in Blood Pressure column with mean value of the blood pressure
    mean_bp = df[df['BloodPressure'] != 0]['BloodPressure'].mean()
    df['BloodPressure'] = df['BloodPressure'].replace(0, mean_bp)

    # Replace 0 values in Skin Thickness column with mean value of the skin thickness
    mean_bp = df[df['SkinThickness'] != 0]['SkinThickness'].mean()
    df['SkinThickness'] = df['SkinThickness'].replace(0, mean_bp)

    # Replace 0 values in Insulin column with mean value of the Insulin
    mean_bp = df[df['Insulin'] != 0]['Insulin'].mean()
    df['Insulin'] = df['Insulin'].replace(0, mean_bp)

    # Replace 0 values in BMI column with mean value of the BMI
    mean_bp = df[df['BMI'] != 0]['BMI'].mean()
    df['BMI'] = df['BMI'].replace(0, mean_bp)

    return df


def get_parkinson_disease_data():
    # Read the data from csv file
    data = pd.read_csv('parkinsons_disease_data.csv')
    data = data.drop(['name'], axis=1)

    return data


# Function to train the machine learning model
def train_model(model_name, disease):
    # Available models
    models = {
        "Support Vector Machines": svm.SVC(kernel='linear'),
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gaussian NB": GaussianNB()

    }
    if disease == "Diabetes":
        # Get the data
        data = get_diabetes_disease_data()

        # Get the dependent and independent features
        X = data.drop(['Outcome'], axis=1)
        y = data['Outcome']

    elif disease == "Heart":
        # Get the data
        data = get_heart_disease_data()

        # Get the dependent and independent features
        X = data.drop(['target'], axis=1)
        y = data['target']

    elif disease == "Parkinson":
        # Get the data
        data = get_parkinson_disease_data()

        # Get the dependent and independent features
        X = data.drop(['status'], axis=1)
        y = data['status']

    # Split the data to training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform standardization on split data. Standardization is performed on independent features
    # only i.e. X_train/X_test
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Get the machine learning model selected by user
    model = models[model_name]

    # Train the model
    model.fit(X_train, y_train)

    # Prediction for test data
    y_pred = model.predict(X_test)

    # Create data frame for storing model performance parameters
    performance_tests = ['Accuracy', 'F1 Score', 'Precision', 'Recall',
                         'ROC AUC Score']  # This will be the name of columns
    df_performance_metric = pd.DataFrame(columns=performance_tests)

    # Get the performance metrics and add them in the data frame
    df_performance_metric.loc[0, 'Accuracy'] = "{:.2f}".format(accuracy_score(y_test, y_pred))
    df_performance_metric.loc[0, 'F1 Score'] = "{:.2f}".format(f1_score(y_test, y_pred))
    df_performance_metric.loc[0, 'Precision'] = "{:.2f}".format(precision_score(y_test, y_pred))
    df_performance_metric.loc[0, 'Recall'] = "{:.2f}".format(recall_score(y_test, y_pred))
    df_performance_metric.loc[0, 'ROC AUC Score'] = "{:.2f}".format(roc_auc_score(y_test, y_pred))

    return model, scaler, df_performance_metric


# Function to predict outcome given the input data nad based on trained machine learning model
def model_predictions(input_data, model, scaler):
    # Convert the input data into a 2D array. This is required for machine learning model
    input_array = np.array(list(input_data.values())).reshape(1, -1)

    # Scale the input data. This is required before making prediction
    input_array_scaled = scaler.transform(input_array)

    # Get prediction from the trained model
    prediction = model.predict(input_array_scaled)

    # Get prediction probability
    probability = model.predict_proba(input_array_scaled)

    return prediction, probability
