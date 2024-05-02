import pickle
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score, balanced_accuracy_score
import joblib

# from pickle5 import pickle 
data = pd.read_csv('breast_cancer2.csv')
data.columns = data.columns.str.replace(' ', '_')
data.dropna(subset=["Teşhis"], inplace=True)


def create_model(data):
    # setting independent and dependent variables
    X = data.drop(['Teşhis'],axis = 1)
    y = data["Teşhis"]

    # scaling values 
    scaler = StandardScaler()
    X = scaler.fit_transform(X) 

    # split the data 
    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=0.2, random_state= 42
    )
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # test the model
    y_preds = model.predict(X_test)
    print("Test set accuracy:",accuracy_score(y_test,y_preds))
    print("Classification Report: \n", classification_report(y_test,y_preds))

    return model, scaler 


def main():
    dataframe = data
    model,scaler = create_model(dataframe) 

    with open('newmodel.pkl','wb') as f:
        pickle.dump(model,f)
    with open('newscaler.pkl', 'wb') as f:
        pickle.dump(scaler,f) 

if __name__ == '__main__':
    main() 
