# Script to train machine learning model.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from preprocess_data import process_data
from clean_data import load_data, cleaned_data
from model import train_model, compute_model_metrics, inference
from joblib import dump

logging.basicConfig(
    filename='./log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def model_slicing(data):
    """
    Slice model for categorical features
    """
    slice_values = []

    for feature in cat_features:
        for value in np.unique(data[feature]):
            temp_df = data[data[feature] == value]
            x, y, _, _ = process_data(
                temp_df, categorical_features=cat_features,
                label="salary", encoder=encoder, lb=lb, training=False)
            y_pred = model.predict(x)
            p, r, f = compute_model_metrics(
                y, y_pred)
            results = "[%s->%s] Precision: %s " \
                "Recall: %s FBeta: %s" % (
                    feature,
                    value,
                    p,
                    r,
                    f)
            slice_values.append(results)

    with open('slice_model_output.txt', 'w') as f_out:
        for slice in slice_values:
            f_out.write(slice + '\n')

def  test_inefernce(y_test, y_pred):
    assert len(y_pred) == len(y_test)
    assert type(y_pred) == type(y_test)

def test_process_data(x):
    assert len(x) > 0
    
def test_model_slicing(path):
    with open(path, 'r') as f_r:
        content = f_r.read()
        assert len(content) > 0
# Add the necessary imports for the starter code.
data = load_data('../data/census.csv')
# Add code to load in the data.
print('check1')
data = cleaned_data(data)
print('check2')

# Optional enhancement, use K-fold cross validation instead of a train-test split.


cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X, y, encoder, lb = process_data(
    data, categorical_features=cat_features, label="salary", training=True
)


dump(encoder, 'encoder.joblib')
dump(lb, 'lb.joblib')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

model = train_model(X_train, y_train)
dump(model, 'model.joblib')
predictions = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, predictions)

model_slicing(data)

test_inefernce(y_test, predictions)

test_process_data(X)

test_model_slicing('slice_model_output.txt')


# Proces the test data with the process_data function.

# Train and save a model.
