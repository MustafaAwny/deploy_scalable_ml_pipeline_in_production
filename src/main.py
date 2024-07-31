# Put the code for your API here.
from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load

import pandas as pd
import numpy as np
from model import train_model, compute_model_metrics, inference
from data import process_data
import os


from sklearn.metrics import fbeta_score, precision_score, recall_score
import logging
from sklearn.ensemble import RandomForestClassifier


from sklearn.preprocessing import LabelBinarizer, OneHotEncoder







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


class User(BaseModel):
    age: int
    workclass: Literal[
        'State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 'Local-gov', 'Self-emp-inc', 'Without-pay']
    education: Literal[
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm', '7th-8th', 'Doctorate', 
        'Assoc-voc', 'Prof-school', '5th-6th', '10th', 'Preschool', '12th', '1st-4th']
    maritalStatus: Literal[
        'Never-married', 'Married-civ-spouse', 'Divorced', 'Married-spouse-absent', 'Separated', 'Married-AF-spouse', 'Widowed']
    occupation: Literal[
        'Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty', 'Other-service', 'Sales', 'Transport-moving',
        'Farming-fishing', 'Machine-op-inspct', 'Tech-support', 'Craft-repair', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv']
    relationship: Literal[
        'Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried', 'Other-relative']
    race: Literal[
        'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
    sex: Literal[
        'Male', 'Female']
    hoursPerWeek: int
    nativeCountry: Literal[
        'United-States', 'Cuba', 'Jamaica', 'India', 'Mexico',
        'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran',
        'Philippines', 'Poland', 'Columbia', 'Cambodia', 'Thailand',
        'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal',
        'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala',
        'Italy', 'China', 'South', 'Japan', 'Yugoslavia', 'Peru',
        'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago',
        'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary',
        'Holand-Netherlands']


app = FastAPI()


@app.get("/")
async def get_items():
    return {"message": "Welcome!"}


@app.post("/")
async def inferences(user_data: User):
    model = load("model/model.joblib")
    encoder = load("model/encoder.joblib")
    lb = load("model/lb.joblib")

    array = np.array([[
                     user_data.age,
                     user_data.workclass,
                     user_data.education,
                     user_data.maritalStatus,
                     user_data.occupation,
                     user_data.relationship,
                     user_data.race,
                     user_data.sex,
                     user_data.hoursPerWeek,
                     user_data.nativeCountry
                     ]])

    df = pd.DataFrame(data=array, columns=["age", "workclass", "education", "marital-status", "occupation", "relationship", "race", "sex",
                      "hours-per-week", "native-country"])
                      
    X, y, e, l = process_data(df, categorical_features=cat_features, encoder=encoder, lb=lb, training=False)
    pred = inference(model, X)
    y = lb.inverse_transform(pred)[0]
    return {"prediction": y}