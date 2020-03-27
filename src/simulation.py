import os
import sys

import numpy as np
import pandas as pd

import streamlit as st 

from sklearn.linear_model import LogisticRegressionCV
from tensorflow.keras import Model

from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.metrics import AUC
from sklearn.metrics import roc_auc_score


def create_model(y, private_data, shared_data, model=None):

    data = pd.concat([shared_data, private_data], axis=0)
    X,y = data.loc[:, data.columns != y], data[y]

    if not model:
        input_layer = Input(X.shape[1])
        batch_norm = BatchNormalization()(input_layer)
        output_layer = Dense(1,activation='sigmoid')(batch_norm)
        model = Model(input_layer,output_layer)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=[['accuracy', AUC()]])
    
    model.fit(
        X,
        y, 
        epochs=20, 
        batch_size=128, 
        callbacks=[EarlyStopping()], 
        validation_split=0.2, 
        verbose=0
        )

    return model


def split_data(data, n, seed):

    df_split = np.array_split(data, n)
    return df_split


def train_model(y, data, N, T, model, SEED=42):
    """
    1. Reads a CSV
    2. Splits into n + 1 folds
    3. Trains an individual model on n folds
    4. Trains an ensemble on the n + 1th fold
    """
    
    df_split = split_data(data, N + 1, SEED)
    
    models = [[None for i in range(N)] for i in range(T)]
    scores = [[None for i in range(N)] for i in range(T)]
    weights = [None for i in range(T)]
    ensemble_scores = []

    for i in range(T):
        for j in range(N):
            if i == 0:
                models[i][j] = create_model(y, df_split[j][:int(len(df_split[j])/ T * i)], df_split[-1])
            else:
                models[i][j] = create_model(y, df_split[j][:int(len(df_split[j])/ T * i)], df_split[-1], models[i-1][j])

        preds = models[i][0].predict(df_split[-1].iloc[:,:-1])
        scores[i][0] = roc_auc_score(df_split[-1].iloc[:,-1],preds)

        for j in range(1, N):
            pred = models[i][j].predict(df_split[-1].iloc[:,:-1])
            scores[i][j] = roc_auc_score(df_split[-1].iloc[:,-1],pred)
            preds = np.hstack((preds,pred))

        catboost = model.fit(
            preds, 
            df_split[-1].iloc[:,-1],
            verbose=False
            )
        ensemble_scores += [roc_auc_score(df_split[-1].iloc[:,-1],catboost.predict_proba(preds)[:,1])]

        weights[i] = catboost.feature_importances_ / sum(catboost.feature_importances_)

    results = pd.DataFrame(np.array(scores))
    results['Ensemble'] = np.array(ensemble_scores)
    results = pd.concat([results, pd.DataFrame(np.array(weights))], axis=1)
    results.columns = [f'Player_{i}' for i in range(N)] + ['Ensemble'] + [f'Player_{i}' for i in range(N)]
    results.to_csv(f'logs/weights_{N}_{T}.csv', index=False)
    
    return results


@st.cache(allow_output_mutation=True)
def ml_loader(y, data, N, T, SEED = 42):

    weights_path = f'logs/weights_{N}_{T}.csv'

    if os.path.exists(weights_path):
        results = pd.read_csv(weights_path)
    else:    
        ensembler = CatBoostClassifier(n_estimators=100, train_dir='logs/ensemble', eval_metric='AUC')
        results = train_model(y, data, N, T, ensembler)

    return results