import os, sys
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegressionCV
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.metrics import AUC

from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier

import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html

from tensorboard import program
import tensorflow as tf




def createModel(privateData, sharedData, userID, model=None, ):
	df = pd.concat([sharedData, privateData],axis=0)
	X,y = df.iloc[:,:-1], df.iloc[:,-1]
	if not model:
		input_layer = Input(X.shape[1])
		batch_norm = BatchNormalization()(input_layer)
		output_layer = Dense(1,activation='sigmoid')(batch_norm)
		model = Model(input_layer,output_layer)
		model.compile(
			optimizer='adam',
			loss='binary_crossentropy',
			metrics=[['accuracy',AUC()]])
	logdir = "logs/scalars/model" + str(userID)

	tensorboard_callback = TensorBoard(log_dir=logdir)
	model.fit(
		X,
		y, 
		epochs=20, 
		batch_size=128, 
		callbacks=[EarlyStopping(), tensorboard_callback], 
		validation_split=0.2, 
		verbose=0
		)
	return model

def splitData(df, n, seed):
	df_split = np.array_split(df, n)
	return  df_split

def test_creditcard_dataset(N, SEED, T, model):
	'''
	1. Reads a CSV
	2. Splits into n + 1 folds
	3. Trains an individual model on n folds
	4. Trains an ensemble on the n + 1th fold

	'''


	df = pd.read_csv("data/UCI_Credit_Card.csv")	
	df_split = splitData(df, N + 1, SEED)
	

	models = [[None for i in range(N)] for i in range(T)]
	scores = [[None for i in range(N)] for i in range(T)]
	weights = [None for i in range(T)]
	ensemble_scores = []
	for i in range(T):
		for j in range(N):
			if i == 0:
				models[i][j] = createModel(df_split[j][:int(len(df_split[j])/ T * i)], df_split[-1],j)
			else:
				models[i][j] = createModel(df_split[j][:int(len(df_split[j])/ T * i)], df_split[-1], j, models[i-1][j])

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


	
	return  models, weights, scores, ensemble_scores




if __name__ == "__main__":
	print("LAUNCHING TENSORBOARD on port 6006")
	print("LAUNCHING DASH on port 8050")
	tf.autograph.set_verbosity(0)
	external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
	ensembledir = "logs/scalars/ensemble"



	N, SEED, T = 3, 42, 5
	adaboost = CatBoostClassifier(
		n_estimators=100,
		train_dir=ensembledir,
		eval_metric="AUC"
		)
	models, weights, scores, ensemble_scores = test_creditcard_dataset(N, 42, T, adaboost)	
	df = pd.DataFrame(np.array(scores))
	df['ensemble_scores'] = np.array(ensemble_scores)
	df = pd.concat([df,pd.DataFrame(np.array(weights))],axis=1)	 
	df.to_csv("logs/weights.csv",index=False)	 
		 
		 


	fig2 = go.Figure()
	for i in range(N):
		fig2.add_trace(go.Scatter(x=list(range(T)), y=df.iloc[:,i],name=f"Participant {i}"))

	fig2.add_trace(go.Scatter(x=list(range(T)), y=df['ensemble_scores'], name="Ensemble Performance"))
	fig2.update_layout(title='Model Performance against Time')
	fig2.update_yaxes(type="log")

	fig3 = go.Figure()
	for i in range(N):
		fig3.add_trace(go.Scatter(x=list(range(T)), y=df.iloc[:,-3+i],name=f"Participant {i}"))
	fig3.update_layout(title='Ensemble Weights against Time')


	app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

	app.layout = html.Div(children=[
	    html.H1(children='Allstate Insurance Claims Simulation'),

	    html.Div(children='''
	        Dash: A web application framework for Python.
	    '''),

	    dcc.Graph(
	        id='performance',
	        figure=fig2
	    ),
	    dcc.Graph(
	        id='weights',
	        figure=fig3
	    )
	])

	logdir = "logs/scalars"
	tb = program.TensorBoard()
	tb.configure(argv=[None, '--logdir', logdir])
	
	tb.launch()
	app.run_server(debug=False)
	# x.start()

	
	


# curl newBlockchain
# curl addBlock
# curl printblockchain

