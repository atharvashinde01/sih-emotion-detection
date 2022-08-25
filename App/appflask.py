#libraries
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
#loading the model
with open("/models/emotion_classifier_pipe_lr_03_june_2021.pkl", 'rb') as file:  
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = model.predict(vect)
	return render_template('request.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)