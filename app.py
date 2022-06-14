from flask import Flask,render_template,url_for,request
from flasgger import Swagger,swag_from
import tensorflow as tf
import numpy as np

# load your machine learning model
model = tf.keras.models.load_model('./lstm_model')

# create app object
app = Flask(__name__)

#define a template Info Object
template = {
    "swagger":"2.0",
    "info":{
        "title":"Application backend",
        "description":"predict and train your RNN with series data",
        "contact":{
            "name": "LocChuong",
            "url": "http://www.swagger.io/support",
            "email": "locchuong123@gmail.com"
        },
        "version":"0.0.1",
        "schemes":['http','https']
    }
}

swagger = Swagger(app,template= template)

# home
@app.route('/')
@app.route('/home')
@swag_from('./docs/home.yml')
def home():
	return render_template('home.html'),200

# predict
@app.route('/predict',methods=  ['GET','POST'])
def predict():
	if request.method == 'POST':
		# get form info
		input_1 = float(request.form['input_1'])
		input_2 = float(request.form['input_2'])
		input_3 = float(request.form['input_3'])
		input_4 = float(request.form['input_4'])
		input_5 = float(request.form['input_5'])
		# preprocess data
		x = np.array([input_1,input_2,input_3,input_4,input_5])
		x = x.reshape(1,x.shape[0],1)
		# predict the value
		y =model.predict(x)
		return f"<h1>your prediction: {round(float(y),2)} $</h1>",201

	return render_template('predict.html'),200

# train your own model
@app.route('/train',methods=  ['GET','POST'])
def train():
	return render_template('train.html'),200

if __name__ == "__main__":
	app.run(debug = False)