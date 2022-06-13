from flask import Flask,render_template
from flasgger import Swagger,swag_from
import tensorflow as tf

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

# train your own model

if __name__ == "__main__":
	app.run(debug = False)