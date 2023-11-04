from flask import Flask, render_template,request,send_file
import requests
import pandas
import utils
from utils import preprocessdata

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict/', methods=['GET', 'POST'])

def predict():  
    if request.method == 'POST': 
        name = request.form.get('name')  
        region = request.form.get('region')  
        days = request.form.get('days')
        choose = request.form.get('choose')   


        prediction = utils.preprocessdata(name,region,days,choose)
        print(prediction)

    return render_template('predict.html', prediction=prediction) 


if __name__ == '__main__':
    app.run()