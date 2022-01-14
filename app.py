# This is basically the heart of my flask 
from flask import Flask, render_template, request, redirect, url_for
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import model


app = Flask(__name__)


@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    Username = request.form.get('Username')
    prediction = model.getOptimizedRecommendations(Username)
    if type(prediction)==list:
        predictionDF = pd.DataFrame(prediction, columns=['Recommended Products'])
        return render_template('index.html', tables=[predictionDF.to_html(classes='Recommendations')], titles = ['For Username : '+ Username])
    else:
        return render_template('index.html', OUTPUT=prediction)

if __name__ == "__main__":
    app.run()