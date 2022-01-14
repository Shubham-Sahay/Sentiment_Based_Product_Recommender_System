# This is basically the heart of my flask 
from flask import Flask, render_template, request, redirect, url_for
import warnings
warnings.filterwarnings("ignore")
import model


app = Flask(__name__)


@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	Username = request.form.get('Username')
	#processed_date = pre_process([data])
	prediction = model.getOptimizedRecommendations(Username)
	return render_template('index.html', OUTPUT=str(prediction))

if __name__ == "__main__":
    app.run(debug=True)