# app.py is the starting point of our application
# app.py connects our frontend and backend code
# Author : Shubham Sahay
# 
# Importing Required Libraries 
from flask import Flask, render_template, request, redirect, url_for
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import model
import pickle

# Initializing Flask Application
app = Flask(__name__)

# Defining function to render our index.html using GET method
@app.route('/')
def home():
	return render_template('index.html')

# Defining function to submit our input to model using POST method
# After getting output from model, it renders updated index.html
@app.route('/predict',methods=['POST'])
def predict():
    # Fetching username from form
    Username = request.form.get('Username')

    # Getting recommendations from model for the input username
    prediction = model.getOptimizedRecommendations(Username)
    if type(prediction)==list:
        predictionDF = pd.DataFrame(prediction, columns=['Recommended Products'])
        return render_template('index.html', tables=[predictionDF.to_html(classes='Recommendations')], titles = ['For Username : '+ Username])
    else:
        return render_template('index.html', OUTPUT=prediction)


# Defining the starting point of our application
if __name__ == "__main__":
    app.run(debug=True)