from flask import Flask, render_template, request, url_for, Markup, jsonify
import pickle
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Model, Sequential, load_model
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences


app = Flask(__name__)

MODEL_PATH ='WORDCNNcyberbullying.h5'
model = load_model(MODEL_PATH)
 
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


@app.route('/')
@app.route('/first') 
def first():
	return render_template('first.html')
@app.route('/login') 
def login():
	return render_template('login.html')    
    
@app.route('/upload') 
def upload():
	return render_template('upload.html') 
@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        # df.set_index('Id', inplace=True)
        return render_template("preview.html",df_view = df)    

@app.route('/home')
def home():
    return render_template('index.html')


labels =  {3 : 'not_cyberbullying', 2: 'gender', 5: 'religion', 
           4: 'other_cyberbullying', 0: 'age', 1: 'ethnicity'}

@app.route('/predict', methods=['POST'])
def predict():
    error = None
    if request.method == 'POST':
        # message
        msg = request.form['message']
        msg = pd.DataFrame(index=[0], data=msg, columns=['data'])
        # transform data
        sequences = tokenizer.texts_to_sequences(msg['data'].astype('U'))
        new_text = pad_sequences(sequences, maxlen=28)

        # model
        result = model.predict(new_text,batch_size=1,verbose=2)

# labels =  {3 : 'not_cyberbullying', 2: 'gender_cyberbullying', 5: 'religion_cyberbullying', 
#            4: 'other_cyberbullying', 0: 'age_cyberbullying', 1: 'ethnicity_cyberbullying'}
        result = np.argmax(result)
        print("result:", result)
        if result == 0:
            result = 'Age_Cyberbullying'
        elif result == 1:
            result = 'Ethnicity_Cyberbullying'
        elif result == 2:
            result = 'Gender_Cyberbullying'
        elif result == 3:
            result = 'Not_Cyberbullying'
        elif result == 4:
            result = 'Other_Cyberbullying'
        elif result == 5:
            result = 'Religion_Cyberbullying'

        return render_template('index.html', prediction_value=result)
    else:
        error = "Invalid message"
        return render_template('index.html', error=error)
    
@app.route('/chart') 
def chart():
	return render_template('chart.html')

if __name__ == "__main__":
    app.run(debug=True)
