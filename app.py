import numpy as np
from flask import Flask, request,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    a= 'BENIGN'
    b='MALIGNANT'
    output = prediction[0]
    if output==2:
        return render_template('index.html', prediction_text='Patient has no Risk of Breast Cancer  \n{}'.format(a))
    else:
        return render_template('index.html', prediction_text='Patient has  Risk of Breast Cancer   \n{}'.format(b))
        

if __name__ == "__main__":
    app.run()