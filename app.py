import numpy as np
from flask import Flask, request, jsonify, render_template
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
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]
    
    if(output==0):
        return render_template('index.html', prediction_text='Extremely Weak')
    elif(output==1):
        return render_template('index.html', prediction_text='Weak')
    elif(output==2):
        return render_template('index.html', prediction_text='Normal')
    elif(output==3):
        return render_template('index.html', prediction_text='Overweight')
    elif(output==4):
        return render_template('index.html', prediction_text='Obesity')
    elif(output==5):
        return render_template('index.html', prediction_text='Extreme Obesity')
    else:
        return render_template('index.html', prediction_text='Can\'t Predict')
    # if(output==):
    #     return render_template('index.html', prediction_text='Chance of Admission: 100%')
    # if(output<0):
    #     return render_template('index.html', prediction_text='Chance of Admission: 0%')

    # return render_template('index.html', prediction_text='Chance of Admission: {}%'.format(output))
    
  


if __name__ == "__main__":
    app.run(port=5001,debug=True)