import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    init_features = [int(x) for x in request.form.values()]
    final_features = [np.array(init_features)]
    prediction = model.predict_proba(final_features)
    
    output = round(prediction[0,0], 2)
    if(output > 0.7):
        result='You are eligible'
    else:
        result='You are not eligible'
    
    return render_template('index.html', prediction_text = result)
   

if __name__ == "__main__": 
    app.run(debug=True)
    
#if __name__ == "__main__":
 #   app.run(host='0.0.0.0', port=8080)