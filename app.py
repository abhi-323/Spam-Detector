import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open('models/model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    int_features = [x for x in request.form.values()]
    prediction = model.predict(int_features)

    output = prediction

    return render_template('index.html', prediction_text='{}'.format(output))


if __name__ == "__main__":
    app.run()
