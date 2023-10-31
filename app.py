# Importing essential libraries
import pickle
import numpy as np
from flask import Flask, render_template, request

# Carregar o modelo de classificação de floresta aleatória
filename = 'diabetes-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

# Criar o aplicativo Flask
app = Flask(__name__)

# Definir a rota para a página inicial
@app.route('/')
def home():
    return render_template('index.html')

# Definir a rota para previsão
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])

        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)

        return render_template('result.html', prediction=my_prediction)

# Executar o aplicativo Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
