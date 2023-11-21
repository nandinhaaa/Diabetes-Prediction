# Importa as bibliotecas necessárias
import pickle
import numpy as np
from flask import Flask, render_template, request

# Carrega o modelo treinado a partir do arquivo usando a biblioteca pickle
filename = 'diabetes-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

# Cria uma instância do Flask
app = Flask(__name__)

# Define a rota inicial para renderizar a página inicial (index.html)
@app.route('/')
def home():
    return render_template('index.html')

# Define a rota '/predict' para lidar com as previsões com base nos dados fornecidos via POST
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Obtém os dados do formulário da solicitação POST
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])

        # Organiza os dados em uma matriz numpy e realiza uma previsão usando o modelo treinado
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)

        # Renderiza a página de resultado (result.html) com a previsão
        return render_template('result.html', prediction=my_prediction)

# Inicia o servidor Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
