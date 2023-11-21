# Importa as bibliotecas necessárias
import numpy as np
import pandas as pd
import pickle

# Carrega o conjunto de dados sobre diabetes do arquivo CSV usando o pandas
df = pd.read_csv('kaggle_diabetes.csv')

# Renomeia a coluna 'DiabetesPedigreeFunction' para 'DPF'
df = df.rename(columns={'DiabetesPedigreeFunction':'DPF'})

# Cria uma cópia profunda do DataFrame original para evitar alterações indesejadas
df_copy = df.copy(deep=True)

# Substitui os valores zero por NaN nas colunas específicas
df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

# Preenche os valores NaN com média ou mediana das respectivas colunas
df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)

# Divide o conjunto de dados em conjuntos de treinamento e teste usando train_test_split
from sklearn.model_selection import train_test_split
X = df.drop(columns='Outcome')
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Cria um modelo de classificação RandomForest com 20 estimadores e o treina com os dados de treinamento
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

# Salva o modelo treinado em um arquivo usando a biblioteca pickle
filename = 'diabetes-prediction-rfc-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))
