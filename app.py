import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Load the Titanic dataset
titanic_data = pd.read_csv('train.csv')

# Data cleaning
titanic_data = titanic_data.drop(columns='Cabin', axis=1)
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

# Data preprocessing
titanic_data.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {'S': 0, 'C': 1, 'Q': 2}}, inplace=True)

# Prepare features and target variable
X = titanic_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
Y = titanic_data['Survived']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Model training
model = LogisticRegression()
model.fit(X_train, Y_train)

# Function to predict survival of a new passenger
def predict_person_survival(features):
    try:
        prediction = model.predict([features])
        return 'Survived' if prediction[0] == 1 else 'Not Survived'
    except Exception as e:
        return f"Error: {e}"

# Streamlit app
st.title('Titanic Survival Prediction App')

# Input fields
pclass = st.selectbox('Passenger Class', [1, 2, 3])
sex = st.selectbox('Sex', ['Male', 'Female'])
age = st.slider('Age', 0, 80, 25)
sibsp = st.number_input('Number of Siblings/Spouses Aboard', 0, 10, 0)
parch = st.number_input('Number of Parents/Children Aboard', 0, 10, 0)
fare = st.number_input('Fare', 0.0, 1000.0, 50.0)
embarked = st.selectbox('Port of Embarkation', ['S', 'C', 'Q'])

# Convert inputs to appropriate format
sex = 0 if sex == 'Male' else 1
embarked = 0 if embarked == 'S' else 1 if embarked == 'C' else 2

# Predict button
if st.button('Predict Survival'):
    features = [pclass, sex, age, sibsp, parch, fare, embarked]
    result = predict_person_survival(features)
    st.write(f'The prediction is: {result}')

# Display data visualizations
st.title('Data Visualizations')

# Survived count
st.subheader('Survived Count')
fig, ax = plt.subplots()
sns.countplot(x='Survived', data=titanic_data, ax=ax)
st.pyplot(fig)

# Gender count
st.subheader('Gender Count')
fig, ax = plt.subplots()
sns.countplot(x='Sex', data=titanic_data, ax=ax)
st.pyplot(fig)

# Survival by gender
st.subheader('Survival by Gender')
fig, ax = plt.subplots()
sns.countplot(x='Sex', hue='Survived', data=titanic_data, ax=ax)
st.pyplot(fig)

# Passenger class count
st.subheader('Passenger Class Count')
fig, ax = plt.subplots()
sns.countplot(x='Pclass', data=titanic_data, ax=ax)
st.pyplot(fig)

# Survival by passenger class
st.subheader('Survival by Passenger Class')
fig, ax = plt.subplots()
sns.countplot(x='Pclass', hue='Survived', data=titanic_data, ax=ax)
st.pyplot(fig)
