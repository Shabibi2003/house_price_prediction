import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LinearRegression

# Load and preprocess the data
df = pd.read_csv('Housing.csv')
scaler = MinMaxScaler()
df.area = scaler.fit_transform(df[['area']])
df['mainroad'] = df['mainroad'].replace({'yes': 1, 'no': 0})
df['guestroom'] = df['guestroom'].replace({'yes':1,'no':0})
df['basement'] = df['basement'].replace({'yes':1,'no':0})
df['hotwaterheating'] = df['hotwaterheating'].replace({'yes':1 , 'no':0})
df['airconditioning'] = df['airconditioning'].replace({'yes':1 , 'no':0})
df['prefarea'] = df['prefarea'].replace({'yes':1, 'no':0})
encoded = LabelEncoder()
df['furnishingstatus'] = encoded.fit_transform(df['furnishingstatus'])
df['price'] = np.log(df['price'])

# Prepare the data for training
from sklearn.model_selection import train_test_split
X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction function
def make_prediction(area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus):
    # Preprocess user input for prediction
    input_data = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'stories': [stories],
        'mainroad': [1 if mainroad == 'yes' else 0],
        'guestroom': [1 if guestroom == 'yes' else 0],
        'basement': [1 if basement == 'yes' else 0],
        'hotwaterheating': [1 if hotwaterheating == 'yes' else 0],
        'airconditioning': [1 if airconditioning == 'yes' else 0],
        'parking': [parking],
        'prefarea': [1 if prefarea == 'yes' else 0],
        'furnishingstatus': [encoded.transform([furnishingstatus])[0]]  # Encode furnishing status
    })
    
    # Scale the area value
    input_data['area'] = scaler.transform(input_data[['area']])

    # Predict the price using the trained model
    predicted_price_log = model.predict(input_data)
    
    # Convert the predicted log price back to the original scale
    predicted_price = np.exp(predicted_price_log)
    
    return predicted_price[0]

# Streamlit UI for user input
st.title('Housing Price Prediction')
st.write('Enter the details of the house to predict the price.')

# User inputs for prediction
area = st.number_input('Area (in sqft)', min_value=0, value=1200)
bedrooms = st.number_input('Number of Bedrooms', min_value=0, value=3)
bathrooms = st.number_input('Number of Bathrooms', min_value=0, value=2)
stories = st.number_input('Number of Stories', min_value=0, value=1)
mainroad = st.selectbox('Is the house on Main Road?', ['yes', 'no'])
guestroom = st.selectbox('Does the house have a Guestroom?', ['yes', 'no'])
basement = st.selectbox('Does the house have a Basement?', ['yes', 'no'])
hotwaterheating = st.selectbox('Does the house have Hot Water Heating?', ['yes', 'no'])
airconditioning = st.selectbox('Does the house have Air Conditioning?', ['yes', 'no'])
parking = st.number_input('Number of Parking Spaces', min_value=0, value=2)
prefarea = st.selectbox('Is the house in a Preferred Area?', ['yes', 'no'])
furnishingstatus = st.selectbox('Furnishing Status', ['unfurnished', 'semi-furnished', 'furnished'])

# Predict button
if st.button('Predict Price'):
    predicted_price = make_prediction(area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus)
    st.write(f"The predicted price of the house is: ₹{predicted_price:,.2f}")
