import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import requests
from flask import render_template,send_file


 
def preprocessdata(symbol,region,next,choose):
    symbol= symbol.upper()
    convert = {'APPLE':'AAPL','TESLA':'TSLA','AMAZON':'AMZN','MICROSOFT':'MSFT','GOOGLE':'GOOGL',"RELIANCE":'RELI','NVDIA':'NVDA'}
    symbol=convert[symbol]
    region = region.upper()
    convert1={'INDIA':'IND','USA':'US','US':'US'}
    region=convert1[region]
    next=int(next)
    choose=choose.upper()
    url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/stock/v3/get-historical-data"

    querystring = {"symbol":symbol,"region":region}

    headers = {
        "X-RapidAPI-Key": "471c0140e1msh69669805eea0ce9p1a03a3jsnbd992334f536",
        "X-RapidAPI-Host": "apidojo-yahoo-finance-v1.p.rapidapi.com"
    }
    try:
        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status() 
        b = response.json()
        data = pd.DataFrame(b["prices"])
        data=data[['date','open','high','low','close','volume']]
        from sklearn.impute import SimpleImputer
        imputer=SimpleImputer(missing_values=np.nan,strategy="mean")
        imputer.fit(data.iloc[:,1:])
        data.iloc[:,1:]=imputer.transform(data.iloc[:,1:])
        x=data.drop('close',axis=1)
        y=data.iloc[:,-2:-1]
        # Extract the relevant feature (e.g., adjusted closing prices)
        price_data = data['close'].values.reshape(-1, 1)
        # print(price_data)
        # Normalize the data
        scaler = MinMaxScaler()
        price_data = scaler.fit_transform(price_data)

        # Define the sequence length (number of past days to consider)
        sequence_length = next

        # Create sequences of historical data
        sequences = []
        target = []
        # print(price_data)
        # print(len(price_data))
        for i in range(len(price_data) - sequence_length):
            sequences.append(price_data[i+1:(i+1)+sequence_length][::-1])
            target.append(price_data[i])


        X = np.array(sequences)
        y = np.array(target)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Build the RNN model
        model = keras.Sequential([
            layers.LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
            layers.Dense(250, activation='relu'),
            layers.Dense(750, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1150, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(550, activation='relu'),
            layers.Dense(250, activation='relu'),
            layers.Dense(1)
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Define the EarlyStopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # Train the model
        model.fit(X_train, y_train, epochs=100, batch_size=32 ,validation_data=(X_test,y_test),callbacks=[early_stopping])

        # Evaluate the model
        loss = model.evaluate(X_test, y_test)
   

        # Make predictions
        predictions = model.predict(X_test)

        from sklearn.metrics import r2_score
        r=r2_score(y_test,predictions)
       

        # Inverse transform the predictions to get real prices
        predictions = scaler.inverse_transform(predictions)

        # You can use the predictions for forecasting stock prices

    
        # Define the number of days to predict (10 in this case)
        num_days_to_predict = next

        # Initialize the last available sequence (e.g., the last sequence from your test data)
        last_sequence = price_data[:num_days_to_predict][::-1]
        # print(last_sequence)
        # Initialize a list to store the predicted prices
        predicted_prices = []
        next_days=[]
        # Make predictions for the next n days
        for _ in range(num_days_to_predict):
            # Reshape the last_sequence to match the model's input shape
            last_sequence_reshaped = last_sequence.reshape(1, sequence_length, 1)
            # print(last_sequence_reshaped)
            # Make a prediction for the next day
            next_day_prediction = model.predict(last_sequence_reshaped)
            # Inverse transform the prediction to get the real price
            next_day_prediction = scaler.inverse_transform(next_day_prediction)
            next_days.append(next_day_prediction[0][0])
            print(next_day_prediction)
            # Append the prediction to the list of predicted prices
            predicted_prices.append(next_day_prediction[0, 0])
            next_1=scaler.transform((next_day_prediction))
            # print("next:",next[0][0])
            # Update the last_sequence with the new prediction for the next day
            last_sequence = np.append(last_sequence[1:],next_1)
            # print(last_sequence)
       
        if choose=="YES":
            return predicted_prices
        else:
            return (scaler.inverse_transform(np.array(last_sequence[-1]).reshape(-1,1)))[0][0]
        # predicted_prices now contains the forecasted stock prices for the next ten days
        
    except requests.exceptions.RequestException as e:
        print(e)
        return f'An error occurred: {str(e)}'

