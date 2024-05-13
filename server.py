from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/triggerml', methods=["POST"])  # Allow both GET and POST requests
def trigger_ml():

    train_errors, test_errors = ml()
    response = {
        "train_errors": train_errors,
        "test_errors": test_errors
    }
    return jsonify(response)

def ml():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, LSTM
    from keras.optimizers import Adam
    from keras.callbacks import ModelCheckpoint
    import math
    from sklearn.metrics import mean_squared_error

    # Load data
    df = pd.read_csv("Coastal Data System - Waves (Mooloolaba) 01-2017 to 06 - 2019.csv")

    # Deleting NaN values
    df.replace(-99.90, np.nan, inplace=True)
    df.drop('Date/Time', axis=1, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Visualizing data
    df_graph = df.loc[0:100]
    plt.figure(figsize=(15,22))
    plt.subplot(6,2,1)
    plt.plot(df_graph['Hs'], color='blue')
    plt.title('Significant Wave Height')

    plt.subplot(6,2,2)
    plt.plot(df_graph['Hmax'], color='red')
    plt.title('Maximum Wave Height')

    plt.subplot(6,2,3)
    plt.plot(df_graph['Tz'], color='orange')
    plt.title('Zero Upcrossing Wave Period')

    plt.subplot(6,2,4)
    plt.plot(df_graph['Tp'], color='brown')
    plt.title('The Peak Energy Wave Period')

    plt.subplot(6,2,5)
    plt.plot(df_graph['Peak Direction'], color='purple')
    plt.title('Direction Related to True North')

    plt.subplot(6,2,6)
    plt.plot(df_graph['SST'], color='green')
    plt.title('Sea Surface Temperature')
    # plt.show();

    # Correlation Matrix Visualization
    plt.figure(figsize=(7,7))
    sns.heatmap(df.corr(), linewidth=.1, annot=True, cmap='YlGnBu')
    plt.title('Correlation Matrix')
    # plt.show();

    # Scaling data
    scaler = MinMaxScaler(feature_range=(0,1))
    data = scaler.fit_transform(df)

    # Separate data into train and test sets
    train = data[:42000,]
    test = data[42000: ,]

    # Prepare data for LSTM
    def prepare_data(data):
        databatch = 30
        x_list = []
        y_list = []
        
        for i in range(len(data)-databatch-1):
            x_list.append(data[i:i+databatch])
            y_list.append(data[i+databatch+1])
            
        X_data = np.array(x_list)
        X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[2], X_data.shape[1]))
        y_data = np.array(y_list)
        
        return X_data, y_data

    X_train, y_train = prepare_data(train)
    X_test, y_test = prepare_data(test)

    # Define LSTM model
    def lstm_model(x_data, y_data, num_epochs, batch_size, learning_rate):
        model = Sequential()
        model.add(LSTM(32, input_shape=(x_data.shape[1], x_data.shape[2]), return_sequences=True))
        model.add(LSTM(16, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(10))
        model.add(Dense(6))
        
        optimizer = Adam(learning_rate=learning_rate)  # Modified this line
        
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
        
        history = model.fit(x_data, y_data, validation_split=0.25, epochs=num_epochs, batch_size=batch_size)
        
        return model, history

    # Train the model
    history = lstm_model(X_train, y_train, num_epochs=15, batch_size=200, learning_rate=0.001)  # Modified this line

    # Predict data
    def predicting(data, y_real):
        predicted_data = history[0].predict(data)
        predicted_data = scaler.inverse_transform(predicted_data)
        y_real = scaler.inverse_transform(y_real)
        
        return predicted_data, y_real

    train_prediction, y_train = predicting(X_train, y_train)
    test_prediction, y_test = predicting(X_test, y_test)

    # RMSE calculation
    def examine_rmse(y_data, predicted_data):
        Score_Hs = math.sqrt(mean_squared_error(y_data[:,0], predicted_data[:,0]))
        Score_Hmax = math.sqrt(mean_squared_error(y_data[:,1], predicted_data[:,1]))
        Score_Tz = math.sqrt(mean_squared_error(y_data[:,2], predicted_data[:,2]))
        Score_Tp = math.sqrt(mean_squared_error(y_data[:,3], predicted_data[:,3]))
        Score_Dir = math.sqrt(mean_squared_error(y_data[:,4], predicted_data[:,4]))
        Score_SST = math.sqrt(mean_squared_error(y_data[:,5], predicted_data[:,5]))
        
        return Score_Hs, Score_Hmax, Score_Tz, Score_Tp, Score_Dir, Score_SST

    train_errors = examine_rmse(y_train, train_prediction)
    test_errors = examine_rmse(y_test, test_prediction)

    # Visualizing real and predicted values
    plt.figure(figsize=(17,25))
    for i in range(6):
        plt.subplot(6,2,i+1)
        plt.plot(test_prediction[1300:,i], color='red', alpha=0.7, label='prediction')
        plt.plot(y_test[1300:,i], color='blue', alpha=0.5, label='real')
        plt.title(df.columns[i])
        plt.legend()
        plt.grid(axis='y')
    # plt.show();

    # Visualizing learning process
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(history[1].history['mean_squared_error'], color='blue', label='Train Evaluation')
    plt.plot(history[1].history['val_mean_squared_error'], color='red', label='Validation Evaluation')
    plt.title('Train vs Validation Evaluation Metrics')
    plt.xlabel('Number of Epochs')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history[1].history['loss'], color='blue', label='Train Loss')
    plt.plot(history[1].history['val_loss'], color='red', label='Validation Loss')
    plt.title('Train vs Validation Loss')
    plt.xlabel('Number of Epochs')
    plt.legend()
    # plt.show();

    return train_errors, test_errors


if __name__ == '__main__':
    app.run(debug=False)
