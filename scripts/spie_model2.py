
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping           
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

import tensorflow as tf
from tensorflow import keras


dict_encoding = { "0:00" : 1,
                    "0:30" : 2,
                    "1:00" : 3,
                    "1:30" : 4,
                    "2:00" : 5,
                    "2:30" : 6,
                    "3:00" : 7,
                    "3:30" : 8,
                    "4:00" : 9,
                    "4:30" : 10,
                    "5:00" : 11,
                    "5:30" : 12,
                    "6:00" : 13,
                    "6:30" : 14,
                    "7:00" : 15,
                    "7:30" : 16,    
                    "8:00" : 17,
                    "8:30" : 18,
                    "9:00" : 19,
                    "9:30" : 20,
                    "10:00" : 21,
                    "10:30" : 22,
                    "11:00" : 23, 
                    "11:30" : 24,
                    "12:00" : 25, 
                    "12:30" : 26, 
                    "13:00" : 27, 
                    "13:30" : 28, 
                    "14:00" : 29, 
                    "14:30" : 30, 
                    "15:00" : 31, 
                    "15:30" : 32, 
                    "16:00" : 33, 
                    "16:30" : 34, 
                    "17:00" : 35, 
                    "17:30" : 36, 
                    "18:00" : 37, 
                    "18:30" : 38, 
                    "19:00" : 39, 
                    "19:30" : 40, 
                    "20:00" : 41, 
                    "20:30" : 42, 
                    "21:00" : 43, 
                    "21:30" : 44, 
                    "22:00" : 45, 
                    "22:30" : 46, 
                    "23:00" : 47, 
                    "23:30" : 48
}

#load the data
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/df.csv')

#delete the column "h_ref"
df = df.drop(['h_ref'], axis=1)

#replace the values of the column "timestamp" with the value of the dictionary
df['timestamp'] = df['timestamp'].map(dict_encoding)

#one hot encoding column h_type
df = pd.get_dummies(df, columns=['h_type'])

#split the data X and y, y is the column consommation
X = df.drop('consommation', axis=1)
y = df['consommation']

# Spliting data into Train, Holdout and Test
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split the train data for holdout set
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

def standardize(X_train, X_valid, X_test):
    """
    :param X_train, X_test: training set and test set
    """
    # scaler object
    scaler = StandardScaler()
    
    scaler.fit(X_train)
    
    X_train_scaled = scaler.transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test) # test or valid data
    
    return X_train_scaled,X_valid_scaled, X_test_scaled
    

# Standardize the data
X_train_scaled, X_valid_scaled, X_test_scaled = standardize(X_train, X_valid, X_test)

def functional_model():
    
    input_layer = keras.layers.Input(shape=(X_train_scaled.shape[1],))
    
    hidden_1 = keras.layers.Dense(30, activation='relu')(input_layer) # Passing layer as parameter
    hidden_2 = keras.layers.Dense(30, activation='relu')(hidden_1)
    
    concat = keras.layers.Concatenate()([input_layer, hidden_2]) # concatinates input and output of second layer
    output_layer = keras.layers.Dense(1)(concat)
    
    model = keras.Model(inputs=[input_layer], outputs=[output_layer])
    
    return model
    
# build model
model_b = functional_model()

model_b.summary()

# Compile the model
model_b.compile(loss='mean_squared_error',
                    optimizer='sgd',
                    metrics=['mae'])

# fit the model with data
history = model_b.fit(X_train_scaled, y_train,
                    epochs=20,
                    batch_size=32,
                    validation_data=(X_valid_scaled, y_valid),
                    callbacks=[keras.callbacks.EarlyStopping(patience=5)],
                    verbose=1)

model_b.save('model4.h5')
