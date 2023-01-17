from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

#load the data
df = pd.read_csv('../data/dataframes/df.csv')

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

#replace the values of the column "timestamp" with the value of the dictionary
df['timestamp'] = df['timestamp'].map(dict_encoding)

#one hot encoding column h_type
df = pd.get_dummies(df, columns=['h_type'])

#split the data X and y, y is the column consommation
X = df.drop('consommation', axis=1)
y = df['consommation']

# Normalisation des données
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

# Séparation des données en entraînement et test
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2)

# Initialisation du modèle
model = Sequential()

# Ajout de la couche d'entrée et de la première couche cachée
model.add(Dense(32, input_dim=7, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Ajout de la deuxième couche cachée
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Ajout de la troisième couche cachée
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Ajout de la quatreème couche cachée
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Ajout de la couche de sortie
model.add(Dense(1))

# Compilation du modèle
model.compile(loss='mean_squared_error', optimizer=Adam())

# Initialisation de l'early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Entraînement du modèle avec early stopping
model.fit(X_train, y_train, epochs=100, batch_size=256, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose = 1)

model.save('model.h5')