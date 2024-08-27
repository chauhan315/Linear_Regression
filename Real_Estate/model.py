import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import pickle

data = pd.read_csv('Real_Estate.csv')

df = data[['Distance to the nearest MRT station', 'Number of convenience stores',
           'House price of unit area']].copy()
df.reset_index()

df.rename(columns={
    'Distance to the nearest MRT station': "Distance to Station",
    'Number of convenience stores': "Stores",
    'House price of unit area': "Unit Price"
}, inplace=True)

X = df.drop('Unit Price', axis=1, inplace=False)
y = df['Unit Price']

# scaling
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)

# model creation and training
lm = LinearRegression()
lm.fit(X_train, y_train)

with open("model.pkl", 'wb') as model_pickle:
    pickle.dump(lm, model_pickle)

with open('scaler.pkl', 'wb') as scaler_picker:
    pickle.dump(scaler, scaler_picker)
