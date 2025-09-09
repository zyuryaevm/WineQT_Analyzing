import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('WineQT.csv')

X = data.drop('quality', axis=1)
y = data['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pd.concat([pd.DataFrame(X_train_scaled), y_train], axis=1).to_csv('train.csv', index=False)
pd.concat([pd.DataFrame(X_test_scaled), y_test], axis=1).to_csv('test.csv', index=False)