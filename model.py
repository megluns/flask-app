import pandas as pd
import numpy as np
import pickle
from sklearn.externals import joblib

dataset = pd.read_csv('Data Skripsi.csv')

#label Encoder
from sklearn import preprocessing
category_col =['Produk', 'Merek', 'Model','Fisik', 'Karet'] 
labelEncoder = preprocessing.LabelEncoder()

mapping_dict={}
for col in category_col:
    dataset[col] = labelEncoder.fit_transform(dataset[col])
    le_name_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
    mapping_dict[col]=le_name_mapping
print(mapping_dict)

X = dataset.drop(["Harga"], axis=1)
X.head()

y=dataset["Harga"]
y.head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=29, random_state=1234)
regressor.fit(X_train, y_train)
y_pred=regressor.predict(X_test)

#save model
pickle.dump(regressor, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2,0,175,0,0]]))
