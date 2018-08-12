import numpy as np
import pandas as pd


dataset = pd.read_csv('train3.csv')
X = dataset.iloc[:, 1: 11].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1= LabelEncoder()
X[:, 0] = labelencoder_X_1.fit_transform(X[:, 0])
X[:, 4] = labelencoder_X_1.fit_transform(X[:, 4])
X[:,5] = labelencoder_X_1.fit_transform(X[:, 5])
X[:, 6] = labelencoder_X_1.fit_transform(X[:, 6])
X[:, 9] = labelencoder_X_1.fit_transform(X[:, 9])

#Gender
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
X=np.delete(X,2, 1)
X=X[:,1:]

#Work
onehotencoder = OneHotEncoder(categorical_features = [5])
X = onehotencoder.fit_transform(X).toarray()
X=X[:,1:]

#Smoker
onehotencoder = OneHotEncoder(categorical_features = [12])
X = onehotencoder.fit_transform(X).toarray()
X=X[:,1:]


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
classifier=Sequential()
classifier.add(Dense(units=32,activation='relu',kernel_initializer='glorot_uniform',input_dim=15))
classifier.add(Dropout(p=0.1))
classifier.add(Dense(units=16,activation='relu',kernel_initializer='glorot_uniform'))
classifier.add(Dropout(p=0.1)) 
classifier.add(Dense(units=16,activation='relu',kernel_initializer='glorot_uniform'))
classifier.add(Dropout(p=0.1)) 
classifier.add(Dense(units=1,activation='sigmoid',kernel_initializer='glorot_uniform'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(X_train,y_train,batch_size=10,epochs=5)

classifier.save_weights('Stroke.h5')

y_pred = classifier.predict(X_test)
y_pred=(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Test
dataset2 = pd.read_csv('test.csv')
X = dataset2.iloc[:, 1: 11].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1= LabelEncoder()
X[:, 0] = labelencoder_X_1.fit_transform(X[:, 0])
X[:, 4] = labelencoder_X_1.fit_transform(X[:, 4])
X[:,5] = labelencoder_X_1.fit_transform(X[:, 5])
X[:, 6] = labelencoder_X_1.fit_transform(X[:, 6])
X[:, 9] = labelencoder_X_1.fit_transform(X[:, 9])

#Gender
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
X=np.delete(X,2, 1)
X=X[:,1:]

#Work
onehotencoder = OneHotEncoder(categorical_features = [5])
X = onehotencoder.fit_transform(X).toarray()
X=X[:,1:]

#Smoker
onehotencoder = OneHotEncoder(categorical_features = [12])
X = onehotencoder.fit_transform(X).toarray()
X=X[:,1:]

X = sc.fit_transform(X)

y_pred_2 = classifier.predict(X)
y_pred_3=(y_pred_2>0.5)
y_pred_3=y_pred_3*1

newDF = pd.DataFrame(y_pred_3) 
newDF.to_csv('submission2.csv', index=False)