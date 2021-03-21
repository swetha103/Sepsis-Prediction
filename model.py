import numpy as np 
import pandas as pd 

data = pd.read_csv('C:\\Users\\shame\\Desktop\\SEPSIS\\Sepsis_final.csv')
print(data.shape)
print(data.head())

import matplotlib.pyplot as plt
count_label = pd.value_counts(data['SepsisLabel'], sort = True)
count_label.plot(kind = 'bar', rot = 0)
plt.title('Data Distribution')
plt.xlabel('Label')
plt.ylabel('Count')

label_0 = data[ data['SepsisLabel'] == 0 ]
label_1 = data[ data['SepsisLabel'] == 1 ]
print(label_0.shape, label_1.shape)

X = data.drop('SepsisLabel',axis = 1).values 
y = data['SepsisLabel'].values 

X = X[:, 1:]
print(X)

from imblearn.under_sampling import NearMiss 
nm = NearMiss()
X_res, y_res = nm.fit_sample(X,y)

print(X_res.shape , y_res.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=0)

import pickle
pickle.dump(X_test, open('X_test.pkl','wb'))



from xgboost import XGBClassifier
model = XGBClassifier(min_child_weight= 3, max_depth=10, learning_rate= 0.3, gamma= 0.4, colsample_bytree= 0.7)
model.fit(X_train,y_train)
print(model.score(X_test,y_test))

import pickle
pickle.dump(model, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

test_data = pickle.load(open('X_test.pkl','rb'))
print(test_data[2])
print(len(test_data))
single_pred = (model.predict(np.array(test_data[11]).reshape(1,-1)))
print(single_pred)
if int(single_pred[0]) == 0:
    print("No chance for sepsis")
else:
    print("There is a chance for sepsis")


 