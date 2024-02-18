import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import parallel_backend
import pickle
import time
import sys

#n=int(input("Enter n_jobs:" ))
# Importing the dataset
dataset = pd.read_csv('/home1/09135/sgupte/dsc520-2022-sakshigupte/project/loan_data.csv')

col_names=dataset.columns.values

for i in range(len(col_names)):
	if pd.isnull(dataset[col_names[i]]).sum()==0:
		print()
	else:
		if i < 5 or i > 7:
			mode=dataset[col_names[i]].mode()
			dataset[col_names[i]].fillna(mode.loc[0], inplace=True)
		else:
			dataset[col_names[i]].fillna(dataset[col_names[i]].mean(), inplace=True)
    			
category_col =['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area','Loan_Status'] 
labelEncoder = preprocessing.LabelEncoder() 

# mapping_dict ={} 
for col in category_col: 
	dataset[col] = labelEncoder.fit_transform(dataset[col]) 

# 	le_name_mapping = dict(zip(labelEncoder.classes_, 
# 	labelEncoder.transform(labelEncoder.classes_))) 

# 	mapping_dict[col]= le_name_mapping 
# print(mapping_dict) 

X = dataset.iloc[:,0:11].values
Y = dataset.iloc[:,11:].values


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 0) 

#n = [sys.argc]
#print(n)
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0, n_jobs = 1000 )
begin=time.time()
classifier.fit(X_train,y_train)
finish=time.time()

print(f"Time taken to fit classsifier is: {finish-begin} ")
#Prediction
y_pred = classifier.predict(X_test)


start= time.time()
print("Random Forest Classifier Accuracy is: ",
			accuracy_score(y_test,y_pred)*100 ) 
end = time.time()

print(f"Time taken by classsifier is: {end-start} ") 

with open('rf_classifier.pickle', 'wb') as file:
	pickle.dump(classifier, file, pickle.HIGHEST_PROTOCOL)
    
    
