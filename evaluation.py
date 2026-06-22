from ml.model import load_model,inferance_model,metrix_cal,train_model
from ml import load_data
import os
import json

# Create Json File For Metrices
metries= os.path.join(r"D:\projectroot","model","metrices.json")

# Load The Dataset
df=load_data(r"data/processed/processed_data1.csv")

#Print The dataset For Refrence
print(df.head())

# Train The Processed Dataset

model,X_test,y_test=train_model(df,'pollutant_max')

# Load The model by Load_Model

model=load_model(r"model/model2.pkl")

# Perform Inferance

y_pred=inferance_model(model,X_test)

print(" The Predicted Data")
print(y_pred)

print(" The Actual Data")
print(y_test)

# Calculate Metrices

mae,mse,r2,mape,evs=metrix_cal(y_test,y_pred)

metrix={
    "MAE":round(mae,3),
    "MSE":round(mse,3),
    "R2":round(r2,3),
    "MAPE":round(mape,3),
    "EVS":round(evs,3),
}

with open(metries, 'w') as json_file:
    json.dump(metrix, json_file, indent=4)





