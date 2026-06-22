from ml import load_data,process_data
from ml import train_model,save_model


# Load the dataset
df1=load_data(r"data\raw\AirQualityData.csv")

x=['pollutant_min',
    'pollutant_max',
    'pollutant_avg']

cat=['state',
    'city',
    'pollutant_id']

round_off=['latitude',
    'longitude']

drop=['country',
    'station',
    'last_update']
     

df,encoders=process_data(df1,x,cat,drop_columns=drop,round_off=round_off) 

model,X_test,y_test=train_model(df,'pollutant_max')
# Dump the model
save_model(model,'model/model2.pkl')

    
    

    
    
