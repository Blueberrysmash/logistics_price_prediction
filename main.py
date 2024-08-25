# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 19:34:33 2024

@author: Asus
"""

import uvicorn
from fastapi import FastAPI, HTTPException
import numpy as np
import pandas as pd
import pickle
from BankNotes import BankNote


app = FastAPI()

# Load the model and scaler
try:
    pickle_in = open("logistics_price_prediction.pkl", "rb")
    lr_model = pickle.load(pickle_in)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="Model not found")
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

# Index route
@app.get('/')
def index():
    return {'message': 'Hello, everyone'}

@app.get('/{name}')
def get_name(name: str):
    return {'message': f'Hello, {name}'}

@app.post('/predict')
def predict_price(data: BankNote):
    try:
        data = data.dict()
        Distance = data['Distance']
        ShipmentType = data['ShipmentType']
        NumPackages = data['NumPackages']
        SpecialHandling = data['SpecialHandling']
        PackageWeight = data['PackageWeight']
        
        # Validate ShipmentType
        valid_shipment_types = [1, 2, 3]
        if ShipmentType not in valid_shipment_types:
            raise HTTPException(status_code=400, detail=f"Invalid ShipmentType: {ShipmentType}. Must be one of {valid_shipment_types}")

        # Validate SpecialHandling
        valid_special_handling = [0, 1]
        if SpecialHandling not in valid_special_handling:
            raise HTTPException(status_code=400, detail=f"Invalid SpecialHandling: {SpecialHandling}. Must be one of {valid_special_handling}")
        
        input_data = np.asarray([Distance, ShipmentType, NumPackages, SpecialHandling, PackageWeight])
        selected_features =['Distance','ShipmentType','NumPackages','SpecialHandling','PackageWeight']
        input_data = pd.DataFrame([input_data], columns=selected_features)

        #nput_data_reshaped = np.asarray(input_data_array).reshape(1, -1)
        #input_data_reshaped = scaler.transform(input_data_reshaped)
        input_data_reshaped = scaler.transform(input_data)
        prediction = lr_model.predict(input_data_reshaped)
        return {'prediction': prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)