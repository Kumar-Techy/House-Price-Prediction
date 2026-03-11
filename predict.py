# predict.py
import joblib
import numpy as np
import pandas as pd

class HousePricePredictor:
    def __init__(self, model_path='house_price_model.pkl'):
        """Initialize the predictor by loading the trained model"""
        print("Loading model...")
        self.artifacts = joblib.load(model_path)
        self.model = self.artifacts['model']
        self.scaler = self.artifacts['scaler']
        self.encoders = self.artifacts['label_encoders']
        self.feature_names = self.artifacts['feature_names']
        print("✅ Model loaded successfully!")
    
    def predict(self, area, bedrooms, bathrooms, floors, year_built, 
                location, condition, garage, current_year=2024):
        """
        Predict house price based on features
        """
        # Calculate engineered features
        age = current_year - year_built
        total_rooms = bedrooms + bathrooms
        area_per_room = area / (total_rooms + 1)
        
        # Encode categorical variables
        location_enc = self.encoders['Location'].transform([location])[0]
        condition_enc = self.encoders['Condition'].transform([condition])[0]
        garage_enc = self.encoders['Garage'].transform([garage])[0]
        
        # Create feature array
        features = np.array([[area, bedrooms, bathrooms, floors, year_built,
                              location_enc, condition_enc, garage_enc,
                              age, total_rooms, area_per_room]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        price_log = self.model.predict(features_scaled)[0]
        price = np.expm1(price_log)  # Convert back from log scale
        
        return price
    
    def predict_batch(self, house_data_list):
        """Predict prices for multiple houses"""
        predictions = []
        for house in house_data_list:
            price = self.predict(**house)
            predictions.append(price)
        return predictions

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = HousePricePredictor()
    
    # Single prediction example
    print("\n" + "="*50)
    print("SINGLE HOUSE PREDICTION")
    print("="*50)
    
    price = predictor.predict(
        area=2500,
        bedrooms=3,
        bathrooms=2,
        floors=2,
        year_built=2010,
        location='Suburban',
        condition='Good',
        garage='Yes'
    )
    print(f"Predicted Price: ${price:,.2f}")
    
    # Batch prediction example
    print("\n" + "="*50)
    print("BATCH PREDICTION")
    print("="*50)
    
    test_houses = [
        {"area": 1500, "bedrooms": 2, "bathrooms": 1, "floors": 1, 
         "year_built": 1995, "location": "Urban", "condition": "Fair", "garage": "No"},
        {"area": 4000, "bedrooms": 5, "bathrooms": 4, "floors": 3, 
         "year_built": 2020, "location": "Downtown", "condition": "Excellent", "garage": "Yes"},
        {"area": 1800, "bedrooms": 3, "bathrooms": 2, "floors": 2, 
         "year_built": 2005, "location": "Rural", "condition": "Good", "garage": "Yes"}
    ]
    
    predictions = predictor.predict_batch(test_houses)
    
    for i, (house, price) in enumerate(zip(test_houses, predictions)):
        print(f"\nHouse {i+1}:")
        print(f"  Features: {house}")
        print(f"  Predicted Price: ${price:,.2f}")