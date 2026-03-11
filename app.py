# app.py
from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
print("Loading model...")
try:
    model_artifacts = joblib.load('house_price_model.pkl')
    model = model_artifacts['model']
    scaler = model_artifacts['scaler']
    encoders = model_artifacts['label_encoders']
    print("✅ Model loaded successfully!")
except:
    print("❌ Error: Model file not found!")
    print("Please run 'py train_model.py' first")
    exit()

# HTML Template for the web interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>House Price Predictor</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 30px 60px rgba(0,0,0,0.3);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 10px;
            font-size: 2.5em;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            color: #555;
            font-weight: 600;
        }
        input, select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        input:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }
        button {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 20px;
            transition: transform 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        }
        .result-card {
            margin-top: 30px;
            padding: 30px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 15px;
            text-align: center;
            display: none;
            animation: slideUp 0.5s ease;
        }
        .result-card.show {
            display: block;
        }
        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        .price {
            font-size: 3.5em;
            font-weight: bold;
            color: #2c3e50;
            margin: 10px 0;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .loading.show {
            display: block;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .footer {
            text-align: center;
            margin-top: 20px;
            color: white;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>House Price Predictor</h1>
        <p class="subtitle">Enter your house details for an instant AI-powered price estimate</p>
        
        <div class="form-grid">
            <div class="form-group">
                <label>Area (sq ft)</label>
                <input type="number" id="area" value="2000" min="500" max="10000">
            </div>
            
            <div class="form-group">
                <label>Bedrooms</label>
                <input type="number" id="bedrooms" value="3" min="1" max="10">
            </div>
            
            <div class="form-group">
                <label>Bathrooms</label>
                <input type="number" id="bathrooms" value="2" min="1" max="10">
            </div>
            
            <div class="form-group">
                <label>Floors</label>
                <input type="number" id="floors" value="2" min="1" max="5">
            </div>
            
            <div class="form-group">
                <label>Year Built</label>
                <input type="number" id="yearBuilt" value="2010" min="1900" max="2024">
            </div>
            
            <div class="form-group">
                <label>Location</label>
                <select id="location">
                    <option value="Urban">Urban</option>
                    <option value="Suburban" selected>Suburban</option>
                    <option value="Rural">Rural</option>
                    <option value="Downtown">Downtown</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>Condition</label>
                <select id="condition">
                    <option value="Excellent">Excellent</option>
                    <option value="Good" selected>Good</option>
                    <option value="Fair">Fair</option>
                    <option value="Poor">Poor</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>Garage</label>
                <select id="garage">
                    <option value="Yes" selected>Yes</option>
                    <option value="No">No</option>
                </select>
            </div>
        </div>
        
        <button onclick="predictPrice()">Predict Price</button>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Calculating prediction...</p>
        </div>
        
        <div class="result-card" id="result">
            <h2>Estimated Price</h2>
            <div class="price" id="price">$0</div>
            <p id="details"></p>
        </div>
    </div>
    
    <div class="footer">
        <p>Powered by Machine Learning | Stacking Ensemble Model</p>
    </div>

    <script>
        async function predictPrice() {
            // Show loading
            document.getElementById('loading').classList.add('show');
            document.getElementById('result').classList.remove('show');
            
            // Get form values
            const data = {
                area: parseFloat(document.getElementById('area').value),
                bedrooms: parseInt(document.getElementById('bedrooms').value),
                bathrooms: parseInt(document.getElementById('bathrooms').value),
                floors: parseInt(document.getElementById('floors').value),
                year_built: parseInt(document.getElementById('yearBuilt').value),
                location: document.getElementById('location').value,
                condition: document.getElementById('condition').value,
                garage: document.getElementById('garage').value
            };
            
            try {
                // Call API
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                // Format and display price
                const formatter = new Intl.NumberFormat('en-US', {
                    style: 'currency',
                    currency: 'USD',
                    minimumFractionDigits: 0,
                    maximumFractionDigits: 0
                });
                
                document.getElementById('price').textContent = formatter.format(result.predicted_price);
                document.getElementById('details').innerHTML = 
                    `Based on ${data.area} sq ft, ${data.bedrooms} bedrooms, ${data.bathrooms} bathrooms`;
                document.getElementById('result').classList.add('show');
            } catch (error) {
                alert('Error predicting price. Please try again.');
                console.error(error);
            } finally {
                document.getElementById('loading').classList.remove('show');
            }
        }

        // Add default values on page load
        window.onload = function() {
            document.getElementById('area').value = 2000;
            document.getElementById('bedrooms').value = 3;
            document.getElementById('bathrooms').value = 2;
            document.getElementById('floors').value = 2;
            document.getElementById('yearBuilt').value = 2010;
        }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Calculate engineered features
        current_year = 2024
        age = current_year - data['year_built']
        total_rooms = data['bedrooms'] + data['bathrooms']
        area_per_room = data['area'] / (total_rooms + 1)
        
        # Encode categorical variables
        location_enc = encoders['Location'].transform([data['location']])[0]
        condition_enc = encoders['Condition'].transform([data['condition']])[0]
        garage_enc = encoders['Garage'].transform([data['garage']])[0]
        
        # Create feature array
        features = np.array([[
            data['area'], data['bedrooms'], data['bathrooms'], data['floors'],
            data['year_built'], location_enc, condition_enc, garage_enc,
            age, total_rooms, area_per_room
        ]])
        
        # Scale and predict
        features_scaled = scaler.transform(features)
        price_log = model.predict(features_scaled)[0]
        price = float(np.expm1(price_log))
        
        return jsonify({'predicted_price': price})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Starting House Price Predictor Web App")
    print("="*60)
    print("Open your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    app.run(debug=True, port=5000)