from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb

# Load trained model and label encoder
model = joblib.load("crops/ml_model/crop_model.pkl")
label_encoder = joblib.load("crops/ml_model/label_encoder.pkl")
label_encoders = joblib.load("crops/ml_model/label_encoders.pkl")

# Crop emoji mapping
crop_emoji_mapping = {
    'Rice': '🌾',
    'Maize (Corn)': '🌽',
    'Chickpea': '🫘',
    'Kidney Beans': '🫘',
    'Pigeon Peas': '🫘',
    'Moth Beans': '🫘',
    'Mung Bean': '🫘',
    'Black Gram': '🫘',
    'Lentil': '🫘',
    'Pomegranate': '🍎',
    'Banana': '🍌',
    'Mango': '🥭',
    'Grapes': '🍇',
    'Watermelon': '🍉',
    'Muskmelon': '🍈',
    'Apple': '🍎',
    'Orange': '🍊',
    'Papaya': '🥭',
    'Coconut': '🥥',
    'Cotton': '🌱',
    'Jute': '🪢',
    'Coffee': '☕'
}

# Soil & weather emoji mapping
emoji_mapping = {
    'N': ('🌿', 'N (Nitrogen ratio)'),
    'P': ('⚡', 'P (Phosphorous ratio)'),
    'K': ('💧', 'K (Potassium ratio)'),
    'temperature': ('🌡️', '°C (Temperature in Celsius)'),
    'humidity': ('💧', '% (Humidity)'),
    'ph': ('🔬', 'pH (Soil pH value)'),
    'rainfall': ('🌧️', 'mm (Rainfall in mm)')
}

def crop_prediction_page(request):
    return render(request, "crop_prediction.html")

def predict_crop(request):
    if request.method == "GET":
        try:
            temperature = request.GET.get("temperature")
            humidity = request.GET.get("humidity")
            rainfall = request.GET.get("rainfall")
            ph = request.GET.get("ph")
            N = request.GET.get("N")
            P = request.GET.get("P")
            K = request.GET.get("K")

            if None in [temperature, humidity, rainfall, ph, N, P, K]:
                return JsonResponse({"error": "Missing required parameters"}, status=400)

            # Convert to float
            temperature = float(temperature)
            humidity = float(humidity)
            rainfall = float(rainfall)
            ph = float(ph)
            N = float(N)
            P = float(P)
            K = float(K)

            # Prediction logic
            input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], 
                                      columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])

            prediction = model.predict(input_data)
            predicted_crop = label_encoder.inverse_transform(prediction)[0]

            crop_emoji = crop_emoji_mapping.get(predicted_crop, "🌱")
            input_data_with_emojis = {
                "N": f"{N} {emoji_mapping['N']}",
                "P": f"{P} {emoji_mapping['P']}",
                "K": f"{K} {emoji_mapping['K']}",
                "ph": f"{ph} {emoji_mapping['ph']}",
                "temperature": f"{temperature} {emoji_mapping['temperature']}",
                "humidity": f"{humidity} {emoji_mapping['humidity']}",
                "rainfall": f"{rainfall} {emoji_mapping['rainfall']}"
            }

            return JsonResponse({
                "predicted_crop": f"{predicted_crop} {crop_emoji}",
                "input_data": input_data_with_emojis
            })
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

def predict_crop1(state, district, season, area):
    state_encoded = label_encoders["State_Name"].transform([state])[0]
    district_encoded = label_encoders["District_Name"].transform([district])[0]
    season_encoded = label_encoders["Season"].transform([season])[0]

    input_data = [[state_encoded, district_encoded, season_encoded, area]]
    predicted_crop_encoded = model.predict(input_data)[0]

    # Convert back to crop name
    predicted_crop = label_encoders["Crop"].inverse_transform([predicted_crop_encoded])[0]
    return predicted_crop



# Load models & encoders
crop_model2 = joblib.load("crops/ml_model2/crop_predict_model.pkl")
yield_model2 = joblib.load("crops/ml_model2/yield_predict_model.pkl")
label_encoders2 = joblib.load("crops/ml_model2/label_encoders.pkl")

@csrf_exempt
def predict_crops(request):
    if request.method == "POST":
        data = json.loads(request.body)

        # Extract input values
        state = data["state"]
        district = data["district"]
        season = data["season"]
        area = float(data["area"])

        # Encode inputs
        state_encoded = label_encoders2["State_Name"].transform([state])[0]
        district_encoded = label_encoders2["District_Name"].transform([district])[0]
        season_encoded = label_encoders2["Season"].transform([season])[0]

        # Predict Crop
        input_data = [[state_encoded, district_encoded, season_encoded, area]]
        predicted_crop_encoded = crop_model2.predict(input_data)[0]
        predicted_crop = label_encoders2["Crop"].inverse_transform([predicted_crop_encoded])[0]

        return JsonResponse({"recommended_crop": predicted_crop})

@csrf_exempt
def predict_yield(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)

            # Extract input values
            state = data.get("state", "").strip()
            district = data.get("district", "").strip()  # District is now optional
            season = data.get("season", "").strip()
            crop = data.get("crop", "").strip()  # Crop is optional
            area = float(data.get("area", 0))

            # Ensure required fields are present
            if not state or not season or area <= 0:
                return JsonResponse({"error": "State, season, and area are required."}, status=400)

            # Function to safely encode inputs
            def safe_encode(label, encoder, default_value=-1):
                return encoder.transform([label])[0] if label in encoder.classes_ else default_value

            # Encode state, district (if provided), and season
            state_encoded = safe_encode(state, label_encoders2["State_Name"])
            print(state_encoded, 'state_encoded')
            district_encoded = safe_encode(district, label_encoders2["District_Name"]) if district else -1
            season_encoded = safe_encode(season, label_encoders2["Season"])
            print(season_encoded, 'season_encoded')
            # Ensure encoding was successful
            if state_encoded == -1 or season_encoded == -1:
                return JsonResponse({"error": "Invalid state or season."}, status=400)

            # **CASE 1: Crop is provided → Predict yield & price**
            if crop:
                crop_encoded = safe_encode(crop, label_encoders2["Crop"])

                if crop_encoded == -1:
                    return JsonResponse({"error": "Invalid crop name."}, status=400)

                input_data = np.array([[state_encoded, district_encoded, season_encoded, crop_encoded, area]])
                prediction = yield_model2.predict(input_data)[0]

                response_data = {
                    "predicted_production": float(prediction[1]),
                    "min_price": float(prediction[2]),
                    "max_price": float(prediction[3]),
                    "modal_price": float(prediction[4]),
                }

                return JsonResponse(response_data)

            # **CASE 2: Crop is missing but District is provided → Return Top 10 Highest-Yielding Crops**
            elif district:
                yield_predictions = []

                for crop_name in label_encoders2["Crop"].classes_:
                    crop_encoded = label_encoders2["Crop"].transform([crop_name])[0]
                    input_data = np.array([[state_encoded, district_encoded, season_encoded, crop_encoded, area]])

                    prediction = yield_model2.predict(input_data)[0]

                    yield_predictions.append({
                        "crop": crop_name,
                        "predicted_production": float(prediction[1]),
                        "min_price": float(prediction[2]),
                        "max_price": float(prediction[3]),
                        "modal_price": float(prediction[4]),
                    })

                top_10_crops = sorted(yield_predictions, key=lambda x: x["predicted_production"], reverse=True)[:10]
                return JsonResponse({"message": f"Top 10 crops in {district}, {state}", "top_yielding_crops": top_10_crops})

            # **CASE 3: Both Crop and District are missing → Return Top 5 Highest-Yielding Crops for Each District**
            else:
                district_to_state = joblib.load("crops/ml_model2/district_to_state.pkl")
                # 🔹 Get districts belonging to the queried state
                valid_districts = [
                    [district for district, state in district_to_state.items() if state == state_encoded]
                ]
                if not valid_districts:
                    return JsonResponse({"error": "No valid districts found for the provided state."}, status=400)
                
                district_yield_predictions = {}

                for district_encoded in valid_districts[0]:
                    yield_predictions = []

                    for crop_name in label_encoders2["Crop"].classes_:
                        crop_encoded = label_encoders2["Crop"].transform([crop_name])[0]
                        input_data = np.array([[state_encoded, district_encoded, season_encoded, crop_encoded, area]])

                        prediction = yield_model2.predict(input_data)[0]

                        yield_predictions.append({
                            "crop": crop_name,
                            "predicted_production": float(prediction[1]),
                            "min_price": float(prediction[2]),
                            "max_price": float(prediction[3]),
                            "modal_price": float(prediction[4]),
                        })

                    # Get top 5 yielding crops for this district
                    # Filter out predictions without predicted production
                    valid_yield_predictions = [yp for yp in yield_predictions if yp["predicted_production"] > 0]
                    district_name = label_encoders2["District_Name"].inverse_transform([district_encoded])[0]

                    district_yield_predictions[district_name] = sorted(
                        valid_yield_predictions, key=lambda x: x["predicted_production"], reverse=True
                    )[:5]

                return JsonResponse({"message": f"Top 5 crops per district in {state}", "district_yield_predictions": district_yield_predictions})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)


def get_states_and_districts(request):
    try:
        # Load the district-to-state mapping
        district_to_state = joblib.load("crops/ml_model2/district_to_state.pkl")

        # Load the label encoder to decode state and district names
        label_encoders = joblib.load("crops/ml_model2/label_encoders.pkl")

        # Dictionary to store states with their corresponding district names
        states_districts = {}

        for district_encoded, state_encoded in district_to_state.items():
            # Decode state and district names
            state_name = label_encoders["State_Name"].inverse_transform([state_encoded])[0]
            district_name = label_encoders["District_Name"].inverse_transform([district_encoded])[0]

            if state_name not in states_districts:
                states_districts[state_name] = []
            states_districts[state_name].append(district_name)

        return JsonResponse({"states": states_districts}, safe=False)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
def recommend_profitable_crops(request):
    try:
        data = json.loads(request.body)

        # Extract input values
        state = data.get("state").strip()
        district = data.get("district").strip() 
        area = float(data.get("area"))

        if not state or not district or area <= 0:
            return JsonResponse({"error": "State, district, and area are required."}, status=400)

        # Load Models
        xgb_model = joblib.load("crops/ml_model_3/revenue_model.pkl")
        state_encoder = joblib.load("crops/ml_model_3/state_encoder.pkl")
        district_encoder = joblib.load("crops/ml_model_3/district_encoder.pkl")
        crop_encoder = joblib.load("crops/ml_model_3/crop_encoder.pkl")

        # Encode Inputs
        state_code = state_encoder.transform([state])[0]
        district_code = district_encoder.transform([district])[0]
        input_data = np.array([[state_code, district_code, area]])

        # Predict Crop & Other Values
        predicted_values = xgb_model.predict(input_data)

        print("Predicted Output:", predicted_values)

        columns = ["Crop_Code", "Cost of Production (Rs./Qtl)", "Implicit Rate (Rs./Qtl.)", "Derived Yield (Qtl./Hectare)", "Expected Profit (Rs)"]
        predictions_df = pd.DataFrame(predicted_values, columns=columns)
        predicted_values = predictions_df.sort_values(by="Expected Profit (Rs)", ascending=False).iloc[0]

        # ✅ Fix: Convert crop code to int before using inverse_transform
        predicted_crop_code = int(round(predicted_values[0]))  # Round for safety
        crop_name = crop_encoder.inverse_transform([predicted_crop_code])[0]

        # ✅ Extract correct numerical values
        cost_of_production = float(predicted_values[1]) * area
        implicit_rate = float(predicted_values[2])
        derived_yield = float(predicted_values[3])* area
        expected_profit = float(predicted_values[4]) * area

        # Prepare response
        result = {
            "Crop": crop_name,
            "Cost of Production (Rs)": round(cost_of_production, 2),
            "Implicit Rate (Rs./Qtl.)": round(implicit_rate, 2),
            "Derived Yield (Qtl./Hectare)": round(derived_yield, 2),
            "Expected Profit (Rs)": round(expected_profit, 2),
            "area": area
        }

        return JsonResponse({"profitable_crop": result})

    except Exception as e:
        return JsonResponse({"error": str(e)})

df = pd.read_csv("crops/ml_model_3/merged_dataset_updated.csv")

def get_states(request):
    """Returns a list of unique states from the dataset."""
    states = df["State"].dropna().unique().tolist()
    return JsonResponse({"states": states})

def get_districts(request):
    """Returns districts based on selected state."""
    state = request.GET.get("state")  # Get state from query params
    if not state:
        return JsonResponse({"error": "State parameter is required"}, status=400)

    districts = df[df["State"] == state]["District_Name"].dropna().unique().tolist()
    return JsonResponse({"districts": districts})