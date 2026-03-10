import os
import random
from typing import Optional
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import joblib 

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


CSV_DF: Optional[pd.DataFrame] = None
MODEL = None

# Valid UP districts from the dataset (these have actual data)
DATASET_UP_DISTRICTS = {
    "lucknow", "kanpur", "ghaziabad", "agra", "aligarh", 
    "bareilly", "gorakhpur", "meerut", "prayagraj", "varanasi",
    "allahabad"  # Prayagraj was formerly Allahabad
}

# Keep for backward compatibility
VALID_UP_DISTRICTS = DATASET_UP_DISTRICTS

# UP state name variations
UP_STATE_NAMES = {
    "uttar pradesh", "up", "uttar pradesh, india", "up, india"
}


def _load_csv_if_available() -> Optional[pd.DataFrame]:
    csv_path = os.getenv("CSV_PATH", "").strip()
    if not csv_path:
        return None
    try:
        df = pd.read_csv(csv_path)
        # Normalize column names we expect
        # Expected columns: District, Latitude, Longitude, Hour, Day_of_Week, Is_Night_Risk, Is_High_Density_Area, Target_Risk_Y
        needed = {"District", "Hour", "Target_Risk_Y"}
        if not needed.issubset(set(df.columns)):
            return None
        # Pre-clean: lower-case district for matching
        df["_district_norm"] = df["District"].astype(str).str.strip().str.lower()
        return df
    except Exception:
        return None


def _is_location_in_up(location: str, geocode_result: dict = None) -> bool:
    """Check if a location is in Uttar Pradesh, India."""
    if not location:
        return False
    
    loc_lower = location.strip().lower()
    
    # Check if it's a known UP district
    if loc_lower in VALID_UP_DISTRICTS:
        return True
    
    # Check if location contains UP district name
    for district in VALID_UP_DISTRICTS:
        if district in loc_lower:
            return True
    
    # If geocode result is provided, check administrative areas
    if geocode_result and "results" in geocode_result:
        for result in geocode_result.get("results", []):
            address_components = result.get("address_components", [])
            for component in address_components:
                types = component.get("types", [])
                long_name = component.get("long_name", "").lower()
                short_name = component.get("short_name", "").lower()
                
                # Check if state is Uttar Pradesh
                if "administrative_area_level_1" in types:
                    if "uttar pradesh" in long_name or "up" in short_name.lower():
                        return True
                
                # Check if it's a known district
                if "locality" in types or "sublocality" in types or "administrative_area_level_2" in types:
                    if long_name in VALID_UP_DISTRICTS or any(district in long_name for district in VALID_UP_DISTRICTS):
                        return True
    
    return False


def _hour_for_time_of_day(time_of_day: str) -> int:
    t = (time_of_day or "").lower()
    return {
        "morning": 9,
        "afternoon": 15,
        "evening": 19,
        "night": 23,
    }.get(t, 15)


def _score_from_csv(location: str, time_of_day: str) -> Optional[int]:
    global CSV_DF
    if CSV_DF is None:
        CSV_DF = _load_csv_if_available()
    if CSV_DF is None:
        return None

    loc_norm = (location or "").strip().lower()
    if not loc_norm:
        return None

    hour = _hour_for_time_of_day(time_of_day)
    
    # Try multiple matching strategies for better location matching
    # 1. Exact match
    subset = CSV_DF[CSV_DF["_district_norm"] == loc_norm]
    
    # 2. If empty, try contains (partial match) - case insensitive
    if subset.empty:
        subset = CSV_DF[CSV_DF["_district_norm"].str.contains(loc_norm, na=False, case=False)]
    
    # 3. If still empty, try reverse - check if any district name is in the location string
    if subset.empty:
        for district in CSV_DF["_district_norm"].unique():
            if district in loc_norm:
                subset = CSV_DF[CSV_DF["_district_norm"] == district]
                break
    
    # 4. If still empty, try matching first word (e.g., "Lucknow City" -> "Lucknow")
    if subset.empty:
        first_word = loc_norm.split()[0] if loc_norm.split() else loc_norm
        subset = CSV_DF[CSV_DF["_district_norm"] == first_word]
        if subset.empty:
            subset = CSV_DF[CSV_DF["_district_norm"].str.contains(first_word, na=False, case=False)]
    
    if subset.empty:
        return None

    # Prefer rows matching the hour; fallback to all rows for that district
    near = subset[subset["Hour"] == hour]
    use_df = near if not near.empty else subset

    # Compute mean of Target_Risk_Y and convert to a 0-100 score
    mean_risk = float(use_df["Target_Risk_Y"].mean())
    score = int(round((1.0 - mean_risk) * 100))  # higher score = safer
    return max(0, min(100, score))


def _predict_with_model(location: str, time_of_day: str) -> Optional[int]:
    global MODEL
    if MODEL is None:
        model_path = os.getenv("MODEL_PATH", "").strip()
        if model_path and os.path.exists(model_path):
            try:
                MODEL = joblib.load(model_path)
            except Exception:
                MODEL = None
    if MODEL is None:
        return None

    # Build single-row dataframe matching training features
    # NEW MODEL: Does NOT use Is_Night_Risk (removed to fix data leakage)
    hour = _hour_for_time_of_day(time_of_day)
    dow = datetime.now().weekday()
    
    # Try to get actual coordinates for the location
    lat, lng = 0.0, 0.0
    if HAS_REQUESTS:
        maps_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
        if maps_key:
            try:
                url = "https://maps.googleapis.com/maps/api/geocode/json"
                params = {
                    "address": location + ", India",
                    "key": maps_key
                }
                response = requests.get(url, params=params, timeout=3)
                if response.ok:
                    data = response.json()
                    if data.get("status") == "OK" and data.get("results"):
                        loc_data = data["results"][0]["geometry"]["location"]
                        lat = loc_data.get("lat", 0.0)
                        lng = loc_data.get("lng", 0.0)
            except Exception:
                pass  # Use default coordinates if geocoding fails
    
    row = pd.DataFrame([
        {
            "District": (location or "").strip() or "Unknown",
            "Latitude": lat,
            "Longitude": lng,
            "Hour": hour,
            "Day_of_Week": dow,
            "Is_High_Density_Area": 1,  # assume city context; adjust if needed
        }
    ])

    try:
        if hasattr(MODEL, "predict_proba"):
            proba = MODEL.predict_proba(row)[0][1]  # probability of risk=1
            score = int(round((1.0 - float(proba)) * 100))
        else:
            pred = int(MODEL.predict(row)[0])  # 0 or 1
            score = 90 if pred == 0 else 40
        return max(0, min(100, score))
    except Exception as e:
        # If model fails (e.g., location not in training data), return None to use fallback
        print(f"Model prediction error for {location}: {e}")
        return None


def create_app() -> Flask:
    app = Flask(
        __name__,
        static_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        static_url_path=""
    )
    CORS(app)

    # Health check
    @app.get("/api/health")
    def health() -> tuple:
        return jsonify({"status": "ok"}), 200

    # Debug endpoint to check dataset loading
    @app.get("/api/debug/dataset")
    def debug_dataset() -> tuple:
        global CSV_DF
        if CSV_DF is None:
            CSV_DF = _load_csv_if_available()
        
        if CSV_DF is None:
            csv_path = os.getenv("CSV_PATH", "")
            return jsonify({
                "loaded": False,
                "csv_path": csv_path,
                "error": "Dataset not loaded. Check CSV_PATH environment variable."
            }), 200
        
        districts = sorted(CSV_DF["District"].unique().tolist())
        district_norm = sorted(CSV_DF["_district_norm"].unique().tolist())
        
        return jsonify({
            "loaded": True,
            "total_rows": len(CSV_DF),
            "districts": districts,
            "districts_normalized": district_norm,
            "columns": list(CSV_DF.columns)
        }), 200

    # Return configuration that frontend can use
    @app.get("/api/config")
    def config() -> tuple:
        # Try multiple sources for API key (in order of priority):
        # 1. Environment variable
        # 2. Direct fallback for development (remove in production!)
        maps_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
        
        # Fallback: If not in env var, use direct key (FOR DEVELOPMENT ONLY)
        # TODO: Remove this in production and use environment variable only
        if not maps_key or maps_key.strip() == "":
            maps_key = "AIzaSyDWerE0UeKSM4KehocchzdE8ZXanzwpSkA"
            print("[WARNING] Using hardcoded API key. Set GOOGLE_MAPS_API_KEY environment variable for production!")
        
        model_path = os.getenv("MODEL_PATH", "")
        csv_path = os.getenv("CSV_PATH", "")
        
        print(f"[DEBUG] Config endpoint called - API key present: {bool(maps_key and maps_key.strip())}")
        
        return jsonify({
            "mapsApiKey": maps_key,  # Frontend will only initialize map if non-empty
            "modelAvailable": bool(model_path),
            "csvAvailable": bool(csv_path)
        }), 200

    # Geocode endpoint to get coordinates for any location (UP only)
    @app.get("/api/geocode")
    def geocode() -> tuple:
        location = request.args.get("location", "")
        if not location:
            return jsonify({"error": "Location parameter required"}), 400
        
        loc_lower = location.strip().lower()
        
        # First check: Is it a known UP district from dataset? If yes, accept immediately
        is_known_district = loc_lower in DATASET_UP_DISTRICTS or any(district in loc_lower for district in DATASET_UP_DISTRICTS)
        
        if is_known_district:
            # It's a known UP district, try to geocode but don't fail if geocoding fails
            maps_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
            if maps_key and HAS_REQUESTS:
                try:
                    url = "https://maps.googleapis.com/maps/api/geocode/json"
                    params = {
                        "address": location + ", Uttar Pradesh, India",
                        "key": maps_key
                    }
                    response = requests.get(url, params=params, timeout=5)
                    data = response.json()
                    
                    if data["status"] == "OK" and data["results"]:
                        result = data["results"][0]
                        location_data = result["geometry"]["location"]
                        return jsonify({
                            "location": location,
                            "lat": location_data["lat"],
                            "lng": location_data["lng"],
                            "formatted_address": result.get("formatted_address", location),
                            "is_valid": True
                        }), 200
                except Exception:
                    pass  # Continue to return default coordinates
            
            # Return default coordinates for known districts if geocoding fails
            # Use approximate coordinates for UP cities
            default_coords = {
                "lucknow": (26.8467, 80.9462),
                "kanpur": (26.4499, 80.3319),
                "agra": (27.1767, 78.0081),
                "aligarh": (27.8974, 78.0880),
                "bareilly": (28.3670, 79.4304),
                "ghaziabad": (28.6692, 77.4538),
                "gorakhpur": (26.7588, 83.3697),
                "meerut": (28.9845, 77.7064),
                "prayagraj": (25.4358, 81.8463),
                "varanasi": (25.3176, 82.9739),
                "allahabad": (25.4358, 81.8463)
            }
            
            for district, coords in default_coords.items():
                if district in loc_lower:
                    return jsonify({
                        "location": location,
                        "lat": coords[0],
                        "lng": coords[1],
                        "formatted_address": location + ", Uttar Pradesh, India",
                        "is_valid": True
                    }), 200
        
        # Not a known district from dataset - try geocoding to validate it's in UP
        # We accept ALL UP locations, not just the ones in our dataset
        maps_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
        if not maps_key:
            # No API key - accept it as valid UP location (user will get fallback prediction)
            return jsonify({
                "location": location,
                "lat": 26.8467,  # Default to Lucknow coordinates
                "lng": 80.9462,
                "formatted_address": location + ", Uttar Pradesh, India",
                "is_valid": True,
                "has_detailed_data": False
            }), 200
        
        if not HAS_REQUESTS:
            return jsonify({
                "location": location,
                "lat": 26.8467,
                "lng": 80.9462,
                "formatted_address": location + ", Uttar Pradesh, India",
                "is_valid": True,
                "has_detailed_data": False
            }), 200
        
        try:
            url = "https://maps.googleapis.com/maps/api/geocode/json"
            params = {
                "address": location + ", Uttar Pradesh, India",
                "key": maps_key
            }
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            
            if data["status"] == "OK" and data["results"]:
                # Validate that location is actually in UP
                if not _is_location_in_up(location, data):
                    return jsonify({
                        "error": "Invalid location",
                        "message": f"'{location}' is not in Uttar Pradesh. Please enter a valid UP location.",
                        "valid_locations": sorted(list(VALID_UP_DISTRICTS))
                    }), 400
                
                result = data["results"][0]
                location_data = result["geometry"]["location"]
                return jsonify({
                    "location": location,
                    "lat": location_data["lat"],
                    "lng": location_data["lng"],
                    "formatted_address": result.get("formatted_address", location),
                    "is_valid": True,
                    "has_detailed_data": is_known_district
                }), 200
            else:
                # Location not in UP - return error
                return jsonify({
                    "error": "Invalid location",
                    "message": f"'{location}' is not in Uttar Pradesh. Please enter a valid UP location.",
                    "note": "We accept all UP locations. Locations with detailed data: " + ", ".join(sorted(DATASET_UP_DISTRICTS)),
                    "valid_locations": sorted(list(DATASET_UP_DISTRICTS))
                }), 400
        except Exception as e:
            return jsonify({
                "error": "Invalid location",
                "message": f"Could not validate location '{location}'. Please enter a valid UP location.",
                "note": "We accept all UP locations. Locations with detailed data: " + ", ".join(sorted(DATASET_UP_DISTRICTS)),
                "valid_locations": sorted(list(DATASET_UP_DISTRICTS))
            }), 400

    # Prediction endpoint (UP locations only)
    @app.post("/api/predict")
    def predict() -> tuple:
        data = request.get_json(silent=True) or {}
        location = data.get("location", "")
        time_of_day = data.get("timeOfDay", "")

        # Validate location is in UP
        if not location:
            return jsonify({
                "error": "Invalid location",
                "message": "Location is required. Please enter a valid UP location.",
                "valid_locations": sorted(list(VALID_UP_DISTRICTS))
            }), 400
        
        # Check if location is in UP (validate via geocoding if needed)
        loc_lower = location.strip().lower()
        is_in_dataset = loc_lower in DATASET_UP_DISTRICTS or any(district in loc_lower for district in DATASET_UP_DISTRICTS)
        
        # 1) Try CSV-based scoring first (most reliable for locations in dataset)
        score = _score_from_csv(location, time_of_day)
        
        # 2) Else try ML model if available
        if score is None:
            score = _predict_with_model(location, time_of_day)
        
        # 3) If still None, validate if location is in UP and provide fallback
        if score is None:
            # If it's in our dataset list, provide time-based default
            if is_in_dataset:
                base = {
                    "morning": 78,
                    "afternoon": 72,
                    "evening": 65,
                    "night": 52,
                }.get((time_of_day or "").lower(), 70)
                score = base
            else:
                # Not in dataset - validate it's actually in UP via geocoding
                if HAS_REQUESTS:
                    maps_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
                    if maps_key:
                        try:
                            url = "https://maps.googleapis.com/maps/api/geocode/json"
                            params = {
                                "address": location + ", Uttar Pradesh, India",
                                "key": maps_key
                            }
                            response = requests.get(url, params=params, timeout=3)
                            if response.ok:
                                geo_data = response.json()
                                if _is_location_in_up(location, geo_data):
                                    # It's a valid UP location, provide time-based fallback
                                    base = {
                                        "morning": 75,
                                        "afternoon": 70,
                                        "evening": 63,
                                        "night": 50,
                                    }.get((time_of_day or "").lower(), 70)
                                    score = base
                                else:
                                    return jsonify({
                                        "error": "Invalid location",
                                        "message": f"'{location}' is not in Uttar Pradesh. Please enter a valid UP location.",
                                        "valid_locations": sorted(list(DATASET_UP_DISTRICTS))
                                    }), 400
                        except Exception:
                            pass  # Continue with error if geocoding fails
                
                # If we still don't have a score, return error
                if score is None:
                    return jsonify({
                        "error": "Invalid location",
                        "message": f"'{location}' could not be validated. Please enter a valid UP location.",
                        "note": "We have detailed data for: " + ", ".join(sorted(DATASET_UP_DISTRICTS)),
                        "valid_locations": sorted(list(DATASET_UP_DISTRICTS))
                    }), 400
        
        status = "Safe" if score >= 75 else ("Medium" if score >= 60 else "Unsafe")

        return jsonify({
            "location": location,
            "timeOfDay": time_of_day,
            "score": score,
            "status": status
        }), 200

    # Serve frontend files
    @app.get("/")
    def serve_index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/dashboard")
    def serve_dashboard():
        return send_from_directory(app.static_folder, "dashboard.html")

    @app.get("/result")
    def serve_result():
        return send_from_directory(app.static_folder, "result.html")

    return app


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app = create_app()
    app.run(host="0.0.0.0", port=port, debug=True)


