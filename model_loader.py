import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Configurable paths using environment variables with defaults
MODEL_DIR = os.environ.get("SANNIF_MODEL_DIR", "./models")
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.json")
ENCODER_PATH = os.path.join(MODEL_DIR, "attack_encoder.pkl")
print("Current Working Directory:", os.getcwd())
print("MODEL_PATH:", os.path.abspath(MODEL_PATH))
print("MODEL EXISTS:", os.path.exists(MODEL_PATH))
print("ENCODER EXISTS:", os.path.exists(ENCODER_PATH))

# Global system health state variables
model = None
encoder = None
model_loaded = False
encoder_loaded = False
prediction_engine_ready = False
features_verified = False
error_message = ""
required_features = []
from xgboost import XGBClassifier
import joblib

@st.cache_resource(show_spinner="Loading SANNIF Detection Models...")
def _load_cached_files(model_path, encoder_path):

    model = XGBClassifier()
    model.load_model(model_path)

    encoder = joblib.load(encoder_path)

    return model, encoder
    
def load_backend():
    """
    Loads the production model and label encoder from the filesystem using joblib.
    Populates system health flags and feature names without crashing.
    """
    global model, encoder, model_loaded, encoder_loaded, prediction_engine_ready, features_verified, error_message, required_features
    
    # Reset states
    model_loaded = False
    encoder_loaded = False
    prediction_engine_ready = False
    features_verified = False
    required_features = []
    error_message = ""
    
    try:
        # Check files existence before calling the cached function
        if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
            missing = []
            if not os.path.exists(MODEL_PATH):
                missing.append(f"'{MODEL_PATH}'")
            if not os.path.exists(ENCODER_PATH):
                missing.append(f"'{ENCODER_PATH}'")
            error_message = f"Missing files: {', '.join(missing)}."
            return
        
        # Load from cache
        model, encoder = _load_cached_files(MODEL_PATH, ENCODER_PATH)
        model_loaded = True
        encoder_loaded = True

        
        # 3. Retrieve and Verify Required Features
        if hasattr(model, "feature_names_in_"):
            required_features = list(model.feature_names_in_)
            features_verified = True
        elif hasattr(model, "feature_names"):
            required_features = list(model.feature_names)
            features_verified = True
        else:
            # Fallback if the object does not have feature_names_in_
            # (e.g. if it is wrapped or serialized differently)
            features_verified = False
            error_message = "Model loaded, but could not extract required feature names automatically."
            return
            
        # If all steps succeeded, prediction engine is ready
        prediction_engine_ready = True
        
    except Exception as e:
        import traceback

        error_message = str(e)

        print("\n========== FULL ERROR ==========")
        traceback.print_exc()
        print("================================\n")

# Attempt to initialize backend on import
load_backend()

def predict_attacks(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Predicts network traffic security status using the loaded XGBoost classifier.
    
    Args:
        input_df (pd.DataFrame): Dataframe containing the network traffic records.
        
    Returns:
        pd.DataFrame: Dataframe with two columns:
            - "Predicted Attack" (str): Predicted category name.
            - "Confidence" (float): Model confidence score (0.0 to 1.0).
    """
    global model, encoder, prediction_engine_ready
    
    if not prediction_engine_ready or model is None or encoder is None:
        raise RuntimeError(f"Prediction engine is not ready. System error: {error_message}")
        
    # Standardize column subsetting to match what the model expects
    X = input_df[required_features]
    
    # Run XGBoost inference
    preds = model.predict(X)
    
    # Obtain confidence values (probabilities)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        confidences = np.max(probs, axis=1)
    else:
        confidences = np.ones(len(preds))
        
    # Convert predictions using the label encoder
    if hasattr(encoder, "inverse_transform"):
        predicted_attacks = encoder.inverse_transform(preds)
    else:
        predicted_attacks = preds
        
    return pd.DataFrame({
        "Predicted Attack": predicted_attacks,
        "Confidence": confidences
    })
