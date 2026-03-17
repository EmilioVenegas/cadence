import sys
from pathlib import Path

# Setup paths
_ENGINE = Path(__file__).resolve().parent
_ROOT = _ENGINE.parent
sys.path.insert(0, str(_ENGINE))
sys.path.insert(0, str(_ROOT / "ode-digitaltwin"))

from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import json
from _paths import DATA_DIR
from digital_twin import rank_interventions

app = FastAPI(title="LAVA Real-Time Inference API")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables/cache
df_raw = None

def load_data():
    global df_raw
    try:
        csv_path = DATA_DIR / 'frailty_index_data.csv'
        if csv_path.exists():
            df_raw = pd.read_csv(csv_path)
            print(f"Loaded {len(df_raw)} records for patient registry.")
        else:
            print(f"Warning: {csv_path} not found.")
            df_raw = pd.DataFrame()
    except Exception as e:
        print(f"Error loading data: {e}")
        df_raw = pd.DataFrame()

@app.on_event("startup")
async def startup_event():
    load_data()

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "patients_loaded": not df_raw.empty if df_raw is not None else False}

@app.get("/api/patients")
async def get_patients(q: str = None):
    """Return a list of unique patient identifiers (latest visits)."""
    if df_raw is None or df_raw.empty:
        load_data()
        if df_raw.empty:
            raise HTTPException(status_code=500, detail="Patient data not loaded.")
    
    # Get unique (cunicah, np) pairs with their latest visit data
    cols = ['cunicah', 'np', 'ronda', 'edad', 'sexo', 'a_o_ent']
    available_cols = [c for c in cols if c in df_raw.columns]
    
    # Filter by q if provided
    if q:
        try:
            mask = df_raw['cunicah'].astype(str).str.contains(q) | \
                   df_raw['edad'].astype(str).str.contains(q)
            df_filtered = df_raw.loc[mask]
        except:
            df_filtered = df_raw
    else:
        df_filtered = df_raw

    # Group by patient ID (cunicah, np) and take the one with highest a_o_ent (latest year)
    # We sort by a_o_ent descending and then drop duplicates on patient ID
    patients = df_filtered[available_cols].sort_values(['cunicah', 'np', 'a_o_ent'], 
                                                      ascending=[True, True, False])
    patients_latest = patients.drop_duplicates(subset=['cunicah', 'np'])
    
    # Return top 100 for browsing
    # We use replace(np.nan, None) to ensure JSON compatibility (NaNs are not valid JSON)
    return patients_latest.head(100).replace({np.nan: None}).to_dict(orient="records")

@app.get("/api/rank/{cunicah}/{np}")
async def get_intervention_ranking(cunicah: float, np_visit: float = Path(..., alias="np")):
    """Run real-time intervention ranking for a specific patient."""
    try:
        # Note: rank_interventions handles its own model loading and data access
        ranking = rank_interventions(cunicah, np_visit)
        
        if ranking is None:
            raise HTTPException(status_code=404, detail="Patient not found or already at targets.")
        
        # Convert numpy arrays to lists for JSON serialization
        # We need to be careful with numpy types
        def sanitize(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.float32) or isinstance(obj, np.float64):
                return float(obj)
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [sanitize(x) for x in obj]
            return obj

        return sanitize(ranking)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
