import sys
from pathlib import Path

# Add the engine directory to the Python path for imports
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "engine"))

import pyreadstat
import pandas as pd
import numpy as np
from _paths import DATA_DIR

filepath = str(DATA_DIR / 'simpleMHAS.sav')
df, meta = pyreadstat.read_sav(filepath)

print("--- RAW DATA STATS ---")
print(f"Total Rows: {len(df)}")
if 'fallecido' in df.columns:
    print("\nFallecido distribution:")
    print(df['fallecido'].value_counts(dropna=False))

if 'tipent' in df.columns:
    print("\nTipent distribution:")
    print(df['tipent'].value_counts(dropna=False))
    
    print("\nFallecido vs Tipent cross-tab:")
    print(pd.crosstab(df['fallecido'], df['tipent'], dropna=False))

# Check for perfect cognitive scores
print("\nRecuerdo1 distribution:")
print(df['recuerdo1'].value_counts(dropna=False).head(10))
