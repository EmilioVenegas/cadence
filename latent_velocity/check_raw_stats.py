import pyreadstat
import pandas as pd
import numpy as np

filepath = '/home/emiliovenegas/Documents/mendel/simpleMHAS/simpleMHAS.sav'
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
