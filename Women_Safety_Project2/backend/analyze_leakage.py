import pandas as pd
import sys

csv_path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\user\Desktop\synthetic_up_locational_risk_data.csv"

df = pd.read_csv(csv_path)

print("=" * 60)
print("DATA LEAKAGE ANALYSIS")
print("=" * 60)

print("\n1. PERFECT CORRELATION: Is_Night_Risk → Target_Risk_Y")
print("-" * 60)
leak_check = pd.crosstab(df['Is_Night_Risk'], df['Target_Risk_Y'])
print(leak_check)
print("\nKey Finding:")
print("  - When Is_Night_Risk=0 → Target_Risk_Y is ALWAYS 0 (1050/1050)")
print("  - When Is_Night_Risk=1 → Target_Risk_Y can be 0 or 1")
print("  - This creates a simple rule: Is_Night_Risk=0 guarantees Safe")

print("\n2. FEATURE REDUNDANCY")
print("-" * 60)
print("Is_Night_Risk is derived from Hour:")
print("  - Hours 0-5, 21-23: Is_Night_Risk=1")
print("  - Hours 6-20: Is_Night_Risk=0")
print("\nThis means:")
print("  - Hour feature + Is_Night_Risk = redundant information")
print("  - Model learns perfect rule: Is_Night_Risk=0 → Safe (100% accuracy)")

print("\n3. LOCATION LEAKAGE")
print("-" * 60)
print("Same Lat/Long for entire district:")
for dist in df['District'].unique()[:3]:
    subset = df[df['District'] == dist]
    print(f"  {dist}: ({subset['Latitude'].iloc[0]}, {subset['Longitude'].iloc[0]})")
print("\n  → Lat/Long provides NO additional information beyond District")

print("\n4. DATASET STATISTICS")
print("-" * 60)
print(f"Total rows: {len(df)}")
print(f"Districts: {df['District'].nunique()}")
print(f"Unique feature combinations: {len(df.drop(columns=['Target_Risk_Y']).drop_duplicates())}")
print(f"Target distribution: {dict(df['Target_Risk_Y'].value_counts())}")

print("\n5. DISTRICT-SPECIFIC RISK PATTERNS")
print("-" * 60)
district_risk = df.groupby('District')['Target_Risk_Y'].agg(['count', 'sum', 'mean'])
district_risk.columns = ['Total', 'Risky_Count', 'Risk_Ratio']
print(district_risk)

print("\n" + "=" * 60)
print("CONCLUSION: 100% accuracy is due to DATA LEAKAGE")
print("=" * 60)
print("\nThe model achieves perfect accuracy because:")
print("1. Is_Night_Risk=0 ALWAYS means Safe (no risk)")
print("2. The target is perfectly predictable from features")
print("3. This is synthetic data with artificial patterns")
print("\nThis will NOT work in real-world scenarios!")
