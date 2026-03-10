"""
Prepare a fixed version of the dataset by removing data leakage.
Main fix: Remove Is_Night_Risk feature which causes perfect correlation.
"""
import pandas as pd
import sys

input_csv = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\user\Desktop\synthetic_up_locational_risk_data.csv"
output_csv = sys.argv[2] if len(sys.argv) > 2 else r"C:\Users\user\Desktop\synthetic_up_locational_risk_data_fixed.csv"

print(f"Reading data from: {input_csv}")
df = pd.read_csv(input_csv)

print(f"Original dataset shape: {df.shape}")
print(f"Original columns: {list(df.columns)}")

# Remove Is_Night_Risk to eliminate data leakage
# Keep: District, Latitude, Longitude, Hour, Day_of_Week, Is_High_Density_Area, Target_Risk_Y
df_fixed = df.drop(columns=['Is_Night_Risk'])

print(f"\nFixed dataset shape: {df_fixed.shape}")
print(f"Fixed columns: {list(df_fixed.columns)}")
print(f"\nRemoving Is_Night_Risk eliminates the perfect correlation with target.")

# Save fixed dataset
df_fixed.to_csv(output_csv, index=False)
print(f"\n✅ Saved fixed dataset to: {output_csv}")

# Show statistics
print("\nTarget distribution:")
print(df_fixed['Target_Risk_Y'].value_counts())
print("\nDistrict risk ratios:")
print(df_fixed.groupby('District')['Target_Risk_Y'].mean().sort_values(ascending=False))
