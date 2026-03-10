import sys
import pandas as pd
p=sys.argv[1]
df=pd.read_csv(p)
print('shape',df.shape)
print(df.head().to_string())
print('\nValue counts for Target_Risk_Y:')
print(df['Target_Risk_Y'].value_counts())
for col in ['Is_Night_Risk','Is_High_Density_Area','Hour','Day_of_Week','District']:
    if col in df.columns:
        print('\nUnique and counts for',col)
        print(df[col].value_counts().head(20))
# check if any single column perfectly separates
for col in df.columns:
    if col=='Target_Risk_Y':
        continue
    try:
        grouped = df.groupby(col)['Target_Risk_Y'].nunique()
        if grouped.max()==1:
            print('\nColumn',col,'is perfectly predictive (each value maps to single target)')
    except Exception:
        pass
