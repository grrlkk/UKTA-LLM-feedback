# XLSX to Parquet 변환 스크립트 (최초 1회만 실행)
# pip install pandas openpyxl pyarrow
import pandas as pd

print("Loading XLSX file...")
df = pd.read_excel("data/feat_train_500.xlsx")

print("Saving to Parquet format...")
df.to_parquet("feat_train_500.parquet", index=False)

print("✅ Conversion complete!")