import os
import openpyxl
import pandas as pd
import numpy as np

os.makedirs("data", exist_ok=True)
excel_path = os.path.join("inputs", "bank_campaign_participant_release.xlsx")

wb = openpyxl.load_workbook(excel_path, data_only=False)

print("=== WORKBOOK INSPECTION ===")
print("Sheet names:", wb.sheetnames)

hidden_sheets = []
for name in wb.sheetnames:
    ws = wb[name]
    state = ws.sheet_state
    if state != "visible":
        hidden_sheets.append((name, state))
print("Hidden sheets:", hidden_sheets if hidden_sheets else "None found (all visible)")

for name in wb.sheetnames:
    ws = wb[name]
    hidden_cols = [col for col, dim in ws.column_dimensions.items() if dim.hidden]
    if hidden_cols:
        print(f"Sheet '{name}' hidden columns:", hidden_cols)
    else:
        print(f"Sheet '{name}': No hidden columns")

print("\n=== EXPORTING SHEETS TO CSV ===")
sheets_data = {}
for name in wb.sheetnames:
    # Read with openpyxl or pandas with dtype=str to preserve strings like "C0273"
    df_raw = pd.read_excel(excel_path, sheet_name=name, dtype=str)
    out_csv = os.path.join("data", f"{name}.csv")
    df_raw.to_csv(out_csv, index=False)
    sheets_data[name] = df_raw
    print(f"Exported '{name}' -> {out_csv} (shape: {df_raw.shape})")

print("\n=== DETAILED DATA INSPECTION ===")
for name, df in sheets_data.items():
    print(f"\n--- Sheet: {name} ---")
    print(f"Shape: {df.shape} (rows: {len(df)}, cols: {len(df.columns)})")
    print(f"Columns: {list(df.columns)}")
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        print(f"Missing values:\n{missing_cols}")
    else:
        print("Missing values: None (no NaN values)")

# Deep dive into Stage1_Train
df_train = sheets_data.get("Stage1_Train")
if df_train is not None:
    print("\n--- Deep Dive: Stage1_Train ---")
    print("Class balance (y):")
    print(df_train['y'].value_counts(dropna=False))
    print(df_train['y'].value_counts(normalize=True) * 100)
    
    # Check for "unknown" categories
    unknown_counts = {}
    for col in df_train.columns:
        cnt = (df_train[col].str.lower() == "unknown").sum()
        if cnt > 0:
            unknown_counts[col] = cnt
    print("Columns with 'unknown' values:", unknown_counts)
    
    # Check pdays
    if 'pdays' in df_train.columns:
        pdays_num = pd.to_numeric(df_train['pdays'], errors='coerce')
        print(f"pdays == -1 count: {(pdays_num == -1).sum()} / {len(pdays_num)}")
        print(f"pdays unique values: {sorted(pdays_num.dropna().unique())[:10]} ...")

# Deep dive into Stage1_Test
df_test = sheets_data.get("Stage1_Test")
if df_test is not None:
    print("\n--- Deep Dive: Stage1_Test ---")
    print("Columns in Stage1_Test:", list(df_test.columns))
    print("Is customer_id in Stage1_Test?", "customer_id" in df_test.columns)

# Deep dive into Stage2_Data & customer_id check
df_s2 = sheets_data.get("Stage2_Data")
if df_train is not None and df_s2 is not None:
    print("\n--- customer_id Cross-Check ---")
    train_ids = set(df_train['customer_id'])
    s2_ids = set(df_s2['customer_id'])
    print(f"Total IDs in Stage1_Train: {len(train_ids)}")
    print(f"Total IDs in Stage2_Data: {len(s2_ids)}")
    
    unmatched_s2_ids = sorted(list(s2_ids - train_ids))
    print(f"IDs in Stage2_Data NOT in Stage1_Train count: {len(unmatched_s2_ids)}")
    print(f"Unmatched IDs list: {unmatched_s2_ids}")

    train_not_in_s2 = train_ids - s2_ids
    print(f"IDs in Stage1_Train NOT in Stage2_Data count: {len(train_not_in_s2)}")
