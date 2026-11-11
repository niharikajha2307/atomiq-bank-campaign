"""
scripts/export_sheets.py
========================
Exports all sheets from inputs/bank_campaign_participant_release (1).xlsx to data/
with dtype=str, keep_default_na=False to strictly preserve cell values and string keys.
"""

import os
import openpyxl
import pandas as pd

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(WORKSPACE)

os.makedirs("data", exist_ok=True)
excel_path = os.path.join("inputs", "bank_campaign_participant_release (1).xlsx")
if not os.path.exists(excel_path):
    excel_path = os.path.join("inputs", "bank_campaign_participant_release.xlsx")

print(f"Reading from {excel_path}...")
wb = openpyxl.load_workbook(excel_path, data_only=False)

for name in wb.sheetnames:
    df_raw = pd.read_excel(excel_path, sheet_name=name, dtype=str, keep_default_na=False)
    out_csv = os.path.join("data", f"{name}.csv")
    df_raw.to_csv(out_csv, index=False)
    print(f"Exported '{name}' -> {out_csv} ({df_raw.shape[0]} rows, {df_raw.shape[1]} cols)")

print("All sheets successfully exported to data/.")
