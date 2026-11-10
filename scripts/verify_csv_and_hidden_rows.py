import os
import openpyxl
import pandas as pd

excel_path = os.path.join("inputs", "bank_campaign_participant_release.xlsx")
wb = openpyxl.load_workbook(excel_path, data_only=True)

# 1. Check hidden rows
hidden_rows_report = {}
for sheetname in wb.sheetnames:
    ws = wb[sheetname]
    hidden_rows = [r for r, dim in ws.row_dimensions.items() if dim.hidden]
    hidden_rows_report[sheetname] = hidden_rows

# 2. Re-export and verify with keep_default_na=False
all_match = True
nan_count = 0
for sheetname in wb.sheetnames:
    df_excel = pd.read_excel(excel_path, sheet_name=sheetname, dtype=str, keep_default_na=False)
    csv_path = os.path.join("data", f"{sheetname}.csv")
    # Save clean csv with keep_default_na=False
    df_excel.to_csv(csv_path, index=False)
    df_csv = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if not df_excel.equals(df_csv):
        all_match = False
    if df_excel.isna().sum().sum() > 0 or df_csv.isna().sum().sum() > 0:
        nan_count += 1

print(f"HIDDEN_ROWS: {hidden_rows_report}")
print(f"ALL_MATCH: {all_match}")
print(f"NAN_COUNT: {nan_count}")
