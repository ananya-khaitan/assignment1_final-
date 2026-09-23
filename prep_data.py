import pandas as pd
import yfinance as yf
import os
import re
from pathlib import Path

# Paths
base_dir = Path('C:/Users/ANANYA/OneDrive/Desktop/BITS/3-2/RC/Assignment 1/critical_minerals_forecasting_team_snapshot_20260909/assignment1_pratham')
excel_file = base_dir / 'AI stocks.xlsx'
processed_dir = base_dir / 'code' / 'data' / 'lithium' / 'processed'
os.makedirs(processed_dir, exist_ok=True)

# 1. Load Excel
df_close = pd.read_excel(excel_file, sheet_name='Close Price', skiprows=[1])
df_close.rename(columns={df_close.columns[0]: 'Date'}, inplace=True)
df_close['Date'] = pd.to_datetime(df_close['Date'])
df_close = df_close.sort_values('Date').reset_index(drop=True)

# Only keep data from 1st Jan 2023 onwards
df_close = df_close[df_close['Date'] >= '2023-01-01'].reset_index(drop=True)

start_date = df_close['Date'].min().strftime('%Y-%m-%d')
end_date = (df_close['Date'].max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

# 2. Fetch Macro Data
print('Fetching macro data from yfinance...')
macro_tickers = ['^NDX', '^VIX', '^TNX', 'XLK']
macro_data = yf.download(macro_tickers, start=start_date, end=end_date)['Close']
macro_data.columns = ['10Y_Treasury', 'NASDAQ_100', 'VIX', 'Tech_ETF_XLK']
macro_data = macro_data.reset_index()
macro_data['Date'] = pd.to_datetime(macro_data['Date']).dt.tz_localize(None)

# 3. Merge
df_merged = pd.merge(df_close, macro_data, on='Date', how='left').ffill().bfill()
df_merged = df_merged.set_index('Date')

# 4. Generate CSVs and config
companies = ['NVDA.O']
exogenous = ['.NQROBO', '.SOLUSAIT', '.IAIQ', 'NASDAQ_100', 'VIX', '10Y_Treasury', 'Tech_ETF_XLK']

config_targets = 'TARGETS = {\n'

for company in companies:
    df_company = df_merged[[company] + exogenous].copy()
    df_company.rename(columns={company: 'Target_Close'}, inplace=True)
    safe_name = company.replace('.', '_')
    csv_filename = f'{safe_name}_panel.csv'
    csv_path = processed_dir / csv_filename
    df_company.to_csv(csv_path)
    
    config_targets += f'''    "{safe_name}": {{
        "label": "{company} Stock Price",
        "processed": LITHIUM_PROCESSED_DIR / "{csv_filename}",
        "modes": 9,
    }},\n'''
    
config_targets += '}\n'

# 5. Patch config
config_path = base_dir / 'code' / 'lithium_config.py'
with open(config_path, 'r', encoding='utf-8') as f:
    config_data = f.read()

config_data = re.sub(r'TARGETS\s*=\s*\{.*?^\}', config_targets, config_data, flags=re.DOTALL | re.MULTILINE)

with open(config_path, 'w', encoding='utf-8') as f:
    f.write(config_data)

print('Done creating datasets and patching config!')
