import pandas as pd

file_path = r'C:\Users\janmi\Documents\VS Code\Gym_log_2024\gym_log_Q1_2024 - workout data.csv'
body_weight = 79

df = pd.read_csv(file_path, sep=';', encoding='latin1')

df['Weight'] = df['Weight'].str.replace(',', '.')
df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
df['Date'] = pd.to_datetime(df['Date'])

df = df[['Date', 'Weight']]

print(df.head())