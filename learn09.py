import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# File path and initial parameters
file_path = r'C:\Users\janmi\Documents\VS Code\Gym_log_2024\gym_log_Q1_2024 - workout data.csv'
body_weight = 79


def daily_view(df, window=8) -> None:
    df_weekly = df.groupby(['Week'])[['Sets']].sum()
    df_monthly = df.groupby(['Month'])[['Sets']].sum()
    df_daily = df.groupby(['Date'])[['Sets']].sum()
    df_daily = df_daily.groupby('Date')['Sets'].transform(lambda x: x.rolling(window=window, min_periods=1).mean())


    fig, ax = plt.subplots(1, 3, figsize=(14, 6))  # Create a figure and two subplots

    sns.barplot(
        data=df_monthly.reset_index(),
        x='Month',
        y='Sets',
        hue='Sets',
        palette='YlOrRd',
        ax=ax[0]
    )
    ax[0].set_title('Monthly Sets')


    sns.barplot(
        data=df_weekly.reset_index(),
        x='Week',
        y='Sets',
        hue='Sets',
        palette='YlOrRd',
        ax=ax[1]
    )
    ax[1].set_title('Weekly Sets')



    sns.lineplot(
        data=df_daily.reset_index(),
        x='Date',
        y='Sets',
        ax=ax[2]
    )
    ax[2].set_title('Daily Sets')
    ax[2].set_xticklabels([])

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def main() -> None:
    try:
        df = pd.read_csv(file_path, sep=';', index_col=None, encoding='latin1')
        
        # Data cleaning
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
        df['Weight'] = df['Weight'].str.replace(',', '.')
        df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
        df['Reps'] = pd.to_numeric(df['Reps'], errors='coerce')
        df['Sets'] = pd.to_numeric(df['Sets'], errors='coerce')

        # Add week number column
        df['Week'] = df['Date'].dt.isocalendar().week
        df['Month'] = df['Date'].dt.month

        df = df.sort_values(by='Date', ascending=True)


    except FileNotFoundError as e:
        print(f"File not found. Details: {e}")
        return

    daily_view(df)


if __name__ == "__main__":
    main()