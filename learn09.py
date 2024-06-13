import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# File path and initial parameters
file_path = r'C:\Users\janmi\Documents\VS Code\Gym_log_2024\gym_log_Q1_2024 - workout data.csv'
body_weight = 79
sns.set_style("white")

border_color = 'lightgrey'
background_color = '#fdfcfc'

def sets_view(df, window=8) -> None:
    df_weekly = df.groupby(['Week'])[['Sets']].sum()
    df_monthly = df.groupby(['Month'])[['Sets']].sum()
    df_daily = df.groupby(['Date'])[['Sets']].sum()
    df_daily = df_daily.groupby('Date')['Sets'].transform(lambda x: x.rolling(window=window, min_periods=1).mean())

    fig = plt.figure(figsize=(11, 7))  # Create a figure
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2])  # Create a gridspec with specific ratios

    ax0 = plt.subplot(gs[0, 0])  # Top row, first column
    ax1 = plt.subplot(gs[0, 1])  # Top row, second column
    ax2 = plt.subplot(gs[1, :])  # Bottom row, spans both columns

    sns.barplot(
        data=df_monthly.reset_index(),
        x='Month',
        y='Sets',
        hue='Sets',
        palette='YlOrRd',
        ax=ax0
    )
    ax0.set_title('Monthly Sets', fontweight='bold', fontsize=12)
    ax0.legend().remove()
    ax0.set_xlabel('Month',fontsize=6)
    ax0.tick_params(axis='x', labelsize=8)
    ax0.tick_params(axis='y', labelsize=8)

    sns.barplot(
        data=df_weekly.reset_index(),
        x='Week',
        y='Sets',
        hue='Sets',
        palette='YlOrRd',
        ax=ax1
    )
    ax1.set_title('Weekly Sets', fontweight='bold', fontsize=12)
    ax1.set_ylabel('')
    ax1.set_xlabel('Week',fontsize=6)
    ax1.legend().remove()
    ax1.tick_params(axis='x', labelsize=8)
    ax1.tick_params(axis='y', labelsize=8)

    sns.barplot(
        data=df_daily.reset_index(),
        x='Date',
        y='Sets',
        hue='Sets',
        palette='YlOrRd',
        ax=ax2
    )
    ax2.set_title('Daily Sets', fontweight='bold', fontsize=12)
    ax2.set_ylabel('')
    ax2.set_xlabel('Day',fontsize=6)
    ax2.tick_params(axis='x', labelsize=5)
    ax2.tick_params(axis='y', labelsize=8)

    for ax in [ax0, ax1, ax2]:
        ax.set_facecolor(background_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)

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

    sets_view(df)

if __name__ == "__main__":
    main()
