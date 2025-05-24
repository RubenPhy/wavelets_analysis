import pandas as pd
import numpy as np
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt

# —-—-—- Ajustes globales Seaborn —-—-—-
sns.set_theme(style="darkgrid")          # fondo gris con rejilla en gris oscuro
plt.rcParams.update({"axes.titleweight": "bold"})  # (opcional) negrita en títulos

# Broad sector categorization based on the industry
broad_sectors = {
    "Consumer Discretionary": [
        "Airlines",
        "Hotels, Motels & Cruise Lines",
        "Restaurants & Bars",
        "Auto Vehicles, Parts & Service Retailers",
        "Computer & Electronics Retailers",
        "Discount Stores",
        "Department Stores",
        "Casinos & Gaming",
        "Leisure & Recreation",
        "Entertainment Production",
        "Apparel & Accessories",
        "Footwear",
        "Miscellaneous Specialty Retailers",
        "Toys & Children's Products",
    ],
    "Consumer Staples": [
        "Food Processing",
        "Brewers",
        "Distillers & Wineries",
        "Non-Alcoholic Beverages",
        "Food Retail & Distribution",
        "Tobacco",
    ],
    "Energy": [
        "Oil & Gas Exploration and Production",
        "Oil & Gas Refining and Marketing",
        "Oil Related Services and Equipment",
        "Oil & Gas Transportation Services",
    ],
    "Financials": [
        "Banks",
        "Consumer Lending",
        "Investment Banking & Brokerage Services",
        "Investment Management & Fund Operators",
        "Life & Health Insurance",
        "Property & Casualty Insurance",
        "Multiline Insurance & Brokers",
        "Financial Technology (Fintech)",
        "Financial & Commodity Market Operators & Service Providers",
    ],
    "Health Care": [
        "Healthcare Facilities & Services",
        "Managed Healthcare",
        "Medical Equipment, Supplies & Distribution",
        "Pharmaceuticals",
        "Biotechnology & Medical Research",
        "Advanced Medical Equipment & Technology",
    ],
    "Industrials": [
        "Aerospace & Defense",
        "Industrial Machinery & Equipment",
        "Electrical Components & Equipment",
        "Construction & Engineering",
        "Heavy Machinery & Vehicles",
        "Business Support Services",
        "Environmental Services & Equipment",
        "Ground Freight & Logistics",
        "Courier, Postal, Air Freight & Land-based Logistics",
    ],
    "Information Technology": [
        "Software",
        "Semiconductors",
        "IT Services & Consulting",
        "Computer Hardware",
        "Communications & Networking",
        "Electronic Equipment & Parts",
        "Semiconductor Equipment & Testing",
        "Financial Technology (Fintech)",
        "Online Services",
    ],
    "Materials": [
        "Commodity Chemicals",
        "Specialty Chemicals",
        "Diversified Chemicals",
        "Construction Materials",
        "Paper Packaging",
        "Non-Paper Containers & Packaging",
        "Metals & Mining",
        "Specialty Mining & Metals",
        "Agricultural Chemicals",
    ],
    "Real Estate": [
        "Real Estate Development",
        "Residential REITs",
        "Commercial REITs",
        "Specialized REITs",
    ],
    "Telecommunication Services": [
        "Integrated Telecommunications Services",
        "Wireless Telecommunications Services",
    ],
    "Utilities": [
        "Electric Utilities",
        "Multiline Utilities",
        "Gas Utilities",
        "Water & Related Utilities",
        "Independent Power Producers",
    ],
}

# Read the S&P_500.xlsx file 
df_sp_info = pd.read_excel(r'C:\Users\Usuario\Documents\PhD_AI\wavelets_analysis\raw\S&P_500.xlsx')

# Path to the folder containing the CSV files
csv_folder = r'C:\Users\Usuario\Documents\PhD_AI\wavelets_analysis\raw'

# Read all CSV files for companies
csv_files = glob.glob(os.path.join(csv_folder, '*.csv'))

# Create a dictionary to store DataFrames for each company
dfs = {}
for file in csv_files:
    ticker = os.path.splitext(os.path.basename(file))[0]
    dfs[ticker.split('-')[0]] = pd.read_csv(file, parse_dates=['Date'])

# Map subsectors to broad sectors
subsector_to_broad = {}
for broad, subsectors in broad_sectors.items():
    for subsector in subsectors:
        subsector_to_broad[subsector] = broad

# Add broad sector column to df_sp_info
df_sp_info['Broad Sector'] = df_sp_info['Industry Name'].map(subsector_to_broad)

# Create synthetic indices for broad sectors (logarithmic returns)
broad_sector_indices = {}
for broad_sector in broad_sectors.keys():
    print(f'Processing broad sector: {broad_sector}')
    tickers = df_sp_info[df_sp_info['Broad Sector'] == broad_sector]['Identifier']
    sector_returns = []
    for ticker in tickers:
        print(f'Processing ticker: {ticker}')
        if ticker not in dfs:
            print(f'Ticker {ticker} not found in DataFrames. Skipping.')
        else:
            df = dfs[ticker].copy()
            df.set_index('Date', inplace=True)
            # Calculate logarithmic returns: log(P_t / P_{t-1})
            df['Log Return'] = np.log(df['CLOSE'] / df['CLOSE'].shift(1))
            sector_returns.append(df['Log Return'])
    if sector_returns:  # Only process if there are valid returns
        sector_df = pd.concat(sector_returns, axis=1)
        sector_mean = sector_df.mean(axis=1)
        # Calculate cumulative return from log returns: cumsum for display
        sector_cumulative = sector_mean.cumsum()
        broad_sector_indices[broad_sector] = (sector_mean, sector_cumulative)

# Create synthetic S&P 500 index (logarithmic return)
all_returns = []
for ticker in df_sp_info['Identifier']:
    if ticker not in dfs:
        print(f'Ticker {ticker} not found in DataFrames. Skipping.')
    else:
        df = dfs[ticker].copy()
        df.set_index('Date', inplace=True)
        # Calculate logarithmic returns
        df['Log Return'] = np.log(df['CLOSE'] / df['CLOSE'].shift(1))
        all_returns.append(df['Log Return'])
sp500_df = pd.concat(all_returns, axis=1)
synthetic_sp500 = sp500_df.mean(axis=1)
synthetic_sp500_cumulative = synthetic_sp500.cumsum()

# Save daily logarithmic returns to CSV
results_df = pd.DataFrame({'S&P 500': synthetic_sp500})
for broad_sector, (returns, _) in broad_sector_indices.items():
    results_df[broad_sector] = returns
results_df.to_csv(r'C:\Users\Usuario\Documents\PhD_AI\wavelets_analysis\sector_log_returns.csv')

# Plot the synthetic S&P 500 and broad sector indices (cumulative log returns)
plt.figure(figsize=(14, 8))
sns.lineplot(data=synthetic_sp500_cumulative, label='Synthetic S&P 500', color='black')
for broad_sector, (_, cumulative) in broad_sector_indices.items():
    sns.lineplot(data=cumulative, label=f'Sector: {broad_sector}', alpha=0.7)
plt.title('Synthetic S&P 500 and Broad Sector Cumulative Log Returns', fontsize=18)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Cumulative Log Returns (%)', fontsize=14)
plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))  # No decimals on y-axis
plt.legend(fontsize=14, loc='upper left')
plt.tick_params(axis='both', labelsize=14)
plt.tight_layout()
# Save the plot
plt.savefig(r'C:\Users\Usuario\Documents\PhD_AI\wavelets_analysis\plots\returns_SP500_and_sectors.png')
plt.show()
print("Analysis complete. Log returns saved to 'sector_log_returns.csv' and plot saved as 'returns_SP500_and_sectors.png'.")