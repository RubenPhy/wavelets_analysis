import pandas as pd
import glob
import os

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
csv_folder = r'C:\Users\Usuario\Documents\PhD_AI\wavelets_analysis\raw'  # Ajusta si es necesario

# Leer todos los archivos CSV de las empresas
csv_files = glob.glob(os.path.join(csv_folder, '*.csv'))

# Crear un diccionario para almacenar los DataFrames de cada empresa
dfs = {}
for file in csv_files:
    ticker = os.path.splitext(os.path.basename(file))[0]
    dfs[ticker.split('-')[0]] = pd.read_csv(file, parse_dates=['Date'])

# Crear un índice sintético de retorno para cada sector
sector_indices = {}
for sector in df_sp_info['Industry Name'].unique():
    print(f'Processing sector: {sector}')
    tickers = df_sp_info[df_sp_info['Industry Name'] == sector]['Identifier']
    sector_returns = []
    for ticker in tickers:
        print(f'Processing ticker: {ticker}')
        if ticker not in dfs:
            print(f'Ticker {ticker} not found in DataFrames. Skipping.')
        else:
            df = dfs[ticker].copy()
            df.set_index('Date', inplace=True)
            df['Return'] = df['CLOSE'].pct_change()
            sector_returns.append(df['Return'])
    sector_df = pd.concat(sector_returns, axis=1)
    sector_indices[sector] = sector_df.mean(axis=1)

# Crear un índice sintético S&P500 con el promedio de los retornos de las 500 empresas
all_returns = []
for ticker in df_sp_info['Identifier']:
    if ticker not in dfs:
        print(f'Ticker {ticker} not found in DataFrames. Skipping.')
    else:
        df = dfs[ticker].copy()
        df.set_index('Date', inplace=True)
        df['Return'] = df['CLOSE'].pct_change()
        all_returns.append(df['Return'])
sp500_df = pd.concat(all_returns, axis=1)
synthetic_sp500 = sp500_df.mean(axis=1)

# Plotear el índice sintético S&P500 y los índices de sectores con seaborn
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 8))
sns.lineplot(data=synthetic_sp500, label='Synthetic S&P 500', color='blue')
for sector, returns in sector_indices.items():
    sns.lineplot(data=returns, label=f'Sector: {sector}', alpha=0.7)
plt.title('Synthetic S&P 500 and Sector Indices')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

