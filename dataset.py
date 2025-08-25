import yfinance as yf
import pandas as pd

# NIFTY 50 stock symbols with company names
nifty50_companies = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services", 
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "INFY.NS": "Infosys",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "SBIN.NS": "State Bank of India",
    "BHARTIARTL.NS": "Bharti Airtel",
    "ITC.NS": "ITC",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "LT.NS": "Larsen & Toubro",
    "AXISBANK.NS": "Axis Bank",
    "BAJFINANCE.NS": "Bajaj Finance",
    "MARUTI.NS": "Maruti Suzuki",
    "ASIANPAINT.NS": "Asian Paints",
    "HCLTECH.NS": "HCL Technologies",
    "SUNPHARMA.NS": "Sun Pharmaceutical",
    "WIPRO.NS": "Wipro",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "NESTLEIND.NS": "Nestle India",
    "TITAN.NS": "Titan Company",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "POWERGRID.NS": "Power Grid Corporation",
    "ONGC.NS": "Oil & Natural Gas Corporation",
    "TATAMOTORS.NS": "Tata Motors",
    "ADANIGREEN.NS": "Adani Green Energy",
    "COALINDIA.NS": "Coal India",
    "TECHM.NS": "Tech Mahindra",
    "ADANIPORTS.NS": "Adani Ports",
    "JSWSTEEL.NS": "JSW Steel",
    "HDFCLIFE.NS": "HDFC Life Insurance",
    "CIPLA.NS": "Cipla",
    "DRREDDY.NS": "Dr. Reddy's Laboratories",
    "GRASIM.NS": "Grasim Industries",
    "TATASTEEL.NS": "Tata Steel",
    "DIVISLAB.NS": "Divi's Laboratories",
    "BRITANNIA.NS": "Britannia Industries",
    "BAJAJ-AUTO.NS": "Bajaj Auto",
    "INDUSINDBK.NS": "IndusInd Bank",
    "NTPC.NS": "NTPC",
    "EICHERMOT.NS": "Eicher Motors",
    "HEROMOTOCO.NS": "Hero MotoCorp",
    "M&M.NS": "Mahindra & Mahindra",
    "UPL.NS": "UPL",
    "SBILIFE.NS": "SBI Life Insurance",
    "ADANIENT.NS": "Adani Enterprises",
    "BPCL.NS": "Bharat Petroleum",
    "SHREECEM.NS": "Shree Cement",
    "HINDALCO.NS": "Hindalco Industries",
    "APOLLOHOSP.NS": "Apollo Hospitals"
}
# nasdaq_companies = {
#     "AAPL": "Apple Inc.",
#     "ABNB": "Airbnb, Inc.",
#     "ADBE": "Adobe Inc.",
#     "ADI": "Analog Devices, Inc.",
#     "ADP": "Automatic Data Processing, Inc.",
#     "ADSK": "Autodesk, Inc.",
#     "AEP": "American Electric Power Company, Inc.",
#     "AMAT": "Applied Materials, Inc.",
#     "AMD": "Advanced Micro Devices, Inc.",
#     "AMGN": "Amgen Inc.",
#     "AMZN": "Amazon.com, Inc.",
#     "APP": "AppLovin Corporation",
#     "ARM": "Arm Holdings plc",
#     "ASML": "ASML Holding N.V.",
#     "AVGO": "Broadcom Inc.",
#     "AXON": "Axon Enterprise, Inc.",
#     "AZN": "AstraZeneca plc",
#     "BIIB": "Biogen Inc.",
#     "BKNG": "Booking Holdings Inc.",
#     "BKR": "Baker Hughes Company",
#     "CCEP": "Coca-Cola Europacific Partners",
#     "CDNS": "Cadence Design Systems, Inc.",
#     "CDW": "CDW Corporation",
#     "CEG": "Constellation Energy, Inc.",
#     "CHTR": "Charter Communications, Inc.",
#     "CMCSA": "Comcast Corporation",
#     "COST": "Costco Wholesale Corporation",
#     "CPRT": "Copart, Inc.",
#     "CRWD": "CrowdStrike Holdings, Inc.",
#     "CSCO": "Cisco Systems, Inc.",
#     "CSGP": "CoStar Group, Inc.",
#     "CSX": "CSX Corporation",
#     "CTAS": "Cintas Corporation",
#     "CTSH": "Cognizant Technology Solutions Corporation",
#     "DASH": "DoorDash, Inc.",
#     "DDOG": "Datadog, Inc.",
#     "DXCM": "DexCom, Inc.",
#     "EA": "Electronic Arts Inc.",
#     "EXC": "Exelon Corporation",
#     "FANG": "Diamondback Energy, Inc.",
#     "FAST": "Fastenal Company",
#     "FTNT": "Fortinet, Inc.",
#     "GEHC": "GE Healthcare",
#     "GFS": "GlobalFoundries Inc.",
#     "GILD": "Gilead Sciences, Inc.",
#     "GOOG": "Alphabet Inc. (Class C)",
#     "GOOGL": "Alphabet Inc. (Class A)",
#     "HON": "Honeywell International Inc.",
#     "IDXX": "Idexx Laboratories, Inc.",
#     "INTC": "Intel Corporation",
#     "INTU": "Intuit Inc.",
#     "ISRG": "Intuitive Surgical, Inc.",
#     "KDP": "Keurig Dr Pepper Inc.",
#     "KHC": "The Kraft Heinz Company",
#     "KLAC": "KLA Corporation",
#     "LIN": "Linde plc",
#     "LRCX": "Lam Research Corporation",
#     "LULU": "Lululemon Athletica Inc.",
#     "MAR": "Marriott International, Inc.",
#     "MCHP": "Microchip Technology Inc.",
#     "MDLZ": "Mondelez International, Inc.",
#     "MELI": "MercadoLibre, Inc.",
#     "META": "Meta Platforms, Inc.",
#     "MNST": "Monster Beverage Corporation",
#     "MRVL": "Marvell Technology, Inc.",
#     "MSFT": "Microsoft Corporation",
#     "MSTR": "MicroStrategy Incorporated",
#     "MU": "Micron Technology, Inc.",
#     "NFLX": "Netflix, Inc.",
#     "NVDA": "NVIDIA Corporation",
#     "NXPI": "NXP Semiconductors N.V.",
#     "ODFL": "Old Dominion Freight Line, Inc.",
#     "ON": "ON Semiconductor Corporation",
#     "ORLY": "O'Reilly Automotive, Inc.",
#     "PANW": "Palo Alto Networks, Inc.",
#     "PAYX": "Paychex, Inc.",
#     "PCAR": "PACCAR Inc.",
#     "PDD": "PDD Holdings Inc.",
#     "PEP": "PepsiCo, Inc.",
#     "PLTR": "Palantir Technologies Inc.",
#     "PYPL": "PayPal Holdings, Inc.",
#     "QCOM": "QUALCOMM Incorporated",
#     "REGN": "Regeneron Pharmaceuticals, Inc.",
#     "ROP": "Roper Technologies, Inc.",
#     "ROST": "Ross Stores, Inc.",
#     "SBUX": "Starbucks Corporation",
#     "SHOP": "Shopify Inc.",
#     "SNPS": "Synopsys, Inc.",
#     "TEAM": "Atlassian Corporation Plc",
#     "TMUS": "T-Mobile US, Inc.",
#     "TSLA": "Tesla, Inc.",
#     "TTD": "The Trade Desk, Inc.",
#     "TTWO": "Take-Two Interactive Software, Inc.",
#     "TXN": "Texas Instruments Incorporated",
#     "VRSK": "Verisk Analytics, Inc.",
#     "VRTX": "Vertex Pharmaceuticals Incorporated",
#     "WBD": "Warner Bros. Discovery, Inc.",
#     "WDAY": "Workday, Inc.",
#     "XEL": "Xcel Energy Inc.",
#     "ZS": "Zscaler, Inc."
# }
# crypto_assets = {
#     "BTC-USD": "Bitcoin",
#     "ETH-USD": "Ethereum",
#     "XRP-USD": "XRP",
#     "USDT-USD": "Tether",
#     "BNB-USD": "BNB",
#     "SOL-USD": "Solana",
#     "USDC-USD": "USD Coin",
#     "TRX-USD": "TRON",
#     "DOGE-USD": "Dogecoin",
#     "ADA-USD": "Cardano",
#     "LINK-USD": "Chainlink",
#     "WBTC-USD": "Wrapped Bitcoin",
#     "XLM-USD": "Stellar",
#     "SUI-USD": "Sui",
#     "BCH-USD": "Bitcoin Cash",
#     "AVAX-USD": "Avalanche",
#     "HBAR-USD": "Hedera",
#     "LEO-USD": "UNUS SED LEO",
#     "LTC-USD": "Litecoin",
#     "TON11419-USD": "Toncoin",  # Updated ticker
#     "DOT-USD": "Polkadot",
#     "SHIB-USD": "Shiba Inu",
#     "MATIC-USD": "Polygon",
#     "UNI7083-USD": "Uniswap",  # Updated ticker
#     "FIL-USD": "Filecoin",
#     "ICP-USD": "Internet Computer",
#     "ETC-USD": "Ethereum Classic",
#     "ALGO-USD": "Algorand",
#     "EOS-USD": "EOS",
#     "YFI-USD": "yearn.finance",
#     "DASH-USD": "Dash",
#     "MKR-USD": "Maker",
#     "ZEC-USD": "Zcash",
#     "XTZ-USD": "Tezos",
#     "XMR-USD": "Monero",
#     "AAVE-USD": "Aave",
#     "NEO-USD": "Neo",
#     "ATOM-USD": "Cosmos",
#     "IOTA-USD": "IOTA",
#     "KSM-USD": "Kusama",
#     "SLP-USD": "Smooth Love Potion",
#     "CELO-USD": "Celo",
#     # Adding more to reach 50
#     "APT21794-USD": "Aptos",
#     "ARB11841-USD": "Arbitrum",
#     "LDO-USD": "Lido DAO",
#     "CRO-USD": "Cronos",
#     "VET-USD": "VeChain",
#     "FTM-USD": "Fantom",
#     "MANA-USD": "Decentraland"
# }

# Time period (last 2 years)
start_date = "2023-01-01"
end_date = "2025-01-01"

all_data = []

# Download data for each company
for ticker, company_name in crypto_assets.items():
    print(f"Downloading data for {company_name} ({ticker})...")
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            print(f"No data available for {ticker}")
            continue
            
        df.reset_index(inplace=True)
        
        # Flatten column names if they are MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df["Company"] = company_name
        
        # Select only required columns
        df = df[["Date", "Company", "Open", "High", "Low", "Close", "Volume"]]
        all_data.append(df)
        
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        continue

if all_data:
    # Combine all into one DataFrame
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by date and company
    combined_df.sort_values(by=["Date", "Company"], inplace=True)
    
    # Save to CSV
    output_file = "CRYPTO50_Combined.csv"
    combined_df.to_csv(output_file, index=False)
    
    print(f"✅ Combined dataset saved as {output_file}")
    print(f"Total records: {len(combined_df)}")
    print(f"Companies downloaded: {len(all_data)}")
else:
    print("No data was downloaded for any company.")
