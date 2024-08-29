import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as mt
from scipy.stats import norm

# Helper Functions
def convert_maturity_to_days(maturity):
    if 'w' in maturity:
        return int(maturity.replace('w', '')) * 7
    elif 'm' in maturity:
        return int(maturity.replace('m', '')) * 30
    else:
        raise ValueError("Invalid maturity format. Use '1w', '1m', '3m', etc.")

def calculate_forward_price_fixed(df, asset, maturity_days, annual_rate=0.05):
    forward_col = f'Forward Price ({asset})'
    spot_col = asset
    df[forward_col] = np.nan
    for i in range(0, len(df), maturity_days):
        if i + maturity_days < len(df):
            forward_price = df.loc[df.index[i], spot_col] * mt.exp(annual_rate * (maturity_days / 365))
            df.loc[df.index[i:i+maturity_days], forward_col] = forward_price
        else:
            remaining_days = len(df) - i
            forward_price = df.loc[df.index[i], spot_col] * mt.exp(annual_rate * (remaining_days / 365))
            df.loc[df.index[i:], forward_col] = forward_price
    return df

def hedge_strategy_corrected(df, start_date, rewards_frequency, reward_amount, maturity, asset):
    maturity_period = convert_maturity_to_days(maturity)
    if asset not in df.columns:
        raise ValueError(f"Asset '{asset}' not found in data. Available assets: {', '.join(df.columns[1:])}")
    
    asset_data_start_date = df[df[asset].notna()]['Date'].min()
    start_date = pd.to_datetime(start_date) if pd.to_datetime(start_date) >= asset_data_start_date else asset_data_start_date
    
    df = df[df['Date'] >= start_date].copy()
    df = calculate_forward_price_fixed(df, asset, maturity_period)
    
    df[f'Notional Exchanged Forward ({asset})'] = 0.0
    df[f'Notional Exchanged Spot ({asset})'] = 0.0
    df[f'Cumulative Forward ({asset})'] = 0.0
    df[f'Cumulative Spot ({asset})'] = 0.0
    
    reward_interval = {'daily': 1, 'weekly': 7, 'monthly': 30}[rewards_frequency]
    forward_accumulation = 0
    last_maturity_date = 0
    
    for i in range(len(df)):
        if i % reward_interval == 0:
            df.loc[df.index[i], f'Notional Exchanged Spot ({asset})'] = df.loc[df.index[i], asset] * reward_amount
            forward_accumulation += reward_amount
            if i - last_maturity_date >= maturity_period and i + maturity_period < len(df):
                df.loc[df.index[i + maturity_period], f'Notional Exchanged Forward ({asset})'] += (
                    df.loc[df.index[last_maturity_date], f'Forward Price ({asset})'] * forward_accumulation
                )
                forward_accumulation = 0
                last_maturity_date = i

        if i > 0:
            df.loc[df.index[i], f'Cumulative Spot ({asset})'] = df.loc[df.index[i-1], f'Cumulative Spot ({asset})'] + df.loc[df.index[i], f'Notional Exchanged Spot ({asset})']
            df.loc[df.index[i], f'Cumulative Forward ({asset})'] = df.loc[df.index[i-1], f'Cumulative Forward ({asset})'] + df.loc[df.index[i], f'Notional Exchanged Forward ({asset})']
        else:
            df.loc[df.index[i], f'Cumulative Spot ({asset})'] = df.loc[df.index[i], f'Notional Exchanged Spot ({asset})']
            df.loc[df.index[i], f'Cumulative Forward ({asset})'] = df.loc[df.index[i], f'Notional Exchanged Forward ({asset})']
    
    if forward_accumulation > 0:
        final_forward_price = df[f'Forward Price ({asset})'].iloc[-1]
        df.loc[df.index[-1], f'Notional Exchanged Forward ({asset})'] += forward_accumulation * final_forward_price
        df.loc[df.index[-1], f'Cumulative Forward ({asset})'] += forward_accumulation * final_forward_price
    
    final_spot_notional = df[f'Cumulative Spot ({asset})'].iloc[-1]
    final_forward_notional = df[f'Cumulative Forward ({asset})'].iloc[-1]
    returns = final_forward_notional/final_spot_notional - 1
    
    st.write(f"Final accumulated notional with spot strategy: {final_spot_notional}")
    st.write(f"Final accumulated notional with forward strategy: {final_forward_notional}")
    st.write(f"Return of forward strategy relative to spot: {returns:.4%}")
    
    return df

def plot_results_adjusted(df, asset):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Date'], df[asset], label=f'{asset} Spot Price', color='blue')
    ax.step(df['Date'], df[f'Forward Price ({asset})'], label=f'{asset} Forward Price', linestyle='-', color='orange')
    ax.set_title(f'Spot vs Forward Prices Over Time for {asset}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Date'], df[f'Cumulative Spot ({asset})'], label=f'Cumulative Spot Notional ({asset})', color='orange')
    ax.step(df['Date'], df[f'Cumulative Forward ({asset})'], label=f'Cumulative Forward Notional ({asset})', color='blue')
    ax.set_title(f'Cumulative Notional Exchanged Over Time for {asset}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Notional Value')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    return price

def payoff_chart(options):
    S = np.linspace(0.5, 1.5, 500)  # Underlying asset price range
    total_payoff = np.zeros_like(S)

    for option in options:
        K = option['strike']
        T = option['maturity']
        r = option['rate']
        sigma = option['volatility']
        option_type = option['type']
        quantity = option['quantity']

        if option_type == 'call':
            payoff = np.maximum(S - K, 0)
        elif option_type == 'put':
            payoff = np.maximum(K - S, 0)

        total_payoff += quantity * payoff

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(S, total_payoff, label='Total Payoff', color='blue')
    ax.set_title('Options Payoff')
    ax.set_xlabel('Underlying Price')
    ax.set_ylabel('Payoff')
    ax.axhline(0, color='black', linestyle='--')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def options_payoff_simulator():
    st.header("Options Payoff Simulator")

    options = []

    if 'option_count' not in st.session_state:
        st.session_state['option_count'] = 0

    if st.button("Add Option"):
        st.session_state['option_count'] += 1

    for i in range(st.session_state['option_count']):
        st.subheader(f"Option {i+1}")
        option_type = st.selectbox(f"Option Type {i+1}", options=['call', 'put'], key=f"type_{i}")
        buy_or_sell = st.selectbox(f"Buy or Sell {i+1}", options=['buy', 'sell'], key=f"buy_sell_{i}")
        strike = st.number_input(f"Strike Price {i+1}", value=1.0, key=f"strike_{i}")
        maturity = st.number_input(f"Maturity (years) {i+1}", value=1.0, key=f"maturity_{i}")
        volatility = st.number_input(f"Volatility {i+1}", value=0.2, key=f"volatility_{i}")
        rate = st.number_input(f"Risk-Free Rate {i+1}", value=0.05, key=f"rate_{i}")
        quantity = st.number_input(f"Quantity {i+1}", value=1, key=f"quantity_{i}")

        # Adjust quantity for sell (negative quantity)
        quantity = -quantity if buy_or_sell == 'sell' else quantity

        # Calculate the option price using Black-Scholes
        price = black_scholes_price(1.0, strike, maturity, rate, volatility, option_type)
        st.write(f"Black-Scholes Price for Option {i+1}: {price:.4f}")

        options.append({
            'type': option_type,
            'strike': strike,
            'maturity': maturity,
            'volatility': volatility,
            'rate': rate,
            'quantity': quantity
        })

    if options:
        payoff_chart(options)

# Streamlit App Interface with Sidebar
st.sidebar.title("DECIMAL HEDGE - STRATEGIES SIMULATOR")
page = st.sidebar.selectbox("Choose a page", ["Hedging Strategy", "Options Payoff Simulation"])

if page == "Hedging Strategy":
    st.title("Hedging Strategy Simulation")
    
    # Replace the file uploader with a direct GitHub file read
    github_url = 'https://raw.githubusercontent.com/your-username/your-repo/main/path/to/your/file.xlsx'
    hp_df = pd.read_excel(github_url)
    st.write("Data Preview:", hp_df.head())

    start_date = st.date_input("Start Date", value=pd.to_datetime("2018-03-01"))
    rewards_frequency = st.selectbox("Rewards Frequency", options=['daily', 'weekly', 'monthly'])
    reward_amount = st.number_input("Reward Amount", value=1.0, min_value=0.0)
    maturity = st.selectbox("Maturity Period", options=['1w', '1m', '3m', '6m', '12m'])
    asset = st.selectbox("Asset", options=hp_df.columns[1:])

    if st.button("Run Hedging Strategy"):
        hedged_df_corrected = hedge_strategy_corrected(hp_df, start_date, rewards_frequency, reward_amount, maturity, asset)
        st.write("Hedging Strategy Results")
        plot_results_adjusted(hedged_df_corrected, asset)

elif page == "Options Payoff Simulation":
    options_payoff_simulator()
