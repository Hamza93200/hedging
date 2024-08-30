import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as mt

# Helper Functions
def convert_maturity_to_days(maturity):
    if 'w' in maturity:
        return int(maturity.replace('w', '')) * 7
    elif 'M' in maturity:
        return int(maturity.replace('M', '')) * 30
    else:
        raise ValueError("Invalid maturity format. Use '1w', '1M', '3M', etc.")

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
    returns = final_forward_notional / final_spot_notional - 1
    
    st.subheader("Results")
    st.write(f"**Final accumulated notional with spot strategy:** {final_spot_notional:.2f}")
    st.write(f"**Final accumulated notional with forward strategy:** {final_forward_notional:.2f}")
    st.write(f"**Return of forward strategy relative to spot:** {returns:.2%}")
    
    return df

def plot_results_adjusted(df, asset):
    st.subheader("Spot vs Forward Prices Over Time")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df['Date'], df[asset], label=f'{asset} Spot Price', color='blue')
    ax.plot(df['Date'], df[f'Forward Price ({asset})'], label=f'{asset} Forward Price', linestyle='--', color='orange')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
    st.subheader("Cumulative Notional Exchanged Over Time")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df['Date'], df[f'Cumulative Spot ({asset})'], label=f'Cumulative Spot Notional ({asset})', color='green')
    ax.plot(df['Date'], df[f'Cumulative Forward ({asset})'], label=f'Cumulative Forward Notional ({asset})', color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Notional Value')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# Function to calculate and plot payoff for vanilla options
def calculate_option_payoff(option_type, is_bought, strike_price, spot_prices, premium):
    if option_type == 'Call':
        intrinsic_values = np.maximum(spot_prices - strike_price, 0)
    else:  # Put
        intrinsic_values = np.maximum(strike_price - spot_prices, 0)
    
    payoff = intrinsic_values - premium if is_bought else premium - intrinsic_values
    return payoff

def plot_payoffs(options):
    if options.empty:
        st.warning("No options added yet. Please add options to see the payoff diagram.")
        return
    
    min_strike = options['Strike Price'].min()
    max_strike = options['Strike Price'].max()
    spot_prices = np.linspace(min_strike * 0.5, max_strike * 1.5, 500)
    total_payoff = np.zeros_like(spot_prices)
    
    plt.figure(figsize=(8, 4))
    
    for idx, option in options.iterrows():
        payoff = calculate_option_payoff(
            option_type=option['Type'],
            is_bought=option['Position'] == 'Buy',
            strike_price=option['Strike Price'],
            spot_prices=spot_prices,
            premium=option['Premium']
        )
        total_payoff += payoff
        plt.plot(spot_prices, payoff, label=f"Option {idx+1}: {option['Position']} {option['Type']} (K={option['Strike Price']}, σ={option['Volatility']}, T={option['Maturity']})")
    
    plt.plot(spot_prices, total_payoff, label='Total Payoff', color='black', linewidth=2, linestyle='--')
    plt.title('Options Payoff Diagram')
    plt.xlabel('Spot Price at Maturity (%)')
    plt.ylabel('Payoff (%)')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

def black_scholes_price(option_type, S, K, T, r, sigma):
    from scipy.stats import norm
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'Call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

# Streamlit App Interface
st.set_page_config(page_title="Decimal Hedge - Strategies Simulator", layout="wide")

st.sidebar.title("DECIMAL HEDGE - STRATEGIES SIMULATOR")
if st.sidebar.button("Forward Backtesting"):
    st.experimental_set_query_params(page="ForwardBacktesting")
if st.sidebar.button("Vanilla Options Payoff Simulator"):
    st.experimental_set_query_params(page="VanillaOptionsPayoffSimulator")

query_params = st.experimental_get_query_params()
page = query_params.get("page", ["ForwardBacktesting"])[0]

if page == "ForwardBacktesting":
    st.title("Forward Backtesting")
    
    # Load data from GitHub
    github_url = 'https://raw.githubusercontent.com/hamza93200/hedging/main/HP.xlsx'
    try:
        hp_df = pd.read_excel(github_url)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
    
    st.subheader("Input Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.date_input("Start Date", value=pd.to_datetime("2018-03-01"))
    with col2:
        rewards_frequency = st.selectbox("Rewards Frequency", options=['daily', 'weekly', 'monthly'])
    with col3:
        asset = st.selectbox("Asset", options=hp_df.columns[1:])
    
    col1, col2 = st.columns(2)
    with col1:
        reward_amount = st.number_input("Reward Amount", value=1.0, min_value=0.0)
    with col2:
        maturity = st.selectbox("Forward Maturity", options=['1w', '1M', '3M', '6M', '12M'])
    
    if st.button("Run Hedging Strategy"):
        with st.spinner("Running hedging strategy simulation..."):
            hedged_df_corrected = hedge_strategy_corrected(hp_df, start_date, rewards_frequency, reward_amount, maturity, asset)
            plot_results_adjusted(hedged_df_corrected, asset)

elif page == "VanillaOptionsPayoffSimulator":
    st.title("Vanilla Options Payoff Simulator")
    
    if 'options_data' not in st.session_state:
        st.session_state.options_data = pd.DataFrame(columns=['Type', 'Position', 'Strike Price', 'Premium', 'Volatility', 'Maturity', 'Risk-Free Rate'])
        # Add a default call option
        st.session_state.options_data = pd.concat([st.session_state.options_data, pd.DataFrame([{
            'Type': 'Call',
            'Position': 'Buy',
            'Strike Price': 100.0,
            'Premium': 0.0,
            'Volatility': 0.2,
            'Maturity': 1.0,
            'Risk-Free Rate': 0.05
        }])], ignore_index=True)
    
    st.subheader("Add New Option")
    with st.form(key='option_form'):
        cols = st.columns(3)
        option_type = cols[0].selectbox("Option Type", options=['Call', 'Put'])
        position = cols[1].selectbox("Position", options=['Buy', 'Sell'])
        strike_price = cols[2].number_input("Strike Price (%)", value=100.0, min_value=0.0)
        
        cols = st.columns(3)
        volatility = cols[0].number_input("Volatility (σ)", value=0.2, min_value=0.0, format="%.2f")
        maturity = cols[1].number_input("Maturity (in years)", value=1.0, min_value=0.01, format="%.2f")
        risk_free_rate = cols[2].number_input("Risk-Free Rate (r)", value=0.05, min_value=0.0, format="%.2f")
        
        underlying_price = st.number_input("Underlying Asset Price (%)", value=100.0, min_value=0.0)
        premium = black_scholes_price(option_type, underlying_price, strike_price, maturity, risk_free_rate, volatility)
        
        if st.form_submit_button(label="Add Option"):
            new_option = {
                'Type': option_type,
                'Position': position,
                'Strike Price': strike_price,
                'Premium': premium,
                'Volatility': volatility,
                'Maturity': maturity,
                'Risk-Free Rate': risk_free_rate
            }
            st.session_state.options_data = pd.concat([st.session_state.options_data, pd.DataFrame([new_option])], ignore_index=True)
            st.success("Option added successfully!")
            plot_payoffs(st.session_state.options_data)
    
    st.subheader("Current Options")
    for idx, option in st.session_state.options_data.iterrows():
        st.write(f"Option {idx+1}: {option['Position']} {option['Type']} (K={option['Strike Price']}%, σ={option['Volatility']}%, T={option['Maturity']} years)")
        if st.button(f"Remove Option {idx+1}"):
            st.session_state.options_data = st.session_state.options_data.drop(idx).reset_index(drop=True)
            st.success(f"Option {idx+1} removed successfully!")
            plot_payoffs(st.session_state.options_data)
    
    if st.button("Reset All Options"):
        st.session_state.options_data = pd.DataFrame(columns=['Type', 'Position', 'Strike Price', 'Premium', 'Volatility', 'Maturity', 'Risk-Free Rate'])
        # Add a default call option
        st.session_state.options_data = pd.concat([st.session_state.options_data, pd.DataFrame([{
            'Type': 'Call',
            'Position': 'Buy',
            'Strike Price': 100.0,
            'Premium': 0.0,
            'Volatility': 0.2,
            'Maturity': 1.0,
            'Risk-Free Rate': 0.05
        }])], ignore_index=True)
        st.success("Options reset successfully!")
        plot_payoffs(st.session_state.options_data)
