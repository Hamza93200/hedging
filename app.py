
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as mt
from scipy.stats import norm
from datetime import datetime
import requests
import time
import os 


def black_scholes_price(option_type, S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'Call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    # Calculate premium as a percentage of the spot price
    premium_percentage = (price / S) * 100
    return premium_percentage

def update_data():
    # ======= CONFIGURATION =======
    EXCEL_FILE = 'HP.xlsx'
    OUTPUT_FILE = 'HP.xlsx'

    # ======= CHARGER LE FICHIER EXCEL =======
    df = pd.read_excel(EXCEL_FILE)

    # ======= TROUVER LA DERNIÈRE DATE =======
    last_date = pd.to_datetime(df['Date']).max()

    # ======= FONCTION POUR RÉCUPÉRER LES DONNÉES DE CLÔTURE BINANCE =======
    def fetch_binance_data(symbol, start_date):
        url = f"https://api.binance.com/api/v3/klines"
        start_time = int(start_date.timestamp() * 1000)
        end_time = int(datetime.utcnow().timestamp() * 1000)
        interval = '1d'
        
        all_data = []
        
        while start_time < end_time:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': start_time,
                'endTime': end_time,
                'limit': 1000
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if not data:
                break
            
            for kline in data:
                date = datetime.fromtimestamp(kline[0] / 1000)
                close_price = float(kline[4])
                all_data.append([date, close_price])
            
            start_time = data[-1][0] + 1
            
            # Évite le rate limit de Binance (12 requêtes/minute)
            time.sleep(0.1)
        return pd.DataFrame(all_data, columns=['Date', symbol])

    # ======= RÉCUPÉRER LES NOUVELLES DONNÉES =======
    start_date = last_date + pd.Timedelta(days=1)
    
    # ETHUSDT
    eth_data = fetch_binance_data('ETHUSDT', start_date).rename(columns={'ETHUSDT': 'ETH'})

    # SOLUSDT
    sol_data = fetch_binance_data('SOLUSDT', start_date).rename(columns={'SOLUSDT': 'SOL'})

    # Fusionner les nouvelles données par date
    new_data = pd.merge(eth_data, sol_data, on='Date', how='outer')

    # Fusionner avec le dataframe existant
    df = pd.concat([df, new_data], ignore_index=True)

    # ======= SAUVEGARDER LE FICHIER =======
    df.to_excel(OUTPUT_FILE, index=False)
    st.write(f"Data Refreshed: {start_date}")
    



def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price



def put_hedge1(put_strike_multiplier,daily_rewards,protocol,option_maturity,hedging_start_date,IR=0.05,sigma=0.6,broker_spread=0.01):
    data = pd.read_excel("HP.xlsx",index_col=0,parse_dates=True)
    data_to_hedge = data[protocol][data.index >= pd.to_datetime(hedging_start_date)]
    data_to_hedge = data_to_hedge.dropna()
    hedging_start_date = data_to_hedge.first_valid_index()

    weekly_offramp_rewards = []
    hedged_offramp_rewards = []

    weekly_offramp_notional = []
    hedged_offramp_notional = []

    put_prices = []

    days_until_maturity = option_maturity
    days_until_week_end = 7

    spot = data_to_hedge[0]
    strike = spot * np.exp(1/12 * IR)
    put_strike = strike *put_strike_multiplier

    put_price = black_scholes_put(spot,put_strike,option_maturity/365,IR,sigma)


    put_prices.append(put_price *(option_maturity*daily_rewards))

    for i in range(len(data_to_hedge)):

        weekly_offramp_rewards.append(daily_rewards)
        hedged_offramp_rewards.append(daily_rewards)
        spot = data_to_hedge[i]

        if days_until_maturity == 0:

            accumulated_rewards = sum(hedged_offramp_rewards)

            if spot <= put_strike:
                hedged_offramp_notional.append(accumulated_rewards * put_strike)

            elif spot > put_strike:
                hedged_offramp_notional.append(accumulated_rewards * spot)

            hedged_offramp_rewards = []

            strike = spot * np.exp(1/12 * IR)
            put_strike = strike *put_strike_multiplier

            put_price = black_scholes_put(spot,put_strike,option_maturity/365,IR,sigma)

            put_prices.append(put_price *(option_maturity*daily_rewards))

            days_until_maturity = option_maturity
        
        if days_until_week_end == 0:
            
            accumulated_rewards = sum(weekly_offramp_rewards)
            weekly_offramp_notional.append(accumulated_rewards * (spot - (broker_spread)) )
            weekly_offramp_rewards = []

            days_until_week_end = 7
        
        days_until_week_end -= 1
        days_until_maturity -= 1

    spot_end_notional = sum(weekly_offramp_notional)
    hedged_end_notional = sum(hedged_offramp_notional)
    options_price_spent = sum(put_prices)

    final_pnl = hedged_end_notional - spot_end_notional - options_price_spent
    final_pnl_perc = (((hedged_end_notional- options_price_spent) / spot_end_notional) - 1) * 100

    return spot_end_notional,hedged_end_notional,options_price_spent,final_pnl,final_pnl_perc,put_prices


def call_hedge(call_strike_multiplier,daily_rewards,protocol,option_maturity,hedging_start_date,IR=0.05,sigma=0.6,broker_spread=0.01):
    data = pd.read_excel("HP.xlsx",index_col=0,parse_dates=True)

    data_to_hedge = data[protocol][data.index >= pd.to_datetime(hedging_start_date)]
    data_to_hedge = data_to_hedge.dropna()
    hedging_start_date = data_to_hedge.first_valid_index()

    weekly_offramp_rewards = []
    hedged_offramp_rewards = []

    weekly_offramp_notional = []
    hedged_offramp_notional = []

    call_prices = []

    days_until_maturity = option_maturity
    days_until_week_end = 7

    spot = data_to_hedge[0]
    strike = spot * np.exp(1/12 * IR)
    call_strike = strike *call_strike_multiplier

    call_price = black_scholes_call(spot,call_strike,option_maturity/365,IR,sigma)


    call_prices.append(call_price *(option_maturity*daily_rewards))

    for i in range(len(data_to_hedge)):

        weekly_offramp_rewards.append(daily_rewards)
        hedged_offramp_rewards.append(daily_rewards)
        spot = data_to_hedge[i]

        if days_until_maturity == 0:

            accumulated_rewards = sum(hedged_offramp_rewards)

            if spot < call_strike:
                hedged_offramp_notional.append(accumulated_rewards * spot)

            elif spot >= call_strike:
                hedged_offramp_notional.append(accumulated_rewards * spot - ((spot - call_strike) * accumulated_rewards))

            hedged_offramp_rewards = []

            strike = spot * np.exp(1/12 * IR)
            call_strike = strike *call_strike_multiplier

            call_price = black_scholes_call(spot,call_strike,option_maturity/365,IR,sigma)

            call_prices.append(call_price *(option_maturity*daily_rewards))

            days_until_maturity = option_maturity
        
        if days_until_week_end == 0:
            
            accumulated_rewards = sum(weekly_offramp_rewards)
            weekly_offramp_notional.append(accumulated_rewards * (spot - (broker_spread)) )
            weekly_offramp_rewards = []

            days_until_week_end = 7
        
        days_until_week_end -= 1
        days_until_maturity -= 1

    spot_end_notional = sum(weekly_offramp_notional)
    hedged_end_notional = sum(hedged_offramp_notional)
    options_price_gained = sum(call_prices)

    final_pnl = hedged_end_notional - spot_end_notional + options_price_gained
    final_pnl_perc = (((hedged_end_notional+ options_price_gained) / spot_end_notional) - 1) * 100

    return spot_end_notional,hedged_end_notional,options_price_gained,final_pnl,final_pnl_perc,call_prices


def convert_maturity_to_years(maturity):
    """Convert maturity strings like '1w', '1M', etc., to a fraction of years."""
    if 'w' in maturity:
        return int(maturity.replace('w', '')) * 7 / 365
    elif 'M' in maturity:
        return int(maturity.replace('M', '')) * 30 / 365
    elif 'y' in maturity:
        return int(maturity.replace('y', ''))
    else:
        raise ValueError("Invalid maturity format. Use '1w', '1M', '3M', etc.")

def convert_maturity_to_days(maturity):
    """Convert maturity strings like '1w', '1M', etc., to days."""
    if 'w' in maturity:
        return int(maturity.replace('w', '')) * 7
    elif 'M' in maturity:
        return int(maturity.replace('M', '')) * 30
    elif 'y' in maturity:
        return int(maturity.replace('y', '')) * 365
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
    
    spot_prices = np.linspace(50, 150, 500)  # Adjusted range centered on 100%
    total_payoff = np.zeros_like(spot_prices)
    
    for _, option in options.iterrows():
        payoff = calculate_option_payoff(
            option_type=option['Type'],
            is_bought=option['Position'] == 'Buy',
            strike_price=option['Strike Price'],
            spot_prices=spot_prices,
            premium=option['Premium']
        )
        total_payoff += payoff
    
    plt.figure(figsize=(8, 4))
    plt.step(spot_prices, total_payoff, label='Total Payoff', color='black', linewidth=2, linestyle='--', where='mid')
    plt.title('Options Payoff Diagram')
    plt.xlabel('Spot Price at Maturity (%)')
    plt.ylabel('Payoff (%)')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)



def collar(call_strike_multiplier,put_strike_multiplier,daily_rewards,protocol,option_maturity,hedging_start_date,percent_to_hedge=0.7,IR=0.05,sigma=0.6,broker_spread=0.10):

    data = pd.read_excel("HP.xlsx",index_col=0,parse_dates=True)
    data_rewards = pd.read_excel("v2_revenue.xlsx",index_col=0,parse_dates=True)

    data_to_hedge = data[protocol][data.index >= pd.to_datetime(hedging_start_date)]
    data_to_hedge = data_to_hedge.dropna()
    hedging_start_date = data_to_hedge.first_valid_index()

    start_window = hedging_start_date - pd.Timedelta(days=30)
    df_base_rewards = data_rewards[protocol][(data_rewards.index > start_window) & (data_rewards.index < hedging_start_date)]
    notional_tohedge_inkind = df_base_rewards.mean()* percent_to_hedge

    mask = data_rewards.index >= pd.to_datetime(hedging_start_date)
    data_rewards_from_start = data_rewards.loc[mask, protocol]
    data_rewards_from_start = data_rewards_from_start.dropna()

    weekly_offramp_rewards = []
    hedged_offramp_rewards = []
    actual_rewards = []

    monthly_actual_rewards = []
    monthly_hedged_rewards = []

    weekly_offramp_notional = []
    hedged_offramp_notional = []

    put_prices = []
    call_prices = []

    days_until_maturity = option_maturity
    days_until_week_end = 7

    spot = data_to_hedge[0]
    strike = spot * np.exp(1/12 * IR)
    put_strike = strike *put_strike_multiplier
    call_strike = strike *call_strike_multiplier

    put_price = black_scholes_put(spot,put_strike,option_maturity/365,IR,sigma)
    call_price = black_scholes_call(spot,call_strike,option_maturity/365,IR,sigma)

    put_prices.append(put_price * (option_maturity*notional_tohedge_inkind))
    call_prices.append(call_price *(option_maturity*notional_tohedge_inkind))

    for i in range(len(data_to_hedge)):

        
        hedged_offramp_rewards.append(notional_tohedge_inkind)
        
        spot = data_to_hedge[i]
        today_reward = data_rewards_from_start[i]
        actual_rewards.append(today_reward)
        weekly_offramp_rewards.append(today_reward)


        if days_until_maturity == 0:

            accumulated_rewards = sum(hedged_offramp_rewards)
            actual_accumulated_rewards = sum(actual_rewards)

            

            if actual_accumulated_rewards < accumulated_rewards:
                if put_strike < spot and spot < call_strike:
                    temp_notional =actual_accumulated_rewards * spot
                elif spot <= put_strike:
                    temp_notional=actual_accumulated_rewards * put_strike
                elif spot >= call_strike:
                    temp_notional=actual_accumulated_rewards * spot - ((spot - call_strike) * actual_accumulated_rewards)
            else:
                if put_strike < spot and spot < call_strike:
                    temp_notional =accumulated_rewards * spot
                elif spot <= put_strike:
                    temp_notional=accumulated_rewards * put_strike
                elif spot >= call_strike:
                    temp_notional=accumulated_rewards * spot - ((spot - call_strike) * accumulated_rewards)
            

            hedged_offramp_notional.append(temp_notional)
            
            monthly_actual_rewards.append(actual_accumulated_rewards)
            monthly_hedged_rewards .append(accumulated_rewards)
            
            hedged_offramp_rewards = []

            strike = spot * np.exp(1/12 * IR)
            put_strike = strike *put_strike_multiplier
            call_strike = strike *call_strike_multiplier

            put_price = black_scholes_put(spot,put_strike,option_maturity/365,IR,sigma)
            call_price = black_scholes_call(spot,call_strike,option_maturity/365,IR,sigma)
            
            put_prices.append(put_price* (option_maturity*notional_tohedge_inkind))
            call_prices.append(call_price *(option_maturity*notional_tohedge_inkind))

            days_until_maturity = option_maturity
            start_window = data_rewards_from_start.index[i] - pd.Timedelta(days=30)
            st.write(f"from : {start_window}")
            df_base_rewards = data_rewards[protocol][(data_rewards.index > start_window) & (data_rewards.index < data_rewards_from_start.index[i])]
            notional_tohedge_inkind = df_base_rewards.mean()* percent_to_hedge
            st.write(f"Daily average rewards: {notional_tohedge_inkind}")
        
        if days_until_week_end == 0:
            
            accumulated_rewards = sum(weekly_offramp_rewards)
            weekly_offramp_notional.append(accumulated_rewards * (spot - (broker_spread)) )
            weekly_offramp_rewards = []

            days_until_week_end = 7
        
        days_until_week_end -= 1
        days_until_maturity -= 1

    spot_end_notional = sum(weekly_offramp_notional)
    hedged_end_notional = sum(hedged_offramp_notional)
    put_options_price = sum(put_prices)
    call_options_price = sum(call_prices)
    options_price_paid = sum(call_prices) - sum(put_prices)

    final_pnl = hedged_end_notional - spot_end_notional + options_price_paid
    final_pnl_perc = ((hedged_end_notional / spot_end_notional) - 1) * 100
    df_hedged_vs_actual_rewards = pd.DataFrame({"Hegded rewards per month":monthly_hedged_rewards,"Actual rewards per month": monthly_actual_rewards})
    
    return spot_end_notional,hedged_end_notional,final_pnl,final_pnl_perc,call_prices,put_options_price,call_options_price,options_price_paid,df_hedged_vs_actual_rewards




def put_hedge(put_strike_multiplier,daily_rewards,protocol,option_maturity,hedging_start_date,hedging_end_date,percent_to_hedge=0.7,IR=0.05,sigma=0.6,broker_spread=0.10):

    data = pd.read_excel("HP.xlsx",index_col=0,parse_dates=True)
    data_rewards = pd.read_excel("v2_revenue.xlsx",index_col=0,parse_dates=True)

    data_to_hedge = data[protocol][data.index >= pd.to_datetime(hedging_start_date)]
    data_to_hedge = data_to_hedge.dropna()
    hedging_start_date = data_to_hedge.first_valid_index()

    start_window = hedging_start_date - pd.Timedelta(days=30)

    df_base_rewards = data_rewards[protocol][(data_rewards.index > start_window) & (data_rewards.index < hedging_start_date)]
    notional_tohedge_inkind = df_base_rewards.mean()* percent_to_hedge

    data_to_hedge_vol = data[protocol][(data.index > start_window) & (data.index < hedging_start_date)]


    sigma = data_to_hedge_vol.pct_change().dropna().std() * np.sqrt(365)
    st.write(f"Vol: {sigma}" )
    mask = data_rewards.index >= pd.to_datetime(hedging_start_date)

    data_rewards_from_start = data_rewards.loc[mask, protocol]
    data_rewards_from_start = data_rewards_from_start.dropna()

    weekly_offramp_rewards = []
    hedged_offramp_rewards = []
    actual_rewards = []

    monthly_actual_rewards = []
    monthly_hedged_rewards = []

    weekly_offramp_notional = []
    hedged_offramp_notional = []

    put_prices = []

    days_until_maturity = option_maturity
    days_until_week_end = 1

    spot = data_to_hedge[0]
    strike = spot * np.exp(1/12 * IR)
    put_strike = strike *put_strike_multiplier

    put_price = black_scholes_put(spot,put_strike,option_maturity/365,IR,sigma)

    put_prices.append(put_price * (option_maturity*notional_tohedge_inkind))

    hedge_strike = []
    spot_strike = []


    for i in range(len(data_to_hedge)):

        if data_to_hedge.index[i]> pd.to_datetime(hedging_end_date):
            break
        
        hedged_offramp_rewards.append(notional_tohedge_inkind)
        
        spot = data_to_hedge[i]
        today_reward = data_rewards_from_start[i]
        actual_rewards.append(today_reward)
        weekly_offramp_rewards.append(today_reward)


        if days_until_maturity == 0:

            accumulated_rewards = sum(hedged_offramp_rewards)
            actual_accumulated_rewards = sum(actual_rewards)

            if actual_accumulated_rewards < accumulated_rewards:
                if spot <= put_strike:
                    hedged_offramp_notional.append(actual_accumulated_rewards * put_strike)
                    hedge_strike.append(put_strike)
                elif spot > put_strike:
                    hedged_offramp_notional.append(actual_accumulated_rewards * spot)
                    spot_strike.append(spot)

            else:

                if spot <= put_strike:
                    temp_not = accumulated_rewards * put_strike
                    hedge_strike.append(put_strike)

                elif spot > put_strike:
                    temp_not=accumulated_rewards * spot
                    spot_strike.append(spot)
                
                temp_not =temp_not + ((actual_accumulated_rewards-accumulated_rewards)*spot)

                hedged_offramp_notional.append(temp_not)



            monthly_actual_rewards.append(actual_accumulated_rewards)
            monthly_hedged_rewards .append(accumulated_rewards)

            hedged_offramp_rewards = []
            actual_rewards = []

            strike = spot * np.exp(1/12 * IR)
            put_strike = strike *put_strike_multiplier


            put_price = black_scholes_put(spot,put_strike,option_maturity/365,IR,sigma)

            put_prices.append(put_price* (option_maturity*notional_tohedge_inkind))

            days_until_maturity = option_maturity
            start_window = data_rewards_from_start.index[i] - pd.Timedelta(days=30)
            df_base_rewards = data_rewards[protocol][(data_rewards.index > start_window) & (data_rewards.index < data_rewards_from_start.index[i])]
            notional_tohedge_inkind = df_base_rewards.mean()* percent_to_hedge
        
        if days_until_week_end == 0:
            
            accumulated_rewards = sum(weekly_offramp_rewards)
            weekly_offramp_notional.append(accumulated_rewards * (spot - (broker_spread)) )
            weekly_offramp_rewards = []

            days_until_week_end = 1
        
        days_until_week_end -= 1
        days_until_maturity -= 1

    spot_end_notional = sum(weekly_offramp_notional)
    hedged_end_notional = sum(hedged_offramp_notional)
    put_options_price = sum(put_prices)

    final_pnl = hedged_end_notional - spot_end_notional - put_options_price
    final_pnl_perc = (((hedged_end_notional- put_options_price) / spot_end_notional) - 1) * 100
    df_hedged_vs_actual_rewards = pd.DataFrame({"Hegded rewards per month":monthly_hedged_rewards,"Actual rewards per month": monthly_actual_rewards})

    
    return spot_end_notional,hedged_end_notional,final_pnl,final_pnl_perc,put_options_price,df_hedged_vs_actual_rewards






EXCEL_FILE = "options_data.xlsx"

# ====== Initialize Cached Data ======
@st.cache_data
def load_options_data():
    df = pd.read_excel(EXCEL_FILE)
    return df


@st.cache_data
def save_options_data(df):
    df.to_excel(EXCEL_FILE, index=False)

@st.cache_data
def load_options_pnl():
    df = pd.read_excel(EXCEL_FILE)
    return df


@st.cache_data
def save_options_pnl(df):
    df.to_excel(EXCEL_FILE, index=False)


# ====== Fetch real-time price from Binance ======
def get_binance_price(symbol):
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    response = requests.get(url)
    data = response.json()
    return float(data['price'])

# ====== Calculate Option Payoff ======
def calculate_option_pnl(option, spot_price):
    option_type = option['Type']
    strike_price = option['Strike Price']
    premium = option['Premium']
    position = option['Position']
    num_options = option['Number of options']
    
    if option_type == 'Call':
        intrinsic_value = max(spot_price - strike_price, 0)
    else:
        intrinsic_value = max(strike_price - spot_price, 0)
    
    pnl_per_option = (intrinsic_value - premium) if position == 'Buy' else (premium - intrinsic_value)
    total_pnl = pnl_per_option * num_options
    return total_pnl

# ====== Options PnL Page ======
def options_pnl_page():


    options_data = load_options_data()
    options_pnl= load_options_pnl()
    


    # ========== Add New Option Form ==========
    st.subheader("Add New Option")
    with st.form(key="add_option_form"):
        col1, col2, col3, col4 = st.columns(4)
        option_type = col1.selectbox("Type", options=["Call", "Put"])
        position = col2.selectbox("Position", options=["Buy", "Sell"])
        symbol = col3.selectbox("Symbol", options=["ETHUSDT", "SOLUSDT"])
        num_options = col4.number_input("Number of Options", min_value=1, value=1)

        col1, col2 = st.columns(2)
        volatility = col1.number_input("Volatility (σ)", min_value=0.01, value=0.6, format="%.2f")
        risk_free_rate = col2.number_input("Risk-Free Rate (r)", min_value=0.0, value=0.02, format="%.2f")

        col1, col2, col3 = st.columns(3)
        strike_price = col1.number_input("Strike Price", min_value=0.0, value=100.0)
        maturity = col2.number_input("Maturity (Days)", min_value=1, value=30)

        # Fetch live price
        spot_price = get_binance_price(symbol)
        strike_price = spot_price*(strike_price/100)

        # Estimate premium using Black-Scholes model
        if option_type == "Call":
            option_price = black_scholes_call(spot_price, strike_price, maturity / 365, risk_free_rate, volatility)
        else:
            option_price = black_scholes_put(spot_price, strike_price, maturity / 365, risk_free_rate, volatility)

        if position=="Buy":

            premium = col3.number_input("Premium", value=-option_price)
        else:
            premium = col3.number_input("Premium", value=option_price)

        if st.form_submit_button("Add Option"):
            
            new_option = {
                "Date": datetime.datetime.now().strftime('%Y-%m-%d'),
                "Type": option_type,
                "Position": position,
                "Strike Price": strike_price,
                "Premium": premium,
                "Volatility": volatility,
                "Maturity (Days)": maturity,
                "Risk-Free Rate": risk_free_rate,
                "Symbol": symbol,
                "Number of options": num_options,
            }

            # Append to DataFrame and save to Excel
            options_data = pd.concat([options_data, pd.DataFrame([new_option])], ignore_index=True)
            save_options_data(options_data)

            st.success(f"Added {option_type} option on {symbol}.")

    # ========== Real-time PnL ==========
    st.subheader("Live PnL Tracking")
    while True:
        if not options_data.empty:
            total_pnl = 0
            pnl_data = []

            for _, option in options_data.iterrows():
                spot_price = get_binance_price(option['Symbol'])
                pnl = calculate_option_pnl(option, spot_price)
                total_pnl += pnl

                pnl_data.append({
                    'Symbol': option['Symbol'],
                    'Spot Price': spot_price,
                    'Strike Price': option['Strike Price'],
                    'PnL': pnl
                })

            # Display PnL table
            pnl_df = pd.DataFrame(pnl_data)
            st.write("**Real-time PnL Data**")
            st.dataframe(pnl_df)

            # Total PnL
            st.write(f"### Total PnL: `{round(total_pnl, 2)} USD`")

            # ========== Plot Payoff ==========
            st.subheader("Payoff Over Spot Price")
            spot_prices = np.linspace(
                min(pnl_df["Spot Price"]) * 0.8,
                max(pnl_df["Spot Price"]) * 1.2,
                100
            )
            total_payoff = np.zeros_like(spot_prices)

            for _, option in options_data.iterrows():
                if option["Type"] == "Call":
                    intrinsic_values = np.maximum(spot_prices - option["Strike Price"], 0)
                else:
                    intrinsic_values = np.maximum(option["Strike Price"] - spot_prices, 0)

                payoff = intrinsic_values - option["Premium"] if option["Position"] == "Buy" else option["Premium"] - intrinsic_values
                total_payoff += payoff * option["Number of options"]

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(spot_prices, total_payoff, label="Total Payoff", color="blue", linewidth=2)
            ax.axhline(0, color="gray", linestyle="--")
            ax.set_xlabel("Spot Price")
            ax.set_ylabel("PnL ($)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            time.sleep(5)

# ====== Display the Page ======
def Options_PnL():
    options_pnl_page()



def Backtesting():
    st.title("Backtesting")
    
    # Explanation of the Forward Backtesting Strategy
    #st.markdown("""
    #In this simulation, we aim to compare the effectiveness of using a spot strategy versus a forward strategy over a specified period. 
    #A spot strategy involves exchanging the asset at its current market price (spot price) when rewards are claimed, while a forward strategy 
    #locks-in a future price (forward price) today for delivery at a later date. By backtesting these strategies on historical data, 
    #we can determine the Pnl you could reach by using a forward hedging strategy.
    
    #You can customize the parameters of the simulation, such as the Hedge start date, the frequency of rewards, and the maturity of the forward 
    #contracts, to see how different strategies would have performed.
    #""")

    # Load data from GitHub

    @st.cache_data
    def load_data():
        github_url = 'HP.xlsx'
        try:
            hp_df = pd.read_excel(github_url)
            temp_df = pd.read_excel(github_url, index_col=0)
            st.write("Data loaded successfully from cache.")
            return hp_df, temp_df
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            return None, None
    hp_df, temp_df = load_data()
        
    if st.button("Refresh Data"):
        update_data()
        hp_df, temp_df = load_data()

    st.subheader("Input Parameters")
    
    col1, col2= st.columns(2)

    with col1:
        vol = st.number_input("Vol", value=0.6, min_value=0.0,max_value = 1.0)
    with col2:
        Risk_free_rate = st.number_input("Risk Free Rate", value=0.05, min_value=-1.0,max_value = 1.0)

    product = st.selectbox("Product", options =['Put','Collar'] )
    
    if product == "Collar":
        


        col1, col2, col3,col4= st.columns(4)
        with col1:
            protocol = st.selectbox("Asset", options=hp_df.columns[1:])
        with col2:
            option_maturity = st.selectbox("Option Maturity", options=['1w', '1M', '3M', '6M', '12M', '24M', '36M'], index=1)  # Default to '12M'
        with col3:
            strike_opt = st.number_input("Lower Strike", value=80.0, min_value=0.0)
        with col4:
            strike2_opt = st.number_input("Upper Strike", value=120.0, min_value=0.0)
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            protocol = st.selectbox("Asset", options=hp_df.columns[1:])
        with col2:
            option_maturity = st.selectbox("Option Maturity", options=['1w', '1M', '3M', '6M', '12M', '24M', '36M'], index=1)  # Default to '12M'
        with col3:
            strike_opt = st.number_input("Strike", value=100.0, min_value=0.0)
    
    temp_df = temp_df[protocol].first_valid_index()
    st.write(f"Data available from : {temp_df}")


    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.date_input("Hedge start date", value=datetime(2025,1,1))
    with col2:
        end_date = st.date_input("Hedge end date", value=datetime(2025,4,1))

    reward_amount=1
    if st.button("Run Strategy"):
        with st.spinner("Running hedging strategy simulation..."):
            
            if product == "Put":

                put_strike_multiplier = strike_opt/100
                option_maturity = convert_maturity_to_days(option_maturity)

                daily_rewards = 1
                
                hedging_start_date = start_date

                spot_end_notional,hedged_end_notional,final_pnl,final_pnl_perc,put_options_price,df_hedged_vs_actual_rewards = put_hedge(put_strike_multiplier,daily_rewards,protocol,option_maturity,hedging_start_date,hedging_end_date=end_date,sigma =vol,IR=Risk_free_rate)
                st.subheader("Strategy Results")

                st.write('### Strategy : Buy Put')
                #st.write(f"Daily {daily_rewards} {protocol} rewards sold each week at spot vs each {option_maturity} days with put options strike {int(put_strike_multiplier*100)}")
                st.write(f"**End USDT notional with daily offramps:** {round(spot_end_notional,2):,} $")
                st.write(f"**End USDT notional with puts and offramps each {option_maturity} days:** {round(hedged_end_notional):,} $")
                st.write(f"**End USDT notional spent buying put options:** {round(put_options_price):,} $")

                st.write(f"**Final PnL:** {round(final_pnl,2):,} $")
                st.write(f"**Final PnL in %:** {round(final_pnl_perc,2):,}%")
                st.dataframe(df_hedged_vs_actual_rewards)
            
            if product == "Call":

                call_strike_multiplier = strike_opt/100
                option_maturity = convert_maturity_to_days(option_maturity)

                daily_rewards = reward_amount
                
                hedging_start_date = start_date

                spot_end_notional,hedged_end_notional,options_price_gained,final_pnl,final_pnl_perc, call_prices = call_hedge(call_strike_multiplier,daily_rewards,protocol,option_maturity,hedging_start_date,sigma =0.85)

                st.subheader("Strategy Results")

                st.write('### Strategy : Buy Call')
                st.write(f"Daily {daily_rewards} {protocol} rewards sold each week at spot vs each {option_maturity} days with call options strike {int(call_strike_multiplier*100)}")
                st.write(f"**End USDT notional with weekly offramps:** {round(spot_end_notional,2):,} $")
                st.write(f"**End USDT notional with calls and offramps each {option_maturity} days:** {round(hedged_end_notional):,} $")
                st.write(f"**End USDT notional gained selling call options:** {round(options_price_gained):,} $")

                st.write(f"**Final PnL:** {round(final_pnl,2):,} $")
                st.write(f"**Final PnL in %:** {round(final_pnl_perc,2):,}%")
            
            if product == "Collar":
                

                put_strike_multiplier = strike_opt/100
                call_strike_multiplier = strike2_opt/100
                option_maturity = convert_maturity_to_days(option_maturity)

                daily_rewards = reward_amount
                
                hedging_start_date = start_date

                spot_end_notional,hedged_end_notional,final_pnl,final_pnl_perc,call_prices,put_options_price,call_options_price,options_price_paid,df_hedgedvsnon = collar(call_strike_multiplier,put_strike_multiplier,daily_rewards,protocol,option_maturity,hedging_start_date,sigma = 0.6)


                st.subheader("Strategy Results")
                

                st.write('### Strategy : Combined Put and Call Strategy')
                #st.write(f"Daily {daily_rewards} {protocol} rewards sold each week at spot vs each {option_maturity} days hedged with Collar")

                st.write(f"**Notional with weekly offramps:** {round(spot_end_notional,2):,} $")
                st.write(f"**Notional with Collar hedged offramps each {option_maturity} days:** {round(hedged_end_notional):,} $")

                st.write(f"**Notional paid to buy put options:** {round(put_options_price):,} $")
                st.write(f"**Notional gained selling call options:** {round(call_options_price):,} $")
                st.write(f"**Options PnL:** {round(options_price_paid):,} $")

                st.write(f"**Final PnL in $:** {round(final_pnl,2):,} $")
                st.write(f"**Final PnL in %:** {round(final_pnl_perc,2):,}%")
                st.dataframe(df_hedgedvsnon)
                st.image("collar.png")

    
                



def VanillaOptionsPayoffSimulator():
    st.title("Vanilla Options Payoff Simulator")
    st.markdown("""
    The Vanilla Options Payoff Simulator allows you to create and visualize different option strategies.
    You can add different option legs, specify their strike prices, maturities, and whether you are buying or selling them.
    
    This tool will calculate the premiums as a percentage of the current spot price and plot the combined payoff diagram for all the options you've added.
    This can help you better understand the potential outcomes of your strategies at different spot prices at maturity.
    """)

    if 'options_data' not in st.session_state:
        st.session_state.options_data = pd.DataFrame(columns=['Type', 'Position', 'Strike Price', 'Premium', 'Volatility', 'Maturity', 'Risk-Free Rate'])
    
    st.subheader("Market Data")
    col1, col2 = st.columns(2)
    with col1:
        volatility = st.number_input("Volatility (σ)", value=0.6, min_value=0.0, format="%.2f")
    with col2:
        risk_free_rate = st.number_input("Risk-Free Rate (r)", value=0.0, min_value=-0.000001, format="%.2f")
    
    st.subheader("Add New Option Leg")
    with st.form(key='option_form'):
        cols = st.columns(4)
        option_type = cols[0].selectbox("Option Type", options=['Call', 'Put'])
        position = cols[1].selectbox("Position", options=['Buy', 'Sell'])
        strike_price = cols[2].number_input("Strike Price (%)", value=100, min_value=0)
        maturity = cols[3].selectbox("Maturity", options=['1w', '1M', '3M', '6M', '12M', '24M', '36M'])

        # Convert maturity to years
        maturity_in_years = convert_maturity_to_years(maturity)

        premium_percentage = black_scholes_price(option_type, 100, strike_price, maturity_in_years, risk_free_rate, volatility)
        if position == "Sell":
            premium_percentage = -premium_percentage
        
        if st.form_submit_button(label="Add Option"):
            new_option = {
                'Type': option_type,
                'Position': position,
                'Strike Price': strike_price,
                'Premium': premium_percentage,  # Premium as a percentage
                'Volatility': volatility,
                'Maturity': maturity,  # Store the original maturity string
                'Risk-Free Rate': risk_free_rate
            }
            st.session_state.options_data = pd.concat([st.session_state.options_data, pd.DataFrame([new_option])], ignore_index=True)

    # Display current option legs and the sum of premiums
    st.subheader("Current Option Legs")
    if not st.session_state.options_data.empty:
        # Display column headers
        cols = st.columns([1, 1, 1, 1, 1, 1, 1])
        cols[0].write("Position")
        cols[1].write("Type")
        cols[2].write("Strike Price")
        cols[3].write("Maturity")
        cols[4].write("Volatility")
        cols[5].write("Premium (%)")  # Updated to indicate percentage
        cols[6].write("Remove")

        for idx, option in st.session_state.options_data.iterrows():
            cols = st.columns([1, 1, 1, 1, 1, 1, 1])
            cols[0].write(option['Position'])
            cols[1].write(option['Type'])
            cols[2].write(option['Strike Price'])
            cols[3].write(option['Maturity'])
            cols[4].write(f"{option['Volatility'] * 100:.2f}%")
            cols[5].write(f"{option['Premium']:.2f} %")  # Display premium as a percentage
            remove_button = cols[6].button("❌", key=f"remove_{idx}")

            # Handle removal of option leg
            if remove_button:
                st.session_state.options_data = st.session_state.options_data.drop(idx).reset_index(drop=True)

    # Display sum of premiums
    total_premium = st.session_state.options_data['Premium'].sum()
    st.write(f"**Total Premium:** {total_premium:.2f} %")

    # Improved plotting section
    st.subheader("Options Payoff Diagram")
    if st.session_state.options_data.empty:
        st.warning("No options added yet. Please add options to see the payoff diagram.")
    else:
        plot_payoffs(st.session_state.options_data)

    # Handle reset action
    if st.button("Reset All Options"):
        st.session_state.options_data = pd.DataFrame(columns=['Type', 'Position', 'Strike Price', 'Premium', 'Volatility', 'Maturity', 'Risk-Free Rate'])



st.sidebar.title("Kiln Hedging Strats")
page = st.sidebar.radio("Go to", ["Option Payoffs", "Backtesting"])

# Display the selected page
if page == "Option Payoffs":
    VanillaOptionsPayoffSimulator()
elif page == "Backtesting":
    Backtesting()
