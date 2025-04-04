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

def save_simulation_to_csv(strategy_name, protocol, simulation_data, parameters=None):
    """
    Save detailed simulation data to CSV files for verification and analysis
    
    Args:
        strategy_name: Name of the strategy (Put/Collar)
        protocol: Asset (ETH/SOL)
        simulation_data: Dict containing all logs and transaction data
        parameters: Dict containing simulation parameters
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "simulation_results"
        
        # Create directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Base filename for all related CSVs
        base_filename = f"{output_dir}/{timestamp}_{strategy_name}_{protocol}"
        saved_files = []
        
        # Save strategy parameters
        if parameters:
            try:
                params_df = pd.DataFrame([parameters])
                params_file = f"{base_filename}_parameters.csv"
                params_df.to_csv(params_file, index=False)
                saved_files.append(params_file)
            except Exception as e:
                st.warning(f"Failed to save parameters: {str(e)}")
        
        # Save transaction logs (options purchases/exercises)
        if 'transaction_log' in simulation_data and not simulation_data['transaction_log'].empty:
            try:
                tx_file = f"{base_filename}_options_transactions.csv"
                simulation_data['transaction_log'].to_csv(tx_file, index=False)
                saved_files.append(tx_file)
            except Exception as e:
                st.warning(f"Failed to save options transactions: {str(e)}")
        
        # Save non-hedged strategy operations (spot sales)
        if 'spot_strategy_log' in simulation_data and not simulation_data['spot_strategy_log'].empty:
            try:
                spot_file = f"{base_filename}_spot_strategy.csv"
                simulation_data['spot_strategy_log'].to_csv(spot_file, index=False)
                saved_files.append(spot_file)
            except Exception as e:
                st.warning(f"Failed to save spot strategy logs: {str(e)}")
        
        # Save hedged strategy operations
        if 'hedged_strategy_log' in simulation_data and not simulation_data['hedged_strategy_log'].empty:
            try:
                hedged_file = f"{base_filename}_hedged_strategy.csv"
                simulation_data['hedged_strategy_log'].to_csv(hedged_file, index=False)
                saved_files.append(hedged_file)
            except Exception as e:
                st.warning(f"Failed to save hedged strategy logs: {str(e)}")
        
        # If parameters includes a dataframe, save it
        if 'parameters' in simulation_data:
            try:
                params_dataframe_file = f"{base_filename}_parameters_df.csv"
                simulation_data['parameters'].to_csv(params_dataframe_file, index=False)
                saved_files.append(params_dataframe_file)
            except Exception as e:
                st.warning(f"Failed to save parameters dataframe: {str(e)}")
        
        # If monthly_rewards is included, save it
        if 'monthly_rewards' in simulation_data:
            try:
                monthly_file = f"{base_filename}_monthly_rewards.csv"
                simulation_data['monthly_rewards'].to_csv(monthly_file, index=False)
                saved_files.append(monthly_file)
            except Exception as e:
                st.warning(f"Failed to save monthly rewards: {str(e)}")
        
        if not saved_files:
            st.warning("No simulation data was saved. Check logs for errors.")
        
        return base_filename
    
    except Exception as e:
        st.error(f"Error saving simulation data: {str(e)}")
        return None



def put_hedge(put_strike_multiplier, daily_rewards, protocol, option_maturity, hedging_start_date, hedging_end_date=None, percent_to_hedge=0.7, IR=0.05, sigma=0.6, broker_spread=0.10):
    # Load price data from CSV files instead of HP.xlsx
    try:
        if protocol.lower() == 'eth':
            price_data = pd.read_csv("prices_eth.csv")
            price_data['date'] = pd.to_datetime(price_data['date'])
            price_data.set_index('date', inplace=True)
        elif protocol.lower() == 'sol':
            price_data = pd.read_csv("prices_sol.csv")
            price_data['date'] = pd.to_datetime(price_data['date'])
            price_data.set_index('date', inplace=True)
        
        # Convert dates to datetime
        hedging_start_date_dt = pd.to_datetime(hedging_start_date)
        
        # Use end date if provided, otherwise use all available data
        if hedging_end_date is not None:
            hedging_end_date_dt = pd.to_datetime(hedging_end_date)
            data_to_hedge = price_data['end_of_day'][
                (price_data.index >= hedging_start_date_dt) & 
                (price_data.index <= hedging_end_date_dt)
            ]
        else:
            data_to_hedge = price_data['end_of_day'][price_data.index >= hedging_start_date_dt]
        
        # Load volatility from the same CSV
        vol_data = price_data.copy()
        # Fill missing volatility values with the default
        vol_data['vol'] = vol_data['vol'].fillna(sigma)
        
        # Get initial volatility for start date (for display only)
        closest_date_idx = vol_data.index.get_indexer([hedging_start_date_dt], method='nearest')[0]
        closest_date = vol_data.index[closest_date_idx]
        
        if not pd.isna(vol_data.loc[closest_date, 'vol']):
            initial_sigma = vol_data.loc[closest_date, 'vol']
            st.write(f"Starting volatility from {closest_date.date()}: {initial_sigma:.2f}")
        else:
            initial_sigma = sigma
            st.warning(f"No volatility data found for {closest_date.date()}. Using default value: {initial_sigma}")
        
    except Exception as e:
        st.error(f"Error loading CSV data: {e}")
        # Fallback to HP.xlsx
        data = pd.read_excel("HP.xlsx", index_col=0, parse_dates=True)
        if hedging_end_date is not None:
            data_to_hedge = data[protocol][
                (data.index >= pd.to_datetime(hedging_start_date)) & 
                (data.index <= pd.to_datetime(hedging_end_date))
            ]
        else:
            data_to_hedge = data[protocol][data.index >= pd.to_datetime(hedging_start_date)]
    
    data_rewards = pd.read_excel("v2_revenue.xlsx", index_col=0, parse_dates=True)
    
    data_to_hedge = data_to_hedge.dropna()
    hedging_start_date = data_to_hedge.first_valid_index()

    # Get previous month rewards to determine hedge amount
    start_window = pd.to_datetime(hedging_start_date) - pd.Timedelta(days=30)
    previous_month_rewards = data_rewards[protocol][
        (data_rewards.index > start_window) & 
        (data_rewards.index < pd.to_datetime(hedging_start_date))
    ]
    
    # Calculate average daily rewards from previous month
    prev_month_avg_daily = previous_month_rewards.mean()
    
    # Calculate hedged amount (fixed based on previous month)
    notional_tohedge_inkind = prev_month_avg_daily * percent_to_hedge
    
    st.write(f"Previous month daily average rewards: {prev_month_avg_daily:.4f}")
    st.write(f"Hedged amount (fixed): {notional_tohedge_inkind:.4f} ({percent_to_hedge*100:.0f}%)")
    
    # Add explanation of the hedging strategy
    st.info("""
    ### Hedging Strategy
    1. For each period, we first accumulate rewards until we reach the total hedged amount.
    2. Only after reaching the hedging threshold, excess rewards are sold weekly on Fridays.
    3. The hedged rewards are exercised or sold at the end of the option period.
    """)
    
    mask = data_rewards.index >= pd.to_datetime(hedging_start_date)

    data_rewards_from_start = data_rewards.loc[mask, protocol]
    data_rewards_from_start = data_rewards_from_start.dropna()

    weekly_offramp_rewards = []
    hedged_offramp_rewards = []
    nonhedged_offramp_rewards = []
    actual_rewards = []

    monthly_actual_rewards = []
    monthly_hedged_rewards = []

    weekly_offramp_notional = []
    hedged_offramp_notional = []
    nonhedged_offramp_notional = []  # Track non-hedged portion separately
    weekly_nonhedged_rewards = []  # Weekly tracking for excess rewards

    # Track total rewards accumulated in the current option period
    period_accumulated_rewards = 0
    # Total hedged amount needed for the current option period
    period_hedge_target = option_maturity * notional_tohedge_inkind
    
    st.write(f"Hedging target per period: {period_hedge_target:.4f}")

    put_prices = []
    
    days_until_maturity = option_maturity
    days_until_week_end = 7

    # Initialize transaction logs
    transaction_log = []
    spot_strategy_log = []
    hedged_strategy_log = []
    
    # Storage for volatilities used at each option purchase
    volatilities_used = []
    
    # Set initial values for the first option
    spot = data_to_hedge[0]
    strike = spot * np.exp(1/12 * IR)
    put_strike = strike * put_strike_multiplier
    
    # Get initial volatility
    current_date = data_to_hedge.index[0]
    try:
        closest_date_idx = vol_data.index.get_indexer([current_date], method='nearest')[0]
        closest_date = vol_data.index[closest_date_idx]
        current_sigma = vol_data.loc[closest_date, 'vol'] if not pd.isna(vol_data.loc[closest_date, 'vol']) else sigma
        volatilities_used.append(current_sigma)
    except:
        current_sigma = sigma
        volatilities_used.append(current_sigma)
    
    # Initial option purchase - FIXED COST CALCULATION
    put_price = black_scholes_put(spot, put_strike, option_maturity/365, IR, current_sigma)
    # We need to multiply by option_maturity because notional_tohedge_inkind is a DAILY amount
    option_cost = put_price * (option_maturity * notional_tohedge_inkind)
    put_prices.append(option_cost)
    
    # Improve transaction log for initial option purchase
    transaction_log.append({
        'date': data_to_hedge.index[0].strftime('%Y-%m-%d'),
        'type': 'Option Purchase',
        'details': f"Bought put option at strike {put_strike:.2f} ({put_strike_multiplier*100:.0f}% of {spot:.2f})",
        'strike': put_strike,
        'spot_price': spot,
        'rewards_amount': option_maturity * notional_tohedge_inkind,
        'premium_per_unit': put_price,
        'cost': option_cost,
        'vol': current_sigma,
        'maturity': option_maturity,
        'expiry_date': (data_to_hedge.index[0] + pd.Timedelta(days=option_maturity)).strftime('%Y-%m-%d')
    })

    # Add a flag to track if final liquidation has happened
    final_liquidation_done = False
    
    for i in range(len(data_to_hedge)):
        spot = data_to_hedge[i]
        current_date_dt = data_to_hedge.index[i]
        current_date = current_date_dt.strftime('%Y-%m-%d')
        today_reward = data_rewards_from_start[i]
        actual_rewards.append(today_reward)
        weekly_offramp_rewards.append(today_reward)
        
        # The hedged amount remains fixed based on previous month calculation
        hedged_offramp_rewards.append(notional_tohedge_inkind)
        
        # Increase total accumulated rewards for the period
        period_accumulated_rewards += today_reward
        
        # Determine if we've accumulated enough rewards to cover the hedged amount
        # We only start selling excess rewards when we've accumulated enough to cover the hedged amount
        excess_reward_available = period_accumulated_rewards > period_hedge_target
        
        # Calculate non-hedged amount only if we have excess rewards
        if excess_reward_available:
            # Today's non-hedged amount is only the portion exceeding what we need for hedging
            # This is the amount that will be sold on the next Friday
            non_hedged_amount = max(0, period_accumulated_rewards - period_hedge_target)
            st.write(f"{current_date}: Accumulated enough rewards ({period_accumulated_rewards:.4f}). Excess: {non_hedged_amount:.4f}")
            # Reset the accumulated amount to exactly the target (since we're considering excess as non-hedged)
            period_accumulated_rewards = period_hedge_target
            # Add to weekly non-hedged rewards for potential Friday selling
            weekly_nonhedged_rewards.append(non_hedged_amount)
        else:
            # If we don't have excess rewards yet, nothing is non-hedged
            # We need to accumulate more rewards to reach the hedging target first
            non_hedged_amount = 0
            if i % 7 == 0:  # Log status weekly to keep track
                st.write(f"{current_date}: Still accumulating rewards ({period_accumulated_rewards:.4f}/{period_hedge_target:.4f})")
        
        nonhedged_offramp_rewards.append(non_hedged_amount)
        
        # Check if this is the last day of the simulation
        is_last_day = (i == len(data_to_hedge) - 1)
        
        # Check if today is Friday (weekday=4) or last day
        is_friday = current_date_dt.weekday() == 4
        
        # Weekly selling for standard strategy - only on Fridays
        if is_friday or is_last_day:
            accumulated_rewards = sum(weekly_offramp_rewards)
            week_value = accumulated_rewards * (spot - broker_spread)
            
            # For weekly spot sales:
            spot_strategy_log.append({
                'date': current_date,
                'action': 'Weekly Spot Sale' if not is_last_day else 'Final Liquidation',
                'rewards_amount': accumulated_rewards,
                'sale_price': spot - broker_spread,
                'spot_price': spot,
                'broker_spread': broker_spread, 
                'value': week_value,
                'note': f"Friday sale with broker spread: {broker_spread:.2f}" + (" (End of simulation)" if is_last_day else "")
            })
            
            weekly_offramp_notional.append(week_value)
            weekly_offramp_rewards = [] if not is_last_day else []
            
            # Only sell non-hedged rewards if we have accumulated enough to cover the hedged amount
            # and there are excess rewards to sell
            if sum(weekly_nonhedged_rewards) > 0:
                weekly_nonhedged_total = sum(weekly_nonhedged_rewards)
                weekly_nonhedged_value = weekly_nonhedged_total * (spot - broker_spread)
                
                hedged_strategy_log.append({
                    'date': current_date,
                    'action': 'Weekly Non-Hedged Sale',
                    'rewards_amount': weekly_nonhedged_total,
                    'sale_price': spot - broker_spread,
                    'spot_price': spot,
                    'value': weekly_nonhedged_value,
                    'strike': None,
                    'protection_value': 0,
                    'note': f"Weekly sale of excess rewards after reaching hedging target ({period_accumulated_rewards:.4f}/{period_hedge_target:.4f})"
                })
                
                nonhedged_offramp_notional.append(weekly_nonhedged_value)
                weekly_nonhedged_rewards = [] if not is_last_day else []
        
        # Monthly option handling
        if days_until_maturity <= 0:
            # Process expired option and accumulated rewards
            accumulated_rewards = sum(hedged_offramp_rewards)
            actual_accumulated_rewards = sum(actual_rewards)
            
            # Record the monthly data
            monthly_actual_rewards.append(actual_accumulated_rewards)
            monthly_hedged_rewards.append(accumulated_rewards)
            
            # Calculate hedged amount payoff
            hedged_amount = min(accumulated_rewards, actual_accumulated_rewards)
            if spot <= put_strike:
                hedged_value = hedged_amount * put_strike
            else:
                hedged_value = hedged_amount * spot
            
            # Add hedged value to the hedged notional
            hedged_offramp_notional.append(hedged_value)
            
            # Log option exercise outcome
            option_exercised = spot <= put_strike
            protection_value = 0
            
            if option_exercised:
                protection_value = hedged_amount * (put_strike - spot)
                transaction_log.append({
                    'date': current_date,
                    'type': 'Put Option Exercised',
                    'details': f"Spot ({spot:.2f}) below strike ({put_strike:.2f}), exercised put option",
                    'strike': put_strike,
                    'spot_price': spot,
                    'rewards_amount': hedged_amount,
                    'protection_value': protection_value,
                    'notional_protected': hedged_amount * put_strike,
                    'savings_vs_spot': protection_value,
                    'original_cost': option_cost
                })
                
                hedged_strategy_log.append({
                    'date': current_date,
                    'action': 'Option Exercise',
                    'rewards_amount': hedged_amount,
                    'sale_price': put_strike,
                    'spot_price': spot,
                    'value': hedged_amount * put_strike,
                    'strike': put_strike,
                    'protection_value': protection_value,
                    'note': f"Put protection: {protection_value:.2f}"
                })
            else:
                transaction_log.append({
                    'date': current_date,
                    'type': 'Put Option Expired',
                    'details': f"Spot ({spot:.2f}) above strike ({put_strike:.2f}), option not exercised",
                    'strike': put_strike,
                    'spot_price': spot,
                    'rewards_amount': hedged_amount,
                    'protection_value': 0,
                    'notional_at_spot': hedged_amount * spot,
                    'original_cost': option_cost
                })
                
                hedged_strategy_log.append({
                    'date': current_date,
                    'action': 'Option Expired',
                    'rewards_amount': hedged_amount,
                    'sale_price': spot,
                    'spot_price': spot,
                    'strike': put_strike,
                    'value': hedged_amount * spot,
                    'protection_value': 0,
                    'note': "Option not exercised (spot > strike)"
                })
            
            # Reset tracking arrays for the next period
            hedged_offramp_rewards = []
            nonhedged_offramp_rewards = []
            actual_rewards = []
            # Reset accumulated rewards for the next period
            period_accumulated_rewards = 0
            
            # Check if there's enough time left for another option period
            days_remaining = len(data_to_hedge) - (i + 1)
            
            if days_remaining >= option_maturity:
                # Recalculate hedge amount for next month based on previous month
                start_window = data_to_hedge.index[i] - pd.Timedelta(days=30)
                end_window = data_to_hedge.index[i]
                
                previous_month_rewards = data_rewards[protocol][
                    (data_rewards.index > start_window) & 
                    (data_rewards.index <= end_window)
                ]
                
                # Calculate new average daily rewards from previous month
                if not previous_month_rewards.empty:
                    prev_month_avg_daily = previous_month_rewards.mean()
                    # Update the notional amount to hedge for the next period
                    notional_tohedge_inkind = prev_month_avg_daily * percent_to_hedge
                    
                    st.write(f"New previous month daily average: {prev_month_avg_daily:.4f}")
                    st.write(f"New hedged amount: {notional_tohedge_inkind:.4f} ({percent_to_hedge*100:.0f}%)")
                else:
                    st.warning(f"No rewards data found for previous month. Using existing hedge amount.")
                
                # Update the period hedge target with the new daily hedge amount
                period_hedge_target = option_maturity * notional_tohedge_inkind
                
                # Calculate new option values with the updated hedging amount
                strike = spot * np.exp(1/12 * IR)
                put_strike = strike * put_strike_multiplier
                
                # Get volatility for new option
                try:
                    closest_date_idx = vol_data.index.get_indexer([data_to_hedge.index[i]], method='nearest')[0]
                    closest_date = vol_data.index[closest_date_idx]
                    current_sigma = vol_data.loc[closest_date, 'vol'] if not pd.isna(vol_data.loc[closest_date, 'vol']) else sigma
                    volatilities_used.append(current_sigma)
                except:
                    current_sigma = sigma
                    volatilities_used.append(current_sigma)
                
                # Calculate option premium for next month - FIXED COST CALCULATION
                put_price = black_scholes_put(spot, put_strike, option_maturity/365, IR, current_sigma)
                option_cost = put_price * (option_maturity * notional_tohedge_inkind)
                put_prices.append(option_cost)
                
                # For the new option purchase after expiry:
                transaction_log.append({
                    'date': current_date,
                    'type': 'Option Purchase',
                    'details': f"Bought put option at strike {put_strike:.2f} ({put_strike_multiplier*100:.0f}% of {spot:.2f})",
                    'strike': put_strike,
                    'spot_price': spot,
                    'rewards_amount': option_maturity * notional_tohedge_inkind,
                    'premium_per_unit': put_price,
                    'cost': option_cost,
                    'vol': current_sigma,
                    'maturity': option_maturity,
                    'expiry_date': (data_to_hedge.index[i] + pd.Timedelta(days=option_maturity)).strftime('%Y-%m-%d')
                })
                
                # CRITICAL FIX: Reset days_until_maturity for the new option period
                days_until_maturity = option_maturity
            else:
                # Not enough time left for a full option period
                st.info(f"Skipping option purchase - only {days_remaining} days left until end date (option maturity: {option_maturity} days)")
                # Mark all remaining rewards as unhedged by setting notional_tohedge_inkind to 0
                notional_tohedge_inkind = 0
                # Reset days_until_maturity to ensure we don't try to buy again
                days_until_maturity = days_remaining + 1
                # Reset period hedge target since we're not hedging anymore
                period_hedge_target = 0
                
                # Log that we're not purchasing a new option
                transaction_log.append({
                    'date': current_date,
                    'type': 'Skip Option Purchase',
                    'details': f"Insufficient time remaining until end date ({days_remaining} days left, need {option_maturity})",
                    'rewards_amount': 0,
                    'cost': 0
                })
        
        # Check if this is the last day of the simulation
        is_last_day = (i == len(data_to_hedge) - 1)
        
        if days_until_week_end == 0 or is_last_day:
            accumulated_rewards = sum(weekly_offramp_rewards)
            week_value = accumulated_rewards * (spot - broker_spread)
            
            # For weekly spot sales:
            spot_strategy_log.append({
                'date': current_date,
                'action': 'Weekly Spot Sale' if not is_last_day else 'Final Liquidation',
                'rewards_amount': accumulated_rewards,
                'sale_price': spot - broker_spread,
                'spot_price': spot,
                'broker_spread': broker_spread, 
                'value': week_value,
                'note': f"Friday sale with broker spread: {broker_spread:.2f}" + (" (End of simulation)" if is_last_day else "")
            })
            
            weekly_offramp_notional.append(week_value)
            weekly_offramp_rewards = [] if not is_last_day else []
        
        # Always decrement the maturity counter
        days_until_maturity -= 1

    # After the loop, check if we need to handle final option exercise/expiry
    # This is needed if the simulation ends before an option matures
    if days_until_maturity > 0 and sum(hedged_offramp_rewards) > 0:
        # We have an active option with accumulated rewards
        final_spot = data_to_hedge[-1]
        final_date = data_to_hedge.index[-1].strftime('%Y-%m-%d')
        
        accumulated_rewards = sum(hedged_offramp_rewards)
        actual_accumulated = sum(actual_rewards)
        
        # For the final exercise, consider if we've accumulated enough rewards
        # to cover the hedged amount for the period
        final_hedge_amount = min(accumulated_rewards, actual_accumulated)
        final_hedge_needed = min(final_hedge_amount, period_hedge_target)
        
        # Calculate hedged amount payoff for final option
        if final_spot <= put_strike:
            hedged_value = final_hedge_needed * put_strike
            protection_value = final_hedge_needed * (put_strike - final_spot)
            
            # Log the final option exercise
            transaction_log.append({
                'date': final_date,
                'type': 'Final Option Exercise',
                'details': f"Final spot ({final_spot:.2f}) below strike ({put_strike:.2f}), exercised put option",
                'strike': put_strike,
                'spot_price': final_spot,
                'rewards_amount': final_hedge_needed,
                'protection_value': protection_value,
                'notional_protected': hedged_value,
                'original_cost': option_cost
            })
            
            hedged_strategy_log.append({
                'date': final_date,
                'action': 'Final Option Exercise',
                'rewards_amount': final_hedge_needed,
                'sale_price': put_strike,
                'spot_price': final_spot,
                'value': hedged_value,
                'strike': put_strike,
                'protection_value': protection_value,
                'note': f"Put protection at end of simulation: {protection_value:.2f}"
            })
        else:
            hedged_value = final_hedge_needed * final_spot
            
            # Log the final option expiry
            transaction_log.append({
                'date': final_date,
                'type': 'Final Option Expiry',
                'details': f"Final spot ({final_spot:.2f}) above strike ({put_strike:.2f}), option not exercised",
                'strike': put_strike,
                'spot_price': final_spot,
                'rewards_amount': final_hedge_needed,
                'protection_value': 0,
                'notional_at_spot': hedged_value,
                'original_cost': option_cost
            })
            
            hedged_strategy_log.append({
                'date': final_date,
                'action': 'Final Option Expiry',
                'rewards_amount': final_hedge_needed,
                'sale_price': final_spot,
                'spot_price': final_spot,
                'value': hedged_value,
                'note': "Option not exercised at end of simulation"
            })
        
        # Add the final hedged values
        hedged_offramp_notional.append(hedged_value)
        
        # Calculate if there are any excess rewards beyond what was needed for hedging
        # This includes any rewards that haven't been sold weekly already
        excess_rewards = actual_accumulated - final_hedge_needed + sum(weekly_nonhedged_rewards)
        
        # Handle any remaining non-hedged rewards that haven't been sold weekly
        if excess_rewards > 0:
            remaining_value = excess_rewards * (final_spot - broker_spread)
            
            hedged_strategy_log.append({
                'date': final_date,
                'action': 'Final Non-Hedged Sale',
                'rewards_amount': excess_rewards,
                'sale_price': final_spot - broker_spread,
                'spot_price': final_spot,
                'value': remaining_value,
                'note': f"Final liquidation of excess rewards with broker spread: {broker_spread:.2f}"
            })
            
            nonhedged_offramp_notional.append(remaining_value)
    
    # Calculate final notionals
    spot_end_notional = sum(weekly_offramp_notional)
    hedged_end_notional = sum(hedged_offramp_notional)
    nonhedged_notional = sum(nonhedged_offramp_notional)
    
    # Calculate spot end notional (non-hedged strategy benchmark)
    non_hedged_strategy_value = spot_end_notional
    
    # Hedged strategy: hedged portion (with put protection) + non-hedged portion - option costs
    hedged_strategy_value = hedged_end_notional + nonhedged_notional - sum(put_prices)
    
    # PnL is the difference between the two strategies
    final_pnl = hedged_strategy_value - non_hedged_strategy_value
    
    # Calculate PnL as percentage of the non-hedged strategy value
    if non_hedged_strategy_value > 0:
        final_pnl_perc = (final_pnl / non_hedged_strategy_value) * 100
    else:
        final_pnl_perc = 0
    
    df_hedged_vs_actual_rewards = pd.DataFrame({
        "Hedged rewards per month": monthly_hedged_rewards,
        "Actual rewards per month": monthly_actual_rewards
    })
    
    # Add volatility information to the results
    avg_volatility = sum(volatilities_used) / max(1, len(volatilities_used))
    min_volatility = min(volatilities_used) if volatilities_used else sigma
    max_volatility = max(volatilities_used) if volatilities_used else sigma
    
    # Convert logs to DataFrames for display
    spot_log_df = pd.DataFrame(spot_strategy_log)
    hedged_log_df = pd.DataFrame(hedged_strategy_log)
    transaction_log_df = pd.DataFrame(transaction_log)
    
    # Return with properly formatted hybrid data
    return spot_end_notional, hedged_end_notional, final_pnl, final_pnl_perc, sum(put_prices), df_hedged_vs_actual_rewards, nonhedged_notional, {
        'avg_vol': avg_volatility,
        'min_vol': min_volatility,
        'max_vol': max_volatility,
        'transaction_log': transaction_log_df,
        'spot_strategy_log': spot_log_df,
        'hedged_strategy_log': hedged_log_df
    }




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
        current_date_dt = data_to_hedge.index[i]
        current_date = current_date_dt.strftime('%Y-%m-%d')
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
        maturity = col2.selectbox("Maturity", options=['1w', '1M', '3M', '6M', '12M', '24M', '36M'])

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
                "Maturity": maturity,
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
    
    # Add a nice header with description
    st.markdown("""
    <div style="background-color:#f0f2f6;padding:15px;border-radius:10px;margin-bottom:20px">
        <h3 style="color:#262730">Options Hedging Strategy Simulator</h3>
        <p>Evaluate different options hedging strategies for cryptocurrency rewards using historical price data.</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data from files
    @st.cache_data
    def load_data():
        github_url = 'HP.xlsx'
        try:
            hp_df = pd.read_excel(github_url)
            temp_df = pd.read_excel(github_url, index_col=0)
            return hp_df, temp_df
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            return None, None
            
    hp_df, temp_df = load_data()
        
    # Add a refresh button with nicer styling
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔄 Refresh Data"):
            with st.spinner("Updating market data..."):
                update_data()
        hp_df, temp_df = load_data()
        st.success("Data refreshed successfully!")

    # Create tabs for different sections
    tab1, tab2 = st.tabs(["Strategy Configuration", "Advanced Settings"])
    
    with tab1:
        st.subheader("Strategy Parameters")
        
        # Choose product type with better UI
        product = st.selectbox(
            "Product Type", 
            options=['Put', 'Collar'],
            format_func=lambda x: {"Put": "Put Option Strategy", "Collar": "Collar Strategy (Put + Call)"}[x]
        )
        
        # Use more organized layout for parameters
    if product == "Collar":
        col1, col2 = st.columns(2)
        with col1:
            protocol = st.selectbox("Asset", options=hp_df.columns[1:])
            option_maturity = st.selectbox(
                    "Option Maturity", 
                    options=['1w', '1M', '3M', '6M', '12M', '24M', '36M'], 
                    index=1,
                    help="Duration of each option contract"
                )
        with col2:
                strike_opt = st.slider("Lower Strike (% of spot)", value=80.0, min_value=50.0, max_value=99.0)
                strike2_opt = st.slider("Upper Strike (% of spot)", value=120.0, min_value=101.0, max_value=200.0)
    else:
        col1, col2 = st.columns(2)
        with col1:
            protocol = st.selectbox("Asset", options=hp_df.columns[1:])
            option_maturity = st.selectbox(
                    "Option Maturity", 
                    options=['1w', '1M', '3M', '6M', '12M', '24M', '36M'], 
                    index=1,
                    help="Duration of each option contract"
                )
        with col2:
                strike_opt = st.slider(
                    "Strike Price (% of spot)", 
                    value=100.0, 
                    min_value=50.0, 
                    max_value=150.0,
                    help="Strike price as a percentage of the spot price"
                )
        
        # Display available data range
        temp_df = temp_df[protocol].first_valid_index()
        st.info(f"📊 Historical data available from: {temp_df}")

        # Add start and end date selectors
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Hedge start date", 
                value=datetime(2025,1,1),
                help="When to begin the hedging strategy"
            )
        with col2:
            end_date = st.date_input(
                "Hedge end date", 
                value=datetime(2025,6,30),
                help="When to end the hedging strategy"
            )
        
        # Better layout for additional parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        percent_to_hedge = st.slider(
                "Percent to Hedge (% of rewards)",
                min_value=0,
                max_value=100,
                value=70,
                step=5,
                help="Percentage of rewards to hedge with options"
            ) / 100  # Convert from percentage (0-100) to decimal (0-1)
    with col2:
        interest_rate = st.number_input(
                "Risk-Free Rate", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.05, 
                step=0.001,
                format="%.3f", 
                help="Annual risk-free interest rate (e.g., 0.05 for 5%)"
            )

    with tab2:
        st.subheader("Advanced Settings")
        # Add more advanced parameters here if needed
        broker_spread = st.slider(
            "Broker Spread", 
            min_value=0.0, 
            max_value=0.5, 
            value=0.1, 
            step=0.01,
            format="%.1f%%",
            help="Trading cost when selling rewards (as % of value)"
        )
        
        # You could add other advanced parameters here

    # Run button with better styling
    if st.button("🚀 Run Simulation", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(0, 101, 10):
            progress_bar.progress(i)
            status_text.text(f"Running simulation... {i}%")
            time.sleep(0.1)  # Simulate computation
            
        with st.spinner("Finalizing results..."):
            if product == "Put":
                put_strike_multiplier = strike_opt/100
                option_maturity_days = convert_maturity_to_days(option_maturity)
                daily_rewards = 1
                hedging_start_date = start_date
                hedging_end_date = end_date

                spot_end_notional, hedged_end_notional, final_pnl, final_pnl_perc, put_options_price, df_hedged_vs_actual_rewards, nonhedged_notional, vol_info = put_hedge(
                    put_strike_multiplier,
                    daily_rewards,
                    protocol,
                    option_maturity_days,
                    hedging_start_date=hedging_start_date,
                    hedging_end_date=hedging_end_date,
                    percent_to_hedge=percent_to_hedge,
                    IR=interest_rate,
                    broker_spread=broker_spread
                )
                
                # Display results with improved strategy breakdown
                st.header("Strategy Results", divider="blue")
                
                # Main strategy comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Non-Hedged Strategy")
                    st.metric(
                        label="Weekly Offramp Notional", 
                        value=f"${spot_end_notional:,.2f}",
                        help="Total USD value from selling rewards weekly at spot price"
                    )
                
                with col2:
                    # Calculate net value (after option costs)
                    hedged_net_value = hedged_end_notional + nonhedged_notional - put_options_price
                    
                    st.subheader("Hedged Strategy (Net)")
                    st.metric(
                        label=f"Net Strategy Value", 
                        value=f"${hedged_net_value:,.2f}",
                        delta=f"{final_pnl_perc:.2f}%",
                        delta_color="normal"
                    )
                
                # Detailed strategy breakdown
                st.subheader("Strategy Breakdown")
                
                # Create a more detailed breakdown with 3 columns
                breakdown_cols = st.columns(3)
                
                # First column: Protected portion details
                with breakdown_cols[0]:
                    # Calculate how much was actually protected by exercised options
                    if 'hedged_strategy_log' in vol_info:
                        hedged_log = vol_info['hedged_strategy_log']
                        exercised_options = hedged_log[hedged_log['action'] == 'Option Exercise']
                        protected_value = exercised_options['value'].sum() if not exercised_options.empty else 0
                        
                        # Calculate how much was sold at spot price (when options weren't exercised)
                        non_exercised = hedged_log[hedged_log['action'] == 'Option Expired']
                        non_exercised_value = non_exercised['value'].sum() if not non_exercised.empty else 0
                        
                        st.markdown("**Protected Portion**")
                        st.metric(
                            label="Protected by Options", 
                            value=f"${protected_value:,.2f}",
                            help="Value from exercised put options"
                        )
                        st.metric(
                            label="Sold at Spot (No Protection)", 
                            value=f"${non_exercised_value:,.2f}",
                            help="Hedged rewards sold at spot price when puts weren't exercised"
                        )
                        st.metric(
                            label="Total Hedged Portion", 
                            value=f"${hedged_end_notional:,.2f}",
                            help="Total value from the hedged portion"
                        )
                    else:
                        st.metric(
                            label="Total Hedged Portion", 
                            value=f"${hedged_end_notional:,.2f}"
                        )
                
                # Second column: Non-hedged portion and options cost
                with breakdown_cols[1]:
                    st.markdown("**Unprotected Portion**")
                    st.metric(
                        label="Non-Hedged Portion", 
                        value=f"${nonhedged_notional:,.2f}",
                        help="Rewards that weren't hedged, sold at spot price"
                    )
                    
                    st.markdown("**Option Costs**")
                    st.metric(
                        label="Options Cost", 
                        value=f"${put_options_price:,.2f}",
                        delta=f"{(put_options_price / spot_end_notional * 100):.2f}% of notional",
                        delta_color="inverse"
                    )
                
                # Third column: Totals and comparison
                with breakdown_cols[2]:
                    st.markdown("**Strategy Totals**")
                    
                    # Calculate correct totals based on the actual transaction logs
                    if 'hedged_strategy_log' in vol_info:
                        hedged_log = vol_info['hedged_strategy_log']
                        
                        # Sum up all values directly from the transaction logs
                        # This ensures we're not double-counting any transactions
                        total_sold = hedged_log['value'].sum()
                        
                        # Get the option costs from transaction logs
                        option_purchase_costs = vol_info['transaction_log'][vol_info['transaction_log']['type'] == 'Option Purchase']['cost'].sum()
                        
                        # Calculate net value after option costs
                        net_after_options = total_sold - option_purchase_costs
                        
                        st.metric(
                            label="Total Sold (Gross)", 
                            value=f"${total_sold:,.2f}",
                            help="Total value from all selling (before option costs)"
                        )
                        
                        st.metric(
                            label="Net After Options", 
                            value=f"${net_after_options:,.2f}",
                            help="Net value after subtracting option costs"
                        )
                        
                        # Calculate protection benefit
                        option_exercises = hedged_log[hedged_log['action'] == 'Option Exercise']
                        total_protection = option_exercises['protection_value'].sum() if not option_exercises.empty else 0
                        
                        protection_benefit = total_protection - option_purchase_costs
                        
                        st.metric(
                            label="Net Protection Benefit", 
                            value=f"${protection_benefit:,.2f}",
                            delta=f"{(protection_benefit / option_purchase_costs * 100):.2f}% on option cost" if option_purchase_costs > 0 else "N/A",
                            help="Value gained from put protection minus cost of options"
                        )
                    else:
                        # Fallback if logs aren't available
                        total_sold = hedged_end_notional + nonhedged_notional
                        net_after_options = total_sold - put_options_price
                        
                        st.metric(
                            label="Total Sold (Gross)", 
                            value=f"${total_sold:,.2f}"
                        )
                        
                        st.metric(
                            label="Net After Options", 
                            value=f"${net_after_options:,.2f}"
                        )
                
                # Final PnL card with clearer information
                pnl_color = "green" if final_pnl > 0 else "red"
                st.markdown(f"""
                <div style="background-color:rgba({0 if final_pnl > 0 else 255}, {255 if final_pnl > 0 else 0}, 0, 0.1);padding:20px;border-radius:10px;text-align:center;margin-top:20px">
                    <h2 style="margin:0;color:{pnl_color}">Final PnL vs Non-Hedged</h2>
                    <h1 style="margin:10px 0;color:{pnl_color}">${final_pnl:,.2f}</h1>
                    <p style="margin:0;color:{pnl_color}">({final_pnl_perc:.2f}% compared to non-hedged strategy)</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Add strategy parameters recap
                st.subheader("Strategy Parameters")
                params_col1, params_col2, params_col3, params_col4 = st.columns(4)
                
                with params_col1:
                    st.metric("Put Strike", f"{int(strike_opt)}%")
                
                with params_col2:
                    # Display volatility range
                    st.metric(
                        "Avg Volatility", 
                        f"{vol_info['avg_vol']:.2f}",
                        delta=f"Range: {vol_info['min_vol']:.2f} - {vol_info['max_vol']:.2f}",
                        delta_color="off"
                    )
                
                with params_col3:
                    st.metric("Risk-Free Rate", f"{interest_rate*100:.1f}%")
                
                with params_col4:
                    st.metric("Option Maturity", f"{option_maturity_days} days")
                
                # Visualization
                st.subheader("Monthly Rewards Comparison")
                fig, ax = plt.subplots(figsize=(10, 5))
                df_hedged_vs_actual_rewards.plot(kind='bar', ax=ax)
                ax.set_ylabel('Rewards')
                ax.set_title('Hedged vs Actual Monthly Rewards')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                st.pyplot(fig)
                
                # Display the detailed data
                with st.expander("View Detailed Data"):
                    st.dataframe(df_hedged_vs_actual_rewards)
            
                # Add transaction logs section
                st.subheader("🔍 Transaction Logs")
                log_tabs = st.tabs(["Option Transactions", "Spot Strategy Sales", "Hedged Strategy Sales"])
                
                with log_tabs[0]:
                    if 'transaction_log' in vol_info and not vol_info['transaction_log'].empty:
                        # Add summary metrics at top
                        option_purchase_costs = vol_info['transaction_log'][vol_info['transaction_log']['type'] == 'Option Purchase']['cost'].sum()
                        option_exercises = vol_info['transaction_log'][vol_info['transaction_log']['type'] == 'Put Option Exercised']
                        total_protection = option_exercises['protection_value'].sum() if not option_exercises.empty else 0
                        
                        metrics_cols = st.columns(3)
                        metrics_cols[0].metric("Total Option Costs", f"${option_purchase_costs:,.2f}")
                        metrics_cols[1].metric("Protection Value", f"${total_protection:,.2f}")
                        metrics_cols[2].metric("Net Option Value", f"${(total_protection - option_purchase_costs):,.2f}")
                        
                        # Show the detailed log with better formatting
                        st.dataframe(vol_info['transaction_log'], use_container_width=True)
                    else:
                        st.info("No option transactions recorded")
                        
                with log_tabs[1]:
                    if 'spot_strategy_log' in vol_info and not vol_info['spot_strategy_log'].empty:
                        # Add a summary at the top
                        spot_total = vol_info['spot_strategy_log']['value'].sum()
                        st.metric("Total Spot Strategy Value", f"${spot_total:,.2f}")
                        
                        # Show the detailed log
                        st.dataframe(vol_info['spot_strategy_log'], use_container_width=True)
                    else:
                        st.info("No spot strategy transactions recorded")
                        
                with log_tabs[2]:
                    if 'hedged_strategy_log' in vol_info and not vol_info['hedged_strategy_log'].empty:
                        # Add better summary metrics
                        hedged_log = vol_info['hedged_strategy_log']
                        option_exercises = hedged_log[hedged_log['action'] == 'Option Exercise']['value'].sum()
                        option_expirations = hedged_log[hedged_log['action'] == 'Option Expired']['value'].sum()
                        non_hedged = hedged_log[hedged_log['action'] == 'Non-Hedged Portion']['value'].sum()
                        
                        # Display summaries
                        summary_cols = st.columns(3)
                        summary_cols[0].metric("Protected Value", f"${option_exercises:,.2f}")
                        summary_cols[1].metric("Unprotected Value", f"${option_expirations+non_hedged:,.2f}")
                        summary_cols[2].metric("Total Gross Value", f"${hedged_log['value'].sum():,.2f}")
                        
                        st.metric("Net Value (after options)", 
                                  f"${hedged_log['value'].sum() - option_purchase_costs:,.2f}")
                        
                        # Show the detailed log
                        st.dataframe(hedged_log, use_container_width=True)
                    else:
                        st.info("No hedged strategy transactions recorded")
                
                # Save simulation results to CSV for verification
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                simulation_parameters = {
                    'strategy': 'Put',
                    'protocol': protocol,
                    'start_date': str(hedging_start_date),
                    'end_date': str(hedging_end_date),
                    'option_maturity': option_maturity_days,
                    'put_strike_pct': strike_opt,
                    'percent_to_hedge': percent_to_hedge*100,
                    'interest_rate': interest_rate,
                    'broker_spread': broker_spread,
                    'total_options_cost': put_options_price,
                    'non_hedged_value': spot_end_notional,
                    'hedged_value': hedged_net_value,
                    'final_pnl': final_pnl,
                    'final_pnl_pct': final_pnl_perc
                }
                
                csv_path = save_simulation_to_csv('Put', protocol, vol_info, simulation_parameters)
                st.success(f"📊 Simulation data saved to CSV files with prefix: {csv_path}")
                st.download_button(
                    label="📥 Download Simulation Details",
                    data=f"Simulation files saved to:\n{csv_path}_parameters.csv\n{csv_path}_options_transactions.csv\n{csv_path}_spot_strategy.csv\n{csv_path}_hedged_strategy.csv",
                    file_name=f"simulation_files_{timestamp}.txt"
                )
            
            # Similar improvements for Call and Collar strategies...
            if product == "Call":
                # [Similar improvements for Call strategy]
                pass
            
            if product == "Collar":
                # [Similar improvements for Collar strategy]
                put_strike_multiplier = strike_opt/100
                call_strike_multiplier = strike2_opt/100
                option_maturity_days = convert_maturity_to_days(option_maturity)
                daily_rewards = 1
                hedging_start_date = start_date
                hedging_end_date = end_date

                spot_end_notional, hedged_end_notional, final_pnl, final_pnl_perc, call_prices, put_options_price, call_options_price, options_price_paid, df_hedgedvsnon = collar(
                    call_strike_multiplier,
                    put_strike_multiplier,
                    daily_rewards,
                    protocol,
                    option_maturity_days,
                    hedging_start_date=hedging_start_date,
                    hedging_end_date=hedging_end_date,
                    percent_to_hedge=percent_to_hedge,
                    IR=interest_rate,
                    broker_spread=broker_spread
                )

                # Calculate options cost as percentage of spot
                put_cost_percentage = (put_options_price / spot_end_notional) * 100
                call_income_percentage = (call_options_price / spot_end_notional) * 100
                net_cost_percentage = ((put_options_price - call_options_price) / spot_end_notional) * 100

                # Display volatility info more prominently
                try:
                    if protocol.lower() == 'eth':
                        vol_data = pd.read_csv("prices_eth.csv")
                        vol_data['date'] = pd.to_datetime(vol_data['date'])
                    elif protocol.lower() == 'sol':
                        vol_data = pd.read_csv("prices_sol.csv")
                        vol_data['date'] = pd.to_datetime(vol_data['date'])
                    
                    # Get volatility for hedging start date (get closest date if exact match not found)
                    hedging_start_date_dt = pd.to_datetime(hedging_start_date)
                    closest_date_idx = vol_data['date'].sub(hedging_start_date_dt).abs().idxmin()
                    closest_date = vol_data.iloc[closest_date_idx]
                    
                    if pd.notna(closest_date['vol']):
                        sigma_value = closest_date['vol']
                        st.info(f"📈 Using volatility of **{sigma_value:.2f}** from {closest_date['date'].date()} for option pricing")
                    else:
                        sigma_value = 0.6  # Default value
                        st.info(f"📈 Using default volatility of **{sigma_value:.2f}** for option pricing")
                except Exception as e:
                    sigma_value = 0.6  # Default value
                    st.info(f"📈 Using default volatility of **{sigma_value:.2f}** for option pricing. Error: {e}")

                # Display results in a more attractive format
                st.header("Strategy Results", divider="blue")
                
                # Summary stats in a card-like container
                st.markdown("""
                <div style="background-color:#f5f5f5;padding:20px;border-radius:10px;border-left:5px solid #4e8df5;">
                    <h3 style="margin-top:0">Strategy: Collar (Put + Call)</h3>
                    <p>Protecting downside with puts while limiting upside with calls to reduce cost</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Create dashboard-like layout for results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Non-Hedged Strategy")
                    st.metric(
                        label="Weekly Offramp Notional", 
                        value=f"${round(spot_end_notional):,}",
                        help="Total USD value from selling rewards weekly at spot price"
                    )
                
                with col2:
                    st.subheader("Hedged Strategy")
                    st.metric(
                        label=f"Hedged Notional ({int(percent_to_hedge*100)}%)", 
                        value=f"${round(hedged_end_notional):,}",
                        delta=f"{round(final_pnl_perc, 2)}%" if final_pnl_perc > 0 else f"{round(final_pnl_perc, 2)}%",
                        delta_color="normal"
                    )
                
                # Options breakdown
                st.subheader("Options Breakdown")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Put Option Cost", 
                        value=f"${round(put_options_price):,}",
                        delta=f"{round(put_cost_percentage, 2)}% of notional",
                        delta_color="off"
                    )
                
                with col2:
                    st.metric(
                        label="Call Option Income", 
                        value=f"${round(call_options_price):,}",
                        delta=f"{round(call_income_percentage, 2)}% of notional",
                        delta_color="off"
                    )
                
                with col3:
                    net_options = call_options_price - put_options_price
                    st.metric(
                        label="Net Options Cost", 
                        value=f"${round(net_options):,}",
                        delta=f"{round(net_cost_percentage, 2)}% of notional",
                        delta_color="off"
                    )
                
                # Add strategy parameters recap
                st.subheader("Strategy Parameters")
                params_col1, params_col2, params_col3, params_col4 = st.columns(4)
                
                with params_col1:
                    st.metric("Put Strike", f"{int(strike_opt)}%")
                    st.metric("Call Strike", f"{int(strike2_opt)}%")
                
                with params_col2:
                    st.metric("Volatility", f"{sigma_value:.2f}")
                
                with params_col3:
                    st.metric("Risk-Free Rate", f"{interest_rate*100:.1f}%")
                
                with params_col4:
                    st.metric("Option Maturity", f"{option_maturity_days} days")
                
                # Final PnL display
                pnl_color = "green" if final_pnl > 0 else "red"
                st.markdown(f"""
                <div style="border-radius:5px;padding:15px;text-align:center;background-color:rgba({0 if final_pnl > 0 else 255}, {255 if final_pnl > 0 else 0}, 0, 0.1);margin-top:20px">
                    <h3 style="margin:0;color:{pnl_color}">Final PnL</h3>
                    <h2 style="margin:5px 0;color:{pnl_color}">${round(final_pnl):,}</h2>
                    <p style="margin:0;color:{pnl_color}">({round(final_pnl_perc, 2)}%)</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Visualization
                st.subheader("Monthly Rewards Comparison")
                fig, ax = plt.subplots(figsize=(10, 5))
                df_hedgedvsnon.plot(kind='bar', ax=ax)
                ax.set_ylabel('Rewards')
                ax.set_title('Hedged vs Actual Monthly Rewards')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                st.pyplot(fig)
                
                # Collar strategy diagram
                st.subheader("Collar Strategy Diagram")
                st.image("collar.png", caption="Collar Strategy Payoff Diagram")
                
                # Display the detailed data
                with st.expander("View Detailed Data"):
                    st.dataframe(df_hedgedvsnon)
                
                # Save simulation results to CSV for verification
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                simulation_parameters = {
                    'strategy': 'Collar',
                    'protocol': protocol,
                    'start_date': str(hedging_start_date),
                    'end_date': str(hedging_end_date),
                    'option_maturity': option_maturity_days,
                    'put_strike_pct': strike_opt,
                    'call_strike_pct': strike2_opt,
                    'percent_to_hedge': percent_to_hedge*100,
                    'interest_rate': interest_rate,
                    'broker_spread': broker_spread,
                    'put_options_cost': put_options_price,
                    'call_options_income': call_options_price,
                    'net_option_cost': options_price_paid,
                    'non_hedged_value': spot_end_notional,
                    'hedged_value': hedged_end_notional,
                    'final_pnl': final_pnl,
                    'final_pnl_pct': final_pnl_perc
                }
                
                # Create a dict for the collar strategy data, similar to vol_info for put strategy
                collar_logs = {
                    'parameters': pd.DataFrame([simulation_parameters]),
                    'monthly_rewards': df_hedgedvsnon
                }
                
                csv_path = save_simulation_to_csv('Collar', protocol, collar_logs, simulation_parameters)
                st.success(f"📊 Simulation data saved to CSV files with prefix: {csv_path}")
                st.download_button(
                    label="📥 Download Simulation Details",
                    data=f"Simulation files saved to:\n{csv_path}_parameters.csv\n{csv_path}_monthly_rewards.csv",
                    file_name=f"simulation_files_{timestamp}.txt"
                )



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

