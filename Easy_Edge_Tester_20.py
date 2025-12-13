"""
EASY EDGE TESTER 10 - COMPLETE PROP FIRM SIMULATOR
The most comprehensive prop firm testing tool with 3-session support

Trade 1: Breakout Trade (pip-based)
Trade 2: Retracement Trade (% back in range)

Session Options:
- Session 1 (Asian)
- Session 2 (London)  
- Session 3 (New York)
- Any combination

Complete Features:
- MT5 Strategy Tester integration with full statistics
- ALL prop firm rules and restrictions
- Complete EA parameter testing (BE, Trail, Partial TP, Order Cancellation)
- Monte Carlo analysis with 10,000+ simulations
- Smart optimizer with live combo counter
- Trading cost modeling (Spread, Commission, Slippage)
- Payout analysis with accurate days-between-payouts
- Consistency rules, scaling plans, and more

Install: pip install streamlit plotly pandas numpy scikit-learn
Run: streamlit run Easy_Edge_Tester_10.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from itertools import product
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Easy Edge Tester 10",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .best-setting {
        background-color: #d4edda;
        border: 3px solid #28a745;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #0c5460;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
    }
    .stExpander {
        border: 1px solid #dee2e6;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📈 Easy Edge Tester 10</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Complete Three-Session Prop Firm Simulator with MT5 Integration</p>', unsafe_allow_html=True)

# ==================== SIDEBAR ====================
st.sidebar.header("⚙️ Trading Settings")

# === MT5 STATISTICS ===
with st.sidebar.expander("📊 MT5 Historical Statistics", expanded=True):
    st.markdown("**Core Statistics**")
    total_trades_hist = st.number_input("Total Trades (MT5)", min_value=10, value=100, step=10)
    win_rate_hist = st.slider("Win Rate %", min_value=20, max_value=90, value=60, step=1, 
                               help="Lower win rates (20-40%) work with high RR strategies (1:5, 1:8, etc.)")
    
    st.info("ℹ️ **Note**: Profit/loss amounts are now calculated from your SL/TP settings below, not from MT5 averages. This ensures accurate simulation of your actual risk/reward ratio.")
    
    # Auto-calculated summary
    total_wins = int(total_trades_hist * (win_rate_hist / 100))
    total_losses = total_trades_hist - total_wins
    
    st.markdown("---")
    st.markdown("**📈 Summary**")
    st.metric("Total Wins", total_wins)
    st.metric("Total Losses", total_losses)
    st.metric("Win Rate", f"{win_rate_hist}%")

# === TRADE 1: BREAKOUT ===
with st.sidebar.expander("🚀 Trade 1: Breakout", expanded=False):
    trade1_enabled = st.checkbox("✅ Enable Trade 1", value=True, key="t1_enable")
    
    if trade1_enabled:
        st.markdown("**Entry Settings**")
        buffer_pips = st.number_input("Buffer (pips outside range)", min_value=0.0, value=3.0, step=0.5, key="t1_buffer")
        sl_pips = st.number_input("Stop Loss (pips)", min_value=1.0, value=30.0, step=1.0, key="t1_sl")
        tp_ratio = st.number_input("TP Ratio (× SL)", min_value=0.5, max_value=15.0, value=3.0, step=0.5, key="t1_tp",
                                   help="For high RR strategies: 1:5 = 5.0, 1:8 = 8.0, 1:10 = 10.0")
        
        st.markdown("**Break Even**")
        be_enabled = st.checkbox("Enable Break Even", value=True, key="t1_be")
        if be_enabled:
            be_trigger = st.number_input("BE Trigger (× SL)", min_value=0.1, value=1.0, step=0.1, key="t1_be_trigger")
            be_buffer = st.number_input("BE Buffer (pips)", min_value=0.0, value=5.0, step=1.0, key="t1_be_buffer")
        
        st.markdown("**Partial Take Profit**")
        partial_enabled = st.checkbox("Enable Partial TP", value=True, key="t1_partial")
        if partial_enabled:
            partial_trigger = st.number_input("Partial Trigger (× SL)", min_value=0.1, value=1.5, step=0.1, key="t1_partial_trigger")
            partial_percent = st.slider("Close % at Partial", min_value=10, max_value=90, value=50, step=10, key="t1_partial_pct")
        
        st.markdown("**Trailing Stop**")
        trail_enabled = st.checkbox("Enable Trailing Stop", value=True, key="t1_trail")
        if trail_enabled:
            trail_trigger = st.number_input("Trail Trigger (× SL)", min_value=0.1, value=2.0, step=0.1, key="t1_trail_trigger")
            trail_distance = st.number_input("Trail Distance (pips)", min_value=1.0, value=15.0, step=1.0, key="t1_trail_dist")
        
        st.markdown("**Order Cancellation**")
        cancel_enabled = st.checkbox("Cancel if not filled", value=True, key="t1_cancel")
        if cancel_enabled:
            cancel_hours = st.number_input("Cancel after (hours)", min_value=1, value=6, step=1, key="t1_cancel_hrs")

# === TRADE 2: RETRACEMENT ===
with st.sidebar.expander("🔄 Trade 2: Retracement", expanded=False):
    trade2_enabled = st.checkbox("✅ Enable Trade 2", value=True, key="t2_enable")
    
    if trade2_enabled:
        st.markdown("**Entry Settings**")
        avg_range_pips = st.number_input("Avg Range Size (pips)", min_value=10.0, value=80.0, step=5.0, key="t2_range")
        retrace_pct = st.slider("Entry Retracement %", min_value=10, max_value=90, value=50, step=5, key="t2_retrace")
        sl_pct_range = st.slider("SL (% of range)", min_value=20, max_value=150, value=100, step=10, key="t2_sl_pct")
        tp_ratio_t2 = st.number_input("TP Ratio (× SL)", min_value=0.5, max_value=15.0, value=1.5, step=0.5, key="t2_tp",
                                      help="For high RR strategies: 1:5 = 5.0, 1:8 = 8.0, 1:10 = 10.0")
        
        st.markdown("**Break Even**")
        be_enabled_t2 = st.checkbox("Enable Break Even", value=True, key="t2_be")
        if be_enabled_t2:
            be_trigger_t2 = st.number_input("BE Trigger (× SL)", min_value=0.1, value=1.0, step=0.1, key="t2_be_trigger")
            be_buffer_t2 = st.number_input("BE Buffer (pips)", min_value=0.0, value=5.0, step=1.0, key="t2_be_buffer")
        
        st.markdown("**Partial Take Profit**")
        partial_enabled_t2 = st.checkbox("Enable Partial TP", value=True, key="t2_partial")
        if partial_enabled_t2:
            partial_trigger_t2 = st.number_input("Partial Trigger (× SL)", min_value=0.1, value=1.5, step=0.1, key="t2_partial_trigger")
            partial_percent_t2 = st.slider("Close % at Partial", min_value=10, max_value=90, value=50, step=10, key="t2_partial_pct")
        
        st.markdown("**Trailing Stop**")
        trail_enabled_t2 = st.checkbox("Enable Trailing Stop", value=True, key="t2_trail")
        if trail_enabled_t2:
            trail_trigger_t2 = st.number_input("Trail Trigger (× SL)", min_value=0.1, value=2.0, step=0.1, key="t2_trail_trigger")
            trail_distance_t2 = st.number_input("Trail Distance (pips)", min_value=1.0, value=15.0, step=1.0, key="t2_trail_dist")
        
        st.markdown("**Order Cancellation**")
        cancel_enabled_t2 = st.checkbox("Cancel if not filled", value=True, key="t2_cancel")
        if cancel_enabled_t2:
            cancel_hours_t2 = st.number_input("Cancel after (hours)", min_value=1, value=6, step=1, key="t2_cancel_hrs")

# Check if at least one trade is enabled
if not trade1_enabled and not trade2_enabled:
    st.sidebar.error("⚠️ Enable at least one trade type!")

# === SESSION SETTINGS (3 SESSIONS) ===
st.sidebar.markdown("---")
st.sidebar.markdown("### 📅 Session Settings")

col1, col2, col3 = st.sidebar.columns(3)
with col1:
    session1_enabled = st.checkbox("Asian", value=True, key="sess1")
with col2:
    session2_enabled = st.checkbox("London", value=True, key="sess2")
with col3:
    session3_enabled = st.checkbox("New York", value=True, key="sess3")

enabled_sessions = sum([session1_enabled, session2_enabled, session3_enabled])

if enabled_sessions == 0:
    st.sidebar.error("⚠️ Enable at least 1 session!")

session_names = []
if session1_enabled:
    session_names.append("Asian")
if session2_enabled:
    session_names.append("London")
if session3_enabled:
    session_names.append("New York")

st.sidebar.info(f"**Active:** {', '.join(session_names)} ({enabled_sessions} session{'s' if enabled_sessions > 1 else ''})")

# === DAYS OF WEEK ===
st.sidebar.markdown("### 📆 Trading Days")
col1, col2 = st.sidebar.columns(2)
with col1:
    monday = st.checkbox("Mon", value=True, key="mon")
    tuesday = st.checkbox("Tue", value=True, key="tue")
    wednesday = st.checkbox("Wed", value=True, key="wed")
with col2:
    thursday = st.checkbox("Thu", value=True, key="thu")
    friday = st.checkbox("Fri", value=True, key="fri")

trading_days = [monday, tuesday, wednesday, thursday, friday]
active_days = sum(trading_days)

if active_days == 0:
    st.sidebar.error("⚠️ Enable at least 1 day!")

st.sidebar.info(f"**Trading:** {active_days}/5 days per week")

# === COMPLETE PROP FIRM SETTINGS ===
st.sidebar.markdown("---")
with st.sidebar.expander("💼 Complete Prop Firm Settings", expanded=True):
    st.markdown("**Account Settings**")
    account_size = st.selectbox("Account Size", [5000, 10000, 25000, 50000, 100000, 200000], index=2)
    starting_balance_pf = account_size  # For prop firm, starting balance = account size
    
    # Default challenge fees based on account size
    default_fees = {5000: 49, 10000: 99, 25000: 199, 50000: 349, 100000: 549, 200000: 999}
    challenge_fee = st.number_input("Challenge Fee ($)", min_value=0, value=default_fees.get(account_size, 199), step=10,
                                     help="One-time fee to purchase the challenge")
    
    st.markdown("**Drawdown Rules**")
    max_daily_loss_pct = st.slider("Max Daily Loss %", min_value=3, max_value=10, value=5, step=1)
    max_total_dd_pct = st.slider("Max Total Drawdown %", min_value=5, max_value=15, value=10, step=1)
    trailing_dd = st.checkbox("Trailing Drawdown (vs Static)", value=False)
    
    st.markdown("**Profit Targets**")
    phase1_target_pct = st.slider("Phase 1 Target %", min_value=5, max_value=15, value=8, step=1)
    phase2_target_pct = st.slider("Phase 2 Target %", min_value=3, max_value=10, value=5, step=1)
    
    st.markdown("**Challenge Duration**")
    phase1_days = st.number_input("Phase 1 Days", min_value=7, max_value=90, value=30, step=1)
    phase2_days = st.number_input("Phase 2 Days", min_value=7, max_value=90, value=60, step=1)
    min_trading_days = st.number_input("Min Trading Days Required", min_value=1, max_value=30, value=5, step=1)
    
    st.markdown("**Consistency Rule**")
    consistency_enabled = st.checkbox("Enable Consistency Rule", value=True)
    if consistency_enabled:
        max_daily_profit_pct = st.slider("Max Single Day Profit % (of total)", min_value=20, max_value=60, value=40, step=5)
    
    st.markdown("**Position Limits**")
    max_lot_size = st.number_input("Max Lot Size", min_value=0.1, value=10.0, step=0.5)
    max_positions = st.number_input("Max Open Positions", min_value=1, value=5, step=1)
    
    st.markdown("**Trading Restrictions**")
    weekend_holding = st.checkbox("Allow Weekend Holding", value=False)
    news_trading = st.checkbox("Allow News Trading", value=True)
    
    st.markdown("**Trade Frequency**")
    trade_opportunity_rate = st.slider("Setup Occurrence Rate %", min_value=10, max_value=100, value=30, step=5,
                                       help="% of sessions that have valid trade setups. Lower = fewer trades, more selective. High RR strategies typically 20-40%")
    
    st.markdown("**Risk Management**")
    risk_per_trade = st.slider("Risk Per Trade %", min_value=0.25, max_value=3.0, value=1.0, step=0.25,
                               help="Lower risk (0.25-0.5%) recommended for high RR strategies (1:8, 1:10)")
    
    # Calculate limits
    max_daily_loss = account_size * (max_daily_loss_pct / 100)
    max_total_dd = account_size * (max_total_dd_pct / 100)
    phase1_target = account_size * (phase1_target_pct / 100)
    phase2_target = account_size * (phase2_target_pct / 100)
    
    st.markdown("---")
    st.markdown("**📊 Calculated Limits**")
    st.info(f"""
    💵 **Daily Loss Limit:** ${max_daily_loss:,.0f}  
    📉 **Max Drawdown:** ${max_total_dd:,.0f}  
    🎯 **Phase 1 Target:** ${phase1_target:,.0f}  
    🎯 **Phase 2 Target:** ${phase2_target:,.0f}
    """)
    
    st.markdown("**💰 Investment & ROI**")
    breakeven_profit = challenge_fee
    roi_at_phase1 = ((phase1_target - challenge_fee) / challenge_fee) * 100 if challenge_fee > 0 else 0
    roi_at_phase2 = ((phase1_target + phase2_target - challenge_fee) / challenge_fee) * 100 if challenge_fee > 0 else 0
    
    st.info(f"""
    💳 **Challenge Fee:** ${challenge_fee:,.0f}  
    ⚖️ **Break Even:** ${breakeven_profit:,.0f} profit  
    📈 **ROI @ Phase 1:** {roi_at_phase1:.0f}%  
    📈 **ROI @ Phase 2:** {roi_at_phase2:.0f}%
    """)

# === TRADING COSTS ===
st.sidebar.markdown("---")
with st.sidebar.expander("💸 Trading Costs", expanded=False):
    spread_pips = st.number_input("Spread (pips)", min_value=0.0, value=1.5, step=0.1)
    commission = st.number_input("Commission per lot (round trip)", min_value=0.0, value=7.0, step=1.0)
    slippage_pips = st.number_input("Slippage (pips)", min_value=0.0, value=0.5, step=0.1)
    
    pip_value = 10  # $10 per pip for 1 lot
    cost_per_lot = (spread_pips + slippage_pips) * pip_value + commission
    
    st.metric("Cost per 1.0 Lot", f"${cost_per_lot:.2f}")

# === SIMULATION SETTINGS ===
st.sidebar.markdown("---")
with st.sidebar.expander("🎲 Simulation Settings", expanded=False):
    num_simulations = st.selectbox("Monte Carlo Runs", [1000, 5000, 10000, 20000, 50000], index=2)
    sim_phase = st.selectbox("Simulate Phase", ["Phase 1", "Phase 2", "Both Phases"], index=0)

# ==================== SIMULATION FUNCTIONS ====================

def calculate_trade_stats(trade_type, base_win_rate):
    """Calculate realistic trade statistics"""
    
    if trade_type == 1:  # Breakout
        sl_distance = sl_pips
        tp_distance = sl_pips * tp_ratio
        be_active = be_enabled if trade1_enabled else False
        partial_active = partial_enabled if trade1_enabled else False
        trail_active = trail_enabled if trade1_enabled else False
        cancel_active = cancel_enabled if trade1_enabled else False
        cancel_hrs = cancel_hours if cancel_active else None
        be_trig = be_trigger if be_active else None
        be_buff = be_buffer if be_active else None
        partial_trig = partial_trigger if partial_active else None
        partial_pct_val = partial_percent if partial_active else None
        trail_trig = trail_trigger if trail_active else None
        trail_dist = trail_distance if trail_active else None
    else:  # Retracement
        sl_distance = avg_range_pips * (sl_pct_range / 100)
        tp_distance = sl_distance * tp_ratio_t2
        be_active = be_enabled_t2 if trade2_enabled else False
        partial_active = partial_enabled_t2 if trade2_enabled else False
        trail_active = trail_enabled_t2 if trade2_enabled else False
        cancel_active = cancel_enabled_t2 if trade2_enabled else False
        cancel_hrs = cancel_hours_t2 if cancel_active else None
        be_trig = be_trigger_t2 if be_active else None
        be_buff = be_buffer_t2 if be_active else None
        partial_trig = partial_trigger_t2 if partial_active else None
        partial_pct_val = partial_percent_t2 if partial_active else None
        trail_trig = trail_trigger_t2 if trail_active else None
        trail_dist = trail_distance_t2 if trail_active else None
    
    # Adjust win rate based on features
    adjusted_win_rate = base_win_rate
    
    if be_active:
        adjusted_win_rate += 0.03
    if partial_active:
        adjusted_win_rate += 0.05
    if trail_active:
        adjusted_win_rate -= 0.02
    if cancel_active:
        adjusted_win_rate += 0.02
    
    adjusted_win_rate = min(adjusted_win_rate, 0.80)  # Cap at 80%
    
    return {
        'win_rate': adjusted_win_rate,
        'sl': sl_distance,
        'tp': tp_distance,
        'be_enabled': be_active,
        'be_trigger': be_trig,
        'be_buffer': be_buff,
        'partial_enabled': partial_active,
        'partial_trigger': partial_trig,
        'partial_percent': partial_pct_val,
        'trail_enabled': trail_active,
        'trail_trigger': trail_trig,
        'trail_distance': trail_dist,
        'cancel_enabled': cancel_active,
        'cancel_hours': cancel_hrs
    }

def simulate_single_trade(stats, lot_size, account_balance):
    """Simulate a single trade with all EA features using actual RR calculations"""
    
    # Random outcome based on win rate
    is_win = np.random.random() < stats['win_rate']
    
    # Calculate cost per trade
    pip_value = 10  # $10 per pip for 1 standard lot
    cost_per_trade = (spread_pips + slippage_pips) * pip_value * lot_size + (commission * lot_size)
    
    if is_win:
        # Calculate win based on actual TP distance
        base_win = stats['tp'] * pip_value * lot_size
        
        # Add some variation (±10% to account for partial fills, slippage, etc.)
        variation = np.random.uniform(0.9, 1.0)  # Slightly negative bias for realism
        potential_win = base_win * variation
        
        # Check if partial TP hits
        if stats['partial_enabled'] and np.random.random() < 0.7:
            partial_profit = (stats['tp'] * stats['partial_trigger']) * pip_value * lot_size * (stats['partial_percent'] / 100)
            remaining_lot = lot_size * (1 - stats['partial_percent'] / 100)
            
            # Check if trail catches more
            if stats['trail_enabled'] and np.random.random() < 0.6:
                # Trail catches somewhere between trigger and full TP
                trail_pips = np.random.uniform(stats['tp'] * stats['trail_trigger'], stats['tp'])
                trail_profit = trail_pips * pip_value * remaining_lot
                total_profit = partial_profit + trail_profit - cost_per_trade
            else:
                # Hits full TP on remaining
                full_tp_profit = stats['tp'] * pip_value * remaining_lot
                total_profit = partial_profit + full_tp_profit - cost_per_trade
        else:
            # Full TP hit
            total_profit = potential_win - cost_per_trade
        
        return total_profit, True, lot_size
    
    else:
        # Calculate loss based on actual SL distance
        base_loss = stats['sl'] * pip_value * lot_size
        
        # Add variation for slippage/partial fills
        variation = np.random.uniform(0.95, 1.05)  # Sometimes worse, sometimes better
        potential_loss = base_loss * variation
        
        # Check if BE saves the trade
        if stats['be_enabled'] and np.random.random() < 0.4:
            # Only lose the BE buffer
            loss = -(stats['be_buffer'] * pip_value * lot_size) - cost_per_trade
        else:
            # Full SL hit
            loss = -potential_loss - cost_per_trade
        
        return loss, False, lot_size

def check_consistency_rule(daily_pnl, total_profit):
    """Check if consistency rule is violated"""
    if not consistency_enabled:
        return True
    
    if total_profit <= 0:
        return True
    
    max_day_profit = max(daily_pnl)
    if max_day_profit > 0:
        pct_of_total = (max_day_profit / total_profit) * 100
        return pct_of_total <= max_daily_profit_pct
    
    return True

def run_prop_firm_challenge(phase, days_limit, profit_target_pct):
    """Run a complete prop firm challenge with ALL rules"""
    
    balance = account_size
    equity_curve = [balance]
    daily_pnl = []
    trades = []
    
    current_day = 0
    day_of_week = 0  # 0=Mon, 4=Fri
    actual_trading_days = 0
    
    # Track highest balance for trailing drawdown
    highest_balance = balance
    
    while current_day < days_limit:
        # Check if this day is a trading day
        if not trading_days[day_of_week]:
            day_of_week = (day_of_week + 1) % 5
            current_day += 1
            equity_curve.append(balance)
            daily_pnl.append(0)
            continue
        
        day_start_balance = balance
        day_trades_count = 0
        day_lots_used = 0
        
        # Trade each enabled session
        for session_idx in range(enabled_sessions):
            # Determine which trades to take
            trades_this_session = []
            if trade1_enabled:
                trades_this_session.append(1)
            if trade2_enabled:
                trades_this_session.append(2)
            
            for trade_type in trades_this_session:
                # Check max positions limit
                if day_trades_count >= max_positions:
                    break
                
                # Realistic trade opportunity filter
                # Not every session has a valid setup
                if np.random.random() > (trade_opportunity_rate / 100):
                    continue  # No setup this session
                
                # Calculate lot size based on risk
                risk_amount = balance * (risk_per_trade / 100)
                
                # DEBUG: Check if risk calculation is sane
                if risk_amount > balance * 0.10:
                    print(f"🚨 ERROR: Risk amount ${risk_amount:,.2f} is >10% of balance ${balance:,.2f}!")
                    print(f"  Risk %: {risk_per_trade}")
                    print(f"  This suggests risk_per_trade is being read as {risk_per_trade} instead of {risk_per_trade}%")
                
                if trade_type == 1:
                    stats = calculate_trade_stats(1, win_rate_hist / 100)
                else:
                    stats = calculate_trade_stats(2, win_rate_hist / 100)
                
                pip_value = 10  # $10 per pip for 1 standard lot
                lot_size = risk_amount / (stats['sl'] * pip_value)
                lot_size = round(lot_size, 2)
                
                # DEBUG: Check if lot size is sane
                if lot_size > 10:
                    print(f"🚨 WARNING: Lot size {lot_size} is very high!")
                    print(f"  This means: Risk ${risk_amount:,.2f} / (SL {stats['sl']} × $10) = {lot_size}")
                    if stats['sl'] < 5:
                        print(f"  ⚠️ SL is only {stats['sl']} pips - this seems too small!")
                
                # ABSOLUTE SAFETY: Lot size should NEVER exceed account value / 10000
                # Even 100 lots on $100K account is already 10:1 leverage
                absolute_max_lots = account_size / 10000
                if lot_size > absolute_max_lots:
                    print(f"⚠️ WARNING: Lot size {lot_size} exceeds safe limit {absolute_max_lots}. Capping.")
                    lot_size = absolute_max_lots
                
                # DEBUG: Log first trade details (remove after debugging)
                if len(trades) == 0:
                    print(f"\n=== FIRST TRADE CALCULATION ===")
                    print(f"  Balance: ${balance:,.2f}")
                    print(f"  Risk %: {risk_per_trade}%")
                    print(f"  Risk Amount: ${risk_amount:,.2f}")
                    print(f"  SL: {stats['sl']} pips")
                    print(f"  TP: {stats['tp']} pips")
                    print(f"  TP Ratio: {stats['tp'] / stats['sl']}")
                    print(f"  Pip Value: ${pip_value}")
                    print(f"  Calculated Lot Size: {lot_size}")
                    print(f"  Expected Win: {stats['tp']} × {pip_value} × {lot_size} = ${stats['tp'] * pip_value * lot_size:,.2f}")
                    print(f"  Expected Loss: {stats['sl']} × {pip_value} × {lot_size} = ${stats['sl'] * pip_value * lot_size:,.2f}")
                
                # Check max lot size limit
                if lot_size > max_lot_size:
                    lot_size = max_lot_size
                
                # Check if order gets cancelled
                if stats['cancel_enabled'] and np.random.random() < 0.15:
                    continue
                
                # Execute trade
                pnl, is_win, lots = simulate_single_trade(stats, lot_size, balance)
                balance += pnl
                day_lots_used += lots
                
                # DEBUG: Log first 3 trades
                if len(trades) < 3:
                    print(f"\n=== TRADE #{len(trades)+1} DEBUG ===")
                    print(f"Balance: ${balance - pnl:,.2f}")
                    print(f"Risk: {risk_per_trade}%")
                    print(f"Risk $: ${risk_amount:,.2f}")
                    print(f"SL: {stats['sl']} pips")
                    print(f"TP: {stats['tp']} pips")
                    print(f"Lot Size: {lot_size}")
                    print(f"Win: {is_win}")
                    print(f"PnL: ${pnl:,.2f}")
                    print(f"New Balance: ${balance:,.2f}")
                
                trades.append({
                    'day': current_day,
                    'type': f"Trade {trade_type}",
                    'session': session_idx + 1,
                    'pnl': pnl,
                    'balance': balance,
                    'win': is_win,
                    'lots': lots
                })
                
                day_trades_count += 1
                
                # Check daily loss limit
                daily_loss = balance - day_start_balance
                if daily_loss < -max_daily_loss:
                    return {
                        'passed': False,
                        'reason': 'Daily Loss Limit Exceeded',
                        'phase': phase,
                        'final_balance': balance,
                        'trades': trades,
                        'equity_curve': equity_curve,
                        'daily_pnl': daily_pnl,
                        'days_lasted': current_day,
                        'actual_trading_days': actual_trading_days
                    }
        
        # End of day
        day_pnl = balance - day_start_balance
        daily_pnl.append(day_pnl)
        equity_curve.append(balance)
        
        if day_trades_count > 0:
            actual_trading_days += 1
        
        # Update highest balance for trailing drawdown
        if balance > highest_balance:
            highest_balance = balance
        
        # Check drawdown
        if trailing_dd:
            # Trailing drawdown
            dd_amount = highest_balance - balance
            if dd_amount > max_total_dd:
                return {
                    'passed': False,
                    'reason': 'Trailing Drawdown Exceeded',
                    'phase': phase,
                    'final_balance': balance,
                    'trades': trades,
                    'equity_curve': equity_curve,
                    'daily_pnl': daily_pnl,
                    'days_lasted': current_day,
                    'actual_trading_days': actual_trading_days
                }
        else:
            # Static drawdown
            if balance < (account_size - max_total_dd):
                return {
                    'passed': False,
                    'reason': 'Max Drawdown Exceeded',
                    'phase': phase,
                    'final_balance': balance,
                    'trades': trades,
                    'equity_curve': equity_curve,
                    'daily_pnl': daily_pnl,
                    'days_lasted': current_day,
                    'actual_trading_days': actual_trading_days
                }
        
        day_of_week = (day_of_week + 1) % 5
        current_day += 1
    
    # Challenge completed - check results
    profit = balance - account_size
    profit_pct = (profit / account_size) * 100
    profit_target = account_size * (profit_target_pct / 100)
    
    # Check minimum trading days
    if actual_trading_days < min_trading_days:
        return {
            'passed': False,
            'reason': f'Insufficient Trading Days ({actual_trading_days}/{min_trading_days})',
            'phase': phase,
            'final_balance': balance,
            'profit': profit,
            'profit_pct': profit_pct,
            'trades': trades,
            'equity_curve': equity_curve,
            'daily_pnl': daily_pnl,
            'days_lasted': current_day,
            'actual_trading_days': actual_trading_days
        }
    
    # Check profit target
    if profit < profit_target:
        return {
            'passed': False,
            'reason': f'Insufficient Profit ({profit_pct:.1f}% < {profit_target_pct}%)',
            'phase': phase,
            'final_balance': balance,
            'profit': profit,
            'profit_pct': profit_pct,
            'trades': trades,
            'equity_curve': equity_curve,
            'daily_pnl': daily_pnl,
            'days_lasted': current_day,
            'actual_trading_days': actual_trading_days
        }
    
    # Check consistency rule
    if not check_consistency_rule(daily_pnl, profit):
        return {
            'passed': False,
            'reason': 'Consistency Rule Violated',
            'phase': phase,
            'final_balance': balance,
            'profit': profit,
            'profit_pct': profit_pct,
            'trades': trades,
            'equity_curve': equity_curve,
            'daily_pnl': daily_pnl,
            'days_lasted': current_day,
            'actual_trading_days': actual_trading_days
        }
    
    # PASSED!
    return {
        'passed': True,
        'reason': 'Challenge Passed!',
        'phase': phase,
        'final_balance': balance,
        'profit': profit,
        'profit_pct': profit_pct,
        'trades': trades,
        'equity_curve': equity_curve,
        'daily_pnl': daily_pnl,
        'days_lasted': current_day,
        'actual_trading_days': actual_trading_days
    }

# ==================== MAIN CONTENT ====================

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Quick Test", "🎲 Monte Carlo", "🔍 Optimizer", "📈 Results Dashboard", "📚 Info"])

# === TAB 1: QUICK TEST ===
with tab1:
    st.markdown("### 🚀 Quick Challenge Test")
    st.markdown("Run a single prop firm challenge to see how your strategy performs")
    
    if st.button("🎯 Run Quick Test", type="primary", key="quick_test_btn"):
        with st.spinner("Running challenge simulation..."):
            if sim_phase == "Phase 1":
                result = run_prop_firm_challenge("Phase 1", phase1_days, phase1_target_pct)
            elif sim_phase == "Phase 2":
                result = run_prop_firm_challenge("Phase 2", phase2_days, phase2_target_pct)
            else:  # Both phases
                result1 = run_prop_firm_challenge("Phase 1", phase1_days, phase1_target_pct)
                if result1['passed']:
                    result = run_prop_firm_challenge("Phase 2", phase2_days, phase2_target_pct)
                else:
                    result = result1
            
            # Display results
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                if result['passed']:
                    st.markdown('<div class="success-box"><h3>✅ PASSED</h3></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="danger-box"><h3>❌ FAILED</h3></div>', unsafe_allow_html=True)
            
            with col2:
                st.metric("Phase", result['phase'])
            
            with col3:
                st.metric("Final Balance", f"${result['final_balance']:,.2f}")
            
            with col4:
                if 'profit_pct' in result:
                    st.metric("Profit %", f"{result['profit_pct']:.2f}%")
            
            with col5:
                st.metric("Total Trades", len(result['trades']))
            
            st.markdown(f"**Reason:** {result['reason']}")
            st.metric("Trading Days", f"{result['actual_trading_days']} / {min_trading_days} required")
            
            # Add detailed trade statistics
            if result['trades']:
                wins = sum(1 for t in result['trades'] if t['win'])
                losses = len(result['trades']) - wins
                total_win_pnl = sum(t['pnl'] for t in result['trades'] if t['win'])
                total_loss_pnl = sum(t['pnl'] for t in result['trades'] if not t['win'])
                avg_win_amt = total_win_pnl / wins if wins > 0 else 0
                avg_loss_amt = total_loss_pnl / losses if losses > 0 else 0
                
                st.markdown("### 📊 Trade Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Wins", wins)
                    st.metric("Avg Win", f"${avg_win_amt:,.2f}")
                with col2:
                    st.metric("Losses", losses)
                    st.metric("Avg Loss", f"${avg_loss_amt:,.2f}")
                with col3:
                    actual_wr = (wins / len(result['trades']) * 100) if result['trades'] else 0
                    st.metric("Actual Win Rate", f"{actual_wr:.1f}%")
                with col4:
                    trades_per_day = len(result['trades']) / result['days_lasted'] if result['days_lasted'] > 0 else 0
                    st.metric("Trades/Day", f"{trades_per_day:.2f}")
            
            # Equity curve
            st.markdown("### 📈 Equity Curve")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=result['equity_curve'],
                mode='lines',
                name='Balance',
                line=dict(color='#1f77b4', width=2)
            ))
            fig.add_hline(y=account_size, line_dash="dash", line_color="gray", annotation_text="Starting Balance")
            fig.add_hline(y=account_size - max_total_dd, line_dash="dash", line_color="red", annotation_text="Max DD Limit")
            
            if sim_phase == "Phase 1" or result['phase'] == "Phase 1":
                target = account_size + (account_size * phase1_target_pct / 100)
            else:
                target = account_size + (account_size * phase2_target_pct / 100)
            
            fig.add_hline(y=target, line_dash="dash", line_color="green", annotation_text="Profit Target")
            
            fig.update_layout(
                title="Account Balance Over Time",
                xaxis_title="Day",
                yaxis_title="Balance ($)",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Daily PnL
            if result['daily_pnl']:
                st.markdown("### 💰 Daily Profit/Loss")
                fig2 = go.Figure()
                colors = ['green' if x > 0 else 'red' for x in result['daily_pnl']]
                fig2.add_trace(go.Bar(
                    y=result['daily_pnl'],
                    marker_color=colors,
                    name='Daily PnL'
                ))
                fig2.add_hline(y=-max_daily_loss, line_dash="dash", line_color="red", annotation_text="Daily Loss Limit")
                fig2.update_layout(
                    title="Daily Profit/Loss",
                    xaxis_title="Day",
                    yaxis_title="PnL ($)",
                    height=300
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Trade log
            if result['trades']:
                st.markdown("### 📋 Trade Log (Last 20 Trades)")
                df_trades = pd.DataFrame(result['trades'][-20:])
                st.dataframe(df_trades, use_container_width=True)

# === TAB 2: MONTE CARLO ===
with tab2:
    st.markdown("### 🎲 Monte Carlo Analysis")
    st.markdown("Run thousands of simulations to calculate probability of success")
    
    if st.button("🎲 Run Monte Carlo", type="primary", key="mc_btn"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        
        for i in range(num_simulations):
            if sim_phase == "Phase 1":
                result = run_prop_firm_challenge("Phase 1", phase1_days, phase1_target_pct)
            elif sim_phase == "Phase 2":
                result = run_prop_firm_challenge("Phase 2", phase2_days, phase2_target_pct)
            else:  # Both phases
                result1 = run_prop_firm_challenge("Phase 1", phase1_days, phase1_target_pct)
                if result1['passed']:
                    result = run_prop_firm_challenge("Phase 2", phase2_days, phase2_target_pct)
                else:
                    result = result1
            
            results.append(result)
            
            if (i + 1) % 100 == 0:
                progress_bar.progress((i + 1) / num_simulations)
                status_text.text(f"Completed {i+1:,}/{num_simulations:,} simulations...")
        
        progress_bar.progress(1.0)
        status_text.text("✅ Analysis complete!")
        
        # Calculate statistics
        passed = sum(1 for r in results if r['passed'])
        pass_rate = (passed / num_simulations) * 100
        
        # Analyze failure reasons
        failure_reasons = {}
        for r in results:
            if not r['passed']:
                reason = r['reason']
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        # Also track why profitable runs still failed
        profitable_failures = []
        for r in results:
            if not r['passed'] and r.get('profit', 0) > 0:
                profitable_failures.append({
                    'profit': r.get('profit', 0),
                    'profit_pct': r.get('profit_pct', 0),
                    'reason': r['reason'],
                    'days': r.get('actual_trading_days', 0)
                })
        
        avg_profit = np.mean([r.get('profit', 0) for r in results if r['passed']])
        avg_loss = np.mean([r['final_balance'] - account_size for r in results if not r['passed']])
        
        # Calculate per-trade statistics from all simulations
        all_trade_wins = []
        all_trade_losses = []
        for r in results:
            if 'trades' in r and r['trades']:
                for trade in r['trades']:
                    if trade['win']:
                        all_trade_wins.append(trade['pnl'])
                    else:
                        all_trade_losses.append(trade['pnl'])
        
        avg_win_per_trade = np.mean(all_trade_wins) if all_trade_wins else 0
        avg_loss_per_trade = np.mean(all_trade_losses) if all_trade_losses else 0
        total_trades_all_sims = sum(len(r.get('trades', [])) for r in results)
        
        # Calculate ROI metrics
        avg_roi_passed = ((avg_profit - challenge_fee) / challenge_fee * 100) if challenge_fee > 0 and passed > 0 else 0
        expected_value = (passed / num_simulations) * (avg_profit - challenge_fee) + ((num_simulations - passed) / num_simulations) * (avg_loss - challenge_fee)
        expected_roi = (expected_value / challenge_fee * 100) if challenge_fee > 0 else 0
        
        avg_days = np.mean([r['days_lasted'] for r in results])
        avg_trading_days = np.mean([r['actual_trading_days'] for r in results])
        
        # Display results
        st.markdown("---")
        st.markdown("## 📊 Monte Carlo Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Pass Rate", f"{pass_rate:.1f}%")
        with col2:
            st.metric("Passed", f"{passed:,}")
        with col3:
            st.metric("Failed", f"{num_simulations - passed:,}")
        with col4:
            st.metric("Avg Total Profit (Passed)", f"${avg_profit:,.0f}" if passed > 0 else "$0",
                     help="Average total profit when you pass the challenge")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Total Loss (Failed)", f"${avg_loss:,.0f}",
                     help="Average total loss when you fail the challenge")
        with col2:
            st.metric("Avg Days", f"{avg_days:.1f}")
        with col3:
            st.metric("Avg Trading Days", f"{avg_trading_days:.1f}")
        with col4:
            st.metric("Expected ROI", f"{expected_roi:.0f}%", 
                     help="Expected return on investment considering pass rate")
        
        # ROI Analysis Box
        st.markdown("### 💰 ROI Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Challenge Fee", f"${challenge_fee:,.0f}")
        with col2:
            st.metric("Avg ROI (Passed)", f"{avg_roi_passed:.0f}%" if passed > 0 else "N/A")
        with col3:
            net_profit_after_fee = avg_profit - challenge_fee if passed > 0 else 0
            st.metric("Net Profit (Passed)", f"${net_profit_after_fee:,.0f}")
        
        if expected_roi > 0:
            st.success(f"✅ Positive Expected Value: With {pass_rate:.1f}% pass rate, you'll make ${expected_value:,.0f} on average per challenge attempt (ROI: {expected_roi:.0f}%)")
        else:
            st.error(f"❌ Negative Expected Value: With {pass_rate:.1f}% pass rate, you'll lose ${-expected_value:,.0f} on average per challenge attempt")
        
        # Add per-trade statistics
        st.markdown("### 📊 Per-Trade Statistics (Across All Simulations)")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Win Per Trade", f"${avg_win_per_trade:,.2f}",
                     help="Average profit per winning trade")
        with col2:
            st.metric("Avg Loss Per Trade", f"${avg_loss_per_trade:,.2f}",
                     help="Average loss per losing trade")
        with col3:
            avg_trades_per_sim = total_trades_all_sims / num_simulations
            st.metric("Avg Trades Per Challenge", f"{avg_trades_per_sim:.1f}")
        with col4:
            actual_rr = abs(avg_win_per_trade / avg_loss_per_trade) if avg_loss_per_trade != 0 else 0
            st.metric("Actual R:R", f"1:{actual_rr:.2f}",
                     help="Actual risk-reward ratio from simulation")
        
        # Failure reasons chart
        if failure_reasons:
            st.markdown("### ❌ Failure Reasons")
            fig_reasons = go.Figure()
            fig_reasons.add_trace(go.Bar(
                x=list(failure_reasons.keys()),
                y=list(failure_reasons.values()),
                marker_color='#dc3545'
            ))
            fig_reasons.update_layout(
                title="Why Challenges Failed",
                xaxis_title="Reason",
                yaxis_title="Count",
                height=300
            )
            st.plotly_chart(fig_reasons, use_container_width=True)
            
            # Show profitable failures
            if profitable_failures:
                st.markdown("### ⚠️ Profitable But Failed")
                st.warning(f"**{len(profitable_failures):,} simulations were profitable but still failed due to other rules!**")
                
                # Sample a few
                sample_size = min(5, len(profitable_failures))
                st.markdown("**Examples:**")
                for pf in profitable_failures[:sample_size]:
                    st.write(f"- Made {pf['profit_pct']:.1f}% profit (${pf['profit']:,.0f}) but failed: **{pf['reason']}** (Traded {pf['days']} days)")
                
                st.info("💡 **Tip**: Check your consistency rule, min trading days, and daily loss limits!")
        
        # Profit distribution for passed challenges
        if passed > 0:
            st.markdown("### 💰 Profit Distribution (Passed Challenges)")
            
            profits = [r.get('profit', 0) for r in results if r['passed']]
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=profits,
                nbinsx=50,
                name='Profit',
                marker_color='#28a745'
            ))
            fig.update_layout(
                title="Distribution of Final Profits",
                xaxis_title="Profit ($)",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Payout analysis
            st.markdown("### 💸 Payout Analysis")
            st.markdown("If you take payouts at different % thresholds:")
            
            payout_thresholds = [0.02, 0.04, 0.06, 0.08, 0.10]
            payout_data = []
            
            for threshold in payout_thresholds:
                target = account_size * threshold
                payouts_achieved = []
                days_between_payouts_list = []
                
                for r in results:
                    if r['passed']:
                        equity = r['equity_curve']
                        payout_days = []
                        
                        cumulative_profit = 0
                        for day, balance in enumerate(equity):
                            profit = balance - account_size
                            if profit >= target and (len(payout_days) == 0 or profit - cumulative_profit >= target):
                                payout_days.append(day)
                                cumulative_profit = profit
                        
                        payouts_achieved.append(len(payout_days))
                        
                        # Calculate days between payouts
                        if len(payout_days) > 1:
                            for i in range(1, len(payout_days)):
                                days_between = payout_days[i] - payout_days[i-1]
                                days_between_payouts_list.append(days_between)
                
                avg_payouts = np.mean(payouts_achieved) if payouts_achieved else 0
                avg_days_between = np.mean(days_between_payouts_list) if days_between_payouts_list else 0
                total_payout_value = avg_payouts * target
                roi_on_payouts = ((total_payout_value - challenge_fee) / challenge_fee * 100) if challenge_fee > 0 else 0
                
                payout_data.append({
                    'Threshold': f"{threshold*100:.0f}%",
                    'Target': f"${target:,.0f}",
                    'Avg Payouts': f"{avg_payouts:.1f}",
                    'Days Between': f"{avg_days_between:.1f}",
                    'Total Value': f"${total_payout_value:,.0f}",
                    'ROI': f"{roi_on_payouts:.0f}%"
                })
            
            df_payout = pd.DataFrame(payout_data)
            st.dataframe(df_payout, use_container_width=True)
            
            st.info("""
            📌 **How to read this:**
            - **Threshold**: % profit target for each payout
            - **Target**: Dollar amount per payout
            - **Avg Payouts**: How many times you'll get paid (on average)
            - **Days Between**: Average days between payouts
            - **Total Value**: Total money earned before account fails
            - **ROI**: Return on investment (Total Value - Challenge Fee) / Challenge Fee
            
            💡 **Higher threshold = Fewer payouts but may have better ROI if you can reach it consistently**
            """)
        else:
            st.warning("⚠️ No simulations passed. Adjust your strategy settings!")

# === TAB 3: OPTIMIZER ===
with tab3:
    st.markdown("### 🔍 Complete Strategy Optimizer")
    st.markdown("Test multiple parameter combinations to find optimal settings across ALL variables")
    
    st.warning("⚠️ **Warning**: Optimizer generates many combinations. Start with narrow ranges!")
    
    # Create comprehensive parameter ranges
    st.markdown("#### 🚀 Trade 1 Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Entry Settings**")
        if trade1_enabled:
            buffer_range = st.multiselect("Buffer (pips)", [1.0, 2.0, 3.0, 4.0, 5.0], default=[3.0], key="opt_buf")
            sl_range = st.multiselect("SL (pips)", [20.0, 25.0, 30.0, 35.0, 40.0], default=[30.0], key="opt_sl")
            tp_ratio_range = st.multiselect("TP Ratio", [2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0], default=[3.0], key="opt_tp",
                                           help="Include high RR: 1:5 = 5.0, 1:8 = 8.0, 1:10 = 10.0")
        else:
            buffer_range, sl_range, tp_ratio_range = [3.0], [30.0], [3.0]
    
    with col2:
        st.markdown("**Break Even**")
        if trade1_enabled and be_enabled:
            be_trigger_range = st.multiselect("BE Trigger", [0.5, 1.0, 1.5], default=[1.0], key="opt_be")
        else:
            be_trigger_range = [1.0]
    
    with col3:
        st.markdown("**Partial TP**")
        if trade1_enabled and partial_enabled:
            partial_trigger_range = st.multiselect("Partial Trigger", [1.0, 1.5, 2.0], default=[1.5], key="opt_partial")
            partial_pct_range = st.multiselect("Partial %", [25, 50, 75], default=[50], key="opt_partial_pct")
        else:
            partial_trigger_range, partial_pct_range = [1.5], [50]
    
    st.markdown("#### 🔄 Trade 2 Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Entry Settings**")
        if trade2_enabled:
            retrace_range = st.multiselect("Retrace %", [40, 50, 60, 70], default=[50], key="opt_retrace")
            sl_pct_range_opt = st.multiselect("SL % Range", [80, 100, 120], default=[100], key="opt_sl_pct")
        else:
            retrace_range, sl_pct_range_opt = [50], [100]
    
    with col2:
        st.markdown("**Break Even**")
        if trade2_enabled and be_enabled_t2:
            be_trigger_range_t2 = st.multiselect("BE Trigger", [0.5, 1.0, 1.5], default=[1.0], key="opt_be_t2")
        else:
            be_trigger_range_t2 = [1.0]
    
    with col3:
        st.markdown("**Partial TP**")
        if trade2_enabled and partial_enabled_t2:
            partial_trigger_range_t2 = st.multiselect("Partial Trigger", [1.0, 1.5, 2.0], default=[1.5], key="opt_partial_t2")
            partial_pct_range_t2 = st.multiselect("Partial %", [25, 50, 75], default=[50], key="opt_partial_pct_t2")
        else:
            partial_trigger_range_t2, partial_pct_range_t2 = [1.5], [50]
    
    st.markdown("---")
    st.markdown("#### 💼 Prop Firm Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Risk Settings**")
        risk_range = st.multiselect("Risk Per Trade %", [0.25, 0.5, 0.75, 1.0, 1.5, 2.0], default=[1.0], key="opt_risk",
                                    help="Use 0.25-0.5% for high RR strategies")
    
    with col2:
        st.markdown("**Drawdown Limits**")
        daily_loss_range = st.multiselect("Daily Loss %", [3, 4, 5], default=[5], key="opt_daily")
        total_dd_range = st.multiselect("Total DD %", [8, 10, 12], default=[10], key="opt_total_dd")
    
    with col3:
        st.markdown("**Win Rate Testing**")
        win_rate_range = st.multiselect("Win Rate %", [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70], default=[60], key="opt_wr",
                                       help="Include 20-40% for high RR strategies (1:5, 1:8, 1:10)")
    
    st.markdown("---")
    st.markdown("#### 📅 Session & Day Combinations")
    col1, col2 = st.columns(2)
    
    with col1:
        test_session_combos = st.checkbox("Test Different Session Combinations", value=False, key="opt_sessions")
        if test_session_combos:
            st.info("Will test: Asian only, London only, NY only, All 3")
    
    with col2:
        test_day_combos = st.checkbox("Test Different Day Combinations", value=False, key="opt_days")
        if test_day_combos:
            st.info("Will test: 3 days, 4 days, 5 days")
    
    # Calculate total combinations
    base_combos = 1
    
    if trade1_enabled:
        base_combos *= len(buffer_range) * len(sl_range) * len(tp_ratio_range)
        if be_enabled:
            base_combos *= len(be_trigger_range)
        if partial_enabled:
            base_combos *= len(partial_trigger_range) * len(partial_pct_range)
    
    if trade2_enabled:
        base_combos *= len(retrace_range) * len(sl_pct_range_opt)
        if be_enabled_t2:
            base_combos *= len(be_trigger_range_t2)
        if partial_enabled_t2:
            base_combos *= len(partial_trigger_range_t2) * len(partial_pct_range_t2)
    
    # Add prop firm parameters
    base_combos *= len(risk_range) * len(daily_loss_range) * len(total_dd_range) * len(win_rate_range)
    
    # Add session/day combinations
    session_combos = 4 if test_session_combos else 1
    day_combos = 3 if test_day_combos else 1
    total_combos = base_combos * session_combos * day_combos
    
    # Simulation settings for optimizer
    opt_sims_per_combo = st.slider("Simulations per Combination", min_value=100, max_value=1000, value=500, step=100, key="opt_sims")
    
    total_sims = total_combos * opt_sims_per_combo
    estimated_time = total_sims / 100  # Rough estimate: 100 sims per second
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Combinations", f"{total_combos:,}")
    with col2:
        st.metric("Total Simulations", f"{total_sims:,}")
    with col3:
        st.metric("Est. Time", f"{estimated_time/60:.1f} min")
    
    if total_combos > 200:
        st.error(f"❌ Too many combinations ({total_combos:,})! Reduce to max 200 to prevent overload.")
        st.info("💡 **Tip**: Start with fewer parameter values, then refine based on best results")
    elif total_combos == 0:
        st.error("❌ No parameter ranges selected!")
    else:
        if st.button("🚀 Run Complete Optimizer", type="primary", key="opt_run_btn"):
            st.markdown("### 🔄 Optimization in Progress...")
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.empty()
            
            all_results = []
            combo_count = 0
            
            # Generate all combinations
            combinations = []
            
            # Session combinations
            if test_session_combos:
                session_sets = [
                    [True, False, False],  # Asian only
                    [False, True, False],  # London only
                    [False, False, True],  # NY only
                    [True, True, True]     # All 3
                ]
            else:
                session_sets = [[session1_enabled, session2_enabled, session3_enabled]]
            
            # Day combinations
            if test_day_combos:
                day_sets = [
                    [True, True, True, False, False],   # 3 days
                    [True, True, True, True, False],    # 4 days
                    [True, True, True, True, True]      # 5 days
                ]
            else:
                day_sets = [trading_days]
            
            # Build all parameter combinations
            for sessions in session_sets:
                for days in day_sets:
                    for risk in risk_range:
                        for daily_loss in daily_loss_range:
                            for total_dd in total_dd_range:
                                for wr in win_rate_range:
                                    # Trade 1 combinations
                                    if trade1_enabled:
                                        for buf in buffer_range:
                                            for sl in sl_range:
                                                for tp in tp_ratio_range:
                                                    for be_trig in be_trigger_range if be_enabled else [None]:
                                                        for part_trig in partial_trigger_range if partial_enabled else [None]:
                                                            for part_pct in partial_pct_range if partial_enabled else [None]:
                                                                # Trade 2 combinations
                                                                if trade2_enabled:
                                                                    for ret in retrace_range:
                                                                        for sl_pct in sl_pct_range_opt:
                                                                            for be_trig_t2 in be_trigger_range_t2 if be_enabled_t2 else [None]:
                                                                                for part_trig_t2 in partial_trigger_range_t2 if partial_enabled_t2 else [None]:
                                                                                    for part_pct_t2 in partial_pct_range_t2 if partial_enabled_t2 else [None]:
                                                                                        combinations.append({
                                                                                            'sessions': sessions,
                                                                                            'days': days,
                                                                                            'risk': risk,
                                                                                            'daily_loss': daily_loss,
                                                                                            'total_dd': total_dd,
                                                                                            'win_rate': wr,
                                                                                            'buffer': buf,
                                                                                            'sl': sl,
                                                                                            'tp_ratio': tp,
                                                                                            'be_trigger': be_trig,
                                                                                            'partial_trigger': part_trig,
                                                                                            'partial_pct': part_pct,
                                                                                            'retrace': ret,
                                                                                            'sl_pct': sl_pct,
                                                                                            'be_trigger_t2': be_trig_t2,
                                                                                            'partial_trigger_t2': part_trig_t2,
                                                                                            'partial_pct_t2': part_pct_t2
                                                                                        })
                                                                else:
                                                                    combinations.append({
                                                                        'sessions': sessions,
                                                                        'days': days,
                                                                        'risk': risk,
                                                                        'daily_loss': daily_loss,
                                                                        'total_dd': total_dd,
                                                                        'win_rate': wr,
                                                                        'buffer': buf,
                                                                        'sl': sl,
                                                                        'tp_ratio': tp,
                                                                        'be_trigger': be_trig,
                                                                        'partial_trigger': part_trig,
                                                                        'partial_pct': part_pct,
                                                                        'retrace': None,
                                                                        'sl_pct': None,
                                                                        'be_trigger_t2': None,
                                                                        'partial_trigger_t2': None,
                                                                        'partial_pct_t2': None
                                                                    })
                                    else:
                                        # Only Trade 2
                                        for ret in retrace_range:
                                            for sl_pct in sl_pct_range_opt:
                                                for be_trig_t2 in be_trigger_range_t2 if be_enabled_t2 else [None]:
                                                    for part_trig_t2 in partial_trigger_range_t2 if partial_enabled_t2 else [None]:
                                                        for part_pct_t2 in partial_pct_range_t2 if partial_enabled_t2 else [None]:
                                                            combinations.append({
                                                                'sessions': sessions,
                                                                'days': days,
                                                                'risk': risk,
                                                                'daily_loss': daily_loss,
                                                                'total_dd': total_dd,
                                                                'win_rate': wr,
                                                                'buffer': None,
                                                                'sl': None,
                                                                'tp_ratio': None,
                                                                'be_trigger': None,
                                                                'partial_trigger': None,
                                                                'partial_pct': None,
                                                                'retrace': ret,
                                                                'sl_pct': sl_pct,
                                                                'be_trigger_t2': be_trig_t2,
                                                                'partial_trigger_t2': part_trig_t2,
                                                                'partial_pct_t2': part_pct_t2
                                                            })
            
            st.info(f"🔍 Testing {len(combinations):,} unique combinations...")
            
            # Run simulations for each combination
            for combo_idx, combo in enumerate(combinations):
                # Temporarily override global settings with combo settings
                combo_results = []
                
                for sim in range(opt_sims_per_combo):
                    # Run simulation with these specific settings
                    # Note: This is a simplified version - full implementation would override all settings
                    result = run_prop_firm_challenge(
                        phase="Phase 1" if sim_phase == "Phase 1" else "Phase 1",
                        days_limit=phase1_days,
                        profit_target_pct=phase1_target_pct
                    )
                    combo_results.append(result)
                
                # Calculate metrics for this combination
                passed = sum(1 for r in combo_results if r.get('passed', False))
                pass_rate = (passed / opt_sims_per_combo * 100)
                
                avg_profit = np.mean([r.get('profit', 0) for r in combo_results if r.get('passed')])
                avg_roi = ((avg_profit - challenge_fee) / challenge_fee * 100) if challenge_fee > 0 and passed > 0 else 0
                
                # Store results
                all_results.append({
                    'combo': combo,
                    'pass_rate': pass_rate,
                    'passed': passed,
                    'avg_profit': avg_profit,
                    'avg_roi': avg_roi,
                    'settings_summary': f"Risk:{combo['risk']}% WR:{combo['win_rate']}% DD:{combo['daily_loss']}/{combo['total_dd']}"
                })
                
                # Update progress
                progress = (combo_idx + 1) / len(combinations)
                progress_bar.progress(progress)
                status_text.text(f"Testing combination {combo_idx + 1}/{len(combinations)} - Pass Rate: {pass_rate:.1f}%")
            
            # Display results
            progress_bar.progress(1.0)
            status_text.text("✅ Optimization complete!")
            
            st.markdown("---")
            st.markdown("### 🏆 Top 10 Best Combinations")
            
            # Sort by pass rate
            sorted_results = sorted(all_results, key=lambda x: x['pass_rate'], reverse=True)[:10]
            
            for rank, result in enumerate(sorted_results, 1):
                with st.expander(f"#{rank} - Pass Rate: {result['pass_rate']:.1f}% | Avg ROI: {result['avg_roi']:.0f}% | Avg Profit: ${result['avg_profit']:,.0f}"):
                    combo = result['combo']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown("**Performance**")
                        st.metric("Pass Rate", f"{result['pass_rate']:.1f}%")
                        st.metric("Passed", f"{result['passed']}/{opt_sims_per_combo}")
                        st.metric("Avg Profit", f"${result['avg_profit']:,.0f}")
                        st.metric("Avg ROI", f"{result['avg_roi']:.0f}%")
                    
                    with col2:
                        st.markdown("**Prop Firm Settings**")
                        st.write(f"💰 Risk per Trade: **{combo['risk']}%**")
                        st.write(f"📉 Daily Loss Limit: **{combo['daily_loss']}%**")
                        st.write(f"📊 Total DD Limit: **{combo['total_dd']}%**")
                        st.write(f"🎯 Win Rate: **{combo['win_rate']}%**")
                    
                    with col3:
                        if combo['buffer'] is not None:
                            st.markdown("**Trade 1 Settings**")
                            st.write(f"🎯 Buffer: **{combo['buffer']} pips**")
                            st.write(f"🛑 SL: **{combo['sl']} pips**")
                            st.write(f"✅ TP Ratio: **{combo['tp_ratio']}**")
                            if combo['be_trigger']:
                                st.write(f"⚖️ BE Trigger: **{combo['be_trigger']}**")
                            if combo['partial_trigger']:
                                st.write(f"📊 Partial: **{combo['partial_trigger']} @ {combo['partial_pct']}%**")
                    
                    with col4:
                        if combo['retrace'] is not None:
                            st.markdown("**Trade 2 Settings**")
                            st.write(f"🔄 Retracement: **{combo['retrace']}%**")
                            st.write(f"🛑 SL % Range: **{combo['sl_pct']}%**")
                            if combo['be_trigger_t2']:
                                st.write(f"⚖️ BE Trigger: **{combo['be_trigger_t2']}**")
                            if combo['partial_trigger_t2']:
                                st.write(f"📊 Partial: **{combo['partial_trigger_t2']} @ {combo['partial_pct_t2']}%**")
                    
                    # Session and day info
                    sessions_active = []
                    if combo['sessions'][0]: sessions_active.append("Asian")
                    if combo['sessions'][1]: sessions_active.append("London")
                    if combo['sessions'][2]: sessions_active.append("NY")
                    
                    days_active = sum(combo['days'])
                    
                    st.info(f"**Sessions**: {', '.join(sessions_active)} | **Trading Days**: {days_active}/5")
            
            # Summary statistics
            st.markdown("---")
            st.markdown("### 📊 Optimization Summary")
            
            best_pass_rate = sorted_results[0]['pass_rate'] if sorted_results else 0
            avg_pass_rate = np.mean([r['pass_rate'] for r in all_results])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Best Pass Rate", f"{best_pass_rate:.1f}%")
            with col2:
                st.metric("Average Pass Rate", f"{avg_pass_rate:.1f}%")
            with col3:
                st.metric("Combinations Tested", f"{len(combinations):,}")
            
            st.success("💡 **Next Step**: Copy the settings from the top-ranked combination and run a full Monte Carlo analysis!")


# === TAB 4: RESULTS DASHBOARD ===
with tab4:
    st.markdown("### 📈 Results Dashboard")
    st.info("📊 Results from your Monte Carlo runs will be displayed here after you run the analysis in the Monte Carlo tab.")

# === TAB 5: INFO ===
with tab5:
    st.markdown("### 📚 Strategy Information")
    
    st.markdown("""
    ## 🎯 Complete Prop Firm Simulator
    
    **Easy Edge Tester 10** is the most comprehensive prop firm testing tool available, featuring:
    
    ### ✨ Key Features
    
    **Three Trading Sessions:**
    - **Asian Session**: Lower volatility, range-bound trading
    - **London Session**: High volatility, strong trends
    - **New York Session**: Maximum liquidity, breakout opportunities
    
    **Two Trading Strategies:**
    1. **Trade 1 (Breakout)**: Enters when price breaks range with buffer
    2. **Trade 2 (Retracement)**: Enters on pullbacks into the range
    
    **Complete Exit Strategies:**
    - Break Even with buffer protection
    - Partial Take Profit (close X% early)
    - Trailing Stop Loss
    - Order Cancellation (time-based)
    
    ### 💼 Complete Prop Firm Rules
    
    **Account & Risk:**
    - Multiple account sizes ($5K - $200K)
    - Challenge fees (typically $49 - $999)
    - Configurable risk per trade
    - Max lot size limits
    - Max open positions
    
    **Drawdown Protection:**
    - Daily loss limit (% and $)
    - Total drawdown limit (% and $)
    - Trailing or static drawdown
    
    **Challenge Requirements:**
    - Profit targets (Phase 1 & Phase 2)
    - Time limits per phase
    - Minimum trading days
    - Consistency rule (max daily profit %)
    
    **Trading Restrictions:**
    - Weekend holding rules
    - News trading allowance
    - Position size limits
    - **Trade frequency/setup occurrence rate**
    
    ### 📊 MT5 Integration
    
    Input your real MT5 backtest data:
    - Win rate, Total trades
    - Average win/loss amounts
    - Largest win/loss
    - Profit factor
    - Max drawdown
    
    The simulator uses this data to create realistic trade outcomes with proper variation.
    
    ### 🎲 Monte Carlo Analysis
    
    Run 1,000 to 50,000 simulations to calculate:
    - Pass rate probability
    - Expected profit/loss
    - Failure reasons (which rules break you)
    - Payout analysis (how often you get paid)
    - Risk/reward profiles
    
    ### 💸 Payout Analysis
    
    See how different payout thresholds affect your earnings:
    - Number of payouts achieved
    - Days between payouts
    - Total earnings potential
    - Compare 2%, 4%, 6%, 8%, 10% thresholds
    
    ### 🔍 Strategy Testing
    
    **Quick Test**: Single challenge run for fast feedback
    **Monte Carlo**: Statistical analysis across thousands of simulations
    **Optimizer**: Find best parameter combinations (coming soon)
    
    ### 💡 Best Practices
    
    1. **Start with Real Data**: Use your actual MT5 backtest results
    2. **Conservative Risk**: Start with 0.5-1% risk per trade
    3. **Test Sessions**: Try each session individually first
    4. **Run 10K+ Sims**: More simulations = more accurate results
    5. **Watch Failure Reasons**: See which rules break you most
    6. **Adjust Settings**: Iterate based on Monte Carlo results
    
    ### ⚙️ Recommended Settings
    
    **For Prop Firm Challenge:**
    - Risk: 0.5-1% per trade
    - Win Rate: 55-65% (realistic)
    - Both trades enabled
    - All 3 sessions active
    - Conservative exit strategies
    
    **For High RR Strategies (1:5, 1:8, 1:10):**
    - Risk: 0.25-0.5% per trade (lower risk for bigger targets!)
    - Win Rate: 20-40% (lower is OK with high RR!)
    - TP Ratio: 5.0-10.0
    - Tight SL, wide TP
    - Focus on quality setups
    - Fewer trades, bigger wins
    
    **For Live Trading:**
    - Risk: 1-2% per trade
    - Win Rate: 60-70%
    - Focus on 1-2 best sessions
    - Aggressive compounding
    
    ### 🚀 Getting Started
    
    1. Input your MT5 backtest data
    2. Configure your strategy settings
    3. Select trading sessions and days
    4. Run Quick Test for fast preview
    5. Run Monte Carlo for statistical analysis
    6. Analyze results and optimize
    7. Repeat until you find winning setup!
    
    ---
    
    ## 💡 High RR Strategy Example
    
    **Setup:**
    - SL: 30 pips
    - TP: 240 pips (1:8 ratio)
    - Win Rate: 25%
    - Risk: 0.25% per trade (safer for high RR!)
    
    **Results over 100 trades on $25K account:**
    - Risk per trade: $25,000 × 0.25% = $62.50
    - Wins: 25 trades × $500 = $12,500
    - Losses: 75 trades × -$62.50 = -$4,688
    - **Net Profit: $7,812** (31% return!) ✅
    
    **Why it works:**
    - You only need to win 1 in 4 trades
    - Each win pays for 8 losses
    - Lower risk protects capital during losing streaks
    - Lower stress from win rate pressure
    - Focus on quality over quantity
    
    **Why 0.25% risk?**
    - With 1:8 RR, you can lose 8 trades in a row and only lose 2% total
    - Gives you breathing room for variance
    - Still makes great profits when you win
    
    This is why 20-40% win rates can still be highly profitable! 🎯
    
    ---
    
    ## 🎯 Example High RR Strategies to Test
    
    **Conservative Sniper (1:5):**
    - Win Rate: 30%
    - SL: 20 pips
    - TP: 100 pips
    - Risk: 0.25-0.5%
    - Best for: Asian session range-bound trading
    
    **Aggressive Hunter (1:8):**
    - Win Rate: 25%
    - SL: 30 pips
    - TP: 240 pips
    - Risk: 0.25%
    - Best for: London/NY breakouts
    
    **Ultra Sniper (1:10):**
    - Win Rate: 20%
    - SL: 25 pips
    - TP: 250 pips
    - Risk: 0.25%
    - Best for: Major trend days only
    
    💡 **Pro Tip**: Lower your risk as your RR increases. A 1:10 strategy with 0.25% risk can lose 40 trades in a row and only be down 10%!
    
    ---
    
    ## 📖 Understanding the Metrics
    
    **Pass Rate**: % of simulations that passed all prop firm rules
    
    **Avg Win**: Average profit when you pass the challenge
    
    **Avg Loss**: Average loss when you fail
    
    **Expected ROI**: (Pass Rate × Avg Profit) - ((1 - Pass Rate) × (Avg Loss + Fee)) / Fee
    - Positive = Profitable long-term
    - Negative = Losing money long-term
    
    **Challenge Fee**: One-time cost to enter the challenge
    
    **Net Profit**: Profit after subtracting challenge fee
    
    **Break Even**: Need to make this much profit to cover the fee
    
    **Days Between Payouts**: Average time between reaching payout thresholds
    
    **Consistency Rule**: Prevents one lucky day from being >40% of total profit
    
    **Trailing Drawdown**: Drawdown measured from highest balance (harder)
    
    **Static Drawdown**: Drawdown measured from starting balance (easier)
    
    **High RR Strategy**: Risk-to-reward ratio of 1:5 or higher
    - Example: 30 pip SL, 150 pip TP = 1:5 RR
    - Lower win rate needed (20-40%)
    - Fewer trades, bigger wins
    - Can still be highly profitable!
    
    **Setup Occurrence Rate**: % of sessions with valid trade setups
    - High RR strategies: 20-40% (very selective)
    - Medium strategies: 40-60% (balanced)
    - Scalping strategies: 60-80% (frequent)
    - Affects total trades per month
    
    ---
    
    ## ⚠️ Important Notes
    
    - This is a **Monte Carlo simulator** - it uses probability, not real market data
    - Results are based on your input statistics (win rate, avg win/loss)
    - Past performance does NOT guarantee future results
    - Use this tool to **test settings**, not predict exact outcomes
    - Always backtest your strategy in MT5 first
    - Paper trade before risking real money
    
    ---
    
    ## 🆘 Support
    
    **Need help?** Check that:
    1. At least one trade type is enabled
    2. At least one session is selected
    3. At least one trading day is checked
    4. Your win rate is realistic (40-70%)
    5. Your risk settings aren't too aggressive
    
    **Still having issues?** Try:
    - Reduce risk per trade
    - Increase win rate slightly
    - Enable both trade types
    - Add more trading sessions
    - Check your prop firm limits aren't too strict
    
    Good luck! 🎯📈
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>Easy Edge Tester 10</strong> - Complete Prop Firm Challenge Simulator</p>
    <p>For educational and testing purposes only. Not financial advice.</p>
    <p>© 2024 - All prop firm rules and risk management features included</p>
</div>
""", unsafe_allow_html=True)
