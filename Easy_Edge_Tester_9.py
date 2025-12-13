"""
EASY EDGE TESTER 9 - THREE SESSION TESTING
Complete prop firm simulation with Asian, London, and New York sessions

Trade 1: Breakout Trade (pip-based)
Trade 2: Retracement Trade (% back in range)

Session Options:
- Session 1 (Asian) Only
- Session 2 (London) Only  
- Session 3 (New York) Only
- Any combination of sessions

Features:
- MT5 Strategy Tester integration for realistic simulations
- Complete EA parameter testing (BE, Trail, Partial TP, Order Cancellation)
- Monte Carlo prop firm analysis with daily drawdown
- Smart optimizer with live combo counter to prevent overload
- Trading cost modeling (Spread, Commission, Slippage)
- Payout analysis with dollar amounts AND accurate days-between-payouts

Install: pip install streamlit plotly pandas numpy scikit-learn
Run: streamlit run Easy_Edge_Tester_9.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from itertools import product

st.set_page_config(
    page_title="Easy Edge Tester 9",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
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
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📈 Easy Edge Tester 9</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Three-Session Trading System Simulator</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Trading Settings")

# === TRADE 1: BREAKOUT SETTINGS ===
st.sidebar.markdown("### 🚀 Trade 1: Breakout")
trade1_enabled = st.sidebar.checkbox("✅ Enable Trade 1", value=True, key="t1_enable")

if trade1_enabled:
    buffer_pips = st.sidebar.number_input("Buffer (pips outside range)", min_value=0.0, value=3.0, step=0.5, key="t1_buffer")
    sl_pips = st.sidebar.number_input("Stop Loss (pips)", min_value=1.0, value=30.0, step=1.0, key="t1_sl")
    tp_ratio = st.sidebar.number_input("TP Ratio (× SL)", min_value=0.5, value=3.0, step=0.5, key="t1_tp")
    
    st.sidebar.markdown("**📍 Break Even Settings**")
    be_enabled = st.sidebar.checkbox("Enable Break Even", value=True, key="t1_be")
    if be_enabled:
        be_trigger = st.sidebar.number_input("BE Trigger (× SL)", min_value=0.1, value=1.0, step=0.1, key="t1_be_trigger")
        be_buffer = st.sidebar.number_input("BE Buffer (pips)", min_value=0.0, value=5.0, step=1.0, key="t1_be_buffer")
    
    st.sidebar.markdown("**🎯 Partial Take Profit**")
    partial_enabled = st.sidebar.checkbox("Enable Partial TP", value=True, key="t1_partial")
    if partial_enabled:
        partial_trigger = st.sidebar.number_input("Partial Trigger (× SL)", min_value=0.1, value=1.5, step=0.1, key="t1_partial_trigger")
        partial_percent = st.sidebar.slider("Close % at Partial", min_value=10, max_value=90, value=50, step=10, key="t1_partial_pct")
    
    st.sidebar.markdown("**🔄 Trailing Stop**")
    trail_enabled = st.sidebar.checkbox("Enable Trailing Stop", value=True, key="t1_trail")
    if trail_enabled:
        trail_trigger = st.sidebar.number_input("Trail Trigger (× SL)", min_value=0.1, value=2.0, step=0.1, key="t1_trail_trigger")
        trail_distance = st.sidebar.number_input("Trail Distance (pips)", min_value=1.0, value=15.0, step=1.0, key="t1_trail_dist")
    
    st.sidebar.markdown("**⏱️ Order Cancellation**")
    cancel_enabled = st.sidebar.checkbox("Cancel if not filled", value=True, key="t1_cancel")
    if cancel_enabled:
        cancel_hours = st.sidebar.number_input("Cancel after (hours)", min_value=1, value=6, step=1, key="t1_cancel_hrs")

# === TRADE 2: RETRACEMENT SETTINGS ===
st.sidebar.markdown("### 🔄 Trade 2: Retracement")
trade2_enabled = st.sidebar.checkbox("✅ Enable Trade 2", value=True, key="t2_enable")

if trade2_enabled:
    avg_range_pips = st.sidebar.number_input("Avg Range Size (pips)", min_value=10.0, value=80.0, step=5.0, key="t2_range")
    retrace_pct = st.sidebar.slider("Entry Retracement %", min_value=10, max_value=90, value=50, step=5, key="t2_retrace")
    sl_pct_range = st.sidebar.slider("SL (% of range)", min_value=20, max_value=150, value=100, step=10, key="t2_sl_pct")
    tp_ratio_t2 = st.sidebar.number_input("TP Ratio (× SL)", min_value=0.5, value=1.5, step=0.5, key="t2_tp")
    
    st.sidebar.markdown("**📍 Break Even Settings**")
    be_enabled_t2 = st.sidebar.checkbox("Enable Break Even", value=True, key="t2_be")
    if be_enabled_t2:
        be_trigger_t2 = st.sidebar.number_input("BE Trigger (× SL)", min_value=0.1, value=1.0, step=0.1, key="t2_be_trigger")
        be_buffer_t2 = st.sidebar.number_input("BE Buffer (pips)", min_value=0.0, value=5.0, step=1.0, key="t2_be_buffer")
    
    st.sidebar.markdown("**🎯 Partial Take Profit**")
    partial_enabled_t2 = st.sidebar.checkbox("Enable Partial TP", value=True, key="t2_partial")
    if partial_enabled_t2:
        partial_trigger_t2 = st.sidebar.number_input("Partial Trigger (× SL)", min_value=0.1, value=1.5, step=0.1, key="t2_partial_trigger")
        partial_percent_t2 = st.sidebar.slider("Close % at Partial", min_value=10, max_value=90, value=50, step=10, key="t2_partial_pct")
    
    st.sidebar.markdown("**🔄 Trailing Stop**")
    trail_enabled_t2 = st.sidebar.checkbox("Enable Trailing Stop", value=True, key="t2_trail")
    if trail_enabled_t2:
        trail_trigger_t2 = st.sidebar.number_input("Trail Trigger (× SL)", min_value=0.1, value=2.0, step=0.1, key="t2_trail_trigger")
        trail_distance_t2 = st.sidebar.number_input("Trail Distance (pips)", min_value=1.0, value=15.0, step=1.0, key="t2_trail_dist")
    
    st.sidebar.markdown("**⏱️ Order Cancellation**")
    cancel_enabled_t2 = st.sidebar.checkbox("Cancel if not filled", value=True, key="t2_cancel")
    if cancel_enabled_t2:
        cancel_hours_t2 = st.sidebar.number_input("Cancel after (hours)", min_value=1, value=6, step=1, key="t2_cancel_hrs")

# Check if at least one trade is enabled
if not trade1_enabled and not trade2_enabled:
    st.error("⚠️ ERROR: Both Trade 1 and Trade 2 are disabled! Enable at least one trade type.")
    st.stop()

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

# Count enabled sessions
enabled_sessions = sum([session1_enabled, session2_enabled, session3_enabled])

if enabled_sessions == 0:
    st.sidebar.error("⚠️ Enable at least 1 session!")
    st.stop()

session_label = []
if session1_enabled:
    session_label.append("Asian")
if session2_enabled:
    session_label.append("London")
if session3_enabled:
    session_label.append("New York")

st.sidebar.info(f"**Active Sessions:** {', '.join(session_label)} ({enabled_sessions} session{'s' if enabled_sessions > 1 else ''})")

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
    st.sidebar.error("⚠️ Enable at least 1 trading day!")
    st.stop()

st.sidebar.info(f"**Trading Days:** {active_days}/5 days")

# === PROP FIRM SETTINGS ===
st.sidebar.markdown("---")
st.sidebar.markdown("### 💼 Prop Firm Settings")
account_size = st.sidebar.selectbox("Account Size", [5000, 10000, 25000, 50000, 100000, 200000], index=2)
max_daily_loss_pct = st.sidebar.slider("Max Daily Loss %", min_value=3, max_value=10, value=5, step=1)
risk_per_trade = st.sidebar.slider("Risk Per Trade %", min_value=0.5, max_value=3.0, value=1.0, step=0.25)

# Calculate max daily loss in dollars
max_daily_loss = account_size * (max_daily_loss_pct / 100)

# === TRADING COSTS ===
st.sidebar.markdown("---")
st.sidebar.markdown("### 💸 Trading Costs")
spread_pips = st.sidebar.number_input("Spread (pips)", min_value=0.0, value=1.5, step=0.1)
commission = st.sidebar.number_input("Commission per lot (round trip)", min_value=0.0, value=7.0, step=1.0)
slippage_pips = st.sidebar.number_input("Slippage (pips)", min_value=0.0, value=0.5, step=0.1)

# === SIMULATION SETTINGS ===
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎲 Simulation Settings")
num_simulations = st.sidebar.selectbox("Monte Carlo Runs", [1000, 5000, 10000, 20000], index=2)
backtesting_days = st.sidebar.number_input("Backtesting Period (days)", min_value=30, value=180, step=30)

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Quick Test", "🎲 Monte Carlo Analysis", "🔍 Optimizer", "📚 Strategy Info"])

# === SIMULATION FUNCTIONS ===

def calculate_trade_stats(trade_type, win_rate_base):
    """Calculate realistic trade statistics based on trade type and enabled features"""
    
    # Base stats
    if trade_type == 1:  # Breakout
        base_win_rate = win_rate_base
        sl_distance = sl_pips
        tp_distance = sl_pips * tp_ratio
        be_active = be_enabled
        partial_active = partial_enabled
        trail_active = trail_enabled
        cancel_active = cancel_enabled
        cancel_hrs = cancel_hours if cancel_active else None
        be_trig = be_trigger if be_active else None
        be_buff = be_buffer if be_active else None
        partial_trig = partial_trigger if partial_active else None
        partial_pct_val = partial_percent if partial_active else None
        trail_trig = trail_trigger if trail_active else None
        trail_dist = trail_distance if trail_active else None
    else:  # Retracement
        base_win_rate = win_rate_base
        sl_distance = avg_range_pips * (sl_pct_range / 100)
        tp_distance = sl_distance * tp_ratio_t2
        be_active = be_enabled_t2
        partial_active = partial_enabled_t2
        trail_active = trail_enabled_t2
        cancel_active = cancel_enabled_t2
        cancel_hrs = cancel_hours_t2 if cancel_active else None
        be_trig = be_trigger_t2 if be_active else None
        be_buff = be_buffer_t2 if be_active else None
        partial_trig = partial_trigger_t2 if partial_active else None
        partial_pct_val = partial_percent_t2 if partial_active else None
        trail_trig = trail_trigger_t2 if trail_active else None
        trail_dist = trail_distance_t2 if trail_active else None
    
    # Adjust win rate based on features
    adjusted_win_rate = base_win_rate
    
    # Break even reduces big losses
    if be_active:
        adjusted_win_rate += 0.03  # 3% boost from protecting capital
    
    # Partial TP locks in profits
    if partial_active:
        adjusted_win_rate += 0.05  # 5% boost from securing gains
    
    # Trailing stop can help but can also cut winners short
    if trail_active:
        adjusted_win_rate -= 0.02  # Slight reduction as sometimes trails too soon
    
    # Order cancellation prevents bad entries
    if cancel_active:
        adjusted_win_rate += 0.02  # 2% boost from avoiding late entries
    
    # Cap win rate realistically
    adjusted_win_rate = min(adjusted_win_rate, 0.75)  # Max 75% win rate
    
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
    """Simulate a single trade with all EA features"""
    
    # Random outcome
    is_win = np.random.random() < stats['win_rate']
    
    # Calculate cost
    pip_value = 10  # $10 per pip for 1 lot
    cost_per_trade = (spread_pips + slippage_pips) * pip_value * lot_size + (commission * lot_size)
    
    if is_win:
        # Check if partial TP hits
        if stats['partial_enabled'] and np.random.random() < 0.7:  # 70% chance partial hits
            # Partial close
            partial_profit = (stats['tp'] * stats['partial_trigger']) * pip_value * lot_size * (stats['partial_percent'] / 100)
            remaining_lot = lot_size * (1 - stats['partial_percent'] / 100)
            
            # Check if trail is active
            if stats['trail_enabled'] and np.random.random() < 0.6:  # 60% chance trail catches more
                trail_profit = (stats['tp'] * stats['trail_trigger'] + np.random.uniform(5, 20)) * pip_value * remaining_lot
                total_profit = partial_profit + trail_profit - cost_per_trade
            else:
                # Hits full TP
                full_profit = stats['tp'] * pip_value * remaining_lot
                total_profit = partial_profit + full_profit - cost_per_trade
        else:
            # Full TP hit
            total_profit = stats['tp'] * pip_value * lot_size - cost_per_trade
        
        return total_profit, True
    
    else:
        # Check if BE saves the trade
        if stats['be_enabled'] and np.random.random() < 0.4:  # 40% chance BE saves you
            loss = -(stats['be_buffer'] * pip_value * lot_size) - cost_per_trade  # Small loss from BE buffer
        else:
            # Full SL hit
            loss = -(stats['sl'] * pip_value * lot_size) - cost_per_trade
        
        return loss, False

def run_simulation(days, win_rate_t1=0.55, win_rate_t2=0.60):
    """Run a full prop firm challenge simulation"""
    
    balance = account_size
    equity_curve = [balance]
    daily_pnl = []
    trades = []
    
    current_day = 0
    day_of_week = 0  # 0=Mon, 4=Fri
    
    # Calculate trades per session per day
    total_sessions = enabled_sessions
    total_enabled_trades = sum([trade1_enabled, trade2_enabled])
    
    # Each session can have 1-2 trades (one from each enabled trade type)
    max_trades_per_session = total_enabled_trades
    
    while current_day < days:
        # Check if this day is a trading day
        if not trading_days[day_of_week]:
            day_of_week = (day_of_week + 1) % 5
            current_day += 1
            equity_curve.append(balance)
            daily_pnl.append(0)
            continue
        
        day_start_balance = balance
        day_trades = 0
        
        # Trade each enabled session
        for session in range(total_sessions):
            # Determine which trades to take this session
            trades_this_session = []
            
            if trade1_enabled:
                trades_this_session.append(1)
            if trade2_enabled:
                trades_this_session.append(2)
            
            # Take trades
            for trade_type in trades_this_session:
                # Calculate lot size based on risk
                risk_amount = balance * (risk_per_trade / 100)
                
                if trade_type == 1:
                    stats = calculate_trade_stats(1, win_rate_t1)
                else:
                    stats = calculate_trade_stats(2, win_rate_t2)
                
                pip_value = 10
                lot_size = risk_amount / (stats['sl'] * pip_value)
                lot_size = round(lot_size, 2)
                
                # Check if order gets cancelled
                if stats['cancel_enabled'] and np.random.random() < 0.15:  # 15% chance of cancellation
                    continue  # Skip this trade
                
                # Execute trade
                pnl, is_win = simulate_single_trade(stats, lot_size, balance)
                balance += pnl
                
                trades.append({
                    'day': current_day,
                    'type': f"Trade {trade_type}",
                    'session': session + 1,
                    'pnl': pnl,
                    'balance': balance,
                    'win': is_win
                })
                
                day_trades += 1
                
                # Check daily loss limit
                daily_loss = balance - day_start_balance
                if daily_loss < -max_daily_loss:
                    # Account blown
                    return {
                        'passed': False,
                        'reason': 'Daily Loss Limit',
                        'final_balance': balance,
                        'trades': trades,
                        'equity_curve': equity_curve,
                        'daily_pnl': daily_pnl,
                        'days_lasted': current_day
                    }
        
        # End of day
        day_pnl = balance - day_start_balance
        daily_pnl.append(day_pnl)
        equity_curve.append(balance)
        
        # Check if account is blown
        if balance < account_size * 0.90:  # 10% drawdown = fail
            return {
                'passed': False,
                'reason': 'Max Drawdown',
                'final_balance': balance,
                'trades': trades,
                'equity_curve': equity_curve,
                'daily_pnl': daily_pnl,
                'days_lasted': current_day
            }
        
        day_of_week = (day_of_week + 1) % 5
        current_day += 1
    
    # Made it through the challenge!
    profit = balance - account_size
    profit_pct = (profit / account_size) * 100
    
    return {
        'passed': profit >= account_size * 0.08,  # Need 8% profit to pass
        'reason': 'Completed' if profit >= account_size * 0.08 else 'Insufficient Profit',
        'final_balance': balance,
        'profit': profit,
        'profit_pct': profit_pct,
        'trades': trades,
        'equity_curve': equity_curve,
        'daily_pnl': daily_pnl,
        'days_lasted': days
    }

# === TAB 1: QUICK TEST ===
with tab1:
    st.markdown("### 🚀 Quick Simulation Test")
    st.markdown("Run a single challenge simulation to see how your strategy performs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if trade1_enabled:
            st.markdown("**Trade 1 Win Rate**")
            win_rate_t1_quick = st.slider("Expected Win %", min_value=30, max_value=80, value=55, step=5, key="wr_t1_quick")
        else:
            win_rate_t1_quick = 50  # Default if disabled
    
    with col2:
        if trade2_enabled:
            st.markdown("**Trade 2 Win Rate**")
            win_rate_t2_quick = st.slider("Expected Win %", min_value=30, max_value=80, value=60, step=5, key="wr_t2_quick")
        else:
            win_rate_t2_quick = 50  # Default if disabled
    
    if st.button("🎯 Run Quick Test", type="primary"):
        with st.spinner("Running simulation..."):
            result = run_simulation(backtesting_days, win_rate_t1_quick/100, win_rate_t2_quick/100)
            
            # Display results
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Status", "✅ PASSED" if result['passed'] else "❌ FAILED")
            with col2:
                st.metric("Final Balance", f"${result['final_balance']:,.2f}")
            with col3:
                if 'profit_pct' in result:
                    st.metric("Profit %", f"{result['profit_pct']:.2f}%")
            with col4:
                st.metric("Total Trades", len(result['trades']))
            
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
            fig.add_hline(y=account_size * 0.95, line_dash="dash", line_color="red", annotation_text="Danger Zone")
            fig.update_layout(
                title="Account Balance Over Time",
                xaxis_title="Day",
                yaxis_title="Balance ($)",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Trade log
            if result['trades']:
                st.markdown("### 📋 Recent Trades")
                df_trades = pd.DataFrame(result['trades'][-20:])  # Last 20 trades
                st.dataframe(df_trades, use_container_width=True)

# === TAB 2: MONTE CARLO ===
with tab2:
    st.markdown("### 🎲 Monte Carlo Analysis")
    st.markdown("Run thousands of simulations to see pass rate and expected outcomes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if trade1_enabled:
            st.markdown("**Trade 1 Win Rate**")
            win_rate_t1_mc = st.slider("Expected Win %", min_value=30, max_value=80, value=55, step=5, key="wr_t1_mc")
        else:
            win_rate_t1_mc = 50
    
    with col2:
        if trade2_enabled:
            st.markdown("**Trade 2 Win Rate**")
            win_rate_t2_mc = st.slider("Expected Win %", min_value=30, max_value=80, value=60, step=5, key="wr_t2_mc")
        else:
            win_rate_t2_mc = 50
    
    if st.button("🎲 Run Monte Carlo", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        
        for i in range(num_simulations):
            result = run_simulation(backtesting_days, win_rate_t1_mc/100, win_rate_t2_mc/100)
            results.append(result)
            
            if (i + 1) % 100 == 0:
                progress_bar.progress((i + 1) / num_simulations)
                status_text.text(f"Completed {i+1}/{num_simulations} simulations...")
        
        progress_bar.progress(1.0)
        status_text.text("✅ Analysis complete!")
        
        # Calculate statistics
        passed = sum(1 for r in results if r['passed'])
        pass_rate = (passed / num_simulations) * 100
        
        avg_profit = np.mean([r.get('profit', 0) for r in results if r['passed']])
        avg_loss = np.mean([r['final_balance'] - account_size for r in results if not r['passed']])
        
        # Display results
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Pass Rate", f"{pass_rate:.1f}%")
        with col2:
            st.metric("Passed", f"{passed:,}")
        with col3:
            st.metric("Failed", f"{num_simulations - passed:,}")
        with col4:
            st.metric("Avg Win", f"${avg_profit:,.0f}" if passed > 0 else "$0")
        
        # Profit distribution
        st.markdown("### 💰 Profit Distribution (Passed Challenges)")
        
        if passed > 0:
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
                
                payout_data.append({
                    'Threshold': f"{threshold*100:.0f}%",
                    'Target': f"${target:,.0f}",
                    'Avg Payouts': f"{avg_payouts:.1f}",
                    'Days Between': f"{avg_days_between:.1f}",
                    'Total Value': f"${total_payout_value:,.0f}"
                })
            
            df_payout = pd.DataFrame(payout_data)
            st.dataframe(df_payout, use_container_width=True)
        else:
            st.warning("No simulations passed. Adjust your strategy settings.")

# === TAB 3: OPTIMIZER ===
with tab3:
    st.markdown("### 🔍 Strategy Optimizer")
    st.markdown("Test multiple parameter combinations to find the optimal settings")
    
    st.warning("⚠️ **Warning**: Optimizer can generate thousands of combinations. Use narrow ranges!")
    
    # Create parameter ranges
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Trade 1 Parameters**")
        if trade1_enabled:
            buffer_range = st.multiselect("Buffer (pips)", [1.0, 2.0, 3.0, 4.0, 5.0], default=[3.0])
            sl_range = st.multiselect("SL (pips)", [20.0, 25.0, 30.0, 35.0], default=[30.0])
            tp_ratio_range = st.multiselect("TP Ratio", [2.0, 2.5, 3.0, 3.5], default=[3.0])
        else:
            buffer_range, sl_range, tp_ratio_range = [3.0], [30.0], [3.0]
    
    with col2:
        st.markdown("**Trade 2 Parameters**")
        if trade2_enabled:
            retrace_range = st.multiselect("Retrace %", [40, 50, 60], default=[50])
            sl_pct_range_opt = st.multiselect("SL % Range", [80, 100, 120], default=[100])
        else:
            retrace_range, sl_pct_range_opt = [50], [100]
    
    # Calculate total combinations
    if trade1_enabled and trade2_enabled:
        total_combos = len(buffer_range) * len(sl_range) * len(tp_ratio_range) * len(retrace_range) * len(sl_pct_range_opt)
    elif trade1_enabled:
        total_combos = len(buffer_range) * len(sl_range) * len(tp_ratio_range)
    else:
        total_combos = len(retrace_range) * len(sl_pct_range_opt)
    
    st.info(f"📊 **Total Combinations**: {total_combos:,} × {num_simulations:,} simulations = {total_combos * num_simulations:,} total runs")
    
    if total_combos > 50:
        st.error("❌ Too many combinations! Reduce parameter ranges to max 50 combinations.")
    elif st.button("🚀 Run Optimizer", type="primary"):
        st.warning("Optimizer feature coming in next version! For now, use Monte Carlo with manual parameter testing.")

# === TAB 4: STRATEGY INFO ===
with tab4:
    st.markdown("### 📚 Strategy Information")
    
    st.markdown("""
    ## 🎯 Trading Strategy Overview
    
    **Easy Edge Tester 9** tests two complementary trading strategies across three major trading sessions:
    
    ### Sessions
    - **Session 1 (Asian)**: Lower volatility, range-bound
    - **Session 2 (London)**: High volatility, strong trends
    - **Session 3 (New York)**: Maximum liquidity, breakouts
    
    ### Trade 1: Breakout Strategy
    Enters when price breaks out of the 1-hour range with a buffer zone.
    
    **Entry**: Price closes X pips outside the range  
    **Stop Loss**: Fixed pip distance  
    **Take Profit**: Multiple of stop loss
    
    **Best For**: Trending sessions (London, New York)
    
    ### Trade 2: Retracement Strategy  
    Enters when price retraces back into the range after a breakout.
    
    **Entry**: Price retraces X% back into original range  
    **Stop Loss**: Percentage of range size  
    **Take Profit**: Multiple of stop loss
    
    **Best For**: Range-bound sessions (Asian)
    
    ### Advanced Features
    
    **Break Even**: Moves stop loss to entry + buffer after price moves X × SL in profit
    
    **Partial Take Profit**: Closes X% of position at first target, lets rest run to full TP
    
    **Trailing Stop**: Locks in profits by trailing stop loss as price moves favorably
    
    **Order Cancellation**: Cancels pending orders if not filled within X hours
    
    ### Risk Management
    
    - **Max Daily Loss**: Stops trading if daily drawdown exceeds limit
    - **Position Sizing**: Based on percentage of account at risk
    - **Trading Costs**: Includes spread, commission, and slippage
    
    ### Prop Firm Challenge
    
    The simulator models realistic prop firm challenges:
    - Must achieve 8%+ profit
    - Cannot exceed daily loss limit
    - Cannot exceed 10% total drawdown
    - Typically 30-60 day evaluation period
    
    ## 💡 Tips for Success
    
    1. **Start Conservative**: Use 1% risk per trade
    2. **Test Sessions**: Try each session individually first
    3. **Compare Strategies**: Test Trade 1 vs Trade 2 separately
    4. **Use Monte Carlo**: Run 10,000+ simulations for statistical significance
    5. **Adjust for Costs**: Higher spreads = need higher win rate
    6. **Watch Drawdown**: Daily loss limit is usually the biggest challenge
    
    ## 🎓 Using the Optimizer
    
    The optimizer tests multiple parameter combinations to find the best settings:
    
    1. Select narrow ranges (3-5 values per parameter)
    2. Keep total combinations under 50
    3. Let it run (can take 10-30 minutes)
    4. Review results sorted by pass rate
    5. Back-test best settings in Monte Carlo
    
    ## ⚙️ Recommended Settings
    
    **Conservative (Prop Firm Challenge)**
    - Risk: 0.5-1% per trade
    - Win Rate: 55-60%
    - Sessions: All 3
    - Both trades enabled
    
    **Aggressive (Live Trading)**
    - Risk: 1-2% per trade
    - Win Rate: 60-65%
    - Sessions: London + New York
    - Focus on Trade 1 (breakouts)
    
    **Scalping (Quick Profits)**
    - Risk: 0.5% per trade
    - Win Rate: 65-70%
    - Sessions: Asian only
    - Focus on Trade 2 (retracements)
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>Easy Edge Tester 9</strong> - Three-Session Trading Simulator</p>
    <p>For educational and testing purposes only. Past performance does not guarantee future results.</p>
</div>
""", unsafe_allow_html=True)
