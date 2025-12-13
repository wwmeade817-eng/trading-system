"""
EASY EDGE TESTER 2 - SESSION-SPECIFIC TRADING SYSTEM
Trade optimization with flexible session selection

Trade 1: Breakout Trade (pip-based)
Trade 2: Retracement Trade (% back in range)

Session Options:
- Session 1 Only
- Session 2 Only  
- Both Sessions

Complete EA simulation with ALL parameters
Monte Carlo with comprehensive prop firm analysis

Install: pip install streamlit plotly pandas numpy scikit-learn
Run: streamlit run Easy_Edge_Tester_2.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Easy Edge Tester 2",
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
    .trade-section {
        background-color: #e7f3ff;
        border: 2px solid #2196F3;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .session-info {
        background-color: #f0f0f0;
        border-left: 5px solid #2196F3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


class Complete2TradeEASimulator:
    """
    Complete EA simulator with 2-trade system:
    - Trade 1: Breakout (pip-based entry)
    - Trade 2: Retracement (% back in range)
    
    Each trade has full exit logic:
    - Break Even with buffer
    - Trailing Stop
    - Partial Take Profit
    
    NEW: Session mode selection
    """
    
    def __init__(self, stats, trade1_settings, trade2_settings, global_settings, session_mode='both'):
        self.stats = stats
        self.trade1 = trade1_settings
        self.trade2 = trade2_settings
        self.global_settings = global_settings
        self.session_mode = session_mode  # 'session1', 'session2', or 'both'
        
        self.total_trades = stats['total_trades']
        self.win_rate = stats['win_rate'] / 100
        self.account_size = stats.get('account_size', 100000)
    
    def simulate_single_trade(self, is_trade1=True):
        """Simulate one trade with complete EA logic"""
        
        settings = self.trade1 if is_trade1 else self.trade2
        
        # Check if trade passes filters
        if not self._passes_filters(settings):
            return None, None
        
        # ORDER CANCELLATION - Check if order gets filled
        if settings.get('use_cancel', False):
            cancel_hours = settings.get('cancel_hours', 4.0)
            # Simulate probability of order filling before cancellation
            # Shorter time = lower fill probability
            fill_probability = min(0.95, 0.3 + (cancel_hours / 24) * 0.65)
            
            if np.random.random() > fill_probability:
                # Order cancelled before fill
                return None, None
        
        # Determine if potential winner
        is_potential_win = np.random.random() < self.win_rate
        
        sl_pips = settings['sl_pips']
        tp_ratio = settings['tp_ratio']
        tp_pips = sl_pips * tp_ratio
        
        # Calculate position size based on risk
        risk_pct = settings['risk_pct']
        risk_amount = self.account_size * (risk_pct / 100)
        position_size = risk_amount / (sl_pips * 10)  # $10 per pip
        
        if not is_potential_win:
            # Loser - hits SL
            return -risk_amount, False
        
        # Winner - simulate price movement
        max_move = tp_pips * 1.3
        favorable_move = np.random.uniform(tp_pips * 0.4, max_move)
        
        current_profit = 0
        remaining_position = 1.0
        stop_at_be = False
        
        # 1. BREAK EVEN
        if settings['use_breakeven']:
            be_trigger = sl_pips * settings['be_ratio']
            if favorable_move >= be_trigger:
                stop_at_be = True
                # Add BE buffer
                be_pips = settings['be_buffer_pips']
        
        # 2. PARTIAL TP
        if settings['use_partial']:
            partial_trigger = sl_pips * settings['partial_ratio']
            if favorable_move >= partial_trigger:
                partial_pct = settings['partial_pct'] / 100
                partial_profit = (partial_trigger * 10) * position_size * partial_pct
                current_profit += partial_profit
                remaining_position -= partial_pct
        
        # 3. TRAILING STOP
        if settings['use_trail'] and favorable_move >= settings['trail_start_pips']:
            # Price reached trail activation
            retrace_amount = np.random.uniform(0, settings['trail_pips'] * 1.5)
            
            if retrace_amount > settings['trail_pips']:
                # Trail stop hit
                exit_pips = max(favorable_move - settings['trail_pips'], 
                               partial_trigger if settings['use_partial'] else 0)
                trail_profit = (exit_pips * 10) * position_size * remaining_position
                current_profit += trail_profit
                return current_profit, current_profit > 0
        
        # 4. FULL TP
        if favorable_move >= tp_pips:
            tp_profit = (tp_pips * 10) * position_size * remaining_position
            current_profit += tp_profit
            return current_profit, True
        
        # 5. RETRACED - check BE
        if stop_at_be:
            # Stopped at BE + buffer
            be_profit = (settings['be_buffer_pips'] * 10) * position_size * remaining_position
            current_profit += be_profit
            return current_profit, current_profit > 0
        
        # 6. STOPPED OUT
        return -risk_amount, False
    
    def _passes_filters(self, settings):
        """Check if trade passes day/session filters - WITH SESSION MODE"""
        
        # Random day (0=Monday, 6=Sunday)
        day = np.random.randint(0, 7)
        day_names = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        
        if not settings['days'][day_names[day]]:
            return False
        
        # SESSION MODE LOGIC
        if self.session_mode == 'session1':
            # Only allow session1
            session = 'session1'
        elif self.session_mode == 'session2':
            # Only allow session2
            session = 'session2'
        else:  # 'both'
            # Random session between session1 and session2 only
            session = np.random.choice(['session1', 'session2'])
        
        if not settings['sessions'][session]:
            return False
        
        # Buffer for Trade 1 only
        if 'buffer_pips' in settings:
            if np.random.random() > 0.7:
                return False
        
        return True
    
    def simulate_full_backtest(self, n_trades):
        """Run complete backtest with both trades"""
        
        results = []
        balance = self.account_size
        
        trades_attempted = 0
        max_attempts = n_trades * 3
        
        while len(results) < n_trades and trades_attempted < max_attempts:
            trades_attempted += 1
            
            trade_type = np.random.choice(['trade1', 'trade2'])
            is_trade1 = (trade_type == 'trade1')
            
            profit, is_win = self.simulate_single_trade(is_trade1)
            
            if profit is None:
                continue
            
            balance += profit
            
            results.append({
                'trade': len(results) + 1,
                'type': trade_type,
                'profit': profit,
                'balance': balance,
                'is_win': is_win
            })
        
        if len(results) == 0:
            return None
        
        return pd.DataFrame(results)
    
    def run_monte_carlo_prop_firm(self, challenge_size=100000, target_pct=10, max_dd_pct=10, 
                                   daily_dd_pct=5, n_sims=1000, max_days=None):
        """
        Monte Carlo simulation for prop firm challenges
        
        Challenge Rules:
        - Profit Target: target_pct% (default 10%)
        - Max Drawdown: max_dd_pct% (default 10%)
        - Daily Drawdown: daily_dd_pct% (default 5%)
        - Optional: Max Days
        
        Returns detailed statistics on pass rates, expected time, failure reasons
        """
        
        results = []
        
        for sim in range(n_sims):
            balance = challenge_size
            starting_balance = challenge_size
            peak_balance = challenge_size
            
            target_profit = challenge_size * (target_pct / 100)
            max_dd_dollars = challenge_size * (max_dd_pct / 100)
            daily_dd_dollars = challenge_size * (daily_dd_pct / 100)
            
            day = 0
            passed = False
            fail_reason = None
            
            # Simulate until pass or fail
            max_trade_days = max_days if max_days else 365
            
            while day < max_trade_days:
                day += 1
                day_start_balance = balance
                
                # Random number of trades per day (0-5)
                trades_today = np.random.randint(0, 6)
                
                for _ in range(trades_today):
                    # Random trade type
                    is_trade1 = np.random.random() < 0.5
                    profit, is_win = self.simulate_single_trade(is_trade1)
                    
                    if profit is None:
                        continue
                    
                    balance += profit
                    
                    # Update peak
                    if balance > peak_balance:
                        peak_balance = balance
                    
                    # Check win condition
                    total_profit = balance - starting_balance
                    if total_profit >= target_profit:
                        passed = True
                        break
                    
                    # Check daily drawdown
                    daily_loss = day_start_balance - balance
                    if daily_loss >= daily_dd_dollars:
                        fail_reason = "Daily Drawdown Hit"
                        break
                    
                    # Check max drawdown
                    drawdown = peak_balance - balance
                    if drawdown >= max_dd_dollars:
                        fail_reason = "Max Drawdown Hit"
                        break
                
                if passed or fail_reason:
                    break
            
            if not passed and fail_reason is None:
                fail_reason = "Time Expired"
            
            results.append({
                'sim': sim + 1,
                'passed': passed,
                'days': day,
                'final_balance': balance,
                'profit': balance - starting_balance,
                'fail_reason': fail_reason
            })
        
        return pd.DataFrame(results)


def main():
    st.markdown('<h1 class="main-header">📈 Easy Edge Tester 2</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="session-info">
        <h3>🎯 Session-Specific Testing</h3>
        <p>Choose to test <strong>Session 1 only</strong>, <strong>Session 2 only</strong>, or <strong>Both Sessions</strong> for maximum flexibility!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # SIDEBAR - EA SETTINGS
    with st.sidebar:
        st.header("⚙️ EA Configuration")
        
        # ===== NEW SESSION MODE SELECTOR =====
        st.markdown("---")
        st.subheader("📅 Session Mode")
        session_mode = st.radio(
            "Select Trading Sessions:",
            options=['both', 'session1', 'session2'],
            format_func=lambda x: {
                'both': '🔄 Both Sessions (1 & 2)',
                'session1': '🌅 Session 1 Only',
                'session2': '🌆 Session 2 Only'
            }[x],
            help="Choose which session(s) to trade"
        )
        
        if session_mode == 'session1':
            st.info("Trading Session 1 only")
        elif session_mode == 'session2':
            st.info("Trading Session 2 only")
        else:
            st.info("Trading both Session 1 & Session 2")
        # =====================================
        
        st.markdown("---")
        st.subheader("📊 Historical Stats")
        total_trades = st.number_input("Total Trades", 100, 10000, 1000, 100)
        win_rate = st.slider("Win Rate %", 0, 100, 65, 5)
        account_size = st.number_input("Account Size", 1000, 1000000, 100000, 10000)
        
        stats = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'account_size': account_size
        }
        
        # TRADE 1 SETTINGS
        st.markdown("---")
        st.markdown('<div class="trade-section"><h3>🚀 Trade 1: Breakout</h3></div>', unsafe_allow_html=True)
        
        with st.expander("⚙️ Trade 1 Settings", expanded=False):
            t1_sl = st.number_input("T1: SL (pips)", 5, 200, 30, 5, key='t1_sl')
            t1_tp = st.slider("T1: TP Ratio", 0.5, 5.0, 2.0, 0.1, key='t1_tp')
            t1_risk = st.slider("T1: Risk %", 0.1, 5.0, 1.0, 0.1, key='t1_risk')
            
            st.markdown("**Break Even**")
            t1_use_be = st.checkbox("Enable BE", True, key='t1_be')
            col1, col2 = st.columns(2)
            with col1:
                t1_be_ratio = st.slider("BE Trigger", 0.3, 2.0, 0.5, 0.1, key='t1_be_ratio')
            with col2:
                t1_be_buffer = st.number_input("BE Buffer", 0, 20, 3, 1, key='t1_be_buffer')
            
            st.markdown("**Partial TP**")
            t1_use_partial = st.checkbox("Enable Partial", True, key='t1_partial')
            col1, col2 = st.columns(2)
            with col1:
                t1_partial_ratio = st.slider("Partial Trigger", 0.3, 2.0, 1.0, 0.1, key='t1_partial_ratio')
            with col2:
                t1_partial_pct = st.slider("Partial %", 10, 90, 50, 10, key='t1_partial_pct')
            
            st.markdown("**Trailing Stop**")
            t1_use_trail = st.checkbox("Enable Trail", True, key='t1_trail')
            col1, col2 = st.columns(2)
            with col1:
                t1_trail_start = st.number_input("Trail Start", 5, 200, 40, 5, key='t1_trail_start')
            with col2:
                t1_trail_pips = st.number_input("Trail Distance", 5, 100, 15, 5, key='t1_trail_pips')
            
            st.markdown("**Order Cancellation**")
            t1_use_cancel = st.checkbox("Enable Cancellation", False, key='t1_cancel')
            t1_cancel_hours = st.slider("Cancel After (hours)", 1.0, 24.0, 4.0, 0.5, key='t1_cancel_hours')
            
            st.markdown("**Trading Days**")
            col1, col2 = st.columns(2)
            with col1:
                t1_mon = st.checkbox("Mon", True, key='t1_mon')
                t1_tue = st.checkbox("Tue", True, key='t1_tue')
                t1_wed = st.checkbox("Wed", True, key='t1_wed')
                t1_thu = st.checkbox("Thu", True, key='t1_thu')
            with col2:
                t1_fri = st.checkbox("Fri", True, key='t1_fri')
                t1_sat = st.checkbox("Sat", False, key='t1_sat')
                t1_sun = st.checkbox("Sun", False, key='t1_sun')
            
            st.markdown("**Trading Sessions**")
            t1_s1 = st.checkbox("Session 1", True, key='t1_s1')
            t1_s2 = st.checkbox("Session 2", True, key='t1_s2')
            t1_s3 = st.checkbox("Session 3", False, key='t1_s3')
            
            t1_buffer = st.number_input("T1: Buffer (pips)", 0, 50, 5, 1, key='t1_buffer')
        
        trade1_settings = {
            'sl_pips': t1_sl,
            'tp_ratio': t1_tp,
            'risk_pct': t1_risk,
            'use_breakeven': t1_use_be,
            'be_ratio': t1_be_ratio,
            'be_buffer_pips': t1_be_buffer,
            'use_partial': t1_use_partial,
            'partial_ratio': t1_partial_ratio,
            'partial_pct': t1_partial_pct,
            'use_trail': t1_use_trail,
            'trail_start_pips': t1_trail_start,
            'trail_pips': t1_trail_pips,
            'use_cancel': t1_use_cancel,
            'cancel_hours': t1_cancel_hours,
            'days': {
                'monday': t1_mon, 'tuesday': t1_tue, 'wednesday': t1_wed,
                'thursday': t1_thu, 'friday': t1_fri, 'saturday': t1_sat, 'sunday': t1_sun
            },
            'sessions': {
                'session1': t1_s1, 'session2': t1_s2, 'session3': t1_s3
            },
            'buffer_pips': t1_buffer
        }
        
        # TRADE 2 SETTINGS
        st.markdown("---")
        st.markdown('<div class="trade-section"><h3>🔄 Trade 2: Retracement</h3></div>', unsafe_allow_html=True)
        
        with st.expander("⚙️ Trade 2 Settings", expanded=False):
            t2_sl = st.number_input("T2: SL (pips)", 5, 200, 25, 5, key='t2_sl')
            t2_tp = st.slider("T2: TP Ratio", 0.5, 5.0, 1.5, 0.1, key='t2_tp')
            t2_risk = st.slider("T2: Risk %", 0.1, 5.0, 1.0, 0.1, key='t2_risk')
            
            st.markdown("**Break Even**")
            t2_use_be = st.checkbox("Enable BE", True, key='t2_be')
            col1, col2 = st.columns(2)
            with col1:
                t2_be_ratio = st.slider("BE Trigger", 0.3, 2.0, 0.5, 0.1, key='t2_be_ratio')
            with col2:
                t2_be_buffer = st.number_input("BE Buffer", 0, 20, 3, 1, key='t2_be_buffer')
            
            st.markdown("**Partial TP**")
            t2_use_partial = st.checkbox("Enable Partial", True, key='t2_partial')
            col1, col2 = st.columns(2)
            with col1:
                t2_partial_ratio = st.slider("Partial Trigger", 0.3, 2.0, 0.8, 0.1, key='t2_partial_ratio')
            with col2:
                t2_partial_pct = st.slider("Partial %", 10, 90, 50, 10, key='t2_partial_pct')
            
            st.markdown("**Trailing Stop**")
            t2_use_trail = st.checkbox("Enable Trail", True, key='t2_trail')
            col1, col2 = st.columns(2)
            with col1:
                t2_trail_start = st.number_input("Trail Start", 5, 200, 30, 5, key='t2_trail_start')
            with col2:
                t2_trail_pips = st.number_input("Trail Distance", 5, 100, 12, 5, key='t2_trail_pips')
            
            st.markdown("**Order Cancellation**")
            t2_use_cancel = st.checkbox("Enable Cancellation", False, key='t2_cancel')
            t2_cancel_hours = st.slider("Cancel After (hours)", 1.0, 24.0, 6.0, 0.5, key='t2_cancel_hours')
            
            st.markdown("**Trading Days**")
            col1, col2 = st.columns(2)
            with col1:
                t2_mon = st.checkbox("Mon", True, key='t2_mon')
                t2_tue = st.checkbox("Tue", True, key='t2_tue')
                t2_wed = st.checkbox("Wed", True, key='t2_wed')
                t2_thu = st.checkbox("Thu", True, key='t2_thu')
            with col2:
                t2_fri = st.checkbox("Fri", True, key='t2_fri')
                t2_sat = st.checkbox("Sat", False, key='t2_sat')
                t2_sun = st.checkbox("Sun", False, key='t2_sun')
            
            st.markdown("**Trading Sessions**")
            t2_s1 = st.checkbox("Session 1", True, key='t2_s1')
            t2_s2 = st.checkbox("Session 2", True, key='t2_s2')
            t2_s3 = st.checkbox("Session 3", False, key='t2_s3')
        
        trade2_settings = {
            'sl_pips': t2_sl,
            'tp_ratio': t2_tp,
            'risk_pct': t2_risk,
            'use_breakeven': t2_use_be,
            'be_ratio': t2_be_ratio,
            'be_buffer_pips': t2_be_buffer,
            'use_partial': t2_use_partial,
            'partial_ratio': t2_partial_ratio,
            'partial_pct': t2_partial_pct,
            'use_trail': t2_use_trail,
            'trail_start_pips': t2_trail_start,
            'trail_pips': t2_trail_pips,
            'use_cancel': t2_use_cancel,
            'cancel_hours': t2_cancel_hours,
            'days': {
                'monday': t2_mon, 'tuesday': t2_tue, 'wednesday': t2_wed,
                'thursday': t2_thu, 'friday': t2_fri, 'saturday': t2_sat, 'sunday': t2_sun
            },
            'sessions': {
                'session1': t2_s1, 'session2': t2_s2, 'session3': t2_s3
            }
        }
        
        global_settings = {}
    
    # MAIN TABS
    tabs = st.tabs(["🎰 Monte Carlo Prop Firm", "📊 Quick Test", "🎯 Optimizer"])
    
    # TAB 1: MONTE CARLO
    with tabs[0]:
        st.header("🎰 Monte Carlo Prop Firm Analysis")
        st.markdown("Simulate thousands of prop firm challenges to predict your success rate")
        
        col1, col2 = st.columns(2)
        
        with col1:
            challenge_size = st.selectbox("Challenge Size", [25000, 50000, 100000, 200000], index=2)
            target_pct = st.slider("Profit Target %", 5, 20, 10, 1)
            max_dd_pct = st.slider("Max Drawdown %", 5, 20, 10, 1)
            daily_dd_pct = st.slider("Daily Drawdown %", 3, 10, 5, 1)
        
        with col2:
            n_sims = st.slider("Simulations", 100, 10000, 1000, 100)
            max_days = st.number_input("Max Days (0=unlimited)", 0, 365, 0, 10)
            challenge_cost = st.number_input("Challenge Cost ($)", 0, 5000, 500, 100)
        
        if max_days == 0:
            max_days = None
        
        if st.button("🚀 Run Monte Carlo Analysis", type="primary"):
            with st.spinner(f"Running {n_sims} simulations..."):
                
                simulator = Complete2TradeEASimulator(stats, trade1_settings, trade2_settings, global_settings, session_mode)
                
                results_df = simulator.run_monte_carlo_prop_firm(
                    challenge_size=challenge_size,
                    target_pct=target_pct,
                    max_dd_pct=max_dd_pct,
                    daily_dd_pct=daily_dd_pct,
                    n_sims=n_sims,
                    max_days=max_days
                )
                
                # Calculate metrics
                passed_df = results_df[results_df['passed'] == True]
                pass_rate = len(passed_df) / len(results_df)
                
                avg_days = passed_df['days'].mean() if len(passed_df) > 0 else 0
                median_days = passed_df['days'].median() if len(passed_df) > 0 else 0
                
                # Expected attempts
                expected_attempts = 1 / pass_rate if pass_rate > 0 else float('inf')
                total_cost = expected_attempts * challenge_cost
                
                # ROI calculation
                expected_profit = challenge_size * (target_pct / 100)
                roi = ((expected_profit - total_cost) / total_cost * 100) if total_cost > 0 else 0
                
                # Payout simulation
                payout_targets = [2, 4, 6, 8, 10]
                payout_results = []
                
                for payout_pct in payout_targets:
                    payout_sims = []
                    for _ in range(100):
                        balance = challenge_size
                        payouts = 0
                        
                        for day in range(365):
                            trades_today = np.random.randint(0, 6)
                            
                            for _ in range(trades_today):
                                is_trade1 = np.random.random() < 0.5
                                profit, _ = simulator.simulate_single_trade(is_trade1)
                                
                                if profit is not None:
                                    balance += profit
                                
                                if balance >= challenge_size * (1 + payout_pct/100):
                                    payouts += 1
                                    balance = challenge_size
                                
                                if balance <= challenge_size * (1 - max_dd_pct/100):
                                    break
                            
                            if balance <= challenge_size * (1 - max_dd_pct/100):
                                break
                        
                        payout_sims.append(payouts)
                    
                    payout_results.append({
                        'payout_pct': payout_pct,
                        'avg_payouts': np.mean(payout_sims)
                    })
                
                payout_df = pd.DataFrame(payout_results)
                
                # Display results
                st.success("✅ Monte Carlo Analysis Complete!")
                
                st.markdown("### 🎯 Challenge Success Metrics")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Pass Rate", f"{pass_rate:.1%}")
                with col2:
                    st.metric("Avg Days", f"{avg_days:.0f}")
                with col3:
                    st.metric("Expected Attempts", f"{expected_attempts:.1f}")
                with col4:
                    st.metric("Total Cost", f"${total_cost:,.0f}")
                with col5:
                    st.metric("Challenge Fee ROI", f"{roi:.0f}%")
                
                # Payout metrics
                st.markdown("### 💰 Payout Analysis")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    payout_4pct = payout_df[payout_df['payout_pct'] == 4.0]['avg_payouts'].values[0] if len(payout_df[payout_df['payout_pct'] == 4.0]) > 0 else 0
                    st.metric("Payouts @ 4%", f"{payout_4pct:.1f}")
                
                with col2:
                    payout_8pct = payout_df[payout_df['payout_pct'] == 8.0]['avg_payouts'].values[0] if len(payout_df[payout_df['payout_pct'] == 8.0]) > 0 else 0
                    st.metric("Payouts @ 8%", f"{payout_8pct:.1f}")
                
                with col3:
                    payout_10pct = payout_df[payout_df['payout_pct'] == 10.0]['avg_payouts'].values[0] if len(payout_df[payout_df['payout_pct'] == 10.0]) > 0 else 0
                    st.metric("Payouts @ 10%", f"{payout_10pct:.1f}")
                
                # Recommendation
                if pass_rate >= 0.6 and roi >= 500:
                    st.markdown(f"""
                    <div class="success-box">
                        <h3>✅ HIGHLY RECOMMENDED</h3>
                        <p><strong>Pass Rate:</strong> {pass_rate:.1%} | <strong>ROI:</strong> {roi:.0f}%</p>
                        <p>Excellent odds! Expected {expected_attempts:.1f} attempts to pass.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif pass_rate >= 0.4 and roi >= 200:
                    st.markdown(f"""
                    <div class="warning-box">
                        <h3>⚠️ PROCEED WITH CAUTION</h3>
                        <p><strong>Pass Rate:</strong> {pass_rate:.1%} | <strong>ROI:</strong> {roi:.0f}%</p>
                        <p>Moderate odds. Budget for {expected_attempts:.1f} attempts.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="danger-box">
                        <h3>❌ NOT RECOMMENDED</h3>
                        <p><strong>Pass Rate:</strong> {pass_rate:.1%} | <strong>ROI:</strong> {roi:.0f}%</p>
                        <p>Optimize your strategy before attempting.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Charts
                st.markdown("---")
                st.markdown("### 📊 Detailed Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pass rate pie
                    pass_counts = results_df['passed'].value_counts()
                    fig = go.Figure(data=[go.Pie(
                        labels=['Passed', 'Failed'],
                        values=[pass_counts.get(True, 0), pass_counts.get(False, 0)],
                        marker_colors=['#28a745', '#dc3545'],
                        hole=0.4
                    )])
                    fig.update_layout(title="Pass/Fail Distribution", height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Failure reasons
                    if len(results_df[results_df['passed'] == False]) > 0:
                        fail_reasons = results_df[results_df['passed'] == False]['fail_reason'].value_counts()
                        fig = go.Figure(data=[go.Bar(
                            x=fail_reasons.index,
                            y=fail_reasons.values,
                            marker_color='#dc3545'
                        )])
                        fig.update_layout(title="Failure Reasons", xaxis_title="Reason", yaxis_title="Count", height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Days to pass
                    if len(passed_df) > 0:
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(x=passed_df['days'], nbinsx=30, marker_color='#3498db'))
                        fig.add_vline(x=avg_days, line_dash="dash", line_color="red", annotation_text=f"Avg: {avg_days:.0f}")
                        fig.update_layout(title="Days to Pass (Successful Attempts)", xaxis_title="Days", yaxis_title="Count", height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Payout expectations
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=payout_df['payout_pct'],
                        y=payout_df['avg_payouts'],
                        mode='lines+markers',
                        marker_color='#2ecc71',
                        line=dict(width=3)
                    ))
                    fig.update_layout(
                        title="Expected Payouts Before Account Fails",
                        xaxis_title="Payout % Target",
                        yaxis_title="Avg # of Payouts",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Download
                st.markdown("---")
                csv = results_df.to_csv(index=False)
                st.download_button("📥 Download Full Results (CSV)", csv, "monte_carlo_results.csv", "text/csv")
    
    # TAB 2: QUICK TEST
    with tabs[1]:
        st.header("📊 Quick Performance Test")
        st.markdown("Test your current settings with 100 simulated trades")
        
        if st.button("🚀 Run Quick Test (100 trades)"):
            
            simulator = Complete2TradeEASimulator(stats, trade1_settings, trade2_settings, global_settings, session_mode)
            
            results_df = simulator.simulate_full_backtest(100)
            
            if results_df is not None and len(results_df) > 0:
                # Calculate metrics
                total_profit = results_df['profit'].sum()
                wins = (results_df['is_win']).sum()
                losses = len(results_df) - wins
                win_rate_result = wins / len(results_df)
                
                # Calculate cancellation rate
                attempted_trades = 100
                actual_trades = len(results_df)
                cancellation_rate = ((attempted_trades - actual_trades) / attempted_trades) * 100
                
                winning_trades = results_df[results_df['profit'] > 0]['profit']
                losing_trades = results_df[results_df['profit'] < 0]['profit']
                
                avg_win_result = winning_trades.mean() if len(winning_trades) > 0 else 0
                avg_loss_result = abs(losing_trades.mean()) if len(losing_trades) > 0 else 0
                
                gross_profit = winning_trades.sum() if len(winning_trades) > 0 else 0
                gross_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0
                pf = gross_profit / gross_loss if gross_loss > 0 else 0
                
                # Drawdown
                equity_curve = results_df['balance'].values
                running_max = np.maximum.accumulate(equity_curve)
                drawdown = (running_max - equity_curve) / running_max * 100
                max_dd = drawdown.max()
                
                return_pct = (total_profit / account_size) * 100
                
                st.success("✅ Quick Test Complete!")
                
                if cancellation_rate > 0:
                    st.warning(f"⚠️ Order Cancellation Rate: {cancellation_rate:.0f}% ({actual_trades} filled out of {attempted_trades} attempted)")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Win Rate", f"{win_rate_result*100:.1f}%")
                    st.metric("Total Trades", len(results_df))
                
                with col2:
                    st.metric("Profit Factor", f"{pf:.2f}")
                    st.metric("Net Profit", f"${total_profit:,.2f}")
                
                with col3:
                    st.metric("Return %", f"{return_pct:.1f}%")
                    st.metric("Max DD", f"{max_dd:.1f}%")
                
                with col4:
                    st.metric("Avg Win", f"${avg_win_result:.2f}")
                    st.metric("Avg Loss", f"${avg_loss_result:.2f}")
                
                # Equity curve
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=results_df['trade'],
                    y=results_df['balance'],
                    mode='lines',
                    name='Balance',
                    line=dict(color='#2196F3', width=2)
                ))
                fig.update_layout(
                    title="Equity Curve (100 Trades)",
                    xaxis_title="Trade #",
                    yaxis_title="Balance ($)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.error("❌ No trades passed filters! Check your day/session settings.")
    
    # TAB 3: OPTIMIZER
    with tabs[2]:
        st.header("🎯 Comprehensive EA Optimizer")
        st.markdown("Test all EA parameters: SL/TP, Break Even, Trailing Stop, Partial TP, and Order Cancellation")
        
        # Optimization mode selector
        opt_mode = st.radio(
            "Optimization Mode:",
            ["Quick Optimize (Core Parameters)", "Full Optimize (All Parameters)"],
            help="Quick = SL/TP/Trail only | Full = All exit strategies + order management"
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📏 Core Parameters**")
            opt_sl_min = st.number_input("Min SL (pips)", 10, 100, 20, 5)
            opt_sl_max = st.number_input("Max SL (pips)", 20, 200, 50, 5)
            opt_sl_step = st.number_input("SL Step", 5, 20, 10, 5)
            
            opt_tp_min = st.slider("Min TP Ratio", 0.5, 3.0, 1.0, 0.5)
            opt_tp_max = st.slider("Max TP Ratio", 1.0, 5.0, 3.0, 0.5)
            opt_tp_step = st.slider("TP Step", 0.1, 1.0, 0.5, 0.1)
        
        with col2:
            st.markdown("**🎯 Trailing Stop**")
            opt_trail_start_min = st.number_input("Min Trail Start", 10, 100, 20, 10)
            opt_trail_start_max = st.number_input("Max Trail Start", 20, 200, 60, 10)
            opt_trail_dist_min = st.number_input("Min Trail Dist", 5, 50, 10, 5)
            opt_trail_dist_max = st.number_input("Max Trail Dist", 10, 100, 30, 5)
            
            if opt_mode == "Full Optimize (All Parameters)":
                st.markdown("**⚡ Break Even**")
                opt_be_ratio_values = st.multiselect("BE Trigger Ratios", 
                    [0.3, 0.5, 0.7, 1.0, 1.2, 1.5], 
                    default=[0.5, 1.0])
                opt_be_buffer_values = st.multiselect("BE Buffer (pips)", 
                    [0, 3, 5, 10], 
                    default=[3, 5])
        
        with col3:
            st.markdown("**⚙️ Optimization Settings**")
            opt_trades = st.number_input("Trades per Test", 50, 500, 100, 50)
            opt_metric = st.selectbox("Optimize For", 
                ["Profit Factor", "Win Rate", "Net Profit", "Return %", "Sharpe Ratio"])
            
            if opt_mode == "Full Optimize (All Parameters)":
                st.markdown("**📦 Partial TP**")
                opt_partial_ratio_values = st.multiselect("Partial Trigger Ratios", 
                    [0.5, 0.7, 1.0, 1.2, 1.5], 
                    default=[0.7, 1.0])
                opt_partial_pct_values = st.multiselect("Partial % Close", 
                    [25, 50, 75], 
                    default=[50])
                
                st.markdown("**⏰ Order Cancellation**")
                opt_cancel_enabled = st.checkbox("Test Order Cancellation", value=True)
                if opt_cancel_enabled:
                    opt_cancel_hours_values = st.multiselect("Cancel Hours", 
                        [2, 4, 6, 8, 12, 24], 
                        default=[4, 8])
        
        if st.button("🚀 Run Comprehensive Optimization", type="primary"):
            
            # Generate parameter combinations
            sl_values = list(range(opt_sl_min, opt_sl_max + 1, opt_sl_step))
            tp_values = [round(x, 1) for x in np.arange(opt_tp_min, opt_tp_max + 0.01, opt_tp_step)]
            trail_start_values = list(range(opt_trail_start_min, opt_trail_start_max + 1, 10))
            trail_dist_values = list(range(opt_trail_dist_min, opt_trail_dist_max + 1, 5))
            
            if opt_mode == "Full Optimize (All Parameters)":
                be_ratio_values = opt_be_ratio_values if opt_be_ratio_values else [0.5]
                be_buffer_values = opt_be_buffer_values if opt_be_buffer_values else [3]
                partial_ratio_values = opt_partial_ratio_values if opt_partial_ratio_values else [1.0]
                partial_pct_values = opt_partial_pct_values if opt_partial_pct_values else [50]
                cancel_hours_values = opt_cancel_hours_values if (opt_cancel_enabled and opt_cancel_hours_values) else [None]
            else:
                be_ratio_values = [0.5]
                be_buffer_values = [3]
                partial_ratio_values = [1.0]
                partial_pct_values = [50]
                cancel_hours_values = [None]
            
            total_combos = (len(sl_values) * len(tp_values) * len(trail_start_values) * 
                           len(trail_dist_values) * len(be_ratio_values) * len(be_buffer_values) *
                           len(partial_ratio_values) * len(partial_pct_values) * len(cancel_hours_values))
            
            st.info(f"🔢 Total combinations to test: {total_combos}")
            
            if total_combos > 2000:
                st.error(f"⚠️ Too many combinations ({total_combos})! Please reduce ranges or use Quick Optimize mode.")
            else:
                st.warning(f"⏳ Optimization running... Testing {total_combos} combinations. This may take several minutes.")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                optimization_results = []
                combo_count = 0
                
                for sl in sl_values:
                    for tp in tp_values:
                        for trail_start in trail_start_values:
                            for trail_dist in trail_dist_values:
                                for be_ratio in be_ratio_values:
                                    for be_buffer in be_buffer_values:
                                        for partial_ratio in partial_ratio_values:
                                            for partial_pct in partial_pct_values:
                                                for cancel_hours in cancel_hours_values:
                                                    combo_count += 1
                                                    
                                                    # Update progress every 10 combos
                                                    if combo_count % 10 == 0:
                                                        progress = combo_count / total_combos
                                                        progress_bar.progress(progress)
                                                        status_text.text(f"Testing {combo_count}/{total_combos} | SL={sl} TP={tp:.1f} Trail={trail_start}/{trail_dist}")
                                                    
                                                    # Update settings
                                                    test_t1 = trade1_settings.copy()
                                                    test_t1['sl_pips'] = sl
                                                    test_t1['tp_ratio'] = tp
                                                    test_t1['trail_start_pips'] = trail_start
                                                    test_t1['trail_pips'] = trail_dist
                                                    test_t1['be_ratio'] = be_ratio
                                                    test_t1['be_buffer_pips'] = be_buffer
                                                    test_t1['partial_ratio'] = partial_ratio
                                                    test_t1['partial_pct'] = partial_pct
                                                    
                                                    if cancel_hours is not None:
                                                        test_t1['use_cancel'] = True
                                                        test_t1['cancel_hours'] = cancel_hours
                                                    else:
                                                        test_t1['use_cancel'] = False
                                                    
                                                    test_t2 = trade2_settings.copy()
                                                    test_t2['sl_pips'] = sl
                                                    test_t2['tp_ratio'] = tp
                                                    test_t2['trail_start_pips'] = trail_start
                                                    test_t2['trail_pips'] = trail_dist
                                                    test_t2['be_ratio'] = be_ratio
                                                    test_t2['be_buffer_pips'] = be_buffer
                                                    test_t2['partial_ratio'] = partial_ratio
                                                    test_t2['partial_pct'] = partial_pct
                                                    
                                                    if cancel_hours is not None:
                                                        test_t2['use_cancel'] = True
                                                        test_t2['cancel_hours'] = cancel_hours
                                                    else:
                                                        test_t2['use_cancel'] = False
                                                    
                                                    # Run simulation
                                                    simulator = Complete2TradeEASimulator(stats, test_t1, test_t2, global_settings, session_mode)
                                                    results_df = simulator.simulate_full_backtest(opt_trades)
                                                    
                                                    if results_df is not None and len(results_df) > 10:
                                                        # Calculate metrics
                                                        total_profit = results_df['profit'].sum()
                                                        wins = (results_df['is_win']).sum()
                                                        win_rate_result = wins / len(results_df) if len(results_df) > 0 else 0
                                                        
                                                        winning_trades = results_df[results_df['profit'] > 0]['profit']
                                                        losing_trades = results_df[results_df['profit'] < 0]['profit']
                                                        
                                                        gross_profit = winning_trades.sum() if len(winning_trades) > 0 else 0
                                                        gross_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0
                                                        pf = gross_profit / gross_loss if gross_loss > 0 else 0
                                                        
                                                        return_pct = (total_profit / account_size) * 100
                                                        
                                                        # Calculate Sharpe Ratio (simplified)
                                                        returns = results_df['profit'] / account_size
                                                        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
                                                        
                                                        optimization_results.append({
                                                            'SL': sl,
                                                            'TP_Ratio': tp,
                                                            'Trail_Start': trail_start,
                                                            'Trail_Dist': trail_dist,
                                                            'BE_Ratio': be_ratio,
                                                            'BE_Buffer': be_buffer,
                                                            'Partial_Ratio': partial_ratio,
                                                            'Partial_Pct': partial_pct,
                                                            'Cancel_Hours': cancel_hours if cancel_hours else 0,
                                                            'Trades': len(results_df),
                                                            'Win_Rate': win_rate_result * 100,
                                                            'Profit_Factor': pf,
                                                            'Net_Profit': total_profit,
                                                            'Return_Pct': return_pct,
                                                            'Sharpe': sharpe
                                                        })
                
                progress_bar.empty()
                status_text.empty()
                
                if len(optimization_results) > 0:
                    opt_df = pd.DataFrame(optimization_results)
                    
                    # Sort by selected metric
                    if opt_metric == "Profit Factor":
                        opt_df = opt_df.sort_values('Profit_Factor', ascending=False)
                    elif opt_metric == "Win Rate":
                        opt_df = opt_df.sort_values('Win_Rate', ascending=False)
                    elif opt_metric == "Net Profit":
                        opt_df = opt_df.sort_values('Net_Profit', ascending=False)
                    elif opt_metric == "Sharpe Ratio":
                        opt_df = opt_df.sort_values('Sharpe', ascending=False)
                    else:  # Return %
                        opt_df = opt_df.sort_values('Return_Pct', ascending=False)
                    
                    st.success(f"✅ Optimization Complete! Tested {len(opt_df)} valid combinations")
                    
                    # Best settings
                    best = opt_df.iloc[0]
                    
                    cancel_text = f"{best['Cancel_Hours']:.0f}h" if best['Cancel_Hours'] > 0 else "Disabled"
                    
                    st.markdown(f"""
                    <div class="best-setting">
                        <h3>🏆 BEST SETTINGS (Optimized for {opt_metric})</h3>
                        <p><strong>SL:</strong> {best['SL']} pips | <strong>TP Ratio:</strong> {best['TP_Ratio']:.1f} | <strong>Trail:</strong> {best['Trail_Start']}/{best['Trail_Dist']} pips</p>
                        <p><strong>Break Even:</strong> {best['BE_Ratio']:.1f}x + {best['BE_Buffer']:.0f}p buffer | <strong>Partial TP:</strong> {best['Partial_Pct']:.0f}% @ {best['Partial_Ratio']:.1f}x</p>
                        <p><strong>Cancel After:</strong> {cancel_text}</p>
                        <hr>
                        <p><strong>Win Rate:</strong> {best['Win_Rate']:.1f}% | <strong>Profit Factor:</strong> {best['Profit_Factor']:.2f} | <strong>Return:</strong> {best['Return_Pct']:.1f}% | <strong>Sharpe:</strong> {best['Sharpe']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Top 20 results
                    st.markdown("### 📊 Top 20 Parameter Combinations")
                    
                    display_cols = ['SL', 'TP_Ratio', 'Trail_Start', 'Trail_Dist', 'BE_Ratio', 'BE_Buffer', 
                                   'Partial_Ratio', 'Partial_Pct', 'Cancel_Hours', 'Trades', 'Win_Rate', 
                                   'Profit_Factor', 'Net_Profit', 'Return_Pct', 'Sharpe']
                    
                    st.dataframe(opt_df[display_cols].head(20).style.format({
                        'TP_Ratio': '{:.1f}',
                        'BE_Ratio': '{:.1f}',
                        'Partial_Ratio': '{:.1f}',
                        'Win_Rate': '{:.1f}%',
                        'Profit_Factor': '{:.2f}',
                        'Net_Profit': '${:,.2f}',
                        'Return_Pct': '{:.1f}%',
                        'Sharpe': '{:.2f}'
                    }), use_container_width=True)
                    
                    # Charts
                    st.markdown("### 📈 Optimization Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # SL vs TP heatmap
                        if opt_metric == "Profit Factor":
                            metric_col = 'Profit_Factor'
                        elif opt_metric == "Win Rate":
                            metric_col = 'Win_Rate'
                        elif opt_metric == "Net Profit":
                            metric_col = 'Net_Profit'
                        elif opt_metric == "Sharpe Ratio":
                            metric_col = 'Sharpe'
                        else:
                            metric_col = 'Return_Pct'
                        
                        pivot = opt_df.pivot_table(values=metric_col, index='SL', columns='TP_Ratio', aggfunc='mean')
                        
                        fig = go.Figure(data=go.Heatmap(
                            z=pivot.values,
                            x=pivot.columns,
                            y=pivot.index,
                            colorscale='RdYlGn',
                            text=np.round(pivot.values, 2),
                            texttemplate='%{text}',
                            textfont={"size": 9}
                        ))
                        
                        fig.update_layout(
                            title=f'{opt_metric} by SL and TP Ratio',
                            xaxis_title='TP Ratio',
                            yaxis_title='SL (pips)',
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Trailing Stop analysis
                        pivot2 = opt_df.pivot_table(values=metric_col, index='Trail_Start', columns='Trail_Dist', aggfunc='mean')
                        
                        fig2 = go.Figure(data=go.Heatmap(
                            z=pivot2.values,
                            x=pivot2.columns,
                            y=pivot2.index,
                            colorscale='RdYlGn',
                            text=np.round(pivot2.values, 2),
                            texttemplate='%{text}',
                            textfont={"size": 9}
                        ))
                        
                        fig2.update_layout(
                            title=f'{opt_metric} by Trail Start and Distance',
                            xaxis_title='Trail Distance (pips)',
                            yaxis_title='Trail Start (pips)',
                            height=400
                        )
                        
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Parameter importance analysis
                    if opt_mode == "Full Optimize (All Parameters)":
                        st.markdown("### 🔍 Parameter Impact Analysis")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # BE Ratio impact
                            be_impact = opt_df.groupby('BE_Ratio')[metric_col].mean().sort_values(ascending=False)
                            fig = go.Figure(data=[go.Bar(x=be_impact.index, y=be_impact.values, marker_color='#3498db')])
                            fig.update_layout(title=f'Break Even Ratio Impact', xaxis_title='BE Ratio', yaxis_title=opt_metric, height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Partial % impact
                            partial_impact = opt_df.groupby('Partial_Pct')[metric_col].mean().sort_values(ascending=False)
                            fig = go.Figure(data=[go.Bar(x=partial_impact.index, y=partial_impact.values, marker_color='#2ecc71')])
                            fig.update_layout(title=f'Partial TP % Impact', xaxis_title='Partial %', yaxis_title=opt_metric, height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col3:
                            # Cancel hours impact
                            if opt_cancel_enabled:
                                cancel_impact = opt_df.groupby('Cancel_Hours')[metric_col].mean().sort_values(ascending=False)
                                fig = go.Figure(data=[go.Bar(x=cancel_impact.index, y=cancel_impact.values, marker_color='#e74c3c')])
                                fig.update_layout(title=f'Cancel Hours Impact', xaxis_title='Hours', yaxis_title=opt_metric, height=300)
                                st.plotly_chart(fig, use_container_width=True)
                    
                    # Download results
                    st.markdown("---")
                    csv = opt_df.to_csv(index=False)
                    st.download_button("📥 Download Full Optimization Results", csv, "optimization_results.csv", "text/csv")
                    
                else:
                    st.error("❌ No valid results! Check your settings and filters.")


if __name__ == "__main__":
    main()
