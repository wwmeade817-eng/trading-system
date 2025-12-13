"""
ULTIMATE TRADING SYSTEM - COMPLETE 2-TRADE EA
Everything you need for optimization and prop firm analysis

Trade 1: Breakout Trade (pip-based)
Trade 2: Retracement Trade (% back in range)

Complete EA simulation with ALL parameters
Monte Carlo with comprehensive prop firm analysis

Install: pip install streamlit plotly pandas numpy scikit-learn
Run: streamlit run ultimate_trading_system.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Ultimate Trading System",
    page_icon="🚀",
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
    """
    
    def __init__(self, stats, trade1_settings, trade2_settings, global_settings):
        self.stats = stats
        self.trade1 = trade1_settings
        self.trade2 = trade2_settings
        self.global_settings = global_settings
        
        self.total_trades = stats['total_trades']
        self.win_rate = stats['win_rate'] / 100
        self.account_size = stats.get('account_size', 100000)
    
    def simulate_single_trade(self, is_trade1=True):
        """Simulate one trade with complete EA logic"""
        
        settings = self.trade1 if is_trade1 else self.trade2
        
        # Check if trade passes filters
        if not self._passes_filters(settings):
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
        """Check if trade passes day/session filters"""
        
        # Random day (0=Monday, 6=Sunday)
        day = np.random.randint(0, 7)
        day_names = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        
        if not settings['days'][day_names[day]]:
            return False
        
        # Random session
        session = np.random.choice(['session1', 'session2', 'session3'])
        
        if not settings['sessions'][session]:
            return False
        
        # Buffer for Trade 1 only
        if 'buffer_pips' in settings and settings['buffer_pips'] > 0:
            # Simulate buffer distance check
            if np.random.random() < 0.1:  # 10% fail buffer
                return False
        
        return True
    
    def simulate_full_backtest(self, num_trades):
        """Simulate complete backtest"""
        
        results = []
        balance = self.account_size
        peak = balance
        
        trade_count = 0
        attempts = 0
        max_attempts = num_trades * 3  # Prevent infinite loop
        
        while trade_count < num_trades and attempts < max_attempts:
            attempts += 1
            
            # Alternate between Trade 1 and Trade 2
            is_trade1 = (trade_count % 2 == 0)
            
            profit, is_win = self.simulate_single_trade(is_trade1)
            
            if profit is None:
                continue  # Trade filtered out
            
            balance += profit
            if balance > peak:
                peak = balance
            
            results.append({
                'trade': trade_count + 1,
                'profit': profit,
                'balance': balance,
                'is_win': is_win,
                'trade_type': 1 if is_trade1 else 2
            })
            
            trade_count += 1
        
        if len(results) == 0:
            return None
        
        return pd.DataFrame(results)
    
    def simulate_prop_firm_challenge(self, challenge_params, max_days=90):
        """Simulate prop firm challenge"""
        
        balance = challenge_params['account_size']
        starting_balance = balance
        peak_balance = balance
        
        total_days = 0
        total_trades = 0
        passed = False
        fail_reason = None
        
        profit_target = balance * (challenge_params['profit_target_pct'] / 100)
        daily_dd_limit = balance * (challenge_params['daily_dd_pct'] / 100)
        max_dd_limit = balance * (challenge_params['max_dd_pct'] / 100)
        
        # Estimate trades per day
        trades_per_day = max(1, self.total_trades / 60)
        
        for day in range(max_days):
            daily_start_balance = balance
            
            num_trades_today = max(1, int(np.random.poisson(trades_per_day)))
            
            for trade_num in range(num_trades_today):
                # Alternate Trade 1 and Trade 2
                is_trade1 = (total_trades % 2 == 0)
                
                profit, is_win = self.simulate_single_trade(is_trade1)
                
                if profit is None:
                    continue
                
                balance += profit
                total_trades += 1
                
                if balance > peak_balance:
                    peak_balance = balance
                
                # Check daily DD
                daily_loss = daily_start_balance - balance
                if daily_loss > daily_dd_limit:
                    fail_reason = "Daily Drawdown"
                    return self._create_result(False, total_days, total_trades,
                                               balance, starting_balance, peak_balance, fail_reason)
                
                # Check max DD
                current_dd = peak_balance - balance
                if current_dd > max_dd_limit:
                    fail_reason = "Max Drawdown"
                    return self._create_result(False, total_days, total_trades,
                                               balance, starting_balance, peak_balance, fail_reason)
            
            total_days += 1
            
            # Check profit target
            total_profit = balance - starting_balance
            if total_profit >= profit_target:
                passed = True
                break
            
            if total_days >= max_days:
                fail_reason = "Max Days"
                break
        
        return self._create_result(passed, total_days, total_trades,
                                   balance, starting_balance, peak_balance, fail_reason)
    
    def _create_result(self, passed, days, trades, balance, starting, peak, fail_reason):
        return {
            'passed': passed,
            'days': days,
            'trades': trades,
            'final_balance': balance,
            'profit': balance - starting,
            'profit_pct': ((balance - starting) / starting) * 100,
            'peak_balance': peak,
            'max_drawdown': peak - balance,
            'max_drawdown_pct': ((peak - balance) / peak) * 100 if peak > 0 else 0,
            'fail_reason': fail_reason if not passed else None
        }


def main():
    st.markdown('<h1 class="main-header">🚀 Ultimate Trading System</h1>', unsafe_allow_html=True)
    st.markdown("### Complete 2-Trade EA: Optimizer + Monte Carlo + Prop Firm Analysis")
    
    # Sidebar - Global Stats
    st.sidebar.header("📊 Your Trading Stats")
    
    total_trades = st.sidebar.number_input("Total Trades", 10, 1000, 93, 1)
    win_rate = st.sidebar.number_input("Win Rate (%)", 30.0, 95.0, 68.82, 0.01)
    avg_win = st.sidebar.number_input("Average Win ($)", 10.0, 10000.0, 989.40, 0.01)
    avg_loss = st.sidebar.number_input("Average Loss ($)", 10.0, 10000.0, 733.56, 0.01)
    account_size = st.sidebar.number_input("Account Size ($)", 10000, 1000000, 100000, 10000)
    
    st.sidebar.markdown("---")
    
    # Main content - Trade Settings
    st.header("⚙️ EA Configuration")
    
    col1, col2 = st.columns(2)
    
    # TRADE 1 SETTINGS
    with col1:
        st.markdown('<div class="trade-section">', unsafe_allow_html=True)
        st.subheader("🎯 Trade 1: Breakout")
        
        t1_sl = st.number_input("SL (pips)", 10, 200, 50, 1, key='t1_sl')
        t1_tp_ratio = st.number_input("TP Ratio", 0.5, 10.0, 2.65, 0.05, key='t1_tp')
        t1_buffer = st.number_input("Buffer (pips)", 0, 50, 2, 1, key='t1_buf')
        t1_risk = st.number_input("Risk %", 0.1, 10.0, 1.0, 0.1, key='t1_risk')
        
        st.markdown("**Break Even:**")
        t1_use_be = st.checkbox("Use BE", value=True, key='t1_be')
        t1_be_ratio = st.number_input("BE Ratio", 0.1, 3.0, 0.75, 0.05, key='t1_be_ratio')
        t1_be_buffer = st.number_input("BE Buffer (pips)", 0, 20, 0, 1, key='t1_be_buf')
        
        st.markdown("**Partial TP:**")
        t1_use_partial = st.checkbox("Use Partial", value=True, key='t1_part')
        t1_partial_ratio = st.number_input("Partial Ratio", 0.5, 5.0, 1.0, 0.1, key='t1_part_ratio')
        t1_partial_pct = st.number_input("Partial %", 10.0, 90.0, 75.0, 5.0, key='t1_part_pct')
        
        st.markdown("**Trailing Stop:**")
        t1_use_trail = st.checkbox("Use Trail", value=True, key='t1_trail')
        t1_trail_start = st.number_input("Trail Start (pips)", 10, 200, 40, 5, key='t1_trail_start')
        t1_trail_pips = st.number_input("Trail Pips", 10, 200, 60, 5, key='t1_trail_pips')
        
        st.markdown("**Filters:**")
        t1_days = {
            'monday': st.checkbox("Mon", value=False, key='t1_mon'),
            'tuesday': st.checkbox("Tue", value=True, key='t1_tue'),
            'wednesday': st.checkbox("Wed", value=True, key='t1_wed'),
            'thursday': st.checkbox("Thu", value=False, key='t1_thu'),
            'friday': st.checkbox("Fri", value=True, key='t1_fri'),
            'saturday': False,
            'sunday': False
        }
        
        t1_sessions = {
            'session1': st.checkbox("Session 1 (London)", value=True, key='t1_s1'),
            'session2': st.checkbox("Session 2 (NY)", value=True, key='t1_s2'),
            'session3': st.checkbox("Session 3 (Asian)", value=True, key='t1_s3')
        }
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # TRADE 2 SETTINGS
    with col2:
        st.markdown('<div class="trade-section">', unsafe_allow_html=True)
        st.subheader("🔄 Trade 2: Retracement")
        
        t2_pct_back = st.number_input("% Back in Range", 10.0, 90.0, 50.0, 5.0, key='t2_back')
        t2_sl_pct = st.number_input("% SL", 10.0, 200.0, 100.0, 5.0, key='t2_sl_pct')
        t2_tp_ratio = st.number_input("TP Ratio", 0.5, 10.0, 2.0, 0.05, key='t2_tp')
        t2_risk = st.number_input("Risk %", 0.1, 10.0, 1.0, 0.1, key='t2_risk')
        
        st.markdown("**Break Even:**")
        t2_use_be = st.checkbox("Use BE", value=True, key='t2_be')
        t2_be_ratio = st.number_input("BE Ratio", 0.1, 3.0, 0.75, 0.05, key='t2_be_ratio')
        t2_be_buffer = st.number_input("BE Buffer (pips)", 0, 20, 0, 1, key='t2_be_buf')
        
        st.markdown("**Partial TP:**")
        t2_use_partial = st.checkbox("Use Partial", value=True, key='t2_part')
        t2_partial_ratio = st.number_input("Partial Ratio", 0.5, 5.0, 1.0, 0.1, key='t2_part_ratio')
        t2_partial_pct = st.number_input("Partial %", 10.0, 90.0, 50.0, 5.0, key='t2_part_pct')
        
        st.markdown("**Trailing Stop:**")
        t2_use_trail = st.checkbox("Use Trail", value=True, key='t2_trail')
        t2_trail_start = st.number_input("Trail Start (pips)", 10, 200, 40, 5, key='t2_trail_start')
        t2_trail_pips = st.number_input("Trail Pips", 10, 200, 60, 5, key='t2_trail_pips')
        
        st.markdown("**Filters:**")
        t2_days = {
            'monday': st.checkbox("Mon", value=False, key='t2_mon'),
            'tuesday': st.checkbox("Tue", value=True, key='t2_tue'),
            'wednesday': st.checkbox("Wed", value=True, key='t2_wed'),
            'thursday': st.checkbox("Thu", value=False, key='t2_thu'),
            'friday': st.checkbox("Fri", value=True, key='t2_fri'),
            'saturday': False,
            'sunday': False
        }
        
        t2_sessions = {
            'session1': st.checkbox("Session 1 (London)", value=True, key='t2_s1'),
            'session2': st.checkbox("Session 2 (NY)", value=True, key='t2_s2'),
            'session3': st.checkbox("Session 3 (Asian)", value=True, key='t2_s3')
        }
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Build settings dictionaries
    stats = {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'account_size': account_size
    }
    
    trade1_settings = {
        'sl_pips': t1_sl,
        'tp_ratio': t1_tp_ratio,
        'buffer_pips': t1_buffer,
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
        'days': t1_days,
        'sessions': t1_sessions
    }
    
    trade2_settings = {
        'pct_back_in_range': t2_pct_back,
        'sl_pips': t1_sl * (t2_sl_pct / 100),  # Calculated from Trade 1 SL
        'tp_ratio': t2_tp_ratio,
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
        'days': t2_days,
        'sessions': t2_sessions
    }
    
    global_settings = {}
    
    # Tabs
    st.markdown("---")
    tabs = st.tabs(["💰 Monte Carlo Prop Firm", "📊 Quick Test", "📈 Optimizer (Coming Soon)"])
    
    # TAB 1: MONTE CARLO
    with tabs[0]:
        st.header("💰 Prop Firm Challenge Simulator")
        st.markdown("### Monte Carlo with Complete 2-Trade EA Logic")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Challenge Parameters")
            
            challenge_preset = st.selectbox("Preset", ["FTMO $100k", "MyForexFunds $100k", "Custom"])
            
            if challenge_preset == "FTMO $100k":
                defaults = (100000, 10.0, 5.0, 10.0, 540, 80)
            elif challenge_preset == "MyForexFunds $100k":
                defaults = (100000, 8.0, 5.0, 12.0, 499, 80)
            else:
                defaults = (account_size, 10.0, 5.0, 10.0, 540, 80)
            
            chal_account = st.number_input("Account Size ($)", 10000, 500000, defaults[0], 10000, key='mc_acc')
            profit_target = st.slider("Profit Target (%)", 1.0, 20.0, defaults[1], 0.5, key='mc_pt')
            daily_dd = st.slider("Daily DD Limit (%)", 1.0, 10.0, defaults[2], 0.5, key='mc_dd')
            max_dd = st.slider("Max DD Limit (%)", 5.0, 20.0, defaults[3], 0.5, key='mc_mdd')
            challenge_fee = st.number_input("Challenge Fee ($)", 0, 5000, defaults[4], 50, key='mc_fee')
            profit_split = st.slider("Profit Split (%)", 50, 100, defaults[5], 5, key='mc_split')
        
        with col2:
            st.subheader("Simulation Settings")
            
            num_simulations = st.selectbox("Simulations", [1000, 5000, 10000], index=2, key='mc_sims')
            max_days = st.slider("Max Days", 10, 90, 60, 5, key='mc_days')
            
            st.markdown("### Payout Analysis")
            min_payout_pct = st.slider("Min Payout %", 1.0, 10.0, 4.0, 0.5, key='mc_min_payout')
            max_payout_pct = st.slider("Max Payout %", 5.0, 20.0, 10.0, 0.5, key='mc_max_payout')
        
        if st.button("🚀 Run 10,000 Simulations", type="primary"):
            st.markdown("---")
            
            with st.spinner(f"Running {num_simulations:,} Monte Carlo simulations..."):
                progress = st.progress(0)
                
                challenge_params = {
                    'account_size': chal_account,
                    'profit_target_pct': profit_target,
                    'daily_dd_pct': daily_dd,
                    'max_dd_pct': max_dd
                }
                
                simulator = Complete2TradeEASimulator(stats, trade1_settings, trade2_settings, global_settings)
                
                results = []
                for i in range(num_simulations):
                    if (i + 1) % 100 == 0:
                        progress.progress((i + 1) / num_simulations)
                    
                    result = simulator.simulate_prop_firm_challenge(challenge_params, max_days)
                    results.append(result)
                
                progress.empty()
            
            results_df = pd.DataFrame(results)
            
            st.success("✅ Monte Carlo Simulation Complete!")
            
            # Calculate comprehensive statistics
            pass_rate = results_df['passed'].mean()
            passed_df = results_df[results_df['passed'] == True]
            
            avg_days = passed_df['days'].mean() if len(passed_df) > 0 else 0
            expected_attempts = 1 / pass_rate if pass_rate > 0 else float('inf')
            total_cost = expected_attempts * challenge_fee
            
            # Monthly/yearly profit estimates
            trades_per_month = (total_trades / 60) * 20
            avg_profit_per_trade = avg_win * win_rate/100 - avg_loss * (1 - win_rate/100)
            monthly_profit = avg_profit_per_trade * trades_per_month * (chal_account / account_size)
            yearly_profit = monthly_profit * 12 * (profit_split / 100)
            
            # ROI
            roi = ((yearly_profit - total_cost) / total_cost) * 100 if total_cost < float('inf') else 0
            
            # Payout analysis
            payout_chances = []
            for payout_pct in np.arange(min_payout_pct, max_payout_pct + 0.5, 0.5):
                payout_target = chal_account * (payout_pct / 100)
                # Simulate reaching payout then account failing
                num_payouts = 0
                for _ in range(100):  # Sub-simulation
                    balance = chal_account
                    payouts = 0
                    while balance > chal_account * 0.5:  # Before account fails
                        # Simulate trading month
                        month_profit = np.random.normal(monthly_profit, monthly_profit * 0.3)
                        balance += month_profit
                        if balance >= chal_account + payout_target:
                            payouts += 1
                            balance -= payout_target  # Payout taken
                        if balance < chal_account * (1 - max_dd/100):
                            break
                    num_payouts += payouts
                avg_payouts = num_payouts / 100
                payout_chances.append({'payout_pct': payout_pct, 'avg_payouts': avg_payouts})
            
            payout_df = pd.DataFrame(payout_chances)
            
            # Display results
            st.markdown("### 🎯 Key Results")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Pass Rate", f"{pass_rate:.1%}")
            with col2:
                st.metric("Avg Days", f"{avg_days:.0f}" if avg_days > 0 else "N/A")
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
            
            simulator = Complete2TradeEASimulator(stats, trade1_settings, trade2_settings, global_settings)
            
            results_df = simulator.simulate_full_backtest(100)
            
            if results_df is not None and len(results_df) > 0:
                # Calculate metrics
                total_profit = results_df['profit'].sum()
                wins = (results_df['is_win']).sum()
                losses = len(results_df) - wins
                win_rate_result = wins / len(results_df)
                
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
        st.header("🎯 EA Optimizer")
        st.markdown("### Coming Soon!")
        st.info("The optimizer will test different SL/TP/Trail combinations and find optimal settings.")


if __name__ == "__main__":
    main()
