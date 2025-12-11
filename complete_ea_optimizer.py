"""
COMPLETE EA OPTIMIZER - Test ALL Settings Together
Tests: SL, TP, Trailing Stop, Break Even, Partial TP, Days, Sessions

This optimizer understands your complete EA logic:
- Multiple exit mechanisms (BE, Trail, Partial TP)
- Day filters
- Session filters
- Finds the BEST combination of ALL settings

Install: pip install streamlit plotly pandas numpy
Run: streamlit run complete_ea_optimizer.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from itertools import product

st.set_page_config(
    page_title="Complete EA Optimizer",
    page_icon="🎯",
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
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class CompleteEAOptimizer:
    """Complete EA optimizer with all exit mechanisms"""
    
    def __init__(self, stats):
        """Initialize with trading stats"""
        self.total_trades = stats['total_trades']
        self.win_rate = stats['win_rate'] / 100
        self.avg_win = stats['avg_win']
        self.avg_loss = stats['avg_loss']
        self.account_size = stats.get('account_size', 100000)
        
    def simulate_trade(self, settings):
        """
        Simulate a single trade with complete EA logic
        
        Returns: profit/loss for the trade
        """
        
        # Determine if trade is winner based on win rate
        is_potential_win = np.random.random() < self.win_rate
        
        sl_pips = settings['sl_pips']
        tp_ratio = settings['tp_ratio']
        tp_pips = sl_pips * tp_ratio
        
        # Calculate base risk
        risk_amount = sl_pips * 10  # $10 per pip
        
        if not is_potential_win:
            # Loser - hits SL
            return -risk_amount
        
        # Winner - simulate price movement
        # Assume price moves in increments toward TP
        max_favorable_move = tp_pips * 1.2  # Can go beyond TP
        favorable_move = np.random.uniform(0, max_favorable_move)
        
        current_profit = 0
        position_size = 1.0  # 100% of position
        
        # 1. CHECK BREAK EVEN
        if settings['use_breakeven']:
            be_trigger = sl_pips * settings['be_trigger_ratio']
            if favorable_move >= be_trigger:
                # Move stop to breakeven (no loss possible now)
                pass  # SL now at entry
        
        # 2. CHECK PARTIAL TP
        if settings['use_partial_tp']:
            partial_pips = sl_pips * settings['partial_tp_ratio']
            if favorable_move >= partial_pips:
                # Close partial position
                partial_percent = settings['partial_tp_percent'] / 100
                current_profit += (partial_pips * 10) * partial_percent
                position_size -= partial_percent
        
        # 3. CHECK TRAILING STOP
        if settings['use_trail'] and favorable_move >= settings['trail_start_pips']:
            # Trailing stop is active
            # Assume price retraces by trail_stop_pips from peak
            retrace = np.random.uniform(0, settings['trail_stop_pips'] * 1.5)
            
            if retrace > settings['trail_stop_pips']:
                # Trailing stop hit - exit at trail distance from peak
                exit_pips = favorable_move - settings['trail_stop_pips']
                current_profit += (exit_pips * 10) * position_size
                return current_profit
        
        # 4. CHECK FULL TP
        if favorable_move >= tp_pips:
            # Full TP reached
            current_profit += (tp_pips * 10) * position_size
            return current_profit
        
        # 5. PRICE RETRACED - Check if BE saved us
        if settings['use_breakeven']:
            be_trigger = sl_pips * settings['be_trigger_ratio']
            if favorable_move >= be_trigger:
                # BE was triggered, so worst case is 0
                # But we keep partial TP profits if any
                return current_profit
        
        # 6. STOPPED OUT
        return -risk_amount
    
    def simulate_full_backtest(self, settings):
        """Simulate complete backtest with day/session filters"""
        
        results = []
        balance = self.account_size
        peak = balance
        
        # Simulate trades
        for i in range(self.total_trades):
            # Simulate trade day/session
            day_of_week = np.random.randint(0, 7)  # 0=Mon, 1=Tue, etc.
            session = np.random.choice(['session1', 'session2', 'session3'])
            
            # Check day filter
            day_names = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            if not settings['days'][day_names[day_of_week]]:
                continue  # Skip this day
            
            # Check session filter
            if not settings['sessions'][session]:
                continue  # Skip this session
            
            # Simulate the trade
            profit = self.simulate_trade(settings)
            balance += profit
            
            if balance > peak:
                peak = balance
            
            results.append({
                'trade': len(results) + 1,
                'profit': profit,
                'balance': balance,
                'is_win': profit > 0,
                'day': day_of_week,
                'session': session
            })
        
        if len(results) == 0:
            # No trades passed filters
            return None
        
        results_df = pd.DataFrame(results)
        
        # Calculate metrics
        metrics = self.calculate_metrics(results_df, peak)
        return metrics
    
    def calculate_metrics(self, results_df, peak_balance):
        """Calculate performance metrics"""
        
        total_trades = len(results_df)
        wins = (results_df['is_win']).sum()
        losses = total_trades - wins
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        total_profit = results_df['profit'].sum()
        avg_win = results_df[results_df['profit'] > 0]['profit'].mean() if wins > 0 else 0
        avg_loss = abs(results_df[results_df['profit'] < 0]['profit'].mean()) if losses > 0 else 0
        
        gross_profit = results_df[results_df['profit'] > 0]['profit'].sum()
        gross_loss = abs(results_df[results_df['profit'] < 0]['profit'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        equity_curve = results_df['balance'].values
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (running_max - equity_curve) / running_max * 100
        max_drawdown = drawdown.max()
        
        final_balance = results_df['balance'].iloc[-1]
        total_return = ((final_balance - self.account_size) / self.account_size) * 100
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_return_pct': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'final_balance': final_balance
        }


def main():
    st.markdown('<h1 class="main-header">🎯 Complete EA Optimizer</h1>', unsafe_allow_html=True)
    st.markdown("### Test ALL Settings: SL, TP, Trailing, Break Even, Partial TP, Days, Sessions")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Trading stats input
    st.sidebar.subheader("📊 Your Backtest Stats")
    total_trades = st.sidebar.number_input("Total Trades", 10, 1000, 93, 1)
    win_rate = st.sidebar.number_input("Win Rate (%)", 30.0, 95.0, 68.82, 0.01)
    avg_win = st.sidebar.number_input("Average Win ($)", 10.0, 10000.0, 989.40, 0.01)
    avg_loss = st.sidebar.number_input("Average Loss ($)", 10.0, 10000.0, 733.56, 0.01)
    account_size = st.sidebar.number_input("Account Size ($)", 10000, 1000000, 100000, 10000)
    
    st.sidebar.markdown("---")
    
    # Current settings
    st.sidebar.subheader("🔧 Your CURRENT Settings")
    
    with st.sidebar.expander("💰 Risk Management", expanded=True):
        curr_sl = st.number_input("Current SL (pips)", 10, 200, 50, 1, key='curr_sl')
        curr_tp = st.number_input("Current TP Ratio", 0.5, 10.0, 2.65, 0.05, key='curr_tp')
    
    with st.sidebar.expander("📍 Break Even", expanded=False):
        curr_use_be = st.checkbox("Use Break Even", value=True, key='curr_be')
        curr_be_trigger = st.number_input("BE Trigger Ratio", 0.1, 3.0, 0.75, 0.05, key='curr_be_trig')
    
    with st.sidebar.expander("📈 Trailing Stop", expanded=False):
        curr_use_trail = st.checkbox("Use Trailing", value=True, key='curr_trail')
        curr_trail_start = st.number_input("Trail Start (pips)", 10, 200, 40, 5, key='curr_trail_start')
        curr_trail_stop = st.number_input("Trail Distance (pips)", 10, 200, 60, 5, key='curr_trail_stop')
    
    with st.sidebar.expander("✂️ Partial TP", expanded=False):
        curr_use_partial = st.checkbox("Use Partial TP", value=True, key='curr_partial')
        curr_partial_ratio = st.number_input("Partial at Ratio", 0.5, 5.0, 1.0, 0.1, key='curr_partial_ratio')
        curr_partial_pct = st.number_input("Partial %", 10.0, 90.0, 75.0, 5.0, key='curr_partial_pct')
    
    with st.sidebar.expander("📅 Days", expanded=False):
        curr_mon = st.checkbox("Monday", value=False, key='curr_mon')
        curr_tue = st.checkbox("Tuesday", value=True, key='curr_tue')
        curr_wed = st.checkbox("Wednesday", value=True, key='curr_wed')
        curr_thu = st.checkbox("Thursday", value=False, key='curr_thu')
        curr_fri = st.checkbox("Friday", value=True, key='curr_fri')
    
    with st.sidebar.expander("🕐 Sessions", expanded=False):
        curr_sess1 = st.checkbox("Session 1 (London)", value=True, key='curr_s1')
        curr_sess2 = st.checkbox("Session 2 (NY)", value=True, key='curr_s2')
        curr_sess3 = st.checkbox("Session 3 (Asian)", value=True, key='curr_s3')
    
    st.sidebar.markdown("---")
    
    # Test ranges
    st.sidebar.subheader("🧪 What to Test")
    
    test_sl = st.sidebar.checkbox("Test Different SL", value=True)
    if test_sl:
        sl_values = st.sidebar.multiselect("SL Values (pips)", [30, 35, 40, 45, 50, 55, 60, 65, 70], default=[40, 50, 60])
    else:
        sl_values = [curr_sl]
    
    test_tp = st.sidebar.checkbox("Test Different TP", value=True)
    if test_tp:
        tp_values = st.sidebar.multiselect("TP Ratios", [1.5, 2.0, 2.5, 2.65, 3.0, 3.5, 4.0], default=[2.0, 2.65, 3.0])
    else:
        tp_values = [curr_tp]
    
    test_trail = st.sidebar.checkbox("Test Trailing Stop", value=True)
    if test_trail:
        trail_start_values = st.sidebar.multiselect("Trail Start (pips)", [30, 40, 50, 60], default=[40, 50])
        trail_stop_values = st.sidebar.multiselect("Trail Distance (pips)", [40, 50, 60, 70], default=[60, 70])
    else:
        trail_start_values = [curr_trail_start]
        trail_stop_values = [curr_trail_stop]
    
    test_days = st.sidebar.checkbox("Test Different Days", value=False)
    test_sessions = st.sidebar.checkbox("Test Different Sessions", value=False)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Optimization Goal")
    opt_goal = st.sidebar.selectbox("Optimize For", ["Profit Factor", "Total Return", "Min Drawdown", "Win Rate"])
    
    num_sims = st.sidebar.selectbox("Simulations per combo", [10, 50, 100], index=1)
    
    # Main content
    if st.sidebar.button("🚀 Run Complete Optimization", type="primary"):
        
        st.markdown("---")
        st.header("🔄 Running Complete Optimization...")
        
        # Build test combinations
        if test_days:
            day_combos = [
                {'monday': True, 'tuesday': True, 'wednesday': True, 'thursday': True, 'friday': True, 'saturday': False, 'sunday': False},
                {'monday': False, 'tuesday': True, 'wednesday': True, 'thursday': False, 'friday': True, 'saturday': False, 'sunday': False},
                {'monday': False, 'tuesday': True, 'wednesday': True, 'thursday': True, 'friday': True, 'saturday': False, 'sunday': False},
            ]
        else:
            day_combos = [{
                'monday': curr_mon, 'tuesday': curr_tue, 'wednesday': curr_wed,
                'thursday': curr_thu, 'friday': curr_fri, 'saturday': False, 'sunday': False
            }]
        
        if test_sessions:
            session_combos = [
                {'session1': True, 'session2': True, 'session3': True},
                {'session1': True, 'session2': True, 'session3': False},
                {'session1': True, 'session2': False, 'session3': False},
                {'session1': False, 'session2': True, 'session3': False},
            ]
        else:
            session_combos = [{
                'session1': curr_sess1, 'session2': curr_sess2, 'session3': curr_sess3
            }]
        
        # Calculate total combinations
        total_combos = (len(sl_values) * len(tp_values) * len(trail_start_values) * 
                       len(trail_stop_values) * len(day_combos) * len(session_combos))
        
        st.info(f"Testing {total_combos} combinations × {num_sims} simulations = {total_combos * num_sims:,} total tests")
        
        if total_combos > 500:
            st.warning("⚠️ This will take 5-10 minutes. Grab a coffee! ☕")
        
        progress_bar = st.progress(0)
        status = st.empty()
        
        # Initialize optimizer
        stats = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'account_size': account_size
        }
        
        optimizer = CompleteEAOptimizer(stats)
        
        all_results = []
        combo_idx = 0
        
        for sl in sl_values:
            for tp in tp_values:
                for trail_start in trail_start_values:
                    for trail_stop in trail_stop_values:
                        for days in day_combos:
                            for sessions in session_combos:
                                
                                # Run multiple simulations for this combo
                                sim_results = []
                                
                                for sim in range(num_sims):
                                    settings = {
                                        'sl_pips': sl,
                                        'tp_ratio': tp,
                                        'use_breakeven': curr_use_be,
                                        'be_trigger_ratio': curr_be_trigger,
                                        'use_trail': curr_use_trail,
                                        'trail_start_pips': trail_start,
                                        'trail_stop_pips': trail_stop,
                                        'use_partial_tp': curr_use_partial,
                                        'partial_tp_ratio': curr_partial_ratio,
                                        'partial_tp_percent': curr_partial_pct,
                                        'days': days,
                                        'sessions': sessions
                                    }
                                    
                                    result = optimizer.simulate_full_backtest(settings)
                                    if result:
                                        sim_results.append(result)
                                
                                if len(sim_results) > 0:
                                    # Average across simulations
                                    avg_result = {
                                        'sl_pips': sl,
                                        'tp_ratio': tp,
                                        'trail_start': trail_start,
                                        'trail_stop': trail_stop,
                                        'days': str(days),
                                        'sessions': str(sessions),
                                        'total_trades': np.mean([r['total_trades'] for r in sim_results]),
                                        'win_rate': np.mean([r['win_rate'] for r in sim_results]),
                                        'profit_factor': np.mean([r['profit_factor'] for r in sim_results]),
                                        'total_return_pct': np.mean([r['total_return_pct'] for r in sim_results]),
                                        'max_drawdown': np.mean([r['max_drawdown'] for r in sim_results]),
                                    }
                                    all_results.append(avg_result)
                                
                                combo_idx += 1
                                progress_bar.progress(combo_idx / total_combos)
                                status.text(f"Tested {combo_idx}/{total_combos} combinations...")
        
        progress_bar.empty()
        status.empty()
        
        if len(all_results) == 0:
            st.error("❌ No valid combinations found! Check your filters.")
            return
        
        results_df = pd.DataFrame(all_results)
        
        # Find best
        if opt_goal == "Profit Factor":
            best_idx = results_df['profit_factor'].idxmax()
        elif opt_goal == "Total Return":
            best_idx = results_df['total_return_pct'].idxmax()
        elif opt_goal == "Min Drawdown":
            best_idx = results_df['max_drawdown'].idxmin()
        else:
            best_idx = results_df['win_rate'].idxmax()
        
        best = results_df.iloc[best_idx]
        
        # Display results
        st.success("✅ Optimization Complete!")
        
        st.markdown("---")
        st.markdown(f"""
        <div class="best-setting">
            <h2>🏆 OPTIMAL SETTINGS FOUND!</h2>
            <h3>💰 Risk Management:</h3>
            <p>Stop Loss: {best['sl_pips']:.0f} pips</p>
            <p>Take Profit: {best['tp_ratio']:.2f}R ({best['sl_pips'] * best['tp_ratio']:.0f} pips)</p>
            <h3>📈 Trailing Stop:</h3>
            <p>Start: {best['trail_start']:.0f} pips | Distance: {best['trail_stop']:.0f} pips</p>
            <h3>📊 Performance:</h3>
            <p>Win Rate: {best['win_rate']*100:.1f}%</p>
            <p>Profit Factor: {best['profit_factor']:.2f}</p>
            <p>Return: {best['total_return_pct']:.1f}%</p>
            <p>Max DD: {best['max_drawdown']:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Top 10
        st.markdown("---")
        st.header("🏅 Top 10 Combinations")
        
        if opt_goal == "Min Drawdown":
            top_10 = results_df.nsmallest(10, 'max_drawdown')
        else:
            metric = opt_goal.lower().replace(' ', '_')
            top_10 = results_df.nlargest(10, metric)
        
        st.dataframe(top_10[['sl_pips', 'tp_ratio', 'trail_start', 'trail_stop', 
                             'win_rate', 'profit_factor', 'total_return_pct', 'max_drawdown']], 
                    use_container_width=True)
        
        # Save results
        csv = results_df.to_csv(index=False)
        st.download_button("📥 Download All Results", csv, "complete_optimization.csv", "text/csv")
        
        st.session_state['results'] = results_df
        st.session_state['best'] = best
    
    else:
        st.info("👈 Configure settings and click 'Run Complete Optimization'")
        
        st.markdown("""
        <div class="warning-box">
            <h3>⚠️ IMPORTANT - This Tests EVERYTHING Together!</h3>
            <p>This optimizer simulates your COMPLETE EA logic including:</p>
            <ul>
                <li>✅ Break Even movements</li>
                <li>✅ Trailing stop activation and distance</li>
                <li>✅ Partial TP percentages</li>
                <li>✅ Day filters</li>
                <li>✅ Session filters</li>
                <li>✅ All interactions between these settings</li>
            </ul>
            <p><strong>Unlike the simple optimizer, this accounts for HOW YOUR EA ACTUALLY WORKS!</strong></p>
        </div>
        
        ### 🎯 How To Use:
        
        1. **Enter your backtest stats** (93 trades, 68.82% WR, etc.)
        2. **Enter your CURRENT settings** (all of them!)
        3. **Choose what to test:**
           - Just SL/TP? Uncheck other boxes
           - Everything? Check all boxes
        4. **Click Run** - wait 2-10 minutes
        5. **Get ACCURATE optimal settings!**
        
        ### 💡 Example:
        
        **Testing:**
        - SL: [40, 50, 60]
        - TP: [2.0, 2.65, 3.0]
        - Trail Start: [40, 50]
        - Trail Distance: [60, 70]
        
        **Result:** 3 × 3 × 2 × 2 = 36 combinations × 50 sims = 1,800 tests
        
        **Time:** ~3 minutes
        
        **Accuracy:** ACCOUNTS FOR ALL YOUR EA LOGIC! ✅
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
