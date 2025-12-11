"""
PROP FIRM CHALLENGE SIMULATOR - WEB APP
Beautiful, interactive web interface for Monte Carlo simulation

Install: pip install streamlit plotly
Run: streamlit run prop_firm_web_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="Prop Firm Challenge Simulator",
    page_icon="💰",
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
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class PropFirmSimulator:
    """Monte Carlo simulator"""
    
    def __init__(self, trading_data, challenge_params):
        self.df = trading_data
        self.params = challenge_params
        
        self.win_rate = (self.df['profit'] > 0).mean()
        self.wins = self.df[self.df['profit'] > 0]['profit'].values
        self.losses = self.df[self.df['profit'] < 0]['profit'].values
        self.avg_win = self.wins.mean() if len(self.wins) > 0 else 0
        self.avg_loss = abs(self.losses.mean()) if len(self.losses) > 0 else 0
        self.trades_per_day = self._estimate_trades_per_day()
    
    def _estimate_trades_per_day(self):
        if 'time' in self.df.columns:
            days = (self.df['time'].max() - self.df['time'].min()).days
            if days > 0:
                return len(self.df) / days
        return 3
    
    def run_simulation(self, num_simulations, max_days, progress_bar=None):
        results = []
        
        for sim in range(num_simulations):
            if progress_bar and (sim + 1) % 100 == 0:
                progress_bar.progress((sim + 1) / num_simulations)
            
            result = self._simulate_single_challenge(max_days)
            results.append(result)
        
        return pd.DataFrame(results)
    
    def _simulate_single_challenge(self, max_days):
        balance = self.params['account_size']
        starting_balance = balance
        peak_balance = balance
        
        total_days = 0
        total_trades = 0
        passed = False
        fail_reason = None
        
        profit_target = balance * (self.params['profit_target_pct'] / 100)
        daily_dd_limit = balance * (self.params['daily_dd_pct'] / 100)
        max_dd_limit = balance * (self.params['max_dd_pct'] / 100)
        
        for step in range(1, self.params['num_steps'] + 1):
            step_starting_balance = balance
            step_peak = balance
            
            for day in range(max_days):
                daily_start_balance = balance
                
                num_trades_today = max(1, np.random.poisson(self.trades_per_day))
                
                for trade in range(num_trades_today):
                    is_win = np.random.random() < self.win_rate
                    
                    if is_win and len(self.wins) > 0:
                        profit = np.random.choice(self.wins)
                    elif not is_win and len(self.losses) > 0:
                        profit = np.random.choice(self.losses)
                    else:
                        profit = 0
                    
                    profit_scaled = profit * (self.params['account_size'] / 10000)
                    balance += profit_scaled
                    total_trades += 1
                    
                    if balance > peak_balance:
                        peak_balance = balance
                    if balance > step_peak:
                        step_peak = balance
                    
                    daily_loss = daily_start_balance - balance
                    if daily_loss > daily_dd_limit:
                        fail_reason = "Daily Drawdown Breached"
                        return self._create_result(False, step, total_days, total_trades, 
                                                   balance, starting_balance, peak_balance, fail_reason)
                    
                    current_drawdown = peak_balance - balance
                    if current_drawdown > max_dd_limit:
                        fail_reason = "Max Drawdown Breached"
                        return self._create_result(False, step, total_days, total_trades, 
                                                   balance, starting_balance, peak_balance, fail_reason)
                
                total_days += 1
                
                step_profit = balance - step_starting_balance
                if step_profit >= profit_target:
                    if step == self.params['num_steps']:
                        passed = True
                        break
                    else:
                        break
                
                if total_days >= max_days:
                    fail_reason = "Max Days Exceeded"
                    break
            
            if passed:
                break
            if fail_reason:
                break
            
            step_profit = balance - step_starting_balance
            if step_profit < profit_target:
                fail_reason = f"Failed Step {step}"
                break
        
        return self._create_result(passed, step, total_days, total_trades, 
                                   balance, starting_balance, peak_balance, fail_reason)
    
    def _create_result(self, passed, step, days, trades, balance, starting, peak, fail_reason):
        return {
            'passed': passed,
            'step_reached': step,
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


def create_sample_data(num_trades=200):
    np.random.seed(42)
    trades = []
    
    for i in range(num_trades):
        win_prob = 0.58
        is_win = np.random.random() < win_prob
        
        if is_win:
            profit = np.random.uniform(20, 100)
        else:
            profit = np.random.uniform(-50, -20)
        
        trades.append({
            'time': datetime.now() - timedelta(days=num_trades-i),
            'profit': profit,
            'type': np.random.choice(['Buy', 'Sell'])
        })
    
    return pd.DataFrame(trades)


def main():
    st.markdown('<h1 class="main-header">💰 Prop Firm Challenge Simulator</h1>', unsafe_allow_html=True)
    st.markdown("### Monte Carlo simulation to predict your chances of passing prop firm challenges")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    data_source = st.sidebar.radio(
        "Data Source",
        ["Use Sample Data", "Upload My MT5 History"]
    )
    
    if data_source == "Upload My MT5 History":
        uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip().str.lower()
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
            st.sidebar.success(f"✅ Loaded {len(df)} trades")
        else:
            st.sidebar.warning("⚠️ Please upload your MT5 history CSV")
            df = create_sample_data()
    else:
        df = create_sample_data()
        st.sidebar.info("📊 Using 200 sample trades for demo")
    
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Challenge Parameters")
    
    challenge_preset = st.sidebar.selectbox(
        "Preset",
        ["Custom", "FTMO $100k", "MyForexFunds $100k"]
    )
    
    if challenge_preset == "FTMO $100k":
        defaults = (100000, 10.0, 5.0, 10.0, 2, 540, 80)
    elif challenge_preset == "MyForexFunds $100k":
        defaults = (100000, 8.0, 5.0, 12.0, 1, 499, 80)
    else:
        defaults = (100000, 10.0, 5.0, 10.0, 2, 540, 80)
    
    account_size = st.sidebar.number_input("Account Size ($)", 10000, 500000, defaults[0], 10000)
    profit_target_pct = st.sidebar.slider("Profit Target (%)", 1.0, 20.0, defaults[1], 0.5)
    daily_dd_pct = st.sidebar.slider("Daily DD Limit (%)", 1.0, 10.0, defaults[2], 0.5)
    max_dd_pct = st.sidebar.slider("Max DD Limit (%)", 5.0, 20.0, defaults[3], 0.5)
    num_steps = st.sidebar.selectbox("Steps", [1, 2], index=1 if defaults[4] == 2 else 0)
    challenge_fee = st.sidebar.number_input("Challenge Fee ($)", 0, 5000, defaults[5], 50)
    profit_split = st.sidebar.slider("Profit Split (%)", 50, 100, defaults[6], 5)
    
    st.sidebar.markdown("---")
    st.sidebar.header("🔬 Simulation Settings")
    
    num_simulations = st.sidebar.selectbox("Simulations", [1000, 5000, 10000], index=2)
    max_days = st.sidebar.slider("Max Days", 10, 90, 60, 5)
    
    run_button = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)
    
    if run_button:
        # Stats
        st.header("📊 Your Trading Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", len(df))
        with col2:
            win_rate = (df['profit'] > 0).mean()
            st.metric("Win Rate", f"{win_rate:.1%}")
        with col3:
            avg_win = df[df['profit'] > 0]['profit'].mean()
            st.metric("Avg Win", f"${avg_win:.2f}")
        with col4:
            avg_loss = abs(df[df['profit'] < 0]['profit'].mean())
            st.metric("Avg Loss", f"${avg_loss:.2f}")
        
        st.markdown("---")
        st.header("🔬 Running Simulation...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        challenge_params = {
            'account_size': account_size,
            'profit_target_pct': profit_target_pct,
            'daily_dd_pct': daily_dd_pct,
            'max_dd_pct': max_dd_pct,
            'num_steps': num_steps,
            'challenge_fee': challenge_fee,
            'profit_split': profit_split
        }
        
        simulator = PropFirmSimulator(df, challenge_params)
        
        status_text.text(f"Running {num_simulations:,} simulations...")
        results = simulator.run_simulation(num_simulations, max_days, progress_bar)
        
        progress_bar.empty()
        status_text.empty()
        
        # Calculate stats
        pass_rate = results['passed'].mean()
        passed_df = results[results['passed'] == True]
        
        if len(passed_df) > 0:
            avg_days = passed_df['days'].mean()
        else:
            avg_days = 0
        
        expected_attempts = 1 / pass_rate if pass_rate > 0 else float('inf')
        total_cost = expected_attempts * challenge_fee
        
        avg_trade_profit = df['profit'].mean()
        trades_per_month = simulator.trades_per_day * 20
        monthly_profit = avg_trade_profit * trades_per_month * (account_size / 10000)
        yearly_profit = monthly_profit * 12 * (profit_split / 100)
        
        roi = ((yearly_profit - total_cost) / total_cost) * 100 if total_cost > 0 and total_cost != float('inf') else 0
        
        st.success("✅ Simulation Complete!")
        
        # Results
        st.header("📈 Key Results")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Pass Rate", f"{pass_rate:.1%}")
        with col2:
            st.metric("Avg Days", f"{avg_days:.1f}" if avg_days > 0 else "N/A")
        with col3:
            st.metric("Expected Attempts", f"{expected_attempts:.1f}")
        with col4:
            st.metric("Total Cost", f"${total_cost:,.0f}")
        with col5:
            st.metric("Yearly Profit", f"${yearly_profit:,.0f}")
        
        st.markdown("---")
        st.header("💰 Financial Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("ROI", f"{roi:.0f}%")
        with col2:
            breakeven = total_cost / (monthly_profit * (profit_split / 100)) if monthly_profit > 0 else float('inf')
            st.metric("Breakeven", f"{breakeven:.1f} mo" if breakeven != float('inf') else "N/A")
        with col3:
            st.metric("Monthly Profit", f"${monthly_profit * (profit_split / 100):,.0f}")
        
        # Recommendation
        st.markdown("---")
        
        if pass_rate >= 0.6 and roi >= 500:
            st.markdown(f"""
            <div class="success-box">
                <h3>✅ HIGHLY RECOMMENDED</h3>
                <p>Excellent odds! {pass_rate:.1%} pass rate, {roi:.0f}% ROI.</p>
            </div>
            """, unsafe_allow_html=True)
        elif pass_rate >= 0.4 and roi >= 200:
            st.markdown(f"""
            <div class="warning-box">
                <h3>⚠️ PROCEED WITH CAUTION</h3>
                <p>Decent odds: {pass_rate:.1%} pass rate, {roi:.0f}% ROI.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="danger-box">
                <h3>❌ NOT RECOMMENDED</h3>
                <p>Low odds: {pass_rate:.1%} pass rate, {roi:.0f}% ROI.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts
        st.markdown("---")
        st.header("📊 Visual Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pass rate pie
            pass_counts = results['passed'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=['Passed', 'Failed'],
                values=[pass_counts.get(True, 0), pass_counts.get(False, 0)],
                marker_colors=['#28a745', '#dc3545'],
                hole=0.4
            )])
            fig.update_layout(title="Pass Rate", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Days histogram
            if len(passed_df) > 0:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=passed_df['days'], nbinsx=30, marker_color='#3498db'))
                fig.update_layout(title="Days to Pass", xaxis_title="Days", yaxis_title="Count", height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Download
        st.markdown("---")
        report = f"""Pass Rate: {pass_rate:.2%}
Avg Days: {avg_days:.1f}
Total Cost: ${total_cost:,.0f}
ROI: {roi:.0f}%
"""
        st.download_button("📄 Download Report", report, "report.txt")
    
    else:
        st.info("👈 Configure parameters and click 'Run Simulation'")


if __name__ == "__main__":
    main()
