"""
COMPLETE TRADING ANALYSIS SYSTEM - ALL-IN-ONE
MT5 Backtest → Machine Learning → Monte Carlo → Prop Firm Simulator

Features:
1. Extract MT5 Strategy Tester results automatically
2. Analyze patterns with Machine Learning
3. Run Monte Carlo simulations
4. Predict prop firm challenge success
5. Beautiful web interface

Install: pip install streamlit plotly pandas numpy scikit-learn
Run: streamlit run complete_trading_system.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import io

st.set_page_config(
    page_title="Complete Trading Analysis System",
    page_icon="📊",
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
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ecc71;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #2ecc71;
        padding-bottom: 0.5rem;
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
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class MT5DataProcessor:
    """Process MT5 Strategy Tester results"""
    
    @staticmethod
    def load_mt5_backtest(file):
        """Load MT5 backtest results from HTML or CSV"""
        try:
            # Try reading as HTML first (MT5 Strategy Tester Report)
            df_list = pd.read_html(file)
            
            # Find the trades table
            for df in df_list:
                if 'Time' in df.columns or 'Profit' in df.columns:
                    df.columns = df.columns.str.strip().str.lower()
                    return df
            
            return None
        except:
            # Try CSV format
            try:
                df = pd.read_csv(file)
                df.columns = df.columns.str.strip().str.lower()
                return df
            except:
                return None
    
    @staticmethod
    def extract_features(df):
        """Extract features from backtest data for ML"""
        df = df.copy()
        
        # Ensure we have required columns
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            df['hour'] = df['time'].dt.hour
            df['day_of_week'] = df['time'].dt.dayofweek
            df['month'] = df['time'].dt.month
        else:
            df['hour'] = 0
            df['day_of_week'] = 0
            df['month'] = 0
        
        # Win/Loss indicator
        if 'profit' in df.columns:
            df['is_win'] = (df['profit'] > 0).astype(int)
        else:
            df['is_win'] = 0
        
        # Trade direction
        if 'type' in df.columns:
            df['is_buy'] = df['type'].str.lower().str.contains('buy', na=False).astype(int)
        else:
            df['is_buy'] = 0
        
        # Rolling statistics
        if 'profit' in df.columns:
            df['win_streak'] = df['is_win'].rolling(5, min_periods=1).sum()
            df['avg_profit_last5'] = df['profit'].rolling(5, min_periods=1).mean()
            df['volatility_last10'] = df['profit'].rolling(10, min_periods=1).std().fillna(0)
        
        return df


class MLAnalyzer:
    """Machine Learning analysis"""
    
    def __init__(self):
        self.model = None
        self.feature_importance = None
    
    def train(self, df):
        """Train ML model to predict trade outcomes"""
        
        # Select features
        feature_cols = ['hour', 'day_of_week', 'month', 'is_buy', 
                       'win_streak', 'avg_profit_last5', 'volatility_last10']
        
        # Filter to available features
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        if len(feature_cols) < 3:
            return None, None
        
        X = df[feature_cols].fillna(0)
        y = df['is_win']
        
        # Remove rows where target is missing
        valid_idx = y.notna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 20:
            return None, None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'feature_importance': self.feature_importance
        }, (X_test, y_test)


class PropFirmSimulator:
    """Monte Carlo simulator for prop firm challenges"""
    
    def __init__(self, trading_data, challenge_params):
        self.df = trading_data
        self.params = challenge_params
        
        # Extract statistics
        if 'profit' in self.df.columns:
            self.win_rate = (self.df['profit'] > 0).mean()
            self.wins = self.df[self.df['profit'] > 0]['profit'].values
            self.losses = self.df[self.df['profit'] < 0]['profit'].values
            self.avg_win = self.wins.mean() if len(self.wins) > 0 else 0
            self.avg_loss = abs(self.losses.mean()) if len(self.losses) > 0 else 0
        else:
            self.win_rate = 0.5
            self.wins = np.array([50])
            self.losses = np.array([-50])
            self.avg_win = 50
            self.avg_loss = 50
        
        self.trades_per_day = self._estimate_trades_per_day()
    
    def _estimate_trades_per_day(self):
        if 'time' in self.df.columns:
            df_time = self.df.dropna(subset=['time'])
            if len(df_time) > 0:
                days = (df_time['time'].max() - df_time['time'].min()).days
                if days > 0:
                    return len(df_time) / days
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
                    
                    # Check daily drawdown
                    daily_loss = daily_start_balance - balance
                    if daily_loss > daily_dd_limit:
                        fail_reason = "Daily Drawdown"
                        return self._create_result(False, step, total_days, total_trades, 
                                                   balance, starting_balance, peak_balance, fail_reason)
                    
                    # Check max drawdown
                    current_drawdown = peak_balance - balance
                    if current_drawdown > max_dd_limit:
                        fail_reason = "Max Drawdown"
                        return self._create_result(False, step, total_days, total_trades, 
                                                   balance, starting_balance, peak_balance, fail_reason)
                
                total_days += 1
                
                # Check profit target
                step_profit = balance - step_starting_balance
                if step_profit >= profit_target:
                    if step == self.params['num_steps']:
                        passed = True
                        break
                    else:
                        break
                
                if total_days >= max_days:
                    fail_reason = "Max Days"
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


def create_sample_data():
    """Create sample trading data"""
    np.random.seed(42)
    trades = []
    
    for i in range(200):
        is_win = np.random.random() < 0.58
        profit = np.random.uniform(20, 100) if is_win else np.random.uniform(-50, -20)
        
        trades.append({
            'time': datetime.now() - timedelta(days=200-i),
            'profit': profit,
            'type': np.random.choice(['Buy', 'Sell'])
        })
    
    return pd.DataFrame(trades)


def main():
    st.markdown('<h1 class="main-header">📊 Complete Trading Analysis System</h1>', unsafe_allow_html=True)
    st.markdown("### MT5 Backtest → Machine Learning → Monte Carlo → Prop Firm Prediction")
    
    # Sidebar
    st.sidebar.header("⚙️ System Configuration")
    
    # Step 1: Data Upload
    st.sidebar.markdown("### 📁 Step 1: Upload MT5 Data")
    
    data_source = st.sidebar.radio(
        "Data Source",
        ["Use Sample Data", "Upload MT5 Backtest Report", "Upload CSV"]
    )
    
    df = None
    
    if data_source == "Use Sample Data":
        df = create_sample_data()
        st.sidebar.success("✅ Using sample data (200 trades)")
    
    elif data_source == "Upload MT5 Backtest Report":
        st.sidebar.markdown("""
        **How to export from MT5:**
        1. Strategy Tester → Results tab
        2. Right-click → Report → Save as HTML
        3. Upload that HTML file here
        """)
        
        uploaded_file = st.sidebar.file_uploader("Upload HTML Report", type=['html', 'htm'])
        if uploaded_file:
            processor = MT5DataProcessor()
            df = processor.load_mt5_backtest(uploaded_file)
            if df is not None:
                st.sidebar.success(f"✅ Loaded {len(df)} trades from backtest")
            else:
                st.sidebar.error("❌ Could not parse file")
    
    else:  # Upload CSV
        uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip().str.lower()
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'], errors='coerce')
            st.sidebar.success(f"✅ Loaded {len(df)} trades")
    
    if df is None:
        st.info("👈 Please upload your MT5 backtest data or use sample data to begin")
        
        st.markdown("""
        <div class="info-box">
            <h3>📖 How This System Works:</h3>
            <ol>
                <li><strong>Upload MT5 Backtest</strong> - Your strategy tester results</li>
                <li><strong>ML Analysis</strong> - AI finds patterns in your trades</li>
                <li><strong>Configure Prop Firm</strong> - Set challenge parameters</li>
                <li><strong>Run Simulation</strong> - See your pass probability</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Main content
    tabs = st.tabs(["📊 Data Overview", "🤖 ML Analysis", "💰 Prop Firm Simulator", "📥 Export"])
    
    # TAB 1: Data Overview
    with tabs[0]:
        st.markdown('<div class="section-header">📊 Trading Data Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", len(df))
        
        with col2:
            if 'profit' in df.columns:
                win_rate = (df['profit'] > 0).mean()
                st.metric("Win Rate", f"{win_rate:.1%}")
            else:
                st.metric("Win Rate", "N/A")
        
        with col3:
            if 'profit' in df.columns:
                avg_win = df[df['profit'] > 0]['profit'].mean()
                st.metric("Avg Win", f"${avg_win:.2f}" if not pd.isna(avg_win) else "N/A")
            else:
                st.metric("Avg Win", "N/A")
        
        with col4:
            if 'profit' in df.columns:
                avg_loss = abs(df[df['profit'] < 0]['profit'].mean())
                st.metric("Avg Loss", f"${avg_loss:.2f}" if not pd.isna(avg_loss) else "N/A")
            else:
                st.metric("Avg Loss", "N/A")
        
        st.markdown("---")
        
        # Show data table
        st.subheader("Recent Trades")
        st.dataframe(df.head(20), use_container_width=True)
        
        # Profit distribution
        if 'profit' in df.columns:
            st.subheader("Profit Distribution")
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df['profit'], nbinsx=50, marker_color='#3498db'))
            fig.update_layout(xaxis_title="Profit ($)", yaxis_title="Frequency", height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: ML Analysis
    with tabs[1]:
        st.markdown('<div class="section-header">🤖 Machine Learning Analysis</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Run ML Analysis", type="primary"):
            with st.spinner("Training ML model..."):
                processor = MT5DataProcessor()
                df_features = processor.extract_features(df)
                
                analyzer = MLAnalyzer()
                ml_results, test_data = analyzer.train(df_features)
                
                if ml_results:
                    st.success("✅ ML Analysis Complete!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Training Accuracy", f"{ml_results['train_score']:.1%}")
                    
                    with col2:
                        st.metric("Testing Accuracy", f"{ml_results['test_score']:.1%}")
                    
                    st.markdown("---")
                    st.subheader("Feature Importance - What Affects Your Win Rate?")
                    
                    fig = go.Figure(go.Bar(
                        x=ml_results['feature_importance']['importance'],
                        y=ml_results['feature_importance']['feature'],
                        orientation='h',
                        marker_color='#2ecc71'
                    ))
                    fig.update_layout(
                        xaxis_title="Importance",
                        yaxis_title="Feature",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    <div class="info-box">
                        <h4>💡 How to Use These Insights:</h4>
                        <ul>
                            <li><strong>High importance features</strong> - Focus on these conditions</li>
                            <li><strong>Hour/Day patterns</strong> - Trade during best times</li>
                            <li><strong>Win streaks</strong> - Manage psychology after wins/losses</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Store in session state
                    st.session_state['ml_results'] = ml_results
                    st.session_state['df_features'] = df_features
                else:
                    st.error("❌ Not enough data for ML analysis (need at least 20 trades)")
        
        else:
            st.info("Click 'Run ML Analysis' to discover patterns in your trading")
    
    # TAB 3: Prop Firm Simulator
    with tabs[2]:
        st.markdown('<div class="section-header">💰 Prop Firm Challenge Simulator</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Challenge Parameters")
            
            challenge_preset = st.selectbox(
                "Preset",
                ["Custom", "FTMO $100k", "MyForexFunds $100k"]
            )
            
            if challenge_preset == "FTMO $100k":
                defaults = (100000, 10.0, 5.0, 10.0, 2, 540)
            elif challenge_preset == "MyForexFunds $100k":
                defaults = (100000, 8.0, 5.0, 12.0, 1, 499)
            else:
                defaults = (100000, 10.0, 5.0, 10.0, 2, 540)
            
            account_size = st.number_input("Account Size ($)", 10000, 500000, defaults[0], 10000)
            profit_target_pct = st.slider("Profit Target (%)", 1.0, 20.0, defaults[1], 0.5)
            daily_dd_pct = st.slider("Daily DD Limit (%)", 1.0, 10.0, defaults[2], 0.5)
            max_dd_pct = st.slider("Max DD Limit (%)", 5.0, 20.0, defaults[3], 0.5)
            num_steps = st.selectbox("Steps", [1, 2], index=1 if defaults[4] == 2 else 0)
            challenge_fee = st.number_input("Challenge Fee ($)", 0, 5000, defaults[5], 50)
        
        with col2:
            st.subheader("Simulation Settings")
            
            num_simulations = st.selectbox("Simulations", [1000, 5000, 10000], index=2)
            max_days = st.slider("Max Days", 10, 90, 60, 5)
            profit_split = st.slider("Profit Split (%)", 50, 100, 80, 5)
        
        if st.button("🚀 Run Prop Firm Simulation", type="primary"):
            st.markdown("---")
            
            with st.spinner(f"Running {num_simulations:,} simulations..."):
                progress_bar = st.progress(0)
                
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
                results = simulator.run_simulation(num_simulations, max_days, progress_bar)
                
                progress_bar.empty()
            
            st.success("✅ Simulation Complete!")
            
            # Calculate statistics
            pass_rate = results['passed'].mean()
            passed_df = results[results['passed'] == True]
            
            avg_days = passed_df['days'].mean() if len(passed_df) > 0 else 0
            expected_attempts = 1 / pass_rate if pass_rate > 0 else float('inf')
            total_cost = expected_attempts * challenge_fee
            
            avg_trade_profit = df['profit'].mean() if 'profit' in df.columns else 0
            trades_per_month = simulator.trades_per_day * 20
            monthly_profit = avg_trade_profit * trades_per_month * (account_size / 10000)
            yearly_profit = monthly_profit * 12 * (profit_split / 100)
            
            roi = ((yearly_profit - total_cost) / total_cost) * 100 if total_cost > 0 and total_cost != float('inf') else 0
            
            # Results
            st.markdown("### 📈 Key Results")
            
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
                st.metric("ROI", f"{roi:.0f}%")
            
            # Recommendation
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
                st.markdown("""
                <div class="warning-box">
                    <h3>❌ NOT RECOMMENDED</h3>
                    <p>Consider improving your strategy first.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Charts
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
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
                if len(passed_df) > 0:
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=passed_df['days'], nbinsx=30, marker_color='#3498db'))
                    fig.update_layout(title="Days to Pass", xaxis_title="Days", yaxis_title="Count", height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Store results
            st.session_state['prop_results'] = results
            st.session_state['prop_stats'] = {
                'pass_rate': pass_rate,
                'roi': roi,
                'total_cost': total_cost,
                'yearly_profit': yearly_profit
            }
    
    # TAB 4: Export
    with tabs[3]:
        st.markdown('<div class="section-header">📥 Export Results</div>', unsafe_allow_html=True)
        
        st.subheader("Download Reports")
        
        # Complete report
        report = f"""
COMPLETE TRADING ANALYSIS REPORT
{'='*70}

TRADING STATISTICS
{'='*70}
Total Trades: {len(df)}
"""
        
        if 'profit' in df.columns:
            win_rate = (df['profit'] > 0).mean()
            avg_win = df[df['profit'] > 0]['profit'].mean()
            avg_loss = abs(df[df['profit'] < 0]['profit'].mean())
            
            report += f"""Win Rate: {win_rate:.2%}
Average Win: ${avg_win:.2f}
Average Loss: ${avg_loss:.2f}
"""
        
        if 'ml_results' in st.session_state:
            ml_results = st.session_state['ml_results']
            report += f"""

MACHINE LEARNING ANALYSIS
{'='*70}
Model Accuracy: {ml_results['test_score']:.2%}

Top Features:
"""
            for idx, row in ml_results['feature_importance'].head().iterrows():
                report += f"  - {row['feature']}: {row['importance']:.3f}\n"
        
        if 'prop_stats' in st.session_state:
            stats = st.session_state['prop_stats']
            report += f"""

PROP FIRM SIMULATION
{'='*70}
Pass Rate: {stats['pass_rate']:.2%}
ROI: {stats['roi']:.0f}%
Total Expected Cost: ${stats['total_cost']:,.0f}
Estimated Yearly Profit: ${stats['yearly_profit']:,.0f}
"""
        
        st.download_button(
            label="📄 Download Complete Report",
            data=report,
            file_name="complete_trading_analysis.txt",
            mime="text/plain"
        )
        
        # CSV export
        if 'prop_results' in st.session_state:
            csv = st.session_state['prop_results'].to_csv(index=False)
            st.download_button(
                label="📊 Download Simulation Data (CSV)",
                data=csv,
                file_name="prop_firm_simulation.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()
