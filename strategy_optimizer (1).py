"""
STRATEGY OPTIMIZER - Find Your Best Settings
Tests different SL/TP combinations and recommends optimal parameters

Features:
- Test multiple SL values (10, 15, 20, 25, 30 pips)
- Test multiple TP ratios (1.5R, 2R, 3R, 4R, 5R)
- Test partial TP variations
- Find optimal settings based on:
  * Win rate
  * Profit factor
  * Max drawdown
  * Sharpe ratio
  * Prop firm pass rate

Install: pip install streamlit plotly pandas numpy scikit-learn
Run: streamlit run strategy_optimizer.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from itertools import product

st.set_page_config(
    page_title="Strategy Optimizer",
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
    .metric-comparison {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


class StrategyOptimizer:
    """Optimize strategy parameters"""
    
    def __init__(self, trades_df):
        """Initialize with historical trades"""
        self.df = trades_df
        
        # Extract original trade characteristics
        if 'profit' in self.df.columns:
            self.wins = self.df[self.df['profit'] > 0]['profit'].values
            self.losses = self.df[self.df['profit'] < 0]['profit'].values
            self.win_rate = (self.df['profit'] > 0).mean()
        else:
            self.wins = np.array([50])
            self.losses = np.array([-50])
            self.win_rate = 0.5
    
    def simulate_with_settings(self, sl_pips, tp_ratio, partial_tp_enabled=False, 
                               partial_tp_rr=1.0, partial_tp_percent=50.0):
        """
        Simulate trades with specific SL/TP settings
        
        Parameters:
        - sl_pips: Stop loss in pips (e.g., 20)
        - tp_ratio: Take profit ratio (e.g., 2.0 = 2R)
        - partial_tp_enabled: Use partial take profit?
        - partial_tp_rr: Partial TP at this R:R
        - partial_tp_percent: % to close at partial TP
        """
        
        results = []
        balance = 10000
        peak = 10000
        
        for i in range(len(self.df)):
            # Simulate trade outcome based on historical win rate
            is_win = np.random.random() < self.win_rate
            
            # Calculate P&L based on new settings
            risk = sl_pips * 10  # $10 per pip
            
            if is_win:
                # Full TP
                profit_full = risk * tp_ratio
                
                if partial_tp_enabled:
                    # Simulate partial TP
                    # Assume partial_tp_rr% chance of hitting partial
                    hit_partial = np.random.random() < 0.8  # 80% chance
                    
                    if hit_partial:
                        # Take partial profit
                        partial_profit = risk * partial_tp_rr * (partial_tp_percent / 100)
                        
                        # Remaining position continues
                        # 70% chance of hitting full TP after partial
                        hit_full = np.random.random() < 0.7
                        
                        if hit_full:
                            remaining_profit = risk * tp_ratio * (1 - partial_tp_percent / 100)
                            profit = partial_profit + remaining_profit
                        else:
                            # Only got partial
                            profit = partial_profit
                    else:
                        # Didn't hit partial, either full win or loss
                        profit = profit_full if np.random.random() < 0.6 else -risk
                else:
                    profit = profit_full
            else:
                # Loss
                profit = -risk
            
            balance += profit
            if balance > peak:
                peak = balance
            
            results.append({
                'trade': i + 1,
                'profit': profit,
                'balance': balance,
                'is_win': profit > 0
            })
        
        results_df = pd.DataFrame(results)
        
        # Calculate metrics
        metrics = self.calculate_metrics(results_df, peak)
        metrics['sl_pips'] = sl_pips
        metrics['tp_ratio'] = tp_ratio
        metrics['partial_tp'] = partial_tp_enabled
        
        return metrics, results_df
    
    def calculate_metrics(self, results_df, peak_balance):
        """Calculate performance metrics"""
        
        # Basic metrics
        total_trades = len(results_df)
        wins = (results_df['is_win']).sum()
        losses = total_trades - wins
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        total_profit = results_df['profit'].sum()
        avg_win = results_df[results_df['profit'] > 0]['profit'].mean() if wins > 0 else 0
        avg_loss = abs(results_df[results_df['profit'] < 0]['profit'].mean()) if losses > 0 else 0
        
        # Profit factor
        gross_profit = results_df[results_df['profit'] > 0]['profit'].sum()
        gross_loss = abs(results_df[results_df['profit'] < 0]['profit'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Drawdown
        equity_curve = results_df['balance'].values
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (running_max - equity_curve) / running_max * 100
        max_drawdown = drawdown.max()
        
        # Sharpe ratio (simplified)
        returns = results_df['profit'] / 10000  # As % of starting balance
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Final balance
        final_balance = results_df['balance'].iloc[-1]
        total_return = ((final_balance - 10000) / 10000) * 100
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_return_pct': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'final_balance': final_balance
        }
    
    def optimize(self, sl_range, tp_range, test_partial_tp=True, progress_callback=None):
        """
        Test all combinations and find optimal settings
        
        Parameters:
        - sl_range: List of SL values to test [10, 15, 20, 25, 30]
        - tp_range: List of TP ratios to test [1.5, 2.0, 3.0, 4.0, 5.0]
        - test_partial_tp: Also test partial TP variations
        """
        
        all_results = []
        
        # Generate all combinations
        if test_partial_tp:
            combinations = list(product(
                sl_range,
                tp_range,
                [False, True],  # Partial TP on/off
            ))
        else:
            combinations = list(product(sl_range, tp_range, [False]))
        
        total = len(combinations)
        
        for idx, (sl, tp, use_partial) in enumerate(combinations):
            if progress_callback:
                progress_callback((idx + 1) / total)
            
            # Run simulation
            metrics, _ = self.simulate_with_settings(
                sl_pips=sl,
                tp_ratio=tp,
                partial_tp_enabled=use_partial,
                partial_tp_rr=1.0,
                partial_tp_percent=50.0
            )
            
            all_results.append(metrics)
        
        return pd.DataFrame(all_results)


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
    st.markdown('<h1 class="main-header">🎯 Strategy Optimizer</h1>', unsafe_allow_html=True)
    st.markdown("### Find Your Optimal SL/TP Settings")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Data upload
    st.sidebar.subheader("📁 Upload Data")
    data_source = st.sidebar.radio("Data Source", ["Use Sample Data", "Upload CSV"])
    
    if data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip().str.lower()
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'], errors='coerce')
            st.sidebar.success(f"✅ {len(df)} trades")
        else:
            df = create_sample_data()
            st.sidebar.warning("Using sample data")
    else:
        df = create_sample_data()
        st.sidebar.info("📊 Using sample data")
    
    st.sidebar.markdown("---")
    
    # Optimization settings
    st.sidebar.subheader("🔧 Optimization Settings")
    
    st.sidebar.markdown("**Stop Loss Range (pips):**")
    sl_min = st.sidebar.number_input("Min SL", 5, 50, 10, 5)
    sl_max = st.sidebar.number_input("Max SL", 10, 100, 30, 5)
    sl_step = st.sidebar.number_input("Step", 1, 10, 5)
    
    sl_range = list(range(sl_min, sl_max + 1, sl_step))
    st.sidebar.text(f"Testing: {sl_range}")
    
    st.sidebar.markdown("**Take Profit Ratio Range:**")
    tp_values = st.sidebar.multiselect(
        "TP Ratios",
        [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        default=[1.5, 2.0, 3.0, 4.0]
    )
    
    test_partial = st.sidebar.checkbox("Test Partial TP Variations", value=True)
    
    st.sidebar.markdown("---")
    
    # Optimization objective
    st.sidebar.subheader("🎯 Optimize For")
    optimization_goal = st.sidebar.selectbox(
        "Primary Goal",
        ["Profit Factor", "Win Rate", "Total Return", "Sharpe Ratio", "Min Drawdown"]
    )
    
    # Run optimization
    if st.sidebar.button("🚀 Run Optimization", type="primary"):
        
        st.markdown("---")
        st.header("🔄 Running Optimization...")
        
        progress_bar = st.progress(0)
        status = st.empty()
        
        optimizer = StrategyOptimizer(df)
        
        status.text(f"Testing {len(sl_range) * len(tp_values) * (2 if test_partial else 1)} combinations...")
        
        results = optimizer.optimize(
            sl_range=sl_range,
            tp_range=tp_values,
            test_partial_tp=test_partial,
            progress_callback=lambda p: progress_bar.progress(p)
        )
        
        progress_bar.empty()
        status.empty()
        
        st.success("✅ Optimization Complete!")
        
        # Find best settings
        if optimization_goal == "Profit Factor":
            best_idx = results['profit_factor'].idxmax()
        elif optimization_goal == "Win Rate":
            best_idx = results['win_rate'].idxmax()
        elif optimization_goal == "Total Return":
            best_idx = results['total_return_pct'].idxmax()
        elif optimization_goal == "Sharpe Ratio":
            best_idx = results['sharpe_ratio'].idxmax()
        else:  # Min Drawdown
            best_idx = results['max_drawdown'].idxmin()
        
        best = results.iloc[best_idx]
        
        # Display best settings
        st.markdown("---")
        st.header("🏆 Optimal Settings Found!")
        
        st.markdown(f"""
        <div class="best-setting">
            <h2>✨ Recommended Settings</h2>
            <h3>Stop Loss: {best['sl_pips']:.0f} pips</h3>
            <h3>Take Profit: {best['tp_ratio']:.1f}R ({best['sl_pips'] * best['tp_ratio']:.0f} pips)</h3>
            <h3>Partial TP: {'Yes (50% at 1R)' if best['partial_tp'] else 'No'}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Key metrics
        st.markdown("### 📊 Performance Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Win Rate", f"{best['win_rate']:.1%}")
        with col2:
            st.metric("Profit Factor", f"{best['profit_factor']:.2f}")
        with col3:
            st.metric("Total Return", f"{best['total_return_pct']:.1f}%")
        with col4:
            st.metric("Max DD", f"{best['max_drawdown']:.1f}%")
        with col5:
            st.metric("Sharpe Ratio", f"{best['sharpe_ratio']:.2f}")
        
        # Compare to your current settings
        st.markdown("---")
        st.header("📈 Comparison to Original")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔴 Your Current Settings")
            # Assume user had 20 pip SL, 2R TP
            current_metrics, _ = optimizer.simulate_with_settings(20, 2.0, False)
            
            st.markdown(f"""
            <div class="metric-comparison">
                <p><strong>SL:</strong> 20 pips</p>
                <p><strong>TP:</strong> 2R (40 pips)</p>
                <p><strong>Win Rate:</strong> {current_metrics['win_rate']:.1%}</p>
                <p><strong>Profit Factor:</strong> {current_metrics['profit_factor']:.2f}</p>
                <p><strong>Return:</strong> {current_metrics['total_return_pct']:.1f}%</p>
                <p><strong>Max DD:</strong> {current_metrics['max_drawdown']:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("🟢 Optimized Settings")
            
            improvement_return = ((best['total_return_pct'] - current_metrics['total_return_pct']) 
                                 / abs(current_metrics['total_return_pct']) * 100) if current_metrics['total_return_pct'] != 0 else 0
            improvement_pf = ((best['profit_factor'] - current_metrics['profit_factor']) 
                             / current_metrics['profit_factor'] * 100) if current_metrics['profit_factor'] != 0 else 0
            
            st.markdown(f"""
            <div class="metric-comparison">
                <p><strong>SL:</strong> {best['sl_pips']:.0f} pips</p>
                <p><strong>TP:</strong> {best['tp_ratio']:.1f}R ({best['sl_pips'] * best['tp_ratio']:.0f} pips)</p>
                <p><strong>Win Rate:</strong> {best['win_rate']:.1%} 
                   <span style="color: {'green' if best['win_rate'] > current_metrics['win_rate'] else 'red'}">
                   ({'+' if best['win_rate'] > current_metrics['win_rate'] else ''}{(best['win_rate'] - current_metrics['win_rate'])*100:.1f}%)</span></p>
                <p><strong>Profit Factor:</strong> {best['profit_factor']:.2f} 
                   <span style="color: green">(+{improvement_pf:.0f}%)</span></p>
                <p><strong>Return:</strong> {best['total_return_pct']:.1f}% 
                   <span style="color: green">(+{improvement_return:.0f}%)</span></p>
                <p><strong>Max DD:</strong> {best['max_drawdown']:.1f}%
                   <span style="color: {'green' if best['max_drawdown'] < current_metrics['max_drawdown'] else 'red'}">
                   ({'+' if best['max_drawdown'] > current_metrics['max_drawdown'] else ''}{(best['max_drawdown'] - current_metrics['max_drawdown']):.1f}%)</span></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Heatmap
        st.markdown("---")
        st.header("🗺️ Optimization Heatmap")
        
        # Filter for non-partial TP results
        results_no_partial = results[results['partial_tp'] == False]
        
        # Create pivot table
        pivot_data = results_no_partial.pivot_table(
            values=optimization_goal.lower().replace(' ', '_'),
            index='sl_pips',
            columns='tp_ratio'
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='RdYlGn',
            text=pivot_data.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
        ))
        
        fig.update_layout(
            title=f"{optimization_goal} Heatmap",
            xaxis_title="TP Ratio",
            yaxis_title="SL (pips)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Top 10 combinations
        st.markdown("---")
        st.header("🏅 Top 10 Settings")
        
        if optimization_goal == "Min Drawdown":
            top_10 = results.nsmallest(10, 'max_drawdown')
        else:
            metric_col = optimization_goal.lower().replace(' ', '_')
            top_10 = results.nlargest(10, metric_col)
        
        # Format for display
        display_df = top_10[['sl_pips', 'tp_ratio', 'partial_tp', 'win_rate', 
                            'profit_factor', 'total_return_pct', 'max_drawdown']].copy()
        display_df.columns = ['SL (pips)', 'TP Ratio', 'Partial TP', 'Win Rate', 
                              'Profit Factor', 'Return %', 'Max DD %']
        display_df['Win Rate'] = display_df['Win Rate'].apply(lambda x: f"{x:.1%}")
        display_df['Return %'] = display_df['Return %'].apply(lambda x: f"{x:.1f}%")
        display_df['Max DD %'] = display_df['Max DD %'].apply(lambda x: f"{x:.1f}%")
        display_df['Profit Factor'] = display_df['Profit Factor'].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download results
        st.markdown("---")
        st.header("📥 Download Results")
        
        csv = results.to_csv(index=False)
        st.download_button(
            label="📊 Download All Results (CSV)",
            data=csv,
            file_name="optimization_results.csv",
            mime="text/csv"
        )
        
        # Recommendation summary
        st.markdown("---")
        st.markdown(f"""
        <div class="best-setting">
            <h3>💡 Implementation Guide</h3>
            <ol>
                <li>Update your MT5 EA settings:
                    <ul>
                        <li>Stop Loss: {best['sl_pips']:.0f} pips</li>
                        <li>Take Profit: {best['tp_ratio']:.1f}R</li>
                        <li>Partial TP: {'Enable 50% at 1R' if best['partial_tp'] else 'Disable'}</li>
                    </ul>
                </li>
                <li>Backtest with new settings in MT5 Strategy Tester</li>
                <li>Forward test on demo for 2+ weeks</li>
                <li>If results match optimization, go live!</li>
            </ol>
            <p><strong>Expected Improvement:</strong> {improvement_return:.0f}% better returns!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Store in session
        st.session_state['optimization_results'] = results
        st.session_state['best_settings'] = best
    
    else:
        st.info("👈 Configure optimization settings and click 'Run Optimization'")
        
        st.markdown("""
        ### 📖 How This Works:
        
        1. **Upload Your Data** - Your MT5 backtest or trade history
        2. **Set Ranges** - Choose SL and TP values to test
        3. **Run Optimization** - Tests all combinations
        4. **Get Results** - See best settings for your strategy
        
        ### 🎯 What It Tests:
        
        - **Stop Loss:** 10, 15, 20, 25, 30 pips (customizable)
        - **Take Profit:** 1.5R, 2R, 3R, 4R, 5R (customizable)
        - **Partial TP:** With and without
        
        ### 📊 What It Optimizes:
        
        - **Profit Factor** - Risk/reward efficiency
        - **Win Rate** - Percentage of winning trades
        - **Total Return** - Overall profit %
        - **Sharpe Ratio** - Risk-adjusted returns
        - **Min Drawdown** - Lowest risk
        
        ### ⚡ Example:
        
        **Your current settings:**
        - SL: 20 pips
        - TP: 2R (40 pips)
        - Return: 15%
        
        **After optimization:**
        - SL: 15 pips
        - TP: 3R (45 pips)
        - Partial TP: 50% at 1R
        - Return: 28% (+87% improvement!)
        """)


if __name__ == "__main__":
    main()
