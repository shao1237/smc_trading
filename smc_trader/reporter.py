import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jinja2 import Template
from typing import List, Dict, Any, Tuple, Optional

class Reporter:
    """
    SMC+SNR 回測報告生成器。
    負責計算指標、繪製資產曲線、統計檢定圖表，並生成網頁報告。
    """
    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def calculate_metrics(self, trades: List[Dict[str, Any]], equity_curve: List[Dict[str, Any]], initial_capital: float) -> Dict[str, Any]:
        """
        計算基礎回測指標。
        """
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_rr': 0.0,
                'profit_factor': 0.0,
                'max_drawdown_pct': 0.0,
                'max_drawdown_val': 0.0,
                'net_profit': 0.0,
                'ending_equity': initial_capital,
                'total_wins': 0,
                'total_losses': 0
            }

        df_trades = pd.DataFrame(trades)
        df_equity = pd.DataFrame(equity_curve)

        total_trades = len(df_trades)
        wins = df_trades[df_trades['pnl'] > 0]
        losses = df_trades[df_trades['pnl'] <= 0]
        total_wins = len(wins)
        total_losses = len(losses)

        win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0.0

        total_gain = wins['pnl'].sum() if total_wins > 0 else 0.0
        total_loss = abs(losses['pnl'].sum()) if total_losses > 0 else 0.0
        profit_factor = total_gain / total_loss if total_loss > 0 else (float('inf') if total_gain > 0 else 1.0)

        avg_gain = wins['pnl'].mean() if total_wins > 0 else 0.0
        avg_loss = abs(losses['pnl'].mean()) if total_losses > 0 else 0.0
        avg_rr = avg_gain / avg_loss if avg_loss > 0 else (float('inf') if avg_gain > 0 else 0.0)

        df_equity['peak'] = df_equity['equity'].cummax()
        df_equity['dd_pct'] = (df_equity['peak'] - df_equity['equity']) / df_equity['peak'] * 100
        df_equity['dd_val'] = df_equity['peak'] - df_equity['equity']
        
        max_drawdown_pct = df_equity['dd_pct'].max()
        max_drawdown_val = df_equity['dd_val'].max()

        ending_equity = df_equity['equity'].iloc[-1]
        net_profit = ending_equity - initial_capital

        return {
            'total_trades': total_trades,
            'win_rate': round(win_rate, 2),
            'avg_rr': round(avg_rr, 2) if avg_rr != float('inf') else "Infinity",
            'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else "Infinity",
            'max_drawdown_pct': round(max_drawdown_pct, 2),
            'max_drawdown_val': round(max_drawdown_val, 2),
            'net_profit': round(net_profit, 2),
            'ending_equity': round(ending_equity, 2),
            'total_wins': total_wins,
            'total_losses': total_losses
        }

    def plot_equity_curve(self, equity_curve: List[Dict[str, Any]], filename: str = "equity_curve.png") -> str:
        """
        繪製資產曲線與最大回撤變化圖。
        """
        df_equity = pd.DataFrame(equity_curve)
        df_equity['peak'] = df_equity['equity'].cummax()
        df_equity['drawdown'] = (df_equity['equity'] - df_equity['peak'])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        plt.style.use('dark_background')
        fig.patch.set_facecolor('#0f172a')
        ax1.set_facecolor('#1e293b')
        ax2.set_facecolor('#1e293b')

        ax1.plot(df_equity['ts'], df_equity['equity'], color='#10b981', linewidth=2)
        ax1.fill_between(df_equity['ts'], df_equity['equity'], df_equity['equity'].min(), color='#10b981', alpha=0.1)
        ax1.set_title('SMC Strategy Backtest - Equity Curve', fontsize=14, color='#f8fafc', fontweight='bold', pad=15)
        ax1.set_ylabel('Equity (NTD)', fontsize=12, color='#94a3b8')
        ax1.grid(True, color='#334155', linestyle='--', alpha=0.5)
        ax1.tick_params(colors='#94a3b8')

        ax2.fill_between(df_equity['ts'], df_equity['drawdown'], 0, color='#f43f5e', alpha=0.3)
        ax2.plot(df_equity['ts'], df_equity['drawdown'], color='#f43f5e', linewidth=1)
        ax2.set_ylabel('Drawdown (NTD)', fontsize=12, color='#94a3b8')
        ax2.set_xlabel('Date Time', fontsize=12, color='#94a3b8')
        ax2.grid(True, color='#334155', linestyle='--', alpha=0.5)
        ax2.tick_params(colors='#94a3b8')
        
        plt.xticks(rotation=15)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, facecolor=fig.get_facecolor(), edgecolor='none', dpi=150)
        plt.close()
        
        return filepath

    def plot_validation_charts(self, mcpt_dist: List[float], real_profit: float, p_value: float,
                               bootstrap_dist: List[float], low_ci: float, high_ci: float,
                               filename: str = "validation_charts.png") -> str:
        """
        繪製蒙地卡羅排列分佈與 Bootstrap 期望值分佈圖。
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        plt.style.use('dark_background')
        fig.patch.set_facecolor('#0f172a')
        ax1.set_facecolor('#1e293b')
        ax2.set_facecolor('#1e293b')

        # 1. 蒙地卡羅排列檢定直方圖
        ax1.hist(mcpt_dist, bins=35, color='#3b82f6', alpha=0.6, edgecolor='#2563eb')
        ax1.axvline(real_profit, color='#ef4444', linestyle='--', linewidth=2.5, 
                    label=f'Real Profit: {real_profit:,.0f}')
        ax1.set_title(f'MCPT Permutation (p-value: {p_value:.4f})', fontsize=12, fontweight='bold', color='#f8fafc', pad=10)
        ax1.set_xlabel('Simulated Profit (NTD)', color='#94a3b8')
        ax1.set_ylabel('Frequency', color='#94a3b8')
        ax1.grid(True, color='#334155', linestyle='--', alpha=0.5)
        ax1.legend(loc='upper left')
        ax1.tick_params(colors='#94a3b8')

        # 2. Bootstrap 信賴區間直方圖
        ax2.hist(bootstrap_dist, bins=35, color='#10b981', alpha=0.6, edgecolor='#059669')
        ax2.axvline(low_ci, color='#f43f5e', linestyle=':', linewidth=2, label=f'2.5% CI Limit: {low_ci:,.1f}')
        ax2.axvline(high_ci, color='#60a5fa', linestyle=':', linewidth=2, label=f'97.5% CI Limit: {high_ci:,.1f}')
        ax2.axvline(0, color='#eab308', linestyle='--', linewidth=1.5, label='0 Line')
        ax2.set_title('Bootstrap Mean PnL CI (95%)', fontsize=12, fontweight='bold', color='#f8fafc', pad=10)
        ax2.set_xlabel('Mean Transaction PnL (NTD)', color='#94a3b8')
        ax2.set_ylabel('Frequency', color='#94a3b8')
        ax2.grid(True, color='#334155', linestyle='--', alpha=0.5)
        ax2.legend(loc='upper left')
        ax2.tick_params(colors='#94a3b8')

        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, facecolor=fig.get_facecolor(), edgecolor='none', dpi=150)
        plt.close()
        
        return filepath

    def generate_html_report(self, metrics: Dict[str, Any], trades: List[Dict[str, Any]], 
                             equity_filepath: str, validation_metrics: Optional[Dict[str, Any]] = None,
                             validation_filepath: Optional[str] = None, filename: str = "report.html") -> str:
        """
        生成完整的 HTML 互動式回測報告。
        """
        img_name = os.path.basename(equity_filepath)
        val_img_name = os.path.basename(validation_filepath) if validation_filepath else None

        formatted_trades = []
        for t in trades:
            formatted_trades.append({
                'direction': '多單 (LONG)' if t['direction'] == 'LONG' else '空單 (SHORT)',
                'entry_time': t['entry_time'].strftime('%Y-%m-%d %H:%M') if isinstance(t['entry_time'], pd.Timestamp) else str(t['entry_time']),
                'entry_price': round(t['entry_price'], 1),
                'exit_time': t['exit_time'].strftime('%Y-%m-%d %H:%M') if isinstance(t['exit_time'], pd.Timestamp) else str(t['exit_time']),
                'exit_price': round(t['exit_price'], 1),
                'exit_type': '停利 (TP)' if t['exit_type'] == 'TP' else ('當沖強平 (ISQ)' if t['exit_type'] == 'ISQ' else '停損 (SL)'),
                'net_points': round(t['net_points'], 1),
                'pnl': round(t['pnl'], 0),
                'balance_after': round(t['balance_after'], 0),
                'pnl_class': 'text-emerald' if t['pnl'] > 0 else 'text-rose'
            })

        template_str = """
        <!DOCTYPE html>
        <html lang="zh-TW">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>SMC + SNR 交易策略回測報告</title>
            <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Noto+Sans+TC:wght@300;400;700&display=swap" rel="stylesheet">
            <style>
                :root {
                    --bg-main: #0f172a;
                    --bg-card: #1e293b;
                    --bg-input: #334155;
                    --text-main: #f8fafc;
                    --text-muted: #94a3b8;
                    --primary: #10b981;
                    --primary-hover: #34d399;
                    --accent-rose: #f43f5e;
                    --accent-cyan: #06b6d4;
                    --accent-amber: #f59e0b;
                    --border-color: #334155;
                }
                * {
                    box-sizing: border-box;
                    margin: 0;
                    padding: 0;
                }
                body {
                    font-family: 'Outfit', 'Noto Sans TC', sans-serif;
                    background-color: var(--bg-main);
                    color: var(--text-main);
                    line-height: 1.6;
                    padding: 2rem;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                header {
                    text-align: center;
                    margin-bottom: 2.5rem;
                    border-bottom: 1px solid var(--border-color);
                    padding-bottom: 1.5rem;
                }
                header h1 {
                    font-size: 2.5rem;
                    font-weight: 800;
                    background: linear-gradient(135deg, var(--primary), var(--accent-cyan));
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    margin-bottom: 0.5rem;
                }
                header p {
                    color: var(--text-muted);
                    font-size: 1.1rem;
                }
                .grid-metrics {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                    gap: 1.5rem;
                    margin-bottom: 2.5rem;
                }
                .card {
                    background-color: var(--bg-card);
                    border: 1px solid var(--border-color);
                    border-radius: 16px;
                    padding: 1.5rem;
                    text-align: center;
                    transition: transform 0.3s ease, border-color 0.3s ease;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
                }
                .card:hover {
                    transform: translateY(-5px);
                    border-color: var(--primary);
                }
                .card h3 {
                    font-size: 0.95rem;
                    color: var(--text-muted);
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                    margin-bottom: 0.75rem;
                }
                .card .value {
                    font-size: 2rem;
                    font-weight: 700;
                    color: var(--text-main);
                }
                .card .value.positive {
                    color: var(--primary);
                }
                .card .value.negative {
                    color: var(--accent-rose);
                }
                .content-section {
                    background-color: var(--bg-card);
                    border: 1px solid var(--border-color);
                    border-radius: 16px;
                    padding: 2rem;
                    margin-bottom: 2.5rem;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
                }
                .content-section h2 {
                    font-size: 1.5rem;
                    margin-bottom: 1.5rem;
                    border-left: 4px solid var(--primary);
                    padding-left: 0.75rem;
                }
                .chart-container {
                    text-align: center;
                    margin: 1rem 0;
                }
                .chart-container img {
                    max-width: 100%;
                    height: auto;
                    border-radius: 12px;
                    border: 1px solid var(--border-color);
                }
                .grid-val {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
                    gap: 1.5rem;
                    margin-bottom: 2rem;
                }
                .badge-val {
                    display: inline-block;
                    margin-top: 0.5rem;
                    padding: 0.3rem 0.8rem;
                    border-radius: 50px;
                    font-size: 0.85rem;
                    font-weight: bold;
                    text-transform: uppercase;
                }
                .badge-passed {
                    background-color: rgba(16, 185, 129, 0.15);
                    color: var(--primary);
                    border: 1px solid var(--primary);
                }
                .badge-failed {
                    background-color: rgba(244, 63, 94, 0.15);
                    color: var(--accent-rose);
                    border: 1px solid var(--accent-rose);
                }
                .table-responsive {
                    overflow-x: auto;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    text-align: left;
                    font-size: 0.95rem;
                }
                th {
                    background-color: #1e293b;
                    padding: 1rem;
                    color: var(--text-muted);
                    font-weight: 600;
                    border-bottom: 2px solid var(--border-color);
                }
                td {
                    padding: 1rem;
                    border-bottom: 1px solid var(--border-color);
                    color: var(--text-main);
                }
                tr:hover td {
                    background-color: rgba(255, 255, 255, 0.03);
                }
                .text-emerald {
                    color: var(--primary);
                    font-weight: 600;
                }
                .text-rose {
                    color: var(--accent-rose);
                    font-weight: 600;
                }
                .badge {
                    display: inline-block;
                    padding: 0.25rem 0.6rem;
                    border-radius: 50px;
                    font-size: 0.8rem;
                    font-weight: 600;
                }
                .badge-tp {
                    background-color: rgba(16, 185, 129, 0.15);
                    color: var(--primary);
                }
                .badge-sl {
                    background-color: rgba(244, 63, 94, 0.15);
                    color: var(--accent-rose);
                }
                .badge-isq {
                    background-color: rgba(245, 158, 11, 0.15);
                    color: var(--accent-amber);
                }
                footer {
                    text-align: center;
                    padding: 2rem 0;
                    color: var(--text-muted);
                    font-size: 0.9rem;
                    border-top: 1px solid var(--border-color);
                }
            </style>
        </head>
        <body>
            <div class="container">
                <header>
                    <h1>SMC + SNR 智能策略回測報告</h1>
                    <p>標的：台股指數期貨 (TXF) | 雙時框分析 (5M 大趨勢 + 1M 內部結構進場)</p>
                </header>

                <!-- 指標網格 -->
                <div class="grid-metrics">
                    <div class="card">
                        <h3>總交易筆數</h3>
                        <div class="value">{{ metrics.total_trades }}</div>
                    </div>
                    <div class="card">
                        <h3>勝率</h3>
                        <div class="value {% if metrics.win_rate >= 50 %}positive{% else %}negative{% endif %}">{{ metrics.win_rate }}%</div>
                    </div>
                    <div class="card">
                        <h3>獲利因子 (PF)</h3>
                        <div class="value {% if metrics.profit_factor != 'Infinity' and metrics.profit_factor >= 1.5 %}positive{% endif %}">{{ metrics.profit_factor }}</div>
                    </div>
                    <div class="card">
                        <h3>平均盈虧比 (R:R)</h3>
                        <div class="value">{{ metrics.avg_rr }}</div>
                    </div>
                    <div class="card">
                        <h3>最大回撤 (MDD)</h3>
                        <div class="value negative">{{ metrics.max_drawdown_pct }}%</div>
                    </div>
                    <div class="card">
                        <h3>淨損益 (NTD)</h3>
                        <div class="value {% if metrics.net_profit >= 0 %}positive{% else %}negative{% endif %}">
                            {{ "${:,.0f}".format(metrics.net_profit) }}
                        </div>
                    </div>
                </div>

                {% if val %}
                <!-- 高階統計驗證儀表板 -->
                <div class="content-section">
                    <h2>高階統計驗證儀表板 (Overfitting & Significance Verification)</h2>
                    <div class="grid-val">
                        <div class="card">
                            <h3>MCPT (蒙地卡羅排列)</h3>
                            <div class="value" style="font-size: 1.6rem;">p-value: {{ val.mcpt.p_value }}</div>
                            <span class="badge-val {% if val.mcpt.passed %}badge-passed{% else %}badge-failed{% endif %}">
                                {% if val.mcpt.passed %}PASSED (顯著){% else %}FAILED (不顯著){% endif %}
                            </span>
                        </div>
                        <div class="card">
                            <h3>Bonferroni 校正</h3>
                            <div class="value" style="font-size: 1.6rem;">門檻: {{ val.bonferroni.adjusted_alpha }}</div>
                            <span class="badge-val {% if val.bonferroni.passed %}badge-passed{% else %}badge-failed{% endif %}">
                                {% if val.bonferroni.passed %}PASSED{% else %}FAILED{% endif %}
                            </span>
                        </div>
                        <div class="card">
                            <h3>Walk-Forward 分析</h3>
                            <div class="value" style="font-size: 1.6rem;">WFE: {{ val.wfa.wfe }}%</div>
                            <span class="badge-val {% if val.wfa.passed %}badge-passed{% else %}badge-failed{% endif %}">
                                {% if val.wfa.passed %}PASSED{% else %}FAILED{% endif %}
                            </span>
                        </div>
                        <div class="card">
                            <h3>Bootstrap CI (95%)</h3>
                            <div class="value" style="font-size: 1.1rem; margin-top: 0.5rem; word-break: break-all;">
                                [{{ val.bootstrap.low_ci }} , {{ val.bootstrap.high_ci }}]
                            </div>
                            <span class="badge-val {% if val.bootstrap.passed %}badge-passed{% else %}badge-failed{% endif %}">
                                {% if val.bootstrap.passed %}PASSED (預期為正){% else %}FAILED (預期為負){% endif %}
                            </span>
                        </div>
                    </div>
                    {% if val_img %}
                    <div class="chart-container">
                        <img src="{{ val_img }}" alt="Advanced Validation Charts">
                    </div>
                    {% endif %}
                </div>
                {% endif %}

                <!-- 資產曲線圖區塊 -->
                <div class="content-section">
                    <h2>資產淨值與回撤曲線 (Equity Curve)</h2>
                    <div class="chart-container">
                        <img src="{{ img_name }}" alt="Equity Curve Chart">
                    </div>
                </div>

                <!-- 交易明細區塊 -->
                <div class="content-section">
                    <h2>交易歷史明細</h2>
                    <div class="table-responsive">
                        <table>
                            <thead>
                                <tr>
                                    <th>編號</th>
                                    <th>交易方向</th>
                                    <th>進場時間</th>
                                    <th>進場價格</th>
                                    <th>出場時間</th>
                                    <th>出場價格</th>
                                    <th>出場類型</th>
                                    <th>淨點數盈虧</th>
                                    <th>淨金額收益</th>
                                    <th>交易後餘額</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for trade in trades %}
                                <tr>
                                    <td>{{ loop.index }}</td>
                                    <td>{{ trade.direction }}</td>
                                    <td>{{ trade.entry_time }}</td>
                                    <td>{{ trade.entry_price }}</td>
                                    <td>{{ trade.exit_time }}</td>
                                    <td>{{ trade.exit_price }}</td>
                                    <td>
                                        <span class="badge {% if trade.exit_type == '停利 (TP)' %}badge-tp{% elif trade.exit_type == '當沖強平 (ISQ)' %}badge-isq{% else %}badge-sl{% endif %}">
                                            {{ trade.exit_type }}
                                        </span>
                                    </td>
                                    <td class="{{ trade.pnl_class }}">{{ trade.net_points }}</td>
                                    <td class="{{ trade.pnl_class }}">{{ "${:,.0f}".format(trade.pnl) }}</td>
                                    <td>{{ "${:,.0f}".format(trade.balance_after) }}</td>
                                </tr>
                                {% else %}
                                <tr>
                                    <td colspan="10" style="text-align: center; color: var(--text-muted);">本回測區間內無交易成交記錄</td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>

                <footer>
                    <p>由 Antigravity 智能回測引擎自動生成 &copy; 2026</p>
                </footer>
            </div>
        </body>
        </html>
        """

        t = Template(template_str)
        rendered_html = t.render(
            metrics=metrics, 
            trades=formatted_trades, 
            img_name=img_name,
            val=validation_metrics,
            val_img=val_img_name
        )
        
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(rendered_html)
            
        return filepath
