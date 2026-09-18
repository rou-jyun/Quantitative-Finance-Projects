import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams

rcParams["font.family"] = "Microsoft JhengHei"
rcParams["axes.unicode_minus"] = False

fees = {"SPY": 0, "BND": 0}
tickers = list(fees.keys())

data = yf.download(tickers, start="2007-04-10", end="2026-06-30", auto_adjust=True)
prices = data["Close"]
returns = prices.pct_change().dropna()

rf = pd.read_csv("par-yield-curve-rates-1990-2023.csv")
rf["date"] = pd.to_datetime(rf["date"], format="%m/%d/%Y")
rf = rf.set_index("date")
rf["3 mo"] = rf["3 mo"] / 100
rf_daily = rf["3 mo"].reindex(returns.index).ffill() / 252


def backtest(prices, returns, fees, portfolio_type="BA"):
    assets = ["SPY", "BND"]
    transaction_cost = 0.001
    total = 100_000_000.0
    fee = np.array([fees.get(t, 0) for t in assets]) / 252.0

    base_weights = {"AA": 0.80, "BA": 0.60, "CA": 0.40}
    state_adjustment = {"extreme_down": 0.20, "bullish": 0.15, "normal": 0.00, "bearish": -0.15}

    base_w = base_weights[portfolio_type]
    current_weight = base_w + state_adjustment["normal"]
    position = total * np.array([current_weight, 1.0 - current_weight])

    price = prices["SPY"]
    sma200 = price.rolling(200).mean()
    bias = (price - sma200) / sma200
    raw_state = pd.Series(index=price.index, dtype="object")

    for date in price.index:
        if pd.isna(sma200.loc[date]):
            continue
        b = bias.loc[date]
        if b < -0.12:
            raw_state.loc[date] = "extreme_down"
        elif b < -0.05:
            raw_state.loc[date] = "bearish"
        elif b > 0.05:
            raw_state.loc[date] = "bullish"
        else:
            raw_state.loc[date] = "normal"

    equity, equity_dates, trade_log = [], [], []
    current_state = "normal"
    consecutive_days = 0
    prev_raw_state = None
    ENTRY_DAYS = 5
    EXIT_DAYS = 5

    for date in returns.index:
        r = returns.loc[date, assets].values
        position *= (1.0 + r - fee)
        total = position.sum()
        equity.append(total)
        equity_dates.append(date)

        if pd.isna(sma200.loc[date]) or pd.isna(raw_state.loc[date]):
            continue

        today_state = raw_state.loc[date]
        consecutive_days = consecutive_days + 1 if today_state == prev_raw_state else 1
        prev_raw_state = today_state

        confirmed_state = None
        if today_state != current_state:
            if current_state != "normal" and today_state == "normal":
                if consecutive_days >= EXIT_DAYS:
                    confirmed_state = today_state
            else:
                if consecutive_days >= ENTRY_DAYS:
                    confirmed_state = today_state

        if confirmed_state is not None:
            raw_target_w = base_w + state_adjustment[confirmed_state]
            current_weight = float(np.clip(raw_target_w, 0.0, 0.80))
            old_position_weight = position / total
            target_weight = np.array([current_weight, 1.0 - current_weight])
            turnover = np.sum(np.abs(target_weight - old_position_weight))
            trading_cost = turnover * transaction_cost
            total *= (1.0 - trading_cost)
            position = total * target_weight

            trade_log.append({
                "Date": date,
                "State": confirmed_state,
                "SPY Price": price.loc[date],
                "MA200": sma200.loc[date],
                "Bias (%)": bias.loc[date] * 100,
                "SPY Before": old_position_weight[0],
                "SPY After": current_weight,
                "BND Before": old_position_weight[1],
                "BND After": 1.0 - current_weight,
                "Turnover": turnover,
                "Transaction Cost": trading_cost,
            })
            current_state = confirmed_state

    equity = pd.Series(equity, index=equity_dates, name="Equity")
    trade_log = pd.DataFrame(trade_log)
    if not trade_log.empty:
        trade_log.set_index("Date", inplace=True)
    return equity, trade_log


def top5_drawdown(e):
    running_max = e.cummax()
    drawdown = e / running_max - 1
    drawdowns = []
    in_drawdown = False
    worst_dd = None

    for dd in drawdown:
        if np.isclose(dd, 0):
            if in_drawdown:
                drawdowns.append(worst_dd)
            in_drawdown = False
            worst_dd = None
        else:
            if not in_drawdown:
                in_drawdown = True
                worst_dd = dd
            else:
                worst_dd = min(worst_dd, dd)

    if in_drawdown:
        drawdowns.append(worst_dd)
    if len(drawdowns) == 0:
        return np.nan

    drawdowns.sort()
    return np.mean(drawdowns[:5])


def metrics(e, rf_daily):
    if isinstance(e, pd.DataFrame):
        e = e["Equity"]
    e = pd.to_numeric(e, errors="coerce").dropna()
    r = e.pct_change().dropna()
    rf_used = rf_daily.loc[r.index]

    years = (e.index[-1] - e.index[0]).days / 365.25
    cagr = (e.iloc[-1] / e.iloc[0]) ** (1 / years) - 1
    vol = r.std() * np.sqrt(252)
    avg_top5_dd = top5_drawdown(e)

    excess_return = r - rf_used
    sharpe = (excess_return.mean() / excess_return.std()) * np.sqrt(252)

    return [f"{cagr:.2%}", f"{vol:.2%}", f"{avg_top5_dd:.2%}", f"{sharpe:.2f}"]


def drawdown_table(e):
    events = []
    peak_date, peak_value = e.index[0], e.iloc[0]
    in_drawdown = False
    valley_date = valley_value = None
    worst_dd = 0

    for date, value in e.items():
        if not in_drawdown:
            if value >= peak_value:
                peak_value, peak_date = value, date
            else:
                in_drawdown = True
                valley_date, valley_value = date, value
                worst_dd = value / peak_value - 1
        else:
            current_dd = value / peak_value - 1
            if current_dd < worst_dd:
                worst_dd = current_dd
                valley_date, valley_value = date, value
            if value >= peak_value:
                recovery_date = date
                recovery_year = (recovery_date - valley_date).days / 365.25
                events.append({
                    "Peak": peak_date, "Valley": valley_date, "Recovery": recovery_date,
                    "Drawdown": worst_dd, "Recovery(Y)": recovery_year,
                })
                peak_date, peak_value = date, value
                in_drawdown = False

    if in_drawdown:
        events.append({
            "Peak": peak_date, "Valley": valley_date, "Recovery": pd.NaT,
            "Drawdown": worst_dd, "Recovery(Y)": np.nan,
        })

    df = pd.DataFrame(events).sort_values("Drawdown").reset_index(drop=True)
    df.index += 1
    return df


def holding_return_table(e):
    start_dates, end_dates = {}, {}
    for y in range(e.index[0].year, e.index[-1].year + 1):
        tmp = e.loc[e.index.year == y]
        if len(tmp) > 0:
            start_dates[y] = tmp.index[0]
            end_dates[y] = tmp.index[-1]

    table = []
    for end_year in sorted(end_dates.keys()):
        row = {"Year": end_year}
        for h in range(1, 6):
            start_year = end_year - h + 1
            if start_year in start_dates:
                start, finish = start_dates[start_year], end_dates[end_year]
                row[f"{h}Y"] = e.loc[finish] / e.loc[start] - 1
            else:
                row[f"{h}Y"] = np.nan
        table.append(row)

    df = pd.DataFrame(table)
    avg = {"Year": "Average"}
    for h in range(1, 6):
        avg[f"{h}Y"] = df[f"{h}Y"].mean()
    return pd.concat([df, pd.DataFrame([avg])], ignore_index=True)


def drawdown_curve(e):
    return e / e.cummax() - 1


def plot_bear_market(start, end, title):
    event = equity_df.loc[start:end].copy()
    spy = prices["SPY"].loc[event.index]
    event = event / event.iloc[0] * 100
    spy = spy / spy.iloc[0] * 100
    event["SPY"] = spy

    colors = {"SPY": "black", "AA": "red", "BA": "blue", "CA": "green"}
    labels = {"SPY": "S&P500", "AA": "AA", "BA": "BA", "CA": "CA"}

    plt.figure(figsize=(13, 7))
    for c in ["SPY", "AA", "BA", "CA"]:
        plt.plot(event.index, event[c], lw=2.5, color=colors[c], label=labels[c])
        valley = event[c].idxmin()
        value = event[c].min()
        plt.scatter(valley, value, color=colors[c], s=45)
        plt.text(valley, value - 3, f"{value-100:.1f}%", fontsize=9, color=colors[c])

        recover = event[event[c] >= 100].loc[valley:]
        if len(recover) > 0:
            plt.scatter(recover.index[0], 100, marker="o", s=60, color=colors[c])

    plt.axhline(100, ls="--", color="gray", alpha=0.7)
    plt.title(title, fontsize=16)
    plt.ylabel("Normalized Portfolio Value (Base = 100)")
    plt.xlabel("Date")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()


def plot_drawdown(start, end, title):
    spy_dd = drawdown_curve(prices["SPY"].loc[start:end])
    curves = [
        ("S&P500", spy_dd, "black"),
        ("AA", drawdown_curve(equity_all["AA"].loc[start:end]), "red"),
        ("BA", drawdown_curve(equity_all["BA"].loc[start:end]), "blue"),
        ("CA", drawdown_curve(equity_all["CA"].loc[start:end]), "green"),
    ]

    plt.figure(figsize=(13, 6))
    for name, curve, color in curves:
        plt.plot(curve.index, curve * 100, linewidth=2, color=color, label=name)
        valley = curve.idxmin()
        mdd = curve.min() * 100
        plt.scatter(valley, mdd, color=color, s=60, zorder=5)
        plt.text(valley, mdd - 2, f"{mdd:.1f}%", color=color, fontsize=9, ha="center")

    plt.axhline(0, color="gray", linestyle="--")
    plt.title(title, fontsize=15)
    plt.ylabel("回撤幅度(%)")
    plt.xlabel("日期")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()


def get_peak_and_trough(data_series, search_start, search_end):
    sub_series = data_series.loc[search_start:search_end]
    cummax = sub_series.cummax()
    drawdown = (sub_series - cummax) / cummax
    trough_date = drawdown.idxmin()
    peak_date = sub_series.loc[:trough_date].idxmax()
    return peak_date, trough_date


def stress_test(start, end, title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    result = []
    for name in fees.keys():
        r = returns.loc[start:end]
        e, _ = backtest(prices, r, fees, portfolio_type=name)
        result.append([name] + metrics(e, rf_daily))

    df = pd.DataFrame(result, columns=["Portfolio", "CAGR", "Volatility", "Avg. Top 5 Drawdown", "Sharpe"])
    print(df)


# ============================================================
# 執行
# ============================================================

result = {}
equity_all = {}

for portfolio_type in ["AA", "BA", "CA"]:
    e, trade_log = backtest(prices, returns, fees, portfolio_type)
    equity_all[portfolio_type] = e
    result[portfolio_type] = metrics(e, rf_daily)

df = pd.DataFrame(result, index=["CAGR", "Volatility", "Avg. Top 5 Drawdown", "Sharpe"]).T
df.index.name = "Portfolio"
print("回測結果")
print(df)

equity_AA, trade_AA = backtest(prices, returns, fees, "AA")
trade_AA.to_csv("換倉紀錄.csv")

#壓力測試
events_search_range = {
    "金融海嘯期間": ("2007-09-01", "2010-01-01"),
    "COVID疫情期間": ("2020-01-01", "2021-12-31"),
    "升息循環期間": ("2021-12-01", "2023-12-31"),
}

for event, (search_s, search_e) in events_search_range.items():
    start, end = get_peak_and_trough(prices["SPY"], search_s, search_e)

    plt.figure(figsize=(12, 6))
    spy = prices["SPY"].loc[start:end]
    spy = spy / spy.iloc[0] - 1
    plt.plot(spy.index, spy * 100, color="black", linewidth=2, label="S&P500")

    colors = {"AA": "red", "BA": "blue", "CA": "green"}
    for name in ["AA", "BA", "CA"]:
        r = equity_all[name].loc[start:end]
        r = r / r.iloc[0] - 1
        plt.plot(r.index, r * 100, linewidth=2, color=colors[name], label=name)

    plt.axhline(0, color="gray", linestyle="--")
    plt.ylabel("累積報酬率 (%)")
    plt.xlabel("日期")
    plt.title(f"{event}累積報酬比較 ({start.strftime('%Y/%m/%d')} ~ {end.strftime('%Y/%m/%d')})")
    plt.legend()
    plt.grid(alpha=0.3)

    print(f"{event}：{start.strftime('%Y-%m-%d')} ~ {end.strftime('%Y-%m-%d')}")

    results = [{"Portfolio": "SPY", "Drawdown": spy.iloc[-1]}]
    for name in ["AA", "BA", "CA"]:
        r = equity_all[name].loc[start:end]
        r = r / r.iloc[0] - 1
        results.append({"Portfolio": name, "Drawdown": r.iloc[-1]})

    result_df = pd.DataFrame(results)
    result_df["Drawdown"] = result_df["Drawdown"].map(lambda x: f"{x:.2%}")
    print(result_df)

equity_df = pd.DataFrame(equity_all)
equity_df.columns = ["AA", "BA", "CA"]
equity_norm = equity_df / equity_df.iloc[0]

# 畫淨值曲線成長圖
plt.figure(figsize=(12, 7))
colors = {"AA": "red", "BA": "blue", "CA": "green"}
for col in equity_norm.columns:
    plt.plot(equity_norm.index, equity_norm[col], linewidth=2, color=colors[col], label=col)

print(equity_norm)
equity_norm.to_csv("資產淨值成長表.csv")
plt.title("價值曲線")
plt.xlabel("日期")
plt.ylabel("資產淨值")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()


#畫季報酬圖
quarter_value = equity_df.resample("QE").last()
quarter_return = quarter_value.pct_change().dropna()

fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True, sharey=True)
for ax, col in zip(axes, quarter_return.columns):
    r = quarter_return[col] * 100
    colors_bar = ["red" if x >= 0 else "green" for x in r]
    ax.bar(r.index, r.values, color=colors_bar, width=75)
    ax.axhline(0, color="black", linewidth=1)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("報酬率(%)")
    ax.set_title(col)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_ylim(-20, 20)

plt.setp(axes[-1].get_xticklabels(), rotation=45, ha="right")
axes[-1].set_xlabel("Date")
plt.tight_layout()

#十年回徹幅度排序及恢復時間
top10 = drawdown_table(equity_all["CA"]).head(10)
top10["Drawdown"] = top10["Drawdown"].map("{:.2%}".format)
top10["Recovery(Y)"] = top10["Recovery(Y)"].map(lambda x: "-" if pd.isna(x) else f"{x:.2f}")
print(top10)

#買進持有報酬表
show = holding_return_table(equity_all["AA"])
for c in ["1Y", "2Y", "3Y", "4Y", "5Y"]:
    show[c] = show[c].map(lambda x: "-" if pd.isna(x) else f"{x:.2%}")
print(show)

plt.show()