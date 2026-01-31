import os
import signal

#  MUST be first (before CrewAI import)
os.environ["CREWAI_TELEMETRY"] = "false"
os.environ["CREWAI_DISABLE_SIGNAL_HANDLERS"] = "true"

#  HARD PATCH: disable signal registration (Streamlit-safe)
def _noop_signal(*args, **kwargs):
    return None

signal.signal = _noop_signal

from openai import OpenAI
import yfinance as yf
import pandas as pd
from crewai import Agent, Task, Crew
import streamlit as st
from dotenv import load_dotenv
from helpercss import css
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import logging
logging.getLogger("crewai").setLevel(logging.ERROR)


# -------------------- ENV --------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------- HELPERS --------------------
def init_ses_states():
    defaults = {
        "chat_history": [],
        "last_assets": [],
        "last_response": "",
        "stock_memory": {}
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

def ensure_stock_memory(ticker):
    if ticker not in st.session_state.stock_memory:
        st.session_state.stock_memory[ticker] = {
            "performance": None,
            "fundamentals": None,
            "risks": None,
            "chat": []
        }

def save_stock_memory(tickers, key, value):
    for ticker in tickers:
        ensure_stock_memory(ticker)
        st.session_state.stock_memory[ticker][key] = value

def save_chat_memory(tickers, question, answer):
    for ticker in tickers:
        ensure_stock_memory(ticker)
        st.session_state.stock_memory[ticker]["chat"].append({
            "question": question,
            "answer": answer
        })
### RETURN-BASED METRICS
## Relative Returns
def relative_returns(df):
    rel = df.pct_change()
    return ((1 + rel).cumprod() - 1).fillna(0)

## Absolute Returns (%)
def absolute_returns(df):
    return ((df.iloc[-1] / df.iloc[0]) - 1) * 100

# Annualized Returns (CAGR %)
def annualized_returns(df, trading_days=252):
    n = len(df)
    if n < 2:
        return 0.0
    return ((df.iloc[-1] / df.iloc[0]) ** (trading_days / n) - 1) * 100

# RISK-BASED METRICS

#Volatility (Annualized)
def volatility(df, trading_days=252):
    return df.pct_change().std() * np.sqrt(trading_days)

# Maximum Drawdown
def max_drawdown(df):
    cum = (1 + df.pct_change()).cumprod()
    drawdown = cum / cum.cummax() - 1
    return drawdown.min()

# Downside Deviation
def downside_deviation(df, trading_days=252):
    returns = df.pct_change()
    downside = returns[returns < 0]
    return downside.std() * np.sqrt(trading_days)

#RISK-ADJUSTED METRICS
# Sharpe Ratio

def sharpe_ratio(df, risk_free_rate=0.0, trading_days=252):
    returns = df.pct_change().mean() * trading_days
    vol = volatility(df, trading_days)
    return (returns - risk_free_rate) / vol

# Sortino Ratio

def sortino_ratio(df, risk_free_rate=0.0, trading_days=252):
    returns = df.pct_change().mean() * trading_days
    downside = downside_deviation(df, trading_days)
    return (returns - risk_free_rate) / downside

# Calmar Ratio

def calmar_ratio(df):
    cagr = annualized_returns(df)
    mdd = abs(max_drawdown(df))
    return cagr / mdd

# MOMENTUM & TREND METRICS

# Rolling Returns (Momentum)

def rolling_returns(df, window=21):
    return df.pct_change(window)

# Moving Averages
def moving_averages(df, windows=[50, 200]):
    return {f"MA_{w}": df.rolling(w).mean() for w in windows}

# Relative Strength Index (RSI)

def rsi(df, window=14):
    delta = df.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# PORTFOLIO-LEVEL METRICS

# Correlation Matrix
def correlation_matrix(df):
    return df.pct_change().corr()

# Beta (vs Market)
def beta(stock_df, market_df):
    stock_ret = stock_df.pct_change().dropna()
    market_ret = market_df.pct_change().dropna()

    cov = np.cov(stock_ret.iloc[:, 0], market_ret.iloc[:, 0])[0][1]
    var = np.var(market_ret.iloc[:, 0])

    return cov / var

def compute_all_metrics(df):
    return {
        "Absolute Return (%)": absolute_returns(df).round(2),
        "CAGR (%)": annualized_returns(df).round(2),
        "Volatility": volatility(df).round(4),
        "Max Drawdown": max_drawdown(df).round(4),
        "Sharpe Ratio": sharpe_ratio(df).round(2),
        "Sortino Ratio": sortino_ratio(df).round(2),
        "Calmar Ratio": calmar_ratio(df).round(2)
    }

def brief_summary(text, max_sentences=2):
    sentences = text.split(". ")
    return ". ".join(sentences[:max_sentences]) + "."

def compute_trend_score(df, lookback_days=60):
    """
    Returns cumulative % return as a scalar float
    """
    series = df.iloc[:, 0]   # force single column Series
    recent = series.tail(lookback_days)

    if len(recent) < 2:
        return 0.0

    ret = (recent.iloc[-1] / recent.iloc[0] - 1) * 100
    return float(round(ret, 2))

def compute_fundamental_score(row):
    """
    Returns a score between -1 (weak) and +1 (strong)
    """
    score = 0

    pe = row.get("P/E")
    margin = row.get("Profit Margin (%)")
    growth = row.get("Revenue Growth (%)")

    if pe and pe < 30:
        score += 1
    elif pe and pe > 50:
        score -= 1

    if margin and margin > 20:
        score += 1

    if growth and growth > 5:
        score += 1

    return score

def compute_anomaly_risk(anomalies, ticker):
    for a in anomalies or []:
        if a["ticker"] == ticker:
            if a["anomaly_count"] >= 5:
                return -1
            elif a["anomaly_count"] >= 2:
                return 0
    return 1

def generate_signal(ticker, price_df, fundamentals_df, anomalies):
    trend = compute_trend_score(price_df[[ticker]])
    fund_row = fundamentals_df[fundamentals_df["Ticker"] == ticker].iloc[0]
    fund_score = compute_fundamental_score(fund_row)
    anomaly_score = compute_anomaly_risk(anomalies, ticker)

    total_score = (
        (1 if trend > 5 else -1 if trend < -5 else 0)
        + fund_score
        + anomaly_score
    )

    if total_score >= 2:
        signal = "BUY"
    elif total_score <= -1:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "ticker": ticker,
        "signal": signal,
        "trend_%": trend,
        "fundamental_score": fund_score,
        "anomaly_score": anomaly_score,
        "total_score": total_score
    }

def fetch_stock_data(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True)
    if df.empty:
        raise ValueError("No stock data found")

    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs("Close", axis=1, level=0)
    elif "Close" in df.columns:
        df = df[["Close"]]

    return df

def detect_price_anomalies(df, z_threshold=3):
    """
    Detect price return anomalies using Z-score
    """
    returns = df.pct_change().dropna()

    anomalies = []

    for col in returns.columns:
        series = returns[col]
        z_scores = (series - series.mean()) / series.std()

        extreme = z_scores[abs(z_scores) > z_threshold]

        if not extreme.empty:
            anomalies.append({
                "ticker": col,
                "anomaly_count": len(extreme),
                "dates": extreme.index.strftime("%Y-%m-%d").tolist(),
                "max_move_%": round((series.loc[extreme.index].abs().max()) * 100, 2)
            })

    return anomalies

def fetch_fundamentals(ticker):
    info = yf.Ticker(ticker).info

    return {
        "Ticker": ticker,
        "P/E": info.get("trailingPE"),
        "Forward P/E": info.get("forwardPE"),
        "EPS (TTM)": info.get("trailingEps"),
        "Profit Margin (%)": (
            round(info.get("profitMargins", 0) * 100, 2)
            if info.get("profitMargins") else None
        ),
        "Operating Margin (%)": (
            round(info.get("operatingMargins", 0) * 100, 2)
            if info.get("operatingMargins") else None
        ),
        "Revenue Growth (%)": (
            round(info.get("revenueGrowth", 0) * 100, 2)
            if info.get("revenueGrowth") else None
        ),
        "ROE (%)": (
            round(info.get("returnOnEquity", 0) * 100, 2)
            if info.get("returnOnEquity") else None
        )
    }

# -------------------- CREW AI AGENTS --------------------
stock_analyst_agent = Agent(
    role="Stock Analyst",
    goal="Analyze price trends and performance",
    backstory="Experienced equity research analyst",
    verbose=True
)

risk_analyst_agent = Agent(
    role="Risk Analyst",
    goal="Identify risks and opportunities for stocks",
    backstory="Macro and sector risk specialist",
    verbose=True
)

# -------------------- TASKS --------------------
performance_task = Task(
    description="""
    Analyze stock price performance using recent price data.
    Identify trends, momentum, and notable movements.
    """,
    expected_output="Concise natural language performance summary",
    agent=stock_analyst_agent
)

anomaly_agent = Agent(
    role="Market Anomaly Analyst",
    goal="Explain unusual or abnormal stock price movements",
    backstory="Quantitative analyst specializing in anomaly and risk detection",
    verbose=True
)

anomaly_task = Task(
    description="""
    Analyze detected price anomalies and explain:
    - What happened
    - Why it is unusual
    - Whether it signals risk or opportunity
    """,
    expected_output="Brief explanation of detected anomalies and implications",
    agent=anomaly_agent
)

fundamental_analyst_agent = Agent(
    role="Fundamental Analyst",
    goal="Analyze valuation, profitability, and growth metrics",
    backstory="Equity analyst specializing in company fundamentals",
    verbose=True
)

fundamental_task = Task(
    description="""
    Analyze company fundamentals using valuation, profitability,
    and growth metrics. Explain what they imply for investors.
    """,
    expected_output="Brief, investor-friendly fundamental analysis",
    agent=fundamental_analyst_agent
)

anomaly_crew = Crew(
    agents=[anomaly_agent],
    tasks=[anomaly_task],
    verbose=True
)

def get_anomaly_analysis_crewai(tickers, start, end):
    df = fetch_stock_data(tickers, start, end)

    anomalies = detect_price_anomalies(df)

    if not anomalies:
        return "No significant price anomalies detected during the selected period."

    context = {
        "anomaly_data": anomalies
    }

    result = anomaly_crew.kickoff(inputs=context)
    return anomalies, result.raw

risk_task = Task(
    description="""
    Analyze risks and opportunities considering:
    - Market conditions
    - Earnings
    - Competition
    - Macroeconomic trends
    """,
    expected_output="Risk and opportunity analysis per stock",
    agent=risk_analyst_agent
)

performance_crew = Crew(
    agents=[stock_analyst_agent],
    tasks=[performance_task],
    verbose=True
)

risk_crew = Crew(
    agents=[risk_analyst_agent],
    tasks=[risk_task],
    verbose=True
)
fundamental_crew = Crew(
    agents=[fundamental_analyst_agent],
    tasks=[fundamental_task],
    verbose=True
)

# -------------------- CREW FUNCTIONS --------------------
def get_stock_summary_crewai(tickers, start, end):
    df = fetch_stock_data(tickers, start, end)
    context = {
        "tickers": tickers,
        "price_data": df.tail(5).to_string()
    }
    result = performance_crew.kickoff(inputs=context)
    return result.raw

def get_risks_and_opportunities_crewai(tickers):
    context = {"tickers": tickers}
    result=risk_crew.kickoff(inputs=context)
    return result.raw

def get_fundamental_analysis_crewai(tickers):
    fundamentals = []
    for ticker in tickers:
        fundamentals.append(fetch_fundamentals(ticker))
    df = pd.DataFrame(fundamentals)
    context = {
        "fundamentals_table": df.to_string(index=False)
    }
    result = fundamental_crew.kickoff(inputs=context)
    return df, result.raw

def style_signal(val):
    if val == "BUY":
        return "background-color: #c6f6d5; color: black;"   # green
    elif val == "SELL":
        return "background-color: #fefcbf; color: black;"   # yellow
    elif val == "HOLD":
        return "background-color: #e5e7eb; color: black;"   # gray
    return ""

# -------------------- FAST CHAT (NO CREW) --------------------
def ask_ai_chat(query, tickers):
    memory_context = ""

    for t in tickers:
        mem = st.session_state.stock_memory.get(t, {})
        if mem:
            memory_context += f"""
            Stock: {t}
            Previous Performance Insight: {mem.get('performance')}
            Previous Fundamental Insight: {mem.get('fundamentals')}
            Previous Risks: {mem.get('risks')}
            """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a financial assistant with memory of past analysis."},
            {"role": "user", "content": f"""
            Context:
            {memory_context}

            New Question:
            {query}
            """}
        ]
    )
    return response.choices[0].message.content

def load_tickers_from_file(filepath="tickers.txt"):
    try:
        with open(filepath, "r") as f:
            tickers = [
                line.strip()
                for line in f
                if line.strip() and not line.startswith("#")
            ]
        return sorted(tickers)
    except FileNotFoundError:
        st.error(f"Ticker file not found: {filepath}")
        return []

# -------------------- STREAMLIT APP --------------------
def main():
    st.set_page_config(page_title="Stock Analysis AI Bot", page_icon="📈")
    st.write(css, unsafe_allow_html=True)
    init_ses_states()

    st.title("Stock Analysis AI Bot")
    st.caption("Data-driven stock analysis with AI-powered insights")

    with st.sidebar:
        with st.expander("Options", expanded=True):
            asset_tickers = load_tickers_from_file("tickers.txt")
            asset_dropdown = st.multiselect(
                "Pick Assets:", 
                asset_tickers
            )
            metric_dropdown = st.selectbox(
                "Metric", ['Adj. Close', 'Relative Returns']
            )

            viz_dropdown = st.multiselect(
                "Pick Charts:", ['Line Chart', 'Area Chart']
            )

            start = st.date_input("Start", value=pd.to_datetime("2024-01-01"))
            end = st.date_input("End", value=pd.to_datetime("today"))

    if not asset_dropdown:
        st.warning("Please select at least one asset.")
        return

    df = yf.download(asset_dropdown, start, end, auto_adjust=True)

    if df.empty:
        st.error("No data available for the selected tickers")
        return

    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs("Close", axis=1, level=0)
    elif "Close" in df.columns:
        df = df["Close"]

    if metric_dropdown == "Relative Returns":
        df = relative_returns(df)

    if metric_dropdown == "Absolute Returns":
        df = absolute_returns(df)
    
    if metric_dropdown == "Annualized Return (CAGR)":
        df = annualized_returns(df)

    if viz_dropdown:
        with st.expander("Charts", expanded=True):
            if "Line Chart" in viz_dropdown:
                st.line_chart(df)
            if "Area Chart" in viz_dropdown:
                st.area_chart(df)

    # -------------------- ADVANCED PERFORMANCE METRICS --------------------
    st.subheader("📊 Advanced Performance Metrics")

    try:
        # IMPORTANT: use RAW price data (not relative returns)
        price_metrics_df = fetch_stock_data(asset_dropdown, start, end)

        metrics = compute_all_metrics(price_metrics_df)
        metrics_df = pd.DataFrame(metrics)

        st.dataframe(metrics_df, width='content')

    except Exception as e:
        st.warning(f"Unable to compute metrics: {e}")

    st.subheader("AI Stock Insights")

    if st.button("📊 Get AI Stock Performance Summary"):
        st.markdown("### 📊 Stock Performance")
        summary = get_stock_summary_crewai(asset_dropdown, start, end)
        save_stock_memory(asset_dropdown, "performance", summary)
        st.success(summary)
    
    if st.button("🚨 Detect Price Anomalies"):
        st.markdown("### 🚨 Anomaly Detection")

        # 1️⃣ Run anomaly detection
        anomalies, explanation = get_anomaly_analysis_crewai(
            asset_dropdown, start, end
        )

        # 2️⃣ Save to per-stock memory
        save_stock_memory(asset_dropdown, "anomalies", anomalies)

        # 3️⃣ Render formatted output
        if anomalies:
            for a in anomalies:
                st.markdown(f"#### 📉 {a['ticker']}")
                st.write(f"- **Abnormal moves detected:** {a['anomaly_count']}")
                st.write(f"- **Max daily move:** {a['max_move_%']}%")

                st.markdown("**Dates:**")
                for d in a["dates"]:
                    st.write(f"• {d}")
        else:
            st.success("No significant price anomalies detected.")

        # 4️⃣ AI explanation
        st.markdown("### 🧠 AI Interpretation")
        st.warning(explanation)

    if st.button("💡 Get AI Risks & Opportunities Analysis"):
        st.markdown("### 💡 AI Risks & Opportunities")
        risks = get_risks_and_opportunities_crewai(asset_dropdown)
        save_stock_memory(asset_dropdown, "risks", risks)
        st.success(risks)

    if st.button("📘 Get Fundamental Analysis"):
        st.markdown("### 📊 Fundamental Metrics")
        fundamentals_df, ai_analysis = get_fundamental_analysis_crewai(asset_dropdown)
        save_stock_memory(asset_dropdown, "fundamentals", fundamentals_df)
        st.dataframe(fundamentals_df)
        st.success(ai_analysis)
        st.markdown("### 🧠 AI Fundamental Insight")
        st.success(ai_analysis)

    if st.button("📈 Generate Buy / Hold / Sell Signals"):
        st.markdown("### 📈 Investment Signals")

        price_df = fetch_stock_data(asset_dropdown, start, end)
        fundamentals_df, _ = get_fundamental_analysis_crewai(asset_dropdown)
        anomalies, _ = get_anomaly_analysis_crewai(asset_dropdown, start, end)

        signals = []

        for t in asset_dropdown:
            signal = generate_signal(
                t, price_df, fundamentals_df, anomalies
            )
            signals.append(signal)
            save_stock_memory([t], "signal", signal)

        signals_df = pd.DataFrame(signals)

        styled_df = signals_df.style.map(
            style_signal, subset=["signal"]
        )


        st.dataframe(styled_df, width='content')

        st.caption(
            "⚠️ Signals are rule-based and for educational purposes only. "
            "Not financial advice."
        )

    user_query = st.text_input("Ask AI about selected stocks:")
    if st.button("Ask AI"):
        if user_query:
            answer = ask_ai_chat(user_query, asset_dropdown)
            save_chat_memory(asset_dropdown, user_query, answer)
            st.write(answer)
        else:
            st.warning("Please enter a question.")

    # -------------------- STOCK MEMORY VIEW --------------------
    st.subheader("🧠 Per-Stock Memory")

    if not asset_dropdown:
        st.info("Select stocks to view stored memory.")
    else:
        for t in asset_dropdown:
            with st.expander(f"📦 Memory for {t}", expanded=False):
                memory = st.session_state.stock_memory.get(t, {})

                if not memory:
                    st.write("No memory stored yet for this stock.")
                    continue

                # ⏱️ Timestamp
                memory["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                st.caption(f"Last updated: {memory['updated_at']}")

                # 📈 Performance
                if memory.get("performance"):
                    st.markdown("### 📈 Performance Summary")
                    st.info(brief_summary(memory["performance"]))

                # 📘 Fundamentals
                if memory.get("fundamentals") is not None:
                    st.markdown("### 📘 Fundamentals")
                    if isinstance(memory["fundamentals"], pd.DataFrame):
                        st.dataframe(memory["fundamentals"], width='content')
                    else:
                        st.code(memory["fundamentals"], language="text")

                # ⚠️ Risks
                if memory.get("risks"):
                    st.markdown("### ⚠️ Risks & Opportunities")
                    st.warning(brief_summary(memory["risks"], max_sentences=3))

                # 🚨 Anomalies
                if memory.get("anomalies"):
                    st.markdown("### 🚨 Detected Anomalies")
                    for a in memory["anomalies"]:
                        st.markdown(
                            f"- **{a['ticker']}**: {a['anomaly_count']} extreme moves, "
                            f"max daily move ≈ {a['max_move_%']}%"
                        )

                # 💬 Chat History
                if memory.get("chat"):
                    st.markdown("### 💬 Previous Questions")
                    for i, chat in enumerate(memory["chat"], 1):
                        st.markdown(f"**Q{i}:** {chat['question']}")
                        st.markdown(f"**A{i}:** {chat['answer']}")
                else:
                    st.caption("No chat history yet.")

                # 📈 Signal
                if memory.get("signal"):
                    s = memory["signal"]
                    st.markdown("### 📈 Investment Signal")

                    if s["signal"] == "BUY":
                        st.success("🟢 BUY")
                    elif s["signal"] == "SELL":
                        st.warning("🟡 SELL")
                    else:
                        st.info("⚪ HOLD")

                    st.write(f"- Trend (60d): {s['trend_%']}%")
                    st.write(f"- Fundamental score: {s['fundamental_score']}")
                    st.write(f"- Anomaly score: {s['anomaly_score']}")

                # ⬇️ Export memory
                st.download_button(
                    label="⬇️ Export Memory",
                    data=json.dumps(memory, indent=4, default=str),
                    file_name=f"{t}_memory.json",
                    mime="application/json"
                )

# -------------------- RUN --------------------
if __name__ == "__main__":
    main()
