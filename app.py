import streamlit as st
import requests
import time
import pandas as pd
import plotly.graph_objects as go

# 1. إعدادات الصفحة
st.set_page_config(page_title="Airline Sentiment AI Pro", page_icon="✈️", layout="wide")

API_BASE_URL = "http://127.0.0.1:8000"


# دالة لرسم مؤشر الثقة (Gauge Chart)
def plot_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': "AI Confidence %", 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#007bff"},
            'steps': [
                {'range': [0, 40], 'color': "#ffcccc"},
                {'range': [40, 70], 'color': "#ffffcc"},
                {'range': [70, 100], 'color': "#ccffcc"}
            ],
        }
    ))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
    return fig


# 2. الواجهة الرئيسية
st.title("✈️ Airline Sentiment Analysis")
st.markdown("Analyze customer feedback in real-time with high-precision AI.")

# القائمة الجانبية (Sidebar)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/784/784306.png", width=100)
    st.title("Dashboard Info")
    st.info("Sentiment Analysis helps airlines improve customer service by identifying pain points.")
    st.divider()
    st.caption("Model Version: v2.4 (Production)")
    st.caption("Status: API Public Access ✅")

# 3. منطقة الإدخال والتحليل
st.subheader("📝 Analyze Tweet")
tweet_text = st.text_area(
    "✏️ Enter tweet text:",
    placeholder="Example: The flight was amazing and the crew was very helpful!",
    height=120
)

if st.button("🚀 Analyze Sentiment", use_container_width=True, type="primary"):
    if not tweet_text.strip():
        st.warning("⚠️ Please enter some text first.")
    else:
        with st.spinner("🧠 AI is processing..."):
            try:
                res = requests.post(f"{API_BASE_URL}/predict", json={"text": tweet_text})

                if res.status_code == 200:
                    data = res.json()
                    st.divider()

                    # عرض النتائج
                    col_metrics, col_chart = st.columns([1, 1])

                    with col_metrics:
                        sentiment = data["sentiment"]
                        if sentiment.lower() == "positive":
                            st.success(f"### Result: {sentiment} 😊")
                            st.balloons()
                        elif sentiment.lower() == "negative":
                            st.error(f"### Result: {sentiment} 😡")
                        else:
                            st.warning(f"### Result: {sentiment} 😐")

                        st.metric("Inference Speed", f"{data['inference_time']}s")
                        st.write("**AI Confidence:** Our model is highly confident in this classification.")

                    with col_chart:
                        st.plotly_chart(plot_gauge(data["confidence"]), use_container_width=True)

                elif res.status_code == 429:
                    st.error("🚨 Rate limit exceeded! Please wait a minute.")
                else:
                    st.error(f"❌ Error: {res.json().get('detail', 'Server Error')}")
            except Exception as e:
                st.error(f"🚨 Connection failed: {e}")

# 4. سجل العمليات والإحصائيات العامة (History & Insights)
st.divider()
st.subheader("📊 Global Insights & History")

if st.button("🔄 Refresh History & Charts", use_container_width=True):
    try:
        # ملاحظة: تأكد من إضافة endpoint /predictions في FastAPI
        history_res = requests.get(f"{API_BASE_URL}/predictions")
        if history_res.status_code == 200:
            history_data = history_res.json()
            if history_data:
                df = pd.DataFrame(history_data)

                c1, c2 = st.columns([1, 2])

                with c1:
                    st.write("**Overall Sentiment Distribution**")
                    sentiment_counts = df['sentiment'].value_counts()
                    fig_pie = go.Figure(
                        data=[go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values, hole=.3)])
                    fig_pie.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig_pie, use_container_width=True)

                with c2:
                    st.write("**Recent Predictions**")
                    st.dataframe(df[['text', 'sentiment', 'confidence', 'created_at']].tail(5),
                                 use_container_width=True)
            else:
                st.info("No data available in history yet.")
    except:
        st.error("Could not load history. Make sure the /predictions endpoint is ready.")

# 5. تذييل الصفحة
st.divider()
st.caption("Developed for Airline Sentiment Enterprise Project | 2026")