"""SupportIQ AI - entry point. Run: streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="SupportIQ AI", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

from src import ui  # noqa: E402

ui.init_session()
ui.apply_theme()
pages = [
    st.Page("pages/1_Executive_Overview.py", title="Executive Overview", icon=":material/dashboard:", url_path="overview", default=True),
    st.Page("pages/2_Ticket_Explorer.py", title="Ticket Explorer", icon=":material/table_view:", url_path="explorer"),
    st.Page("pages/3_AI_Ticket_Analyzer.py", title="AI Ticket Analyzer", icon=":material/psychology:", url_path="analyzer"),
    st.Page("pages/4_SLA_Risk.py", title="SLA Risk Intelligence", icon=":material/timer:", url_path="sla"),
    st.Page("pages/5_Volume_Forecasting.py", title="Volume Forecasting", icon=":material/trending_up:", url_path="forecast"),
    st.Page("pages/6_Similar_Incidents.py", title="Similar Incident Search", icon=":material/manage_search:", url_path="similar"),
    st.Page("pages/7_Customer_Insights.py", title="Customer & Product Insights", icon=":material/groups:", url_path="insights"),
    st.Page("pages/8_Model_Explainability.py", title="Model Performance", icon=":material/analytics:", url_path="model"),
    st.Page("pages/9_Data_Import.py", title="Data Import & Config", icon=":material/upload_file:", url_path="import"),
]
page = st.navigation(pages)
st.session_state["filtered"] = ui.global_filters()
ui.sidebar_status()
page.run()
