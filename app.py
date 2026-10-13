import requests
import streamlit as st

st.set_page_config(
    page_title="Pharmaceutical Performance Predictor (NIPER Scheme)",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 20px; }
    .card { background-color: #F3F4F6; padding: 20px; border-radius: 10px; }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-header"> NIPER Scheme Performance Predictor</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">AI Decision Support System for Evaluating Grant'
    ' Utilization & Financial Risk Flags</div>',
    unsafe_allow_html=True,
)
st.divider()

# Sidebar Configuration
st.sidebar.header("⚙️ Server Settings")
backend_url = st.sidebar.text_input(
    "FastAPI Backend Endpoint", value="http://127.0.0.1:8000/predict"
)

# Main UI Layout
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
  st.subheader("📋 Grant Proposal Inputs")

  with st.form("proposal_form"):
    campus_name = st.selectbox(
        "Select NIPER Campus",
        options=[
            "NIPER SAS Nagar (Mohali)",
            "NIPER Ahmedabad",
            "NIPER Hyderabad",
            "NIPER Guwahati",
            "NIPER Kolkata",
            "NIPER Raebareli",
            "NIPER Hajipur",
        ],
    )

    grant_category = st.selectbox(
        "Grant Category",
        options=[
            "Campus Infrastructure & Civil",
            "R&D Equipment & Instruments",
            "Consumables & Operational Grants",
            "Clinical Trial & Testing Facilities",
            "Faculty Research & Fellowships",
        ],
    )

    quarter_released = st.select_slider(
        "Quarter of Fund Release", options=["Q1", "Q2", "Q3", "Q4"], value="Q2"
    )

    budget_allocated_cr = st.number_input(
        "Budget Allocation (₹ Crores)",
        min_value=1.0,
        max_value=100.0,
        value=25.0,
        step=0.5,
    )

    release_delay_days = st.slider(
        "Fund Release Delay (Days)", min_value=0, max_value=120, value=15
    )

    procurement_completion_pct = st.slider(
        "Procurement Completion (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
    )

    prior_year_utilization_pct = st.slider(
        "Prior Year Utilization Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=88.0,
    )

    active_projects_count = st.number_input(
        "Active R&D Projects Count", min_value=1, max_value=50, value=15
    )

    submit = st.form_submit_button(
        "📊 Evaluate Grant Proposal", use_container_width=True
    )

with right_col:
  st.subheader("🎯 Model Prediction Output")

  if submit:
    payload = {
        "campus_name": campus_name,
        "grant_category": grant_category,
        "quarter_released": quarter_released,
        "budget_allocated_cr": float(budget_allocated_cr),
        "release_delay_days": int(release_delay_days),
        "procurement_completion_pct": float(procurement_completion_pct),
        "prior_year_utilization_pct": float(prior_year_utilization_pct),
        "active_projects_count": int(active_projects_count),
    }

    try:
      with st.spinner("Executing model inference via FastAPI..."):
        response = requests.post(backend_url, json=payload, timeout=5)

      if response.status_code == 200:
        res = response.json()

        m1, m2 = st.columns(2)
        m1.metric(
            "Forecasted Utilization", f"{res['predicted_utilization_pct']}%"
        )
        m2.metric(
            "Estimated Expenditure", f"₹ {res['estimated_expenditure_cr']} Cr"
        )

        st.write("---")
        st.write("**Utilization Gauge:**")
        st.progress(min(int(res["predicted_utilization_pct"]), 100))

        risk = res["risk_level"]
        st.write("**Risk Analysis & Recommendation:**")
        if risk == "LOW_RISK":
          st.success(f"🟢 **LOW RISK** — {res['recommendation']}")
        elif risk == "MEDIUM_RISK":
          st.warning(f"🟡 **MEDIUM RISK** — {res['recommendation']}")
        else:
          st.error(f"🔴 **HIGH RISK** — {res['recommendation']}")

        with st.expander("🔍 View Raw JSON Response"):
          st.json(res)
      else:
        st.error(f"API Error {response.status_code}: {response.text}")
    except Exception as e:
      st.error(f"Failed to connect to backend: {e}")
  else:
    st.info("👈 Fill in the proposal parameters and click **Evaluate Grant Proposal**.")