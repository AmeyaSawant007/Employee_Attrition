import streamlit as st

st.set_page_config(page_title="Attrition EWS", layout="wide")
st.title("🏠 Employee Attrition Early Warning System")
st.caption("About • How to use • Feature glossary • FAQ")

st.markdown("""
### What this app does
This app forecasts employees who are at **risk of leaving next year** (Early Warning System) and helps HR plan interventions.

### Quick start
1. Open **Dashboard** from the left sidebar.
2. Upload `MFG10YearTerminationData.csv` (or use local file).
3. Pick the prediction mode (Next-year recommended).
4. Use the tabs: High Risk → Alerts → Interventions → (optional) Slack.

### Jump to features
- [High Risk + Deep Dive](#high-risk--deep-dive)
- [Insights](#insights)
- [Alerts + Slack](#alerts--slack)
- [Intervention Tracker](#intervention-tracker)
- [Data QA](#data-qa)
""")

st.header("High Risk + Deep Dive")
with st.expander("What it is + how to use", expanded=True):
    st.write("""
- Shows top high-risk employee snapshots with a risk score.
- Pick an EmployeeID to view history across years.
- Generates recommended retention actions and lets you add them to the tracker.
""")

st.header("Insights")
with st.expander("What it is + how to use"):
    st.write("""
- Risk score distribution chart.
- Feature importance (what signals the model used most).
""")

st.header("Alerts + Slack")
with st.expander("What it is + how to use"):
    st.write("""
- Groups risk by department/city/store/business unit.
- Optional: send a summary alert to Slack using an Incoming Webhook URL.
""")

st.header("Intervention Tracker")
with st.expander("What it is + how to use"):
    st.write("""
- Editable tracker where HR can assign an owner, set status (Planned/In Progress/Done), and add notes.
- Export the tracker as CSV for reporting.
""")

st.header("Data QA")
with st.expander("What it is + how to use"):
    st.write("""
- Shows missing value rates and basic sanity checks.
- Helps you explain data quality and model reliability.
""")

st.markdown("---")
st.subheader("FAQ")
with st.expander("How is 'next-year risk' computed?"):
    st.write("We build a label using each employee's next year's STATUS, then train on the current year snapshot.")

with st.expander("Can we retrain automatically?"):
    st.write("Yes—use GitHub Actions schedule / cron to run training and update model artifacts, then the Dashboard loads the latest model.")
