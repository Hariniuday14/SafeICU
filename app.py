
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="SafeICU Dashboard",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 SafeICU")
st.subheader("Safe Reinforcement Learning for ICU Resource Allocation")
st.caption(
    "Synthetic educational simulation — not a clinical decision system."
)

DATA = Path(__file__).parent


def load_csv(filename):
    path = DATA / filename
    if path.exists():
        return pd.read_csv(path)
    return None


initial = load_csv("safeicu_initial_comparison.csv")
followup = load_csv("safeicu_agentic_followup.csv")
agent_report = load_csv("safeicu_agentic_report.csv")
dqn_tests = load_csv("safeicu_dqn_safety_tests.csv")
ppo_tests = load_csv("safeicu_ppo_safety_tests.csv")

if initial is None:
    st.error(
        "Evaluation data not found. Place the CSV files "
        "beside app.py and restart the dashboard."
    )
    st.stop()

st.sidebar.header("Dashboard Controls")

scenario = st.sidebar.selectbox(
    "Select demand scenario",
    ["All", "normal", "surge", "shortage"]
)

if scenario == "All":
    filtered = initial.copy()
else:
    filtered = initial[
        initial["Scenario"] == scenario
    ].copy()

# Summary metrics
c1, c2, c3 = st.columns(3)

c1.metric("Evaluation episodes", len(filtered))

c2.metric(
    "Policies compared",
    filtered["Policy"].nunique()
)

c3.metric(
    "Scenarios",
    filtered["Scenario"].nunique()
)

st.divider()

# Policy comparison
st.header("1. Policy Performance")

reward_summary = (
    filtered.groupby("Policy")
    .agg(
        Mean_Reward=("Reward", "mean"),
        Mean_Admissions=("Admissions", "mean"),
        Mean_Critical_Waiting_Burden=(
            "Critical Waiting Burden", "mean"
        ),
        Mean_Shield_Interventions=(
            "Shield Interventions", "mean"
        )
    )
    .reset_index()
)

st.dataframe(
    reward_summary.round(2),
    use_container_width=True
)

fig, ax = plt.subplots(figsize=(10, 4))

reward_summary.plot(
    x="Policy",
    y="Mean_Reward",
    kind="bar",
    ax=ax,
    legend=False
)

ax.set_title("Mean Cumulative Reward")
ax.set_ylabel("Reward")
ax.set_xlabel("")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

# Waiting burden
st.header("2. Critical-Patient Waiting Burden")

fig, ax = plt.subplots(figsize=(10, 4))

reward_summary.plot(
    x="Policy",
    y="Mean_Critical_Waiting_Burden",
    kind="bar",
    ax=ax,
    legend=False
)

ax.set_title("Mean Cumulative Waiting Burden")
ax.set_ylabel("Patient-hours")
ax.set_xlabel("")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

# Agentic AI report
st.header("3. Agentic AI Experiment Supervisor")

if agent_report is not None:
    st.write(
        "The supervisor's follow-up experiment results:"
    )
    st.dataframe(
        agent_report.round(2),
        use_container_width=True
    )
else:
    st.info("Agentic AI report CSV not found.")

if followup is not None:
    st.download_button(
        "Download agentic follow-up data",
        data=followup.to_csv(index=False),
        file_name="safeicu_agentic_followup.csv",
        mime="text/csv"
    )

# Safety test reports
st.header("4. Safety Regression Tests")

for label, tests in [
    ("DQN", dqn_tests),
    ("PPO", ppo_tests)
]:
    st.subheader(label)

    if tests is not None:
        st.dataframe(tests, use_container_width=True)
    else:
        st.info(f"{label} safety test data not found.")

# Download evaluation
st.header("5. Export Results")

st.download_button(
    "Download initial evaluation data",
    data=initial.to_csv(index=False),
    file_name="safeicu_initial_comparison.csv",
    mime="text/csv"
)

st.success(
    "Dashboard loaded. Interpret all metrics in the "
    "context of the simplified synthetic environment."
)
