import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------
# SafeICU | Research dashboard for synthetic RL experiments
# ------------------------------------------------------------
st.set_page_config(
    page_title="SafeICU | Safety-Aware RL Lab",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent

# A restrained dark research-console theme; charts use Streamlit theme defaults.
st.markdown(
    """
    <style>
      .stApp { background: #0b1020; }
      [data-testid="stSidebar"] { background: #11182a; }
      .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px; }
      .hero {
        padding: 1.4rem 1.6rem; border-radius: 18px;
        background: linear-gradient(115deg, #14243b 0%, #123b43 100%);
        border: 1px solid rgba(111, 231, 213, .25);
        margin-bottom: 1.1rem;
      }
      .eyebrow { color: #6fe7d5; text-transform: uppercase; letter-spacing: .13em;
        font-size: .74rem; font-weight: 700; }
      .hero h1 { margin: .3rem 0 .35rem 0; font-size: 2.35rem; color: #f4f8ff; }
      .hero p { margin: 0; color: #c5d4e7; font-size: 1rem; }
      .small-note { color: #9eacc2; font-size: .84rem; }
      div[data-testid="stMetric"] {
        background: #121b2d; border: 1px solid #25354b; padding: 1rem;
        border-radius: 14px;
      }
      div[data-testid="stMetricLabel"] { color: #aabbd0; }
      div[data-testid="stMetricValue"] { color: #f4f8ff; }
      .status-pill {
        display: inline-block; border: 1px solid #326b67; color: #8af0d8;
        background: #102f34; padding: .28rem .65rem; border-radius: 999px;
        font-size: .78rem; font-weight: 650;
      }
      .section-intro { color: #aabbd0; margin-top: -.5rem; }
      .stTabs [data-baseweb="tab-list"] { gap: .4rem; }
      .stTabs [data-baseweb="tab"] { background: #121b2d; border-radius: 8px 8px 0 0; padding: .7rem 1rem; }
      hr { border-color: #26344a; }
    </style>
    """,
    unsafe_allow_html=True,
)

def load_csv(filename):
    path = DATA_DIR / filename
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read `{filename}`: {exc}")
        return None

# Load both original and improved evaluation files if available.
initial = load_csv("safeicu_final_evaluation.csv")
if initial is None:
    initial = load_csv("safeicu_initial_comparison.csv")

final_summary = load_csv("safeicu_final_summary.csv")
followup = load_csv("safeicu_agentic_followup.csv")
agent_report = load_csv("safeicu_agentic_report.csv")
dqn_tests = load_csv("safeicu_dqn_safety_tests.csv")
ppo_tests = load_csv("safeicu_ppo_safety_tests.csv")

if initial is None or initial.empty:
    st.error(
        "Evaluation data was not found. Upload `safeicu_initial_comparison.csv` "
        "or `safeicu_final_evaluation.csv` beside `app.py` in the GitHub repository."
    )
    st.stop()

# Normalize scenario/policy labels without fabricating values.
if "Scenario" not in initial.columns or "Policy" not in initial.columns:
    st.error("The evaluation CSV must contain `Scenario` and `Policy` columns.")
    st.stop()

# Sidebar
st.sidebar.markdown("## 🎛️ Experiment controls")
st.sidebar.caption("Filter stored synthetic evaluation results.")
scenarios = ["All"] + sorted(initial["Scenario"].dropna().astype(str).unique().tolist())
scenario = st.sidebar.selectbox("Demand scenario", scenarios, index=0)
policies = ["All"] + sorted(initial["Policy"].dropna().astype(str).unique().tolist())
selected_policy = st.sidebar.selectbox("Policy", policies, index=0)

st.sidebar.divider()
st.sidebar.markdown("### About this demo")
st.sidebar.caption(
    "SafeICU is a synthetic educational simulation. It is not a clinical "
    "decision-support system and must not be used for real patient care."
)
st.sidebar.caption("Results are loaded from versioned CSV files; this dashboard does not retrain models.")

filtered = initial.copy()
if scenario != "All":
    filtered = filtered[filtered["Scenario"].astype(str) == scenario]
if selected_policy != "All":
    filtered = filtered[filtered["Policy"].astype(str) == selected_policy]

# Header
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Safety-aware reinforcement learning lab</div>
      <h1>🏥 SafeICU</h1>
      <p>Evaluating resource-allocation policies under normal demand, emergency surge,
      and constrained-capacity scenarios.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<span class="status-pill">SYNTHETIC DATA</span> '
    '<span class="small-note">Research prototype · Not a clinical decision system</span>',
    unsafe_allow_html=True,
)

# KPI row: use available columns, clearly describe episode count.
k1, k2, k3, k4 = st.columns(4)
k1.metric("Evaluation episodes", f"{len(filtered):,}")
k2.metric("Policies in view", int(filtered["Policy"].nunique()))
k3.metric("Scenarios in view", int(filtered["Scenario"].nunique()))
if "Seed" in filtered.columns:
    k4.metric("Unique evaluation seeds", int(filtered["Seed"].nunique()))
else:
    k4.metric("Source", "CSV results")

st.divider()

tab_overview, tab_policies, tab_safety, tab_agent, tab_data = st.tabs(
    ["🏠 Overview", "📊 Policy laboratory", "🛡️ Safety audit", "🤖 Agentic supervisor", "📁 Data & methodology"]
)

def first_existing_column(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None

with tab_overview:
    st.subheader("Experiment overview")
    st.markdown(
        '<p class="section-intro">Use the filters on the left to inspect a specific demand scenario or policy.</p>',
        unsafe_allow_html=True,
    )
    a, b = st.columns([1.15, 1])
    with a:
        st.markdown("#### Mean reward by policy")
        reward_col = first_existing_column(filtered, ["Reward", "Mean_Reward"])
        if reward_col:
            reward = filtered.groupby("Policy", as_index=False)[reward_col].mean()
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(reward["Policy"], reward[reward_col])
            ax.set_ylabel("Mean cumulative reward")
            ax.tick_params(axis="x", rotation=25)
            ax.grid(axis="y", alpha=.2)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("Reward column is not available in this CSV.")
    with b:
        st.markdown("#### Critical-patient waiting burden")
        waiting_col = first_existing_column(
            filtered,
            ["Critical Waiting Patient-Hours", "Critical Waiting Burden",
             "Mean_Critical_Waiting_Burden", "Critical_Waiting_Patient_Hours"]
        )
        if waiting_col:
            waiting = filtered.groupby("Policy", as_index=False)[waiting_col].mean()
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(waiting["Policy"], waiting[waiting_col])
            ax.set_ylabel("Waiting burden (patient-hours where applicable)")
            ax.tick_params(axis="x", rotation=25)
            ax.grid(axis="y", alpha=.2)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("A critical-waiting metric is not available in this CSV.")
    st.markdown("#### Filtered evaluation records")
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download filtered records",
        data=filtered.to_csv(index=False),
        file_name="safeicu_filtered_evaluation.csv",
        mime="text/csv",
    )

with tab_policies:
    st.subheader("Policy laboratory")
    st.markdown(
        "Compare the observed performance of the baseline, DQN, PPO, and shielded variants. "
        "Higher reward is not automatically safer; review safety and waiting metrics together."
    )
    numeric_cols = [c for c in filtered.select_dtypes(include="number").columns
                    if c not in {"Seed"}]
    if not numeric_cols:
        st.info("No numeric metrics found in the selected data.")
    else:
        default_metric = (
            "Reward" if "Reward" in numeric_cols
            else "Mean_Reward" if "Mean_Reward" in numeric_cols
            else numeric_cols[0]
        )
        metric = st.selectbox("Metric to compare", numeric_cols, index=numeric_cols.index(default_metric))
        summary = filtered.groupby("Policy", as_index=False)[metric].agg(["mean", "std"]).reset_index()
        summary.columns = ["Policy", "Mean", "StdDev"]
        st.dataframe(summary.round(3), use_container_width=True, hide_index=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(summary["Policy"], summary["Mean"])
        ax.set_title(f"Mean {metric} by policy")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=.2)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("Standard deviation is shown in the table when multiple rows are available per policy.")

with tab_safety:
    st.subheader("Safety audit")
    st.markdown(
        "The safety shield is a rule-based action filter. Intervention counts represent rejected/modified "
        "proposals; executed violations are a separate metric and should not be conflated."
    )
    violation_col = first_existing_column(
        filtered, ["Executed Violations", "constraint_violations", "Constraint Violations"]
    )
    intervention_col = first_existing_column(
        filtered, ["Shield Interventions", "Mean_Shield_Interventions",
                   "shield_interventions", "interventions"]
    )
    delay_col = first_existing_column(
        filtered, ["Mean Critical Admission Delay", "mean_critical_wait",
                   "Mean_Critical_Admission_Delay"]
    )
    s1, s2, s3 = st.columns(3)
    if violation_col:
        s1.metric("Mean executed violations", f"{filtered[violation_col].mean():.2f}")
    else:
        s1.metric("Executed violations", "Not recorded")
    if intervention_col:
        s2.metric("Mean shield interventions", f"{filtered[intervention_col].mean():.2f}")
    else:
        s2.metric("Shield interventions", "Not recorded")
    if delay_col:
        s3.metric("Mean critical admission delay", f"{filtered[delay_col].mean():.2f}")
    else:
        s3.metric("Admission delay", "Not recorded")
    if dqn_tests is not None:
        st.markdown("#### DQN safety regression tests")
        st.dataframe(dqn_tests, use_container_width=True, hide_index=True)
    if ppo_tests is not None:
        st.markdown("#### PPO safety regression tests")
        st.dataframe(ppo_tests, use_container_width=True, hide_index=True)
    if not violation_col:
        st.warning(
            "The selected evaluation file does not include executed-violation counts. "
            "Do not claim zero violations based only on a missing column."
        )

with tab_agent:
    st.subheader("Agentic AI experiment supervisor")
    st.markdown(
        "This prototype uses a rule-based supervisor that analyzes saved results, chooses a follow-up "
        "scenario, calls a bounded experiment runner during notebook execution, and summarizes outcomes. "
        "The deployed dashboard displays stored outputs; it does not execute training or experiments."
    )
    if agent_report is not None and not agent_report.empty:
        st.markdown("#### Follow-up experiment summary")
        st.dataframe(agent_report, use_container_width=True, hide_index=True)
    else:
        st.info("Agentic report CSV is not present.")
    if followup is not None and not followup.empty:
        st.markdown("#### Follow-up experiment records")
        st.dataframe(followup, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download agentic follow-up data",
            data=followup.to_csv(index=False),
            file_name="safeicu_agentic_followup.csv",
            mime="text/csv",
        )
    if final_summary is not None and not final_summary.empty:
        st.markdown("#### Improved evaluation summary")
        st.dataframe(final_summary, use_container_width=True, hide_index=True)

with tab_data:
    st.subheader("Data, assumptions & methodology")
    st.markdown(
        """
        **Environment:** synthetic hourly ICU simulation with discrete patient-priority categories
        and constrained beds, monitoring units, and staff capacity.

        **Policies:** priority-based baseline, DQN, PPO, and safety-shielded variants where available.

        **Scenarios:** normal arrivals, surge arrivals, and reduced resource capacity.

        **Interpretation:** results describe only the implemented simulator and reward function. They do
        not establish clinical validity, patient outcomes, or safety in a real hospital.
        """
    )
    st.markdown("#### Available data files")
    files = [
        "safeicu_initial_comparison.csv",
        "safeicu_final_evaluation.csv",
        "safeicu_final_summary.csv",
        "safeicu_agentic_followup.csv",
        "safeicu_agentic_report.csv",
        "safeicu_dqn_safety_tests.csv",
        "safeicu_ppo_safety_tests.csv",
    ]
    for filename in files:
        path = DATA_DIR / filename
        st.write(("✅ " if path.exists() else "— ") + filename)
    st.download_button(
        "⬇️ Download current filtered dataset",
        data=filtered.to_csv(index=False),
        file_name="safeicu_evaluation_export.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "SafeICU research prototype · Synthetic data only · Not for real clinical use. "
    "Interpret every metric within the simulator's assumptions."
)
