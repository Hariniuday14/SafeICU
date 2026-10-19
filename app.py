import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="SafeICU | Safety-Aware RL Lab",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent

st.markdown("""
<style>
.stApp { background: #0a1020; }
[data-testid="stSidebar"] { background: #111a2d; }
.block-container { max-width: 1500px; padding-top: 1.5rem; padding-bottom: 3rem; }
.hero { padding: 1.5rem 1.7rem; border-radius: 18px;
  background: linear-gradient(115deg,#14243b,#124149);
  border: 1px solid #285d66; margin-bottom: 1rem; }
.eyebrow { color: #79e5d4; font-size: .75rem; letter-spacing: .14em;
  font-weight: 800; text-transform: uppercase; }
.hero h1 { color: #f5f8ff; margin: .3rem 0; font-size: 2.5rem; }
.hero p { color: #c4d4e8; margin: 0; }
div[data-testid="stMetric"] { background:#121d31; border:1px solid #263750;
  border-radius:14px; padding: .9rem 1rem; }
div[data-testid="stMetricLabel"] { color:#aabbd0; }
div[data-testid="stMetricValue"] { color:#f5f8ff; }
.note { color:#9cacc2; font-size:.86rem; }
</style>
""", unsafe_allow_html=True)

def load_csv(filename):
    path = DATA_DIR / filename
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

initial = load_csv("safeicu_final_evaluation.csv")
if initial is None:
    initial = load_csv("safeicu_initial_comparison.csv")
final_summary = load_csv("safeicu_final_summary.csv")
followup = load_csv("safeicu_agentic_followup.csv")
agent_report = load_csv("safeicu_agentic_report.csv")
dqn_tests = load_csv("safeicu_dqn_safety_tests.csv")
ppo_tests = load_csv("safeicu_ppo_safety_tests.csv")

st.sidebar.markdown("## 🎛️ Experiment controls")
st.sidebar.caption("Filter stored evaluation results.")
if initial is not None and not initial.empty and {"Scenario", "Policy"}.issubset(initial.columns):
    scenarios = ["All"] + sorted(initial["Scenario"].dropna().astype(str).unique().tolist())
    scenario_filter = st.sidebar.selectbox("Demand scenario", scenarios)
    policies = ["All"] + sorted(initial["Policy"].dropna().astype(str).unique().tolist())
    policy_filter = st.sidebar.selectbox("Policy", policies)
else:
    scenario_filter, policy_filter = "All", "All"

st.sidebar.divider()
st.sidebar.markdown("### About this demo")
st.sidebar.caption("Synthetic educational simulation only. Not a clinical decision-support system.")
st.sidebar.caption("The live simulator below uses a transparent rule-based demo policy. It does not load trained DQN/PPO weights.")

st.markdown("""
<div class="hero">
  <div class="eyebrow">Safety-aware reinforcement learning lab</div>
  <h1>🏥 SafeICU</h1>
  <p>Explore synthetic ICU resource allocation, compare stored RL evaluations, and run an interactive 24-hour demonstration.</p>
</div>
""", unsafe_allow_html=True)
st.markdown('<span class="note">SYNTHETIC DATA · RESEARCH PROTOTYPE · NOT FOR REAL CLINICAL USE</span>', unsafe_allow_html=True)

if initial is None or initial.empty:
    st.error("No evaluation CSV found. Upload safeicu_initial_comparison.csv or safeicu_final_evaluation.csv beside app.py.")
    st.stop()

filtered = initial.copy()
if scenario_filter != "All":
    filtered = filtered[filtered["Scenario"].astype(str) == scenario_filter]
if policy_filter != "All":
    filtered = filtered[filtered["Policy"].astype(str) == policy_filter]

metric_reward = "Reward" if "Reward" in filtered.columns else "Mean_Reward" if "Mean_Reward" in filtered.columns else None
metric_wait = next((c for c in ["Critical Waiting Patient-Hours", "Critical Waiting Burden", "Mean_Critical_Waiting_Burden"] if c in filtered.columns), None)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Evaluation episodes", f"{len(filtered):,}")
k2.metric("Policies in view", int(filtered["Policy"].nunique()))
k3.metric("Scenarios in view", int(filtered["Scenario"].nunique()))
k4.metric("Evaluation seeds", int(filtered["Seed"].nunique()) if "Seed" in filtered.columns else "Not recorded")

tabs = st.tabs(["🏠 Overview", "📊 Policy laboratory", "🛡️ Safety audit", "🧪 Live simulation", "🤖 Agentic supervisor", "📁 Data & methodology"])

with tabs[0]:
    st.subheader("Experiment overview")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Mean reward by policy")
        if metric_reward:
            g = filtered.groupby("Policy", as_index=False)[metric_reward].mean()
            fig, ax = plt.subplots(figsize=(7, 3.7))
            ax.bar(g["Policy"], g[metric_reward])
            ax.set_ylabel("Mean cumulative reward")
            ax.tick_params(axis="x", rotation=25)
            ax.grid(axis="y", alpha=.2)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("Reward metric unavailable in this dataset.")
    with right:
        st.markdown("#### Critical-patient waiting burden")
        if metric_wait:
            g = filtered.groupby("Policy", as_index=False)[metric_wait].mean()
            fig, ax = plt.subplots(figsize=(7, 3.7))
            ax.bar(g["Policy"], g[metric_wait])
            ax.set_ylabel("Waiting burden (source metric)")
            ax.tick_params(axis="x", rotation=25)
            ax.grid(axis="y", alpha=.2)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("Waiting-burden metric unavailable.")
    st.markdown("#### Filtered evaluation records")
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download filtered evaluation", filtered.to_csv(index=False), "safeicu_filtered_evaluation.csv", "text/csv")

with tabs[1]:
    st.subheader("Policy laboratory")
    st.write("Explore stored evaluation metrics. Results are read from CSV files; this page does not retrain the RL models.")
    numeric_cols = [c for c in filtered.select_dtypes(include="number").columns if c != "Seed"]
    if numeric_cols:
        default = "Reward" if "Reward" in numeric_cols else numeric_cols[0]
        chosen_metric = st.selectbox("Metric to compare", numeric_cols, index=numeric_cols.index(default), key="lab_metric")
        grouped = filtered.groupby("Policy")[chosen_metric].agg(["mean", "std"]).reset_index()
        grouped = grouped.rename(columns={"mean": "Mean", "std": "StdDev"})
        st.dataframe(grouped.round(3), use_container_width=True, hide_index=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(grouped["Policy"], grouped["Mean"])
        ax.set_title(f"Mean {chosen_metric} by policy")
        ax.set_ylabel(chosen_metric)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=.2)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("No numeric metrics available.")

with tabs[2]:
    st.subheader("Safety audit")
    st.write("Shield interventions and executed violations are different. A blocked proposal is not an executed violation.")
    viol_col = next((c for c in ["Executed Violations", "constraint_violations", "Constraint Violations"] if c in filtered.columns), None)
    inter_col = next((c for c in ["Shield Interventions", "Mean_Shield_Interventions", "shield_interventions"] if c in filtered.columns), None)
    delay_col = next((c for c in ["Mean Critical Admission Delay", "Mean_Critical_Admission_Delay"] if c in filtered.columns), None)
    a, b, c = st.columns(3)
    a.metric("Executed violations", f"{filtered[viol_col].mean():.2f}" if viol_col else "Not recorded")
    b.metric("Mean shield interventions", f"{filtered[inter_col].mean():.2f}" if inter_col else "Not recorded")
    c.metric("Mean critical admission delay", f"{filtered[delay_col].mean():.2f}" if delay_col else "Not recorded")
    if dqn_tests is not None:
        st.markdown("#### DQN regression tests")
        st.dataframe(dqn_tests, use_container_width=True, hide_index=True)
    if ppo_tests is not None:
        st.markdown("#### PPO regression tests")
        st.dataframe(ppo_tests, use_container_width=True, hide_index=True)
    if not viol_col:
        st.warning("Executed-violation counts are absent from the selected CSV. Do not infer zero violations from missing data.")

with tabs[3]:
    st.subheader("Interactive 24-hour ICU simulation")
    st.markdown("This is a new synthetic demonstration episode. It animates the simulated resource and queue trajectory when you run it. **It uses an explicit priority-rule demo policy, not the trained DQN/PPO models.**")
    c1, c2, c3 = st.columns(3)
    sim_scenario = c1.selectbox("Simulation scenario", ["normal", "surge", "shortage"], key="sim_scenario")
    sim_seed = c2.number_input("Random seed", min_value=1, max_value=99999, value=42, step=1)
    horizon = c3.selectbox("Duration (hours)", [12, 24, 48], index=1)
    run = st.button("▶ Run synthetic simulation", type="primary")
    if run:
        rng = np.random.default_rng(int(sim_seed))
        beds, monitors, staff = ({"normal": (10, 8, 6), "surge": (10, 8, 6), "shortage": (5, 4, 3)})[sim_scenario]
        arrival_rate = {"normal": 1.0, "surge": 2.2, "shortage": 1.5}[sim_scenario]
        if sim_scenario == "shortage":
            arrival_rate = 1.5
        waiting = [0, 0, 0]  # critical, high, moderate
        active = []
        records = []
        arrivals_total = 0
        admissions_total = 0
        violations = 0
        for hour in range(horizon):
            active = [r - 1 for r in active if r - 1 > 0]
            arrivals = int(rng.poisson(arrival_rate))
            cats = rng.choice(3, size=arrivals, p=[0.20, 0.35, 0.45]) if arrivals else []
            for cat in cats:
                waiting[int(cat)] += 1
            arrivals_total += arrivals
            occupied = len(active)
            capacity_available = occupied < min(beds, monitors, staff)
            admitted_now = 0
            if capacity_available and sum(waiting) > 0:
                # Transparent priority rule: critical > high > moderate.
                category = next(i for i, count in enumerate(waiting) if count > 0)
                waiting[category] -= 1
                active.append(int(rng.integers(2, 7)))
                admissions_total += 1
                admitted_now = 1
            occupied = len(active)
            records.append({
                "Hour": hour + 1,
                "Arrivals": arrivals,
                "Critical waiting": waiting[0],
                "High waiting": waiting[1],
                "Moderate waiting": waiting[2],
                "Active care slots": occupied,
                "Bed utilization": occupied / beds,
                "Monitor utilization": occupied / monitors,
                "Staff utilization": occupied / staff,
                "Admissions this hour": admitted_now,
                "Waiting burden": waiting[0]
            })
        sim_df = pd.DataFrame(records)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total arrivals", arrivals_total)
        m2.metric("Admissions", admissions_total)
        m3.metric("End-of-run critical queue", int(sim_df["Critical waiting"].iloc[-1]))
        m4.metric("Mean bed utilization", f"{sim_df['Bed utilization'].mean():.0%}")
        st.markdown("#### Queue and resource trajectory")
        fig, ax = plt.subplots(figsize=(10, 4))
        for col in ["Critical waiting", "High waiting", "Moderate waiting"]:
            ax.plot(sim_df["Hour"], sim_df[col], marker="o", markersize=3, label=col)
        ax.set_xlabel("Simulated hour")
        ax.set_ylabel("Patients waiting")
        ax.legend()
        ax.grid(alpha=.2)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(sim_df["Hour"], sim_df["Bed utilization"], label="Beds")
        ax.plot(sim_df["Hour"], sim_df["Monitor utilization"], label="Monitors")
        ax.plot(sim_df["Hour"], sim_df["Staff utilization"], label="Staff")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Simulated hour")
        ax.set_ylabel("Utilization")
        ax.legend()
        ax.grid(alpha=.2)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("#### Hour-by-hour simulation log")
        st.dataframe(sim_df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download simulation log", sim_df.to_csv(index=False), "safeicu_live_demo_log.csv", "text/csv")
        st.info("This is a live-generated synthetic episode using a rule-based priority policy. It is not inference from your trained DQN/PPO weights. To demonstrate the trained agents live, their saved model files and matching environment implementation must also be deployed.")
    else:
        st.info("Choose a scenario and click **Run synthetic simulation** to generate a new episode. The result is generated on demand, not read from the saved evaluation CSV.")

with tabs[4]:
    st.subheader("Agentic AI experiment supervisor")
    st.write("The notebook supervisor analyzed saved results and generated a bounded follow-up experiment. The deployed dashboard shows those saved outputs; it does not currently execute the supervisor or train agents on the server.")
    if agent_report is not None:
        st.markdown("#### Follow-up summary")
        st.dataframe(agent_report, use_container_width=True, hide_index=True)
    else:
        st.info("safeicu_agentic_report.csv is missing.")
    if followup is not None:
        st.markdown("#### Follow-up experiment records")
        st.dataframe(followup, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download follow-up data", followup.to_csv(index=False), "safeicu_agentic_followup.csv", "text/csv")
    if final_summary is not None:
        st.markdown("#### Improved evaluation summary")
        st.dataframe(final_summary, use_container_width=True, hide_index=True)

with tabs[5]:
    st.subheader("Data, assumptions & methodology")
    st.markdown("""
**Environment:** synthetic hourly ICU simulation with discrete priority categories and constrained resources.

**Policies:** priority baseline, DQN, PPO, and shielded variants where the saved data includes them.

**Scenarios:** normal demand, surge demand, and reduced resource capacity.

**Live simulation tab:** generates a new synthetic episode using a transparent priority rule. It is deliberately not labelled as trained-agent inference.

**Limitations:** this simulator is not medically validated. Results describe only the code, reward function, and synthetic assumptions used in this project.
""")
    st.markdown("#### Data files in repository")
    expected_files = [
        "safeicu_initial_comparison.csv", "safeicu_final_evaluation.csv",
        "safeicu_final_summary.csv", "safeicu_agentic_followup.csv",
        "safeicu_agentic_report.csv", "safeicu_dqn_safety_tests.csv",
        "safeicu_ppo_safety_tests.csv"
    ]
    for name in expected_files:
        st.write(("✅ " if (DATA_DIR / name).exists() else "— ") + name)
    st.download_button("⬇️ Download filtered data", filtered.to_csv(index=False), "safeicu_evaluation_export.csv", "text/csv")

st.divider()
st.caption("SafeICU research prototype · Synthetic data only · Not for real clinical use.")
