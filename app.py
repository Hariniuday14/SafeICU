
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="SafeICU Dashboard",
    page_icon="🏥",
    layout="wide"
)

DATA = Path(__file__).parent

st.title("🏥 SafeICU")
st.subheader("Safe Reinforcement Learning for ICU Resource Allocation")
st.caption(
    "Synthetic educational simulation — not a clinical decision system."
)


# --------------------------------------------------
# FILE LOADING
# --------------------------------------------------

def load_csv(filename):
    path = DATA / filename
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as exc:
            st.warning(f"Could not read {filename}: {exc}")
    return None


def file_exists(filename):
    return (DATA / filename).exists()


# Prefer the genuinely new evaluation file.
final_evaluation = load_csv("safeicu_final_evaluation.csv")
initial_comparison = load_csv("safeicu_initial_comparison.csv")

if final_evaluation is not None:
    initial = final_evaluation.copy()
    evaluation_source = "Independent evaluation"
else:
    initial = initial_comparison
    evaluation_source = "Initial comparison (fallback)"

followup = load_csv("safeicu_agentic_followup.csv")
agent_report = load_csv("safeicu_agentic_report.csv")
dqn_tests = load_csv("safeicu_dqn_safety_tests.csv")
ppo_tests = load_csv("safeicu_ppo_safety_tests.csv")
final_summary = load_csv("safeicu_final_summary.csv")

if initial is None:
    st.error(
        "No evaluation CSV found. Upload "
        "safeicu_final_evaluation.csv or "
        "safeicu_initial_comparison.csv beside app.py."
    )
    st.stop()


# Validate columns required by the dashboard.
required_columns = [
    "Policy",
    "Scenario",
    "Reward",
    "Admissions",
    "Critical Waiting Burden",
    "Shield Interventions",
]

missing_columns = [
    col for col in required_columns
    if col not in initial.columns
]

if missing_columns:
    st.error(
        "Evaluation CSV is missing required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Dashboard Controls")

scenario_options = ["All", "normal", "surge", "shortage"]

scenario = st.sidebar.selectbox(
    "Select demand scenario",
    scenario_options
)

if scenario == "All":
    filtered = initial.copy()
else:
    filtered = initial[
        initial["Scenario"].astype(str).str.lower() == scenario
    ].copy()

st.sidebar.caption(f"Data source: {evaluation_source}")
st.sidebar.caption(f"Records loaded: {len(initial)}")


# --------------------------------------------------
# TAB LAYOUT
# --------------------------------------------------

(
    overview_tab,
    laboratory_tab,
    safety_tab,
    live_tab,
    agentic_tab,
    methodology_tab,
) = st.tabs([
    "Overview",
    "Policy Laboratory",
    "Safety Audit",
    "Live Simulation",
    "Agentic Supervisor",
    "Data & Methodology",
])


# --------------------------------------------------
# TAB 1: OVERVIEW
# --------------------------------------------------

with overview_tab:
    st.header("Evaluation Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Evaluation episodes", len(filtered))
    c2.metric("Policies compared", filtered["Policy"].nunique())
    c3.metric("Scenarios", filtered["Scenario"].nunique())
    c4.metric("Data source", evaluation_source)

    st.info(
        "Results come from a synthetic environment. "
        "They are not clinical evidence."
    )

    reward_summary = (
        filtered.groupby("Policy", as_index=False)
        .agg(
            Mean_Reward=("Reward", "mean"),
            Mean_Admissions=("Admissions", "mean"),
            Mean_Critical_Waiting_Burden=(
                "Critical Waiting Burden", "mean"
            ),
            Mean_Shield_Interventions=(
                "Shield Interventions", "mean"
            ),
        )
    )

    st.subheader("Policy Performance Summary")
    st.dataframe(
        reward_summary.round(2),
        use_container_width=True,
        hide_index=True,
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(
        reward_summary["Policy"],
        reward_summary["Mean_Reward"],
    )
    ax.set_title("Mean Cumulative Reward")
    ax.set_ylabel("Reward")
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(
        reward_summary["Policy"],
        reward_summary["Mean_Critical_Waiting_Burden"],
    )
    ax.set_title("Mean Critical-Patient Waiting Burden")
    ax.set_ylabel("Cumulative waiting burden")
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    if "Seed" in filtered.columns:
        st.subheader("Evaluation Coverage")
        st.write(
            "Distinct seeds in selected data:",
            filtered["Seed"].nunique(),
        )

    st.download_button(
        "Download displayed evaluation data",
        data=filtered.to_csv(index=False),
        file_name="safeicu_displayed_evaluation.csv",
        mime="text/csv",
    )


# --------------------------------------------------
# TAB 2: POLICY LABORATORY
# --------------------------------------------------

with laboratory_tab:
    st.header("Policy Laboratory")
    st.write(
        "Compare simulated policy outcomes for the selected scenario."
    )

    lab_metrics = [
        ("Reward", "Mean reward"),
        ("Admissions", "Mean admissions"),
        ("Critical Waiting Burden", "Mean critical waiting burden"),
        ("Shield Interventions", "Mean shield interventions"),
    ]

    selected_metric = st.selectbox(
        "Metric to compare",
        options=[m[0] for m in lab_metrics],
        format_func=lambda value: dict(lab_metrics)[value],
    )

    lab_summary = (
        filtered.groupby("Policy", as_index=False)
        .agg(Value=(selected_metric, "mean"))
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(lab_summary["Policy"], lab_summary["Value"])
    ax.set_title(dict(lab_metrics)[selected_metric])
    ax.set_ylabel(dict(lab_metrics)[selected_metric])
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Detailed policy metrics")
    st.dataframe(
        lab_summary.round(3),
        use_container_width=True,
        hide_index=True,
    )


# --------------------------------------------------
# TAB 3: SAFETY AUDIT
# --------------------------------------------------

with safety_tab:
    st.header("Safety Audit")
    st.caption(
        "Recorded simulation checks only; these do not establish "
        "clinical safety or prove the absence of all violations."
    )

    for label, tests in [
        ("DQN Safety Tests", dqn_tests),
        ("PPO Safety Tests", ppo_tests),
    ]:
        st.subheader(label)

        if tests is None:
            st.warning(f"{label} CSV not found.")
            continue

        st.dataframe(
            tests,
            use_container_width=True,
            hide_index=True,
        )

        for col in [
            "resource_limit_checks_passed",
            "shield_checks_passed",
        ]:
            if col in tests.columns:
                passed = tests[col].fillna(False).astype(bool)
                st.write(
                    f"{col}: {int(passed.sum())}/{len(tests)} "
                    "recorded checks passed"
                )

        if "issues" in tests.columns:
            issues = tests["issues"].dropna().astype(str)
            if len(issues):
                st.write("Recorded issues:", issues.tolist())
            else:
                st.info(
                    "No issue details were recorded in this file. "
                    "Missing issue values do not prove zero violations."
                )

    st.subheader("Independent Evaluation Resource Checks")

    check_col = "Resource Limit Check Passed"
    if check_col in initial.columns:
        checks = initial[check_col].fillna(False).astype(bool)
        st.metric(
            "Resource checks passed",
            f"{int(checks.sum())}/{len(initial)}",
        )
        st.dataframe(
            initial[
                [
                    c for c in [
                        "Policy",
                        "Scenario",
                        "Seed",
                        check_col,
                        "Shield Check Passed",
                        "Reported Violations",
                    ]
                    if c in initial.columns
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "The selected evaluation dataset does not contain "
            "resource-check columns."
        )

    if "Reported Violations" in initial.columns:
        st.warning(
            "The environment's reported violation counter is not "
            "an independent proof of safety. Interpret it alongside "
            "the actual environment implementation."
        )


# --------------------------------------------------
# TAB 4: LIVE SIMULATION
# --------------------------------------------------

with live_tab:
    st.header("Live Simulation")
    st.write(
        "Inspect individual rows from the saved simulation results. "
        "This tab does not run a new RL episode."
    )

    if filtered.empty:
        st.warning("No records for this scenario.")
    else:
        policy_options = ["All"] + sorted(
            filtered["Policy"].dropna().unique().tolist()
        )

        chosen_policy = st.selectbox(
            "Select policy",
            policy_options,
            key="live_policy",
        )

        live_data = filtered.copy()
        if chosen_policy != "All":
            live_data = live_data[
                live_data["Policy"] == chosen_policy
            ]

        st.dataframe(
            live_data,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download selected simulation records",
            data=live_data.to_csv(index=False),
            file_name="safeicu_simulation_records.csv",
            mime="text/csv",
        )


# --------------------------------------------------
# TAB 5: AGENTIC SUPERVISOR
# --------------------------------------------------

with agentic_tab:
    st.header("Agentic AI Experiment Supervisor")

    st.write(
        "Review the saved follow-up experiment and supervisor report."
    )

    if agent_report is not None:
        st.subheader("Supervisor Report")
        st.dataframe(
            agent_report,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Agentic report CSV not found.")

    if followup is not None:
        st.subheader("Follow-up Experiment Data")
        st.dataframe(
            followup,
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download agentic follow-up data",
            data=followup.to_csv(index=False),
            file_name="safeicu_agentic_followup.csv",
            mime="text/csv",
        )

    if final_summary is not None:
        st.subheader("Saved Evaluation Summary")
        st.dataframe(
            final_summary,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Final summary CSV not found.")


# --------------------------------------------------
# TAB 6: DATA & METHODOLOGY
# --------------------------------------------------

with methodology_tab:
    st.header("Data & Methodology")

    st.subheader("Project Files")

    expected_files = [
        "app.py",
        "requirements.txt",
        "safeicu_env.py",
        "safeicu_initial_comparison.csv",
        "safeicu_final_evaluation.csv",
        "safeicu_final_summary.csv",
        "safeicu_agentic_followup.csv",
        "safeicu_agentic_report.csv",
        "safeicu_dqn_safety_tests.csv",
        "safeicu_ppo_safety_tests.csv",
        "models/safeicu_dqn.zip",
        "models/safeicu_ppo.zip",
    ]

    for filename in expected_files:
        if file_exists(filename):
            st.write(f"✅ {filename}")
        else:
            st.write(f"⬜ {filename}")

    st.subheader("Current Evaluation Source")
    st.write(evaluation_source)
    st.write(f"Rows loaded: {len(initial)}")

    if final_evaluation is not None:
        st.success(
            "The independent evaluation CSV is loaded."
        )
    else:
        st.warning(
            "The independent evaluation CSV is missing. "
            "The dashboard is using the initial comparison instead."
        )

    st.subheader("Evaluation Methodology")
    st.markdown(
        """
        - Synthetic ICU resource-allocation environment.
        - Policies include DQN, DQN + Shield, PPO, PPO + Shield,
          and a Priority Baseline.
        - Scenarios: normal demand, surge demand, and resource shortage.
        - Evaluation records include reward, admissions, critical waiting
          burden, and shield interventions.
        - The independent evaluation uses new random seeds compared with
          the original comparison dataset.
        - Results are educational and do not establish clinical safety.
        """
    )

    st.subheader("Download Datasets")

    for filename, label in [
        ("safeicu_final_evaluation.csv", "Download independent evaluation"),
        ("safeicu_initial_comparison.csv", "Download initial comparison"),
        ("safeicu_final_summary.csv", "Download evaluation summary"),
        ("safeicu_agentic_followup.csv", "Download agentic follow-up"),
        ("safeicu_agentic_report.csv", "Download agentic report"),
        ("safeicu_dqn_safety_tests.csv", "Download DQN safety tests"),
        ("safeicu_ppo_safety_tests.csv", "Download PPO safety tests"),
    ]:
        path = DATA / filename
        if path.exists():
            st.download_button(
                label,
                data=path.read_bytes(),
                file_name=filename,
                mime="text/csv",
                key=f"download_{filename}",
            )

    st.subheader("Saved Models")

    for filename in [
        "models/safeicu_dqn.zip",
        "models/safeicu_ppo.zip",
    ]:
        path = DATA / filename
        if path.exists():
            st.write(f"✅ {filename} ({path.stat().st_size:,} bytes)")
        else:
            st.write(f"⬜ {filename}")

    st.warning(
        "This dashboard is a research/educational demonstration. "
        "It is not intended for real patient care."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()
st.caption(
    "SafeICU — Synthetic educational RL project. "
    "Results depend on the environment, model training, and evaluation seeds."
)
