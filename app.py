
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from stable_baselines3 import DQN, PPO
except Exception:
    DQN = PPO = None

try:
    from safeicu_env import SafeICURLEnv
except Exception:
    SafeICURLEnv = None

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

st.set_page_config(
    page_title="SafeICU | Safety-Aware RL Lab",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at 80% 0%, #132b42 0, #0a1020 38%);
    color: #edf4ff;
}
[data-testid="stSidebar"] {
    background: #0e1728;
    border-right: 1px solid #24354b;
}
.block-container {
    max-width: 1580px;
    padding-top: 1.25rem;
    padding-bottom: 3rem;
}
.hero {
    padding: 1.55rem 1.8rem;
    border-radius: 20px;
    background: linear-gradient(115deg,#14243b,#124149);
    border: 1px solid #2b6570;
    margin-bottom: 1rem;
}
.eyebrow {
    color: #79e5d4;
    font-size: .72rem;
    letter-spacing: .16em;
    font-weight: 800;
    text-transform: uppercase;
}
.hero h1 {
    color: #f5f8ff;
    margin: .25rem 0;
    font-size: 2.65rem;
}
.hero p { color: #c4d4e8; margin: 0; }
div[data-testid="stMetric"] {
    background: linear-gradient(145deg,#142138,#101a2b);
    border: 1px solid #2a3d57;
    border-radius: 15px;
    padding: .85rem 1rem;
}
div[data-testid="stMetricLabel"] { color: #aabbd0; }
div[data-testid="stMetricValue"] { color: #f5f8ff; }
.note {
    color: #9cacc2;
    font-size: .82rem;
    letter-spacing: .04em;
}
.room-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    background: #111e31;
    border: 1px solid #2a3d57;
    border-radius: 14px;
    margin: 8px 0 14px;
}
.room-label {
    font-weight: 800;
    letter-spacing: .08em;
    font-size: .82rem;
    color: #b9cadf;
}
.pill {
    display: inline-block;
    border-radius: 999px;
    padding: 4px 9px;
    font-size: .72rem;
    font-weight: 800;
    margin: 2px;
}
.pill-green {
    background: #123d35;
    color: #8cf0cf;
    border: 1px solid #286b5a;
}
.pill-blue {
    background: #142f4d;
    color: #9bd2ff;
    border: 1px solid #315a80;
}
.bed-card {
    min-height: 145px;
    border-radius: 14px;
    padding: 12px;
    margin: 4px 0 10px;
    border: 1px solid #30435a;
    background: linear-gradient(150deg,#17243a,#111b2b);
}
.bed-card.occupied {
    border-color: #28796f;
    background: linear-gradient(150deg,#123a3a,#142536);
}
.bed-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: #a9bbd0;
    font-size: .68rem;
    font-weight: 800;
    letter-spacing: .07em;
}
.bed-icon { font-size: 1.75rem; margin: 8px 0 4px; }
.bed-status { font-size: .8rem; font-weight: 800; color: #e7f2ff; }
.bed-meta { font-size: .72rem; color: #a8bad0; margin-top: 4px; }
.queue-card {
    border: 1px solid #2b3e56;
    border-radius: 14px;
    padding: 14px;
    background: #111d30;
}
.queue-count { font-size: 1.9rem; font-weight: 800; color: #f5f8ff; }
.queue-name { color: #b5c7dc; font-size: .78rem; }
.stButton > button { border-radius: 10px; }
hr { border-color: #27384e; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD FILES
# --------------------------------------------------

def load_csv(filename):
    path = DATA_DIR / filename
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# Prefer the new, independently evaluated results.
initial = load_csv("safeicu_final_evaluation.csv")
if initial is None or initial.empty:
    initial = load_csv("safeicu_initial_comparison.csv")
    evaluation_source = "Initial comparison (fallback)"
else:
    evaluation_source = "Independent evaluation"

final_summary = load_csv("safeicu_final_summary.csv")
followup = load_csv("safeicu_agentic_followup.csv")
agent_report = load_csv("safeicu_agentic_report.csv")
dqn_tests = load_csv("safeicu_dqn_safety_tests.csv")
ppo_tests = load_csv("safeicu_ppo_safety_tests.csv")

st.sidebar.markdown("## 🎛️ Experiment controls")
st.sidebar.caption("Filter stored evaluation results.")

if (
    initial is not None
    and not initial.empty
    and {"Scenario", "Policy"}.issubset(initial.columns)
):
    scenarios = ["All"] + sorted(
        initial["Scenario"].dropna().astype(str).unique().tolist()
    )
    policies = ["All"] + sorted(
        initial["Policy"].dropna().astype(str).unique().tolist()
    )
    scenario_filter = st.sidebar.selectbox("Demand scenario", scenarios)
    policy_filter = st.sidebar.selectbox("Policy", policies)
else:
    scenario_filter, policy_filter = "All", "All"

st.sidebar.divider()
st.sidebar.caption(f"Data source: {evaluation_source}")
st.sidebar.caption("Synthetic educational simulation only.")
st.sidebar.caption("Not a clinical decision-support system.")

st.markdown("""
<div class="hero">
  <div class="eyebrow">Safety-aware reinforcement learning lab</div>
  <h1>🏥 SafeICU</h1>
  <p>Explore synthetic ICU resource allocation, compare stored RL evaluations,
  and run an interactive 24-hour demonstration.</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<span class="note">SYNTHETIC DATA · RESEARCH PROTOTYPE · '
    'NOT FOR REAL CLINICAL USE</span>',
    unsafe_allow_html=True,
)

if initial is None or initial.empty:
    st.error(
        "No evaluation CSV found. Upload safeicu_initial_comparison.csv "
        "or safeicu_final_evaluation.csv beside app.py."
    )
    st.stop()

required = {
    "Policy", "Scenario", "Reward", "Admissions",
    "Critical Waiting Burden", "Shield Interventions"
}
missing = required - set(initial.columns)

if missing:
    st.error("Evaluation CSV missing columns: " + ", ".join(sorted(missing)))
    st.stop()

filtered = initial.copy()

if scenario_filter != "All":
    filtered = filtered[
        filtered["Scenario"].astype(str) == scenario_filter
    ]

if policy_filter != "All":
    filtered = filtered[
        filtered["Policy"].astype(str) == policy_filter
    ]

metric_reward = (
    "Reward" if "Reward" in filtered.columns
    else "Mean_Reward" if "Mean_Reward" in filtered.columns
    else None
)
metric_wait = next(
    (
        c for c in [
            "Critical Waiting Patient-Hours",
            "Critical Waiting Burden",
            "Mean_Critical_Waiting_Burden",
        ]
        if c in filtered.columns
    ),
    None,
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Evaluation episodes", f"{len(filtered):,}")
k2.metric("Policies in view", int(filtered["Policy"].nunique()))
k3.metric("Scenarios in view", int(filtered["Scenario"].nunique()))
k4.metric(
    "Evaluation seeds",
    int(filtered["Seed"].nunique()) if "Seed" in filtered.columns
    else "Not recorded",
)

tabs = st.tabs([
    "🏠 Overview",
    "📊 Policy laboratory",
    "🛡️ Safety audit",
    "🧪 Live simulation",
    "🤖 Agentic supervisor",
    "📁 Data & methodology",
])


# --------------------------------------------------
# TAB 1: OVERVIEW
# --------------------------------------------------

with tabs[0]:
    st.subheader("Experiment overview")

    left, right = st.columns(2)

    with left:
        st.markdown("#### Mean reward by policy")
        if metric_reward:
            grouped = filtered.groupby(
                "Policy", as_index=False
            )[metric_reward].mean()

            fig, ax = plt.subplots(figsize=(7, 3.7))
            ax.bar(grouped["Policy"], grouped[metric_reward])
            ax.set_ylabel("Mean cumulative reward")
            ax.tick_params(axis="x", rotation=25)
            ax.grid(axis="y", alpha=.2)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("Reward metric unavailable.")

    with right:
        st.markdown("#### Critical-patient waiting burden")
        if metric_wait:
            grouped = filtered.groupby(
                "Policy", as_index=False
            )[metric_wait].mean()

            fig, ax = plt.subplots(figsize=(7, 3.7))
            ax.bar(grouped["Policy"], grouped[metric_wait])
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

    st.download_button(
        "⬇️ Download filtered evaluation",
        filtered.to_csv(index=False),
        "safeicu_filtered_evaluation.csv",
        "text/csv",
    )


# --------------------------------------------------
# TAB 2: POLICY LABORATORY
# --------------------------------------------------

with tabs[1]:
    st.subheader("Policy laboratory")
    st.write(
        "Explore stored evaluation metrics. This tab does not retrain models."
    )

    numeric_cols = [
        c for c in filtered.select_dtypes(include="number").columns
        if c != "Seed"
    ]

    if numeric_cols:
        default = "Reward" if "Reward" in numeric_cols else numeric_cols[0]
        chosen_metric = st.selectbox(
            "Metric to compare",
            numeric_cols,
            index=numeric_cols.index(default),
            key="lab_metric",
        )

        grouped = filtered.groupby("Policy")[chosen_metric].agg(
            ["mean", "std"]
        ).reset_index()
        grouped = grouped.rename(
            columns={"mean": "Mean", "std": "StdDev"}
        )

        st.dataframe(
            grouped.round(3),
            use_container_width=True,
            hide_index=True,
        )

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


# --------------------------------------------------
# TAB 3: SAFETY AUDIT
# --------------------------------------------------

with tabs[2]:
    st.subheader("Safety audit")
    st.write(
        "Shield interventions and executed violations are different. "
        "A blocked proposal is not an executed violation."
    )

    viol_col = next(
        (
            c for c in [
                "Executed Violations",
                "constraint_violations",
                "Constraint Violations",
                "Reported Violations",
            ]
            if c in filtered.columns
        ),
        None,
    )
    inter_col = next(
        (
            c for c in [
                "Shield Interventions",
                "Mean_Shield_Interventions",
                "shield_interventions",
            ]
            if c in filtered.columns
        ),
        None,
    )

    a, b = st.columns(2)
    a.metric(
        "Mean reported violations",
        f"{filtered[viol_col].mean():.2f}"
        if viol_col else "Not recorded",
    )
    b.metric(
        "Mean shield interventions",
        f"{filtered[inter_col].mean():.2f}"
        if inter_col else "Not recorded",
    )

    if dqn_tests is not None:
        st.markdown("#### DQN regression tests")
        st.dataframe(dqn_tests, use_container_width=True, hide_index=True)
    else:
        st.info("DQN safety-test file is missing.")

    if ppo_tests is not None:
        st.markdown("#### PPO regression tests")
        st.dataframe(ppo_tests, use_container_width=True, hide_index=True)
    else:
        st.info("PPO safety-test file is missing.")

    check_col = "Resource Limit Check Passed"
    if check_col in filtered.columns:
        checks = filtered[check_col].fillna(False).astype(bool)
        st.metric(
            "Recorded resource-limit checks passed",
            f"{int(checks.sum())}/{len(checks)}",
        )

    if not viol_col:
        st.warning(
            "Executed-violation counts are absent. Missing data must not "
            "be interpreted as proof of zero violations."
        )
    else:
        st.warning(
            "Interpret reported violation counts with the environment code; "
            "the current environment's violation counter may not capture "
            "every possible unsafe condition."
        )


# --------------------------------------------------
# TAB 4: LIVE SIMULATION
# --------------------------------------------------

with tabs[3]:
    st.subheader("Live ICU Digital Twin")
    st.markdown(
        "Run a step-by-step synthetic ICU episode. Select a policy, "
        "scenario and seed, then advance the simulation. This is an "
        "educational prototype, not a clinical system."
    )

    model_dir = DATA_DIR / "models"
    dqn_path = model_dir / "safeicu_dqn.zip"
    ppo_path = model_dir / "safeicu_ppo.zip"

    dqn_available = dqn_path.exists() and DQN is not None
    ppo_available = ppo_path.exists() and PPO is not None
    env_available = SafeICURLEnv is not None

    a, b, c = st.columns(3)
    a.metric("DQN checkpoint", "READY" if dqn_available else "NOT FOUND")
    b.metric("PPO checkpoint", "READY" if ppo_available else "NOT FOUND")
    c.metric("Environment", "READY" if env_available else "NOT FOUND")

    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 1.5])
    live_scenario = c1.selectbox(
        "Demand scenario",
        ["normal", "surge", "shortage"],
        key="live_scenario",
    )
    live_policy = c2.selectbox(
        "Live policy",
        ["DQN", "PPO", "Priority baseline"],
        key="live_policy",
    )
    horizon = c3.selectbox(
        "Episode length", [12, 24, 48], index=1, key="live_horizon"
    )
    seed = c4.number_input(
        "Random seed", min_value=1, max_value=999999,
        value=42, step=1, key="live_seed",
    )

    config = (live_policy, live_scenario, int(horizon), int(seed))
    model_ready = {
        "DQN": dqn_available,
        "PPO": ppo_available,
        "Priority baseline": True,
    }[live_policy]

    def baseline_action(environment):
        # Actions: 0=wait, 1=critical, 2=high, 3=moderate.
        for category, action in enumerate((1, 2, 3)):
            if environment.waiting[category] > 0:
                return action
        return 0

    def new_live_state(policy_name, scenario_name, episode_horizon, episode_seed):
        environment = SafeICURLEnv(
            scenario=scenario_name,
            safety_enabled=True,
            horizon=int(episode_horizon),
        )
        observation, _ = environment.reset(seed=int(episode_seed))

        model = None
        if policy_name == "DQN":
            model = DQN.load(str(dqn_path), device="cpu")
        elif policy_name == "PPO":
            model = PPO.load(str(ppo_path), device="cpu")

        return {
            "config": (
                policy_name, scenario_name,
                int(episode_horizon), int(episode_seed)
            ),
            "env": environment,
            "model": model,
            "obs": observation,
            "records": [],
            "done": False,
            "policy": policy_name,
            "scenario": scenario_name,
            "horizon": int(episode_horizon),
            "seed": int(episode_seed),
        }

    def advance_one_hour(state):
        environment = state["env"]
        if state["done"]:
            return

        if state["model"] is None:
            proposed = baseline_action(environment)
        else:
            action_value, _ = state["model"].predict(
                state["obs"], deterministic=True
            )
            proposed = int(np.asarray(action_value).item())

        observation, reward, terminated, truncated, info = (
            environment.step(proposed)
        )
        state["obs"] = observation

        used = len(environment.active)
        cap_beds, cap_monitors, cap_staff = (
            environment.capacity[state["scenario"]]
        )

        state["records"].append({
            "Hour": int(environment.hour),
            "Critical waiting": int(environment.waiting[0]),
            "High-priority waiting": int(environment.waiting[1]),
            "Moderate-priority waiting": int(environment.waiting[2]),
            "Occupied beds": used,
            "Available beds": max(cap_beds - used, 0),
            "Bed utilization": used / cap_beds,
            "Monitor utilization": used / cap_monitors,
            "Staff utilization": used / cap_staff,
            "Proposed action": int(info["proposed_action"]),
            "Executed action": int(info["executed_action"]),
            "Shield intervention": bool(info["shield_intervention"]),
            "Reward this hour": float(reward),
            "Cumulative reward": float(info["total_reward"]),
            "Admissions total": int(info["admissions"]),
            "Critical waiting burden": int(info["critical_wait_total"]),
            "Executed violations": int(info.get("executed_violations", 0)),
            "Active durations": ", ".join(map(str, environment.active)),
        })

        state["done"] = bool(
            terminated or truncated
            or environment.hour >= environment.horizon
        )

    b1, b2, b3, b4, b5 = st.columns(5)
    start_clicked = b1.button(
        "↻ Start / reset", type="primary", key="live_start",
        disabled=not (env_available and model_ready),
    )
    step_clicked = b2.button(
        "▶ Advance 1 hour", key="live_step",
        disabled=not (env_available and model_ready),
    )
    batch_clicked = b3.button(
        "⏩ Advance 6 hours", key="live_batch",
        disabled=not (env_available and model_ready),
    )
    finish_clicked = b4.button(
        "⏭ Run to completion", key="live_finish",
        disabled=not (env_available and model_ready),
    )
    if b5.button("⏹ Clear episode", key="live_clear"):
        st.session_state.pop("safeicu_live_state", None)
        st.session_state["safeicu_autoplay"] = False
        st.rerun()

    if not env_available:
        st.error(
            "Cannot import SafeICURLEnv. Ensure safeicu_env.py is "
            "beside app.py."
        )

    if live_policy in ("DQN", "PPO") and not model_ready:
        st.error(
            f"{live_policy} is unavailable. Check the model ZIP and "
            "stable-baselines3 installation."
        )

    if start_clicked:
        try:
            st.session_state["safeicu_live_state"] = new_live_state(
                live_policy, live_scenario, int(horizon), int(seed)
            )
            st.session_state["safeicu_autoplay"] = False
        except Exception as exc:
            st.error("Could not start the simulation.")
            st.exception(exc)

    state = st.session_state.get("safeicu_live_state")

    if state and state.get("config") != config:
        st.warning(
            "Your controls differ from the current episode. "
            "Click Start / reset to apply the new selections."
        )

    if state and state.get("config") == config:
        if step_clicked or batch_clicked or finish_clicked:
            st.session_state["safeicu_autoplay"] = False
            try:
                count = (
                    1 if step_clicked else
                    6 if batch_clicked else
                    int(state["horizon"])
                )
                for _ in range(count):
                    if state["done"]:
                        break
                    advance_one_hour(state)
                st.session_state["safeicu_live_state"] = state
            except Exception as exc:
                st.error("Simulation stopped due to an environment error.")
                st.exception(exc)

    # Autoplay controls: use streamlit-autorefresh if installed.
    state = st.session_state.get("safeicu_live_state")
    if state and state.get("config") == config:
        play_col, pause_col = st.columns(2)

        if play_col.button(
            "▶ Play autoplay", key="live_play",
            disabled=state["done"] or st_autorefresh is None,
        ):
            st.session_state["safeicu_autoplay"] = True
            st.rerun()

        if pause_col.button("⏸ Pause", key="live_pause"):
            st.session_state["safeicu_autoplay"] = False
            st.rerun()

        if st_autorefresh is None:
            st.caption(
                "Autoplay requires streamlit-autorefresh in requirements.txt. "
                "Manual step controls are still available."
            )

        if st.session_state.get("safeicu_autoplay") and not state["done"]:
            st_autorefresh(interval=1200, limit=None, key="safeicu_auto_refresh")
            try:
                advance_one_hour(state)
                st.session_state["safeicu_live_state"] = state
            except Exception as exc:
                st.session_state["safeicu_autoplay"] = False
                st.error("Autoplay stopped due to an error.")
                st.exception(exc)

    state = st.session_state.get("safeicu_live_state")

    if state and state.get("config") == config:
        env_live = state["env"]
        records = pd.DataFrame(state["records"])
        cap_beds, cap_monitors, cap_staff = (
            env_live.capacity[state["scenario"]]
        )
        occupied = len(env_live.active)
        hour = env_live.hour
        done = state["done"]
        status = "COMPLETE" if done else "LIVE EPISODE"
        pill_class = "pill-green" if done else "pill-blue"

        if done:
            st.session_state["safeicu_autoplay"] = False

        st.markdown(
            f"""
            <div class="room-head">
              <div>
                <div class="room-label">SAFEICU · RESOURCE FLOOR</div>
                <div style="font-size:1.2rem;font-weight:800;margin-top:5px;">
                    Synthetic intensive care unit
                </div>
              </div>
              <div style="text-align:right;">
                <span class="pill {pill_class}">{status}</span>
                <div style="color:#aabbd0;font-size:.76rem;margin-top:7px;">
                    Hour {hour} / {state['horizon']} ·
                    {state['scenario'].upper()} · {state['policy']}
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Simulation hour", f"{hour}/{state['horizon']}")
        m2.metric("Beds occupied", f"{occupied}/{cap_beds}")
        m3.metric("Admissions", int(env_live.admissions))
        m4.metric("Cumulative reward", f"{env_live.total_reward:.1f}")
        m5.metric("Shield interventions", int(env_live.interventions))

        st.progress(
            min(hour / max(state["horizon"], 1), 1.0),
            text=f"Episode progress · {hour} of {state['horizon']} hours",
        )

        st.markdown("#### ICU bed bay")
        st.caption(
            "Each occupied card is a synthetic active care slot. "
            "No real patient data or clinical vital signs are used."
        )

        durations = list(env_live.active)
        bed_columns = st.columns(5)

        for index in range(cap_beds):
            if index < occupied:
                remaining = (
                    durations[index] if index < len(durations) else "—"
                )
                card = f"""
                <div class="bed-card occupied">
                    <div class="bed-top">
                        <span>BED {index+1:02d}</span>
                        <span class="pill pill-green">OCCUPIED</span>
                    </div>
                    <div class="bed-icon">🛏️ 🧑</div>
                    <div class="bed-status">Care slot active</div>
                    <div class="bed-meta">Monitor assigned · {remaining} h remaining</div>
                </div>
                """
            else:
                card = f"""
                <div class="bed-card">
                    <div class="bed-top">
                        <span>BED {index+1:02d}</span>
                        <span class="pill pill-blue">AVAILABLE</span>
                    </div>
                    <div class="bed-icon">🛏️</div>
                    <div class="bed-status">Ready for allocation</div>
                    <div class="bed-meta">Monitor available · Staff capacity permitting</div>
                </div>
                """
            with bed_columns[index % 5]:
                st.markdown(card, unsafe_allow_html=True)

        st.markdown("#### Patient queues")
        q1, q2, q3 = st.columns(3)
        q1.markdown(
            f'<div class="queue-card"><div class="queue-name">🔴 CRITICAL PRIORITY</div>'
            f'<div class="queue-count">{env_live.waiting[0]}</div>'
            f'<div class="queue-name">Synthetic patients waiting</div></div>',
            unsafe_allow_html=True,
        )
        q2.markdown(
            f'<div class="queue-card"><div class="queue-name">🟠 HIGH PRIORITY</div>'
            f'<div class="queue-count">{env_live.waiting[1]}</div>'
            f'<div class="queue-name">Synthetic patients waiting</div></div>',
            unsafe_allow_html=True,
        )
        q3.markdown(
            f'<div class="queue-card"><div class="queue-name">🔵 MODERATE PRIORITY</div>'
            f'<div class="queue-count">{env_live.waiting[2]}</div>'
            f'<div class="queue-name">Synthetic patients waiting</div></div>',
            unsafe_allow_html=True,
        )

        u1, u2, u3 = st.columns(3)
        u1.metric("Bed utilization", f"{occupied/cap_beds:.0%}")
        u2.metric("Monitor utilization", f"{occupied/cap_monitors:.0%}")
        u3.metric("Staff utilization", f"{occupied/cap_staff:.0%}")

        if not records.empty:
            left, right = st.columns(2)

            with left:
                st.markdown("#### Waiting queue trajectory")
                fig, ax = plt.subplots(figsize=(7, 3.5))
                for col in [
                    "Critical waiting",
                    "High-priority waiting",
                    "Moderate-priority waiting",
                ]:
                    ax.plot(
                        records["Hour"], records[col],
                        marker="o", markersize=3, label=col,
                    )
                ax.set_xlabel("Simulated hour")
                ax.set_ylabel("Patients waiting")
                ax.grid(alpha=.2)
                ax.legend(fontsize=8)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            with right:
                st.markdown("#### Resource utilization")
                fig, ax = plt.subplots(figsize=(7, 3.5))
                for col, label in [
                    ("Bed utilization", "Beds"),
                    ("Monitor utilization", "Monitors"),
                    ("Staff utilization", "Staff"),
                ]:
                    ax.plot(
                        records["Hour"], records[col],
                        marker="o", markersize=3, label=label,
                    )
                ax.set_ylim(0, 1.05)
                ax.set_xlabel("Simulated hour")
                ax.set_ylabel("Utilization")
                ax.grid(alpha=.2)
                ax.legend(fontsize=8)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            latest = records.iloc[-1]
            action_names = {
                0: "No admission",
                1: "Admit critical",
                2: "Admit high-priority",
                3: "Admit moderate-priority",
            }
            proposed = action_names.get(
                int(latest["Proposed action"]), "Unknown"
            )
            executed = action_names.get(
                int(latest["Executed action"]), "Unknown"
            )

            st.markdown("#### Latest agent decision")
            d1, d2, d3 = st.columns(3)
            d1.info(f"**Proposed action**\n\n{proposed}")
            if bool(latest["Shield intervention"]):
                d2.warning(f"**Executed action**\n\n{executed}")
            else:
                d2.info(f"**Executed action**\n\n{executed}")
            d3.metric(
                "Shield intervened this hour",
                "Yes" if bool(latest["Shield intervention"]) else "No",
            )

            if bool(latest["Shield intervention"]):
                st.warning(
                    "The safety shield modified the proposed action "
                    "under the environment's critical-priority rule."
                )

            with st.expander("View full episode event log", expanded=False):
                log_view = records.copy()
                log_view["Proposed action"] = log_view[
                    "Proposed action"
                ].map(action_names)
                log_view["Executed action"] = log_view[
                    "Executed action"
                ].map(action_names)
                st.dataframe(
                    log_view, use_container_width=True, hide_index=True
                )

            st.download_button(
                "⬇ Download episode log",
                records.to_csv(index=False),
                "safeicu_live_episode.csv",
                "text/csv",
            )

        if done:
            st.success(
                "Episode complete. Change the scenario or seed and "
                "start a new episode to compare behaviour."
            )
    else:
        st.info(
            "Click Start / reset, then advance one hour or six hours "
            "to watch the ICU room update. You can also run the whole episode."
        )


# --------------------------------------------------
# TAB 5: AGENTIC SUPERVISOR
# --------------------------------------------------

with tabs[4]:
    st.subheader("Agentic AI experiment supervisor")
    st.write(
        "The notebook supervisor analyzed saved results and generated "
        "a bounded follow-up experiment. This deployed dashboard displays "
        "those saved outputs; it does not train agents on the server."
    )

    if agent_report is not None:
        st.markdown("#### Follow-up summary")
        st.dataframe(
            agent_report, use_container_width=True, hide_index=True
        )
    else:
        st.info("safeicu_agentic_report.csv is missing.")

    if followup is not None:
        st.markdown("#### Follow-up experiment records")
        st.dataframe(followup, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download follow-up data",
            followup.to_csv(index=False),
            "safeicu_agentic_followup.csv",
            "text/csv",
        )

    if final_summary is not None:
        st.markdown("#### Evaluation summary")
        st.dataframe(
            final_summary, use_container_width=True, hide_index=True
        )


# --------------------------------------------------
# TAB 6: DATA & METHODOLOGY
# --------------------------------------------------

with tabs[5]:
    st.subheader("Data, assumptions & methodology")

    st.markdown("""
**Environment:** Synthetic hourly ICU simulation with discrete priority
categories and constrained resources.

**Policies:** Priority baseline, DQN, PPO, and shielded variants.

**Scenarios:** Normal demand, surge demand, and reduced resource capacity.

**Live simulation:** Runs the selected saved DQN/PPO model or priority
baseline in the synthetic environment when model files are available.

**Limitations:** This simulator is not medically validated. Results describe
only the code, reward function, and synthetic assumptions used in this project.
""")

    st.markdown("#### Data files in repository")

    expected_files = [
        "safeicu_initial_comparison.csv",
        "safeicu_final_evaluation.csv",
        "safeicu_final_summary.csv",
        "safeicu_agentic_followup.csv",
        "safeicu_agentic_report.csv",
        "safeicu_dqn_safety_tests.csv",
        "safeicu_ppo_safety_tests.csv",
        "safeicu_env.py",
        "models/safeicu_dqn.zip",
        "models/safeicu_ppo.zip",
    ]

    for name in expected_files:
        exists = (DATA_DIR / name).exists()
        st.write(("✅ " if exists else "— ") + name)

    st.markdown(f"**Evaluation data currently used:** {evaluation_source}")

    st.download_button(
        "⬇️ Download filtered data",
        filtered.to_csv(index=False),
        "safeicu_evaluation_export.csv",
        "text/csv",
    )

    for filename, label in [
        ("safeicu_final_evaluation.csv", "Download independent evaluation"),
        ("safeicu_initial_comparison.csv", "Download initial comparison"),
        ("safeicu_final_summary.csv", "Download final summary"),
        ("safeicu_agentic_report.csv", "Download agentic report"),
        ("safeicu_dqn_safety_tests.csv", "Download DQN safety tests"),
        ("safeicu_ppo_safety_tests.csv", "Download PPO safety tests"),
    ]:
        path = DATA_DIR / filename
        if path.exists():
            st.download_button(
                label,
                data=path.read_bytes(),
                file_name=filename,
                mime="text/csv",
                key=f"download_{filename}",
            )

st.divider()
st.caption(
    "SafeICU research prototype · Synthetic data only · "
    "Not for real clinical use."
)
