import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ml_model import train_model
from game_theory import analyze_game


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# DARK THEME
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #05080b;
        color: #f1f5f9;
    }

    section[data-testid="stSidebar"] {
        background-color: #080d12;
        border-right: 1px solid #1b2832;
    }

    section[data-testid="stSidebar"] * {
        color: #d8e3ea;
    }

    h1 {
        color: #f1f5f9 !important;
        font-weight: 800 !important;
    }

    h2 {
        color: #00e6a0 !important;
        font-weight: 700 !important;
    }

    h3 {
        color: #dce8ef !important;
    }

    p {
        color: #9eafb9;
    }

    div[data-testid="stMetric"] {
        background-color: #0b1117;
        border: 1px solid #1b2a35;
        border-radius: 12px;
        padding: 12px;
    }

    div[data-testid="stMetricLabel"] {
        color: #8497a3 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #00e6a0 !important;
        font-weight: 800 !important;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #1b2a35;
        border-radius: 10px;
    }

    .stButton > button {
        background-color: #0d171e;
        color: #00e6a0;
        border: 1px solid #00e6a0;
        border-radius: 8px;
    }

    .stButton > button:hover {
        background-color: #00e6a0;
        color: #05080b;
    }

    hr {
        border-color: #1b2832 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CHART SIZES
# ============================================================

# Normal charts
SMALL_CHART = (4.0, 2.2)

# ML performance graph specifically
# This is the SMALL size you asked for.
ML_PERFORMANCE_CHART = (2.4, 1.35)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return train_model()


with st.spinner("🔄 Loading AI intrusion detection model..."):
    results = load_model()


# ============================================================
# MODEL RESULTS
# ============================================================

accuracy = results["accuracy"]
report = results["report"]
attack_distribution = results["attack_distribution"]
data = results["data"]


# ============================================================
# DATA COUNTS
# ============================================================

normal_count = 0
attack_count = 0

for attack_type, count in attack_distribution.items():

    if str(attack_type).lower() == "normal":
        normal_count += count
    else:
        attack_count += count


total_records = len(data)


# ============================================================
# DEFAULT GAME
# ============================================================

default_game = analyze_game(
    0.85,
    0.05,
    0.30
)

default_ids_strategy = default_game["ids_strategy"]
default_attacker_strategy = default_game["attacker_strategy"]
default_game_value = default_game["game_value"]


# ============================================================
# CHART STYLE
# ============================================================

def style_chart(ax):

    ax.set_facecolor("#080d12")

    ax.tick_params(
        colors="#8ea2af",
        labelsize=7
    )

    ax.xaxis.label.set_color("#8ea2af")
    ax.yaxis.label.set_color("#8ea2af")

    ax.title.set_color("#eaf2f7")

    for spine in ax.spines.values():
        spine.set_color("#1e2b34")

    ax.grid(
        axis="y",
        color="#1e2b34",
        alpha=0.4,
        linewidth=0.5
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🛡️ IDS Dashboard")

    st.caption(
        "AI + Game Theory Network Security"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🤖 ML Analysis",
            "🎮 Game Theory",
            "🎯 Strategy Analysis",
            "📊 Dataset Explorer",
            "ℹ️ About"
        ]
    )

    st.divider()

    st.subheader("Technology Stack")

    st.write("🐍 Python")
    st.write("⚡ Streamlit")
    st.write("🌲 Random Forest")
    st.write("📊 Pandas / NumPy")
    st.write("🧮 SciPy")
    st.write("🎮 Game Theory")

    st.divider()

    st.success("🟢 SYSTEM OPERATIONAL")


# ============================================================
# 1. DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title(
        "🛡️ AI-Based Network Intrusion Detection"
    )

    st.subheader(
        "Machine Learning + Game Theory + Mixed Strategies"
    )

    st.write(
        """
        A cybersecurity analytics system that combines
        Machine Learning and Game Theory to detect malicious
        network traffic and model attacker–IDS interactions.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # SYSTEM METRICS
    # --------------------------------------------------------

    st.subheader("📌 System Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "🎯 Model Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    with col2:

        st.metric(
            "📁 Dataset Records",
            f"{total_records:,}"
        )

    with col3:

        st.metric(
            "🚨 Attack Records",
            f"{attack_count:,}"
        )

    with col4:

        st.metric(
            "🧮 Game Value",
            f"{default_game_value:.3f}"
        )

    st.divider()

    # --------------------------------------------------------
    # TRAFFIC CLASSIFICATION
    # --------------------------------------------------------

    st.subheader("📊 Traffic Classification")

    st.caption(
        "Normal versus malicious network traffic"
    )

    col1, col2 = st.columns(2)

    # NORMAL VS ATTACK
    with col1:

        fig, ax = plt.subplots(
            figsize=SMALL_CHART
        )

        labels = [
            "Normal",
            "Attack"
        ]

        values = [
            normal_count,
            attack_count
        ]

        bars = ax.bar(
            labels,
            values,
            width=0.45,
            color=[
                "#00e6a0",
                "#ff5267"
            ]
        )

        ax.set_ylabel(
            "Records",
            fontsize=7
        )

        ax.set_title(
            "Traffic Classification",
            fontsize=10
        )

        style_chart(ax)

        for bar, value in zip(
            bars,
            values
        ):

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                bar.get_height(),
                f"{value:,}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#dce8ef"
            )

        plt.tight_layout()

        st.pyplot(
            fig,
            clear_figure=True
        )

    # ATTACK DISTRIBUTION
    with col2:

        fig, ax = plt.subplots(
            figsize=SMALL_CHART
        )

        distribution_df = pd.DataFrame(
            list(attack_distribution.items()),
            columns=[
                "Attack Type",
                "Count"
            ]
        )

        ax.bar(
            distribution_df["Attack Type"],
            distribution_df["Count"],
            color="#00bfff",
            width=0.55
        )

        ax.set_title(
            "🚨 Attack Distribution",
            fontsize=10
        )

        ax.set_ylabel(
            "Records",
            fontsize=7
        )

        style_chart(ax)

        ax.tick_params(
            axis="x",
            rotation=25,
            labelsize=6
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            clear_figure=True
        )

    st.divider()

    # --------------------------------------------------------
    # STRATEGIC OVERVIEW
    # --------------------------------------------------------

    st.subheader("🎮 Strategic Overview")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("### 🛡️ IDS Strategy")

        st.caption(
            "Mixed strategy probabilities"
        )

        st.metric(
            "Low Detection",
            f"{default_ids_strategy[0] * 100:.1f}%"
        )

        st.metric(
            "High Detection",
            f"{default_ids_strategy[1] * 100:.1f}%"
        )

    with col2:

        st.markdown("### ⚔️ Attacker Strategy")

        st.caption(
            "Mixed strategy probabilities"
        )

        st.metric(
            "Stealth Attack",
            f"{default_attacker_strategy[0] * 100:.1f}%"
        )

        st.metric(
            "Aggressive Attack",
            f"{default_attacker_strategy[1] * 100:.1f}%"
        )

    with col3:

        st.markdown("### 🧮 Game Equilibrium")

        st.caption(
            "Zero-sum game solution"
        )

        st.metric(
            "Game Value",
            f"{default_game_value:.3f}"
        )

        st.caption(
            "Expected payoff from IDS perspective"
        )

    st.divider()

    st.success(
        "🟢 AI model and game-theoretic analysis are operational."
    )


# ============================================================
# 2. ML ANALYSIS
# ============================================================

elif page == "🤖 ML Analysis":

    st.title(
        "🤖 Machine Learning Analysis"
    )

    st.write(
        """
        The Random Forest classifier analyses network traffic
        and classifies observations as Normal or Attack.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Random Forest Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Dataset Size",
            f"{total_records:,}"
        )

    with col3:

        st.metric(
            "Attack Records",
            f"{attack_count:,}"
        )

    st.divider()

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    st.subheader(
        "📋 Classification Report"
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # SMALL MODEL PERFORMANCE GRAPH
    # --------------------------------------------------------

    st.subheader(
        "📈 Model Performance"
    )

    st.caption(
        "Precision, Recall and F1-score"
    )

    class_names = [
        name
        for name in [
            "Normal",
            "Attack"
        ]
        if name in report_df.index
    ]

    if len(class_names) > 0:

        metric_df = report_df.loc[
            class_names,
            [
                "precision",
                "recall",
                "f1-score"
            ]
        ]

        # ----------------------------------------------
        # CENTER THE SMALL GRAPH
        # ----------------------------------------------

        left, center, right = st.columns(
            [1.3, 1, 1.3]
        )

        with center:

            fig, ax = plt.subplots(
                figsize=ML_PERFORMANCE_CHART,
                dpi=90
            )

            x = np.arange(
                len(metric_df)
            )

            width = 0.16

            # Precision
            ax.bar(
                x - width,
                metric_df["precision"],
                width,
                label="Precision",
                color="#00bfff"
            )

            # Recall
            ax.bar(
                x,
                metric_df["recall"],
                width,
                label="Recall",
                color="#00e6a0"
            )

            # F1
            ax.bar(
                x + width,
                metric_df["f1-score"],
                width,
                label="F1",
                color="#ffbd5c"
            )

            ax.set_xticks(x)

            ax.set_xticklabels(
                metric_df.index,
                fontsize=5,
                color="#8ea2af"
            )

            ax.set_ylim(
                0,
                1.05
            )

            ax.set_ylabel(
                "Score",
                fontsize=5,
                color="#8ea2af"
            )

            ax.set_title(
                "Model Performance",
                fontsize=7,
                color="#eaf2f7",
                pad=3
            )

            ax.tick_params(
                axis="y",
                labelsize=5,
                colors="#718694"
            )

            ax.tick_params(
                axis="x",
                labelsize=5
            )

            ax.grid(
                axis="y",
                color="#1e2b34",
                alpha=0.4,
                linewidth=0.4
            )

            for spine in ax.spines.values():
                spine.set_color("#1e2b34")

            ax.legend(
                fontsize=4.5,
                loc="lower right",
                frameon=True,
                facecolor="#0b1117",
                edgecolor="#26343e",
                labelcolor="#dce8ef"
            )

            plt.tight_layout(
                pad=0.3
            )

            st.pyplot(
                fig,
                clear_figure=True,
                use_container_width=False
            )


# ============================================================
# 3. GAME THEORY
# ============================================================

elif page == "🎮 Game Theory":

    st.title(
        "🎮 Game-Theoretic Analysis"
    )

    st.write(
        """
        The attacker and IDS are represented as two players
        in a zero-sum game. The payoff matrix represents the
        IDS payoff for different combinations of strategies.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # PARAMETERS
    # --------------------------------------------------------

    st.subheader(
        "⚙️ Game Parameters"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        detection_rate = st.slider(
            "IDS Detection Rate",
            min_value=0.50,
            max_value=0.99,
            value=0.85,
            step=0.01,
            key="game_detection"
        )

    with col2:

        false_positive_rate = st.slider(
            "False Positive Rate",
            min_value=0.01,
            max_value=0.30,
            value=0.05,
            step=0.01,
            key="game_false_positive"
        )

    with col3:

        attack_success_rate = st.slider(
            "Attack Success Rate",
            min_value=0.05,
            max_value=0.80,
            value=0.30,
            step=0.01,
            key="game_attack_success"
        )

    # --------------------------------------------------------
    # CALCULATE
    # --------------------------------------------------------

    game = analyze_game(
        detection_rate,
        false_positive_rate,
        attack_success_rate
    )

    payoff_matrix = game[
        "payoff_matrix"
    ]

    ids_strategy = game[
        "ids_strategy"
    ]

    attacker_strategy = game[
        "attacker_strategy"
    ]

    game_value = game[
        "game_value"
    ]

    st.divider()

    # --------------------------------------------------------
    # PAYOFF MATRIX
    # --------------------------------------------------------

    st.subheader(
        "💰 Payoff Matrix"
    )

    payoff_df = pd.DataFrame(
        payoff_matrix,
        index=[
            "Stealth Attack",
            "Aggressive Attack"
        ],
        columns=[
            "Low Detection",
            "High Detection"
        ]
    )

    col1, col2 = st.columns(2)

    with col1:

        st.dataframe(
            payoff_df.round(3),
            use_container_width=True
        )

    with col2:

        fig, ax = plt.subplots(
            figsize=SMALL_CHART
        )

        sns.heatmap(
            payoff_df,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            center=0,
            cbar=False,
            linewidths=1,
            ax=ax,
            annot_kws={
                "fontsize": 8
            }
        )

        ax.set_xlabel(
            "IDS Strategy",
            fontsize=7
        )

        ax.set_ylabel(
            "Attacker Strategy",
            fontsize=7
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            clear_figure=True
        )

    st.divider()

    # --------------------------------------------------------
    # MIXED STRATEGIES
    # --------------------------------------------------------

    st.subheader(
        "🎲 Optimal Mixed Strategies"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "IDS: Low Detection",
            f"{ids_strategy[0] * 100:.2f}%"
        )

        st.metric(
            "IDS: High Detection",
            f"{ids_strategy[1] * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Attacker: Stealth",
            f"{attacker_strategy[0] * 100:.2f}%"
        )

        st.metric(
            "Attacker: Aggressive",
            f"{attacker_strategy[1] * 100:.2f}%"
        )

    with col3:

        st.metric(
            "Game Value",
            f"{game_value:.3f}"
        )


# ============================================================
# 4. STRATEGY ANALYSIS
# ============================================================

elif page == "🎯 Strategy Analysis":

    st.title(
        "🎯 Strategy Analysis"
    )

    st.write(
        """
        The following charts show the probability assigned
        to each IDS and attacker strategy by the game-theoretic
        solution.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # PARAMETERS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        detection_rate = st.slider(
            "Detection Rate",
            0.50,
            0.99,
            0.85,
            0.01,
            key="strategy_detection"
        )

    with col2:

        false_positive_rate = st.slider(
            "False Positive Rate",
            0.01,
            0.30,
            0.05,
            0.01,
            key="strategy_false_positive"
        )

    with col3:

        attack_success_rate = st.slider(
            "Attack Success Rate",
            0.05,
            0.80,
            0.30,
            0.01,
            key="strategy_attack_success"
        )

    game = analyze_game(
        detection_rate,
        false_positive_rate,
        attack_success_rate
    )

    ids_strategy = game[
        "ids_strategy"
    ]

    attacker_strategy = game[
        "attacker_strategy"
    ]

    game_value = game[
        "game_value"
    ]

    st.divider()

    # --------------------------------------------------------
    # STRATEGY GRAPHS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    # IDS
    with col1:

        st.subheader(
            "🛡️ IDS Strategy"
        )

        fig, ax = plt.subplots(
            figsize=SMALL_CHART
        )

        labels = [
            "Low Detection",
            "High Detection"
        ]

        bars = ax.bar(
            labels,
            ids_strategy,
            color=[
                "#ffbd5c",
                "#00e6a0"
            ],
            width=0.4
        )

        ax.set_ylim(
            0,
            1
        )

        ax.set_ylabel(
            "Probability",
            fontsize=7
        )

        ax.set_title(
            "IDS Mixed Strategy",
            fontsize=10
        )

        style_chart(ax)

        for bar, value in zip(
            bars,
            ids_strategy
        ):

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                value + 0.03,
                f"{value * 100:.1f}%",
                ha="center",
                fontsize=7,
                color="#f1f5f9"
            )

        plt.tight_layout()

        st.pyplot(
            fig,
            clear_figure=True
        )

    # ATTACKER
    with col2:

        st.subheader(
            "⚔️ Attacker Strategy"
        )

        fig, ax = plt.subplots(
            figsize=SMALL_CHART
        )

        labels = [
            "Stealth",
            "Aggressive"
        ]

        bars = ax.bar(
            labels,
            attacker_strategy,
            color=[
                "#8090ff",
                "#ff5c70"
            ],
            width=0.4
        )

        ax.set_ylim(
            0,
            1
        )

        ax.set_ylabel(
            "Probability",
            fontsize=7
        )

        ax.set_title(
            "Attacker Mixed Strategy",
            fontsize=10
        )

        style_chart(ax)

        for bar, value in zip(
            bars,
            attacker_strategy
        ):

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                value + 0.03,
                f"{value * 100:.1f}%",
                ha="center",
                fontsize=7,
                color="#f1f5f9"
            )

        plt.tight_layout()

        st.pyplot(
            fig,
            clear_figure=True
        )

    st.divider()

    # --------------------------------------------------------
    # VALUES
    # --------------------------------------------------------

    st.subheader(
        "📌 Strategy Probabilities"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "IDS Low",
            f"{ids_strategy[0] * 100:.2f}%"
        )

    with col2:

        st.metric(
            "IDS High",
            f"{ids_strategy[1] * 100:.2f}%"
        )

    with col3:

        st.metric(
            "Attacker Stealth",
            f"{attacker_strategy[0] * 100:.2f}%"
        )

    with col4:

        st.metric(
            "Attacker Aggressive",
            f"{attacker_strategy[1] * 100:.2f}%"
        )

    st.info(
        f"🧮 Game value from IDS perspective: "
        f"{game_value:.3f}"
    )


# ============================================================
# 5. DATASET EXPLORER
# ============================================================

elif page == "📊 Dataset Explorer":

    st.title(
        "📊 Dataset Explorer"
    )

    st.write(
        """
        Explore the network traffic dataset used by the
        intrusion detection model.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # DATASET METRICS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Records",
            f"{data.shape[0]:,}"
        )

    with col2:

        st.metric(
            "Total Columns",
            f"{data.shape[1]:,}"
        )

    with col3:

        st.metric(
            "Attack Records",
            f"{attack_count:,}"
        )

    st.divider()

    # --------------------------------------------------------
    # DATA PREVIEW
    # --------------------------------------------------------

    st.subheader(
        "🔍 Dataset Preview"
    )

    number_rows = st.slider(
        "Rows to display",
        min_value=5,
        max_value=50,
        value=10,
        key="dataset_rows"
    )

    st.dataframe(
        data.head(number_rows),
        use_container_width=True,
        height=350
    )

    st.divider()

    # --------------------------------------------------------
    # ATTACK SUMMARY
    # --------------------------------------------------------

    st.subheader(
        "🚨 Attack Category Summary"
    )

    distribution_df = pd.DataFrame(
        list(attack_distribution.items()),
        columns=[
            "Attack Type",
            "Count"
        ]
    )

    st.dataframe(
        distribution_df,
        use_container_width=True
    )

# --------------------------------------------------------
# SMALL DATASET DISTRIBUTION GRAPH
# --------------------------------------------------------

    st.subheader(
        "📊 Dataset Distribution"
    )

    # Center the graph and make it compact
    left, center, right = st.columns(
        [1.5, 1, 1.5]
    )

    with center:

        fig, ax = plt.subplots(
            figsize=(2.4, 1.35),
            dpi=90
        )

        ax.bar(
            distribution_df["Attack Type"],
            distribution_df["Count"],
            color="#00bfff",
            width=0.45
        )

        ax.set_ylabel(
            "Records",
            fontsize=5
        )

        ax.set_title(
            "Attack Distribution",
            fontsize=7,
            color="#eaf2f7",
            pad=3
        )

        style_chart(ax)

        ax.tick_params(
            axis="x",
            rotation=25,
            labelsize=5
        )

        ax.tick_params(
            axis="y",
            labelsize=5
        )

        plt.tight_layout(
            pad=0.3
        )

        st.pyplot(
            fig,
            clear_figure=True,
            use_container_width=False
        )

    st.divider()


    # --------------------------------------------------------
    # FEATURE INFORMATION
    # --------------------------------------------------------

    st.subheader(
        "📋 Feature Information"
    )

    feature_df = pd.DataFrame(
        {
            "Feature": data.columns,
            "Data Type": [
                str(dtype)
                for dtype in data.dtypes
            ]
        }
    )

    st.dataframe(
        feature_df,
        use_container_width=True
    )


# ============================================================
# 6. ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.title(
        "ℹ️ About This Project"
    )

    st.subheader(
        "🛡️ AI-Based Network Intrusion Detection"
    )

    st.write(
        """
        This GTDS mini project combines Artificial Intelligence
        and Game Theory for network security analysis.
        """
    )

    st.divider()

    st.subheader(
        "🎯 Objective"
    )

    st.write(
        """
        The system detects malicious network traffic using
        a Random Forest classifier and models the strategic
        interaction between an attacker and an Intrusion
        Detection System using a zero-sum game.
        """
    )

    st.subheader(
        "🔄 Project Pipeline"
    )

    st.code(
        """
NSL-KDD Dataset
       ↓
Data Preprocessing
       ↓
Random Forest
       ↓
Normal / Attack Classification
       ↓
Payoff Matrix
       ↓
Zero-Sum Game
       ↓
Linear Programming
       ↓
Mixed Strategies
       ↓
IDS Strategic Analysis
        """,
        language="text"
    )

    st.divider()

    st.subheader(
        "🧰 Technology Stack"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            **Machine Learning**

            🐍 Python  
            🐼 Pandas  
            🔢 NumPy  
            🤖 Scikit-learn  
            🌲 Random Forest
            """
        )

    with col2:

        st.markdown(
            """
            **Game Theory & UI**

            🧮 SciPy  
            🎮 Zero-Sum Game  
            ⚡ Streamlit  
            📈 Matplotlib  
            📊 Seaborn
            """
        )

    st.divider()

    st.success(
        "🟢 AI + Game Theory IDS Analysis Ready"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🛡️ AI-Based Network Intrusion Detection | "
    "GTDS Mini Project | Machine Learning + Game Theory"
)
