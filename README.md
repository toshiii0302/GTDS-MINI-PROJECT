# 🛡️ AI-Based Network Intrusion Detection Using Game Theory and Mixed Strategies

An AI-based Network Intrusion Detection System (NIDS) that combines **Machine Learning and Game Theory** to identify suspicious network traffic and model strategic interactions between attackers and defenders using optimal mixed strategies.

## 📌 Project Overview

This project implements an intelligent network intrusion detection prototype using a **Random Forest Classifier** and **Game-Theoretic Optimization**.

The machine learning component classifies network traffic as either Normal or Attack using the NSL-KDD dataset. The game theory component models the interaction between an attacker and an Intrusion Detection System (IDS) as a 2 × 2 zero-sum game and calculates optimal mixed strategies using linear programming.

An interactive Streamlit dashboard brings both components together, allowing users to explore model performance, attack distributions, payoff matrices, and strategic decisions.

## 🎯 Objectives

- Develop a prototype Network Intrusion Detection System using Python.
- Classify network traffic into Normal and Attack categories using Random Forest.
- Preprocess and encode categorical network traffic features.
- Model attacker-defender interactions using a 2 × 2 zero-sum game.
- Calculate optimal mixed strategies using linear programming.
- Visualize machine learning results and game-theoretic analysis through an interactive dashboard.

## ✨ Key Features

### 1. Machine Learning-Based Intrusion Detection
- Uses a Random Forest classifier for network traffic classification.
- Classifies traffic as Normal or Attack.
- Uses One-Hot Encoding for categorical features.
- Displays accuracy, precision, recall, F1-score, and confusion matrix.
- Provides feature importance and attack category distribution.

### 2. Game-Theoretic Analysis
- Models attacker and IDS interactions as a 2 × 2 zero-sum game.
- Attacker strategies:
  - Stealth Attack
  - Aggressive Attack
- IDS strategies:
  - Low Detection
  - High Detection
- Generates a payoff matrix based on detection rate, false-positive rate, and attack-success rate.

### 3. Optimal Mixed Strategy Calculation
- Uses SciPy Linear Programming (`linprog`) to calculate optimal strategies.
- Determines the attacker's optimal mixed strategy.
- Determines the IDS's optimal mixed strategy.
- Calculates the game value (expected payoff).

### 4. Interactive Streamlit Dashboard

The application consists of the following sections:

- **Dashboard:** Overview of model performance and project statistics.
- **ML Analysis:** Classification metrics and model evaluation.
- **Game Theory:** Payoff matrix and game solution.
- **Strategy Analysis:** Optimal mixed strategy probabilities.
- **Dataset Explorer:** Dataset inspection and attack distribution.
- **About:** Project information.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive dashboard |
| Scikit-learn | Random Forest classification and preprocessing |
| Pandas | Dataset handling |
| NumPy | Numerical computations |
| SciPy | Linear programming optimization |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualizations |
| NSL-KDD | Network intrusion detection dataset |

## ⚙️ System Workflow

1. Load the NSL-KDD dataset.
2. Preprocess network traffic records.
3. Encode categorical features.
4. Convert attack labels into binary classes (Normal / Attack).
5. Train and test the Random Forest classifier.
6. Evaluate model performance.
7. Generate the attacker-defender payoff matrix.
8. Apply linear programming to calculate optimal mixed strategies.
9. Display results through the Streamlit dashboard.

## 📂 Dataset

The project is designed to use the **NSL-KDD KDDTrain+.txt** dataset.

- The dataset contains network traffic records and corresponding attack labels.
- The first 41 fields are used as network features.
- Attack labels are grouped into categories such as DoS, Probe, R2L, U2R, and Other Attack.
- The final classification target is converted into two classes: Normal (0) and Attack (1).

**Note:** If `KDDTrain+.txt` is unavailable, the application generates synthetic demonstration data as a fallback. Results obtained using synthetic data should not be interpreted as actual NSL-KDD model performance.

## 🚀 Installation and Execution

### Prerequisites
- Python 3.9 or above
- pip package manager

### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### Step 2: Navigate to the Project Directory

```bash
cd <PROJECT_FOLDER_NAME>
```

### Step 3: Install Dependencies

```bash
pip install streamlit pandas numpy scikit-learn scipy matplotlib seaborn
```

### Step 4: Add the Dataset

Place `KDDTrain+.txt` in the project directory.

If the dataset is not provided, the application can use its built-in synthetic data fallback.

### Step 5: Run the Application

```bash
streamlit run app.py
```

The application will open in your browser, displaying the interactive dashboard.

## 📁 Project Structure

```text
AI-Based-Network-Intrusion-Detection/
│
├── app.py                  # Main Streamlit application
├── ml_model.py             # Machine learning and classification module
├── game_theory.py          # Game-theoretic model and optimization
├── KDDTrain+.txt           # NSL-KDD dataset (if available)
├── README.md               # Project documentation
└── requirements.txt        # Project dependencies (if included)
```

## 🔮 Future Scope

- Integrate real-time network traffic monitoring.
- Explore advanced machine learning and deep learning models.
- Extend the game-theoretic model to support multiple attacker and defender strategies.
- Introduce adaptive strategies based on changing attack patterns.
- Integrate automated threat response and network defense mechanisms.
- Improve detection performance using larger and more diverse datasets.

## ⚠️ Limitations

- This is an educational prototype and not a production-grade security solution.
- It does not directly monitor live network traffic or block malicious activity.
- Game-theoretic payoff values depend on the selected scenario parameters.
- Model performance depends on whether the actual NSL-KDD dataset or synthetic fallback data is used.

**Developed as part of the Game Theory in Data Science (GTDS) Mini Project.**
