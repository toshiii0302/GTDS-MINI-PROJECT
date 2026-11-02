import numpy as np
from scipy.optimize import linprog


# ============================================================
# PAYOFF MATRIX
# ============================================================

def calculate_payoff_matrix(
    detection_rate: float,
    false_positive_rate: float,
    attack_success_rate: float
) -> np.ndarray:
    """
    Creates a 2x2 zero-sum payoff matrix.

    Rows    -> Attacker strategies
    Columns -> IDS strategies

    Payoff is measured from the IDS perspective.

    Attacker:
        0 = Stealth Attack
        1 = Aggressive Attack

    IDS:
        0 = Low Detection
        1 = High Detection
    """

    # --------------------------------------------------------
    # High detection against stealth
    # --------------------------------------------------------

    stealth_high = (
        detection_rate * 10
        - (1 - detection_rate) * 8
        - false_positive_rate * 2
    )

    # --------------------------------------------------------
    # High detection against aggressive attack
    # --------------------------------------------------------

    aggressive_high = (
        detection_rate * 12
        - attack_success_rate * 10
        - false_positive_rate * 2
    )

    # --------------------------------------------------------
    # Low detection against stealth
    # --------------------------------------------------------

    stealth_low = (
        -attack_success_rate * 8
        -false_positive_rate * 2
    )

    # --------------------------------------------------------
    # Low detection against aggressive attack
    # --------------------------------------------------------

    aggressive_low = (
        -attack_success_rate * 12
        -false_positive_rate * 2
    )

    matrix = np.array([
        [
            stealth_low,
            stealth_high
        ],
        [
            aggressive_low,
            aggressive_high
        ]
    ])

    return matrix


# ============================================================
# ZERO-SUM GAME SOLVER
# ============================================================

def solve_zero_sum_game(
    payoff_matrix: np.ndarray
) -> dict:

    matrix = np.asarray(
        payoff_matrix,
        dtype=float
    )

    rows, cols = matrix.shape

    if rows != 2 or cols != 2:
        raise ValueError(
            "This implementation expects a 2x2 payoff matrix."
        )

    # ========================================================
    # IDS OPTIMIZATION
    # ========================================================
    #
    # Maximize v
    #
    # subject to:
    #
    # p^T A[:,j] >= v
    #
    # ========================================================

    objective = np.zeros(rows + 1)

    # linprog minimizes, so maximize v
    objective[-1] = -1

    A_ub = []
    b_ub = []

    for column in range(cols):

        constraint = np.zeros(rows + 1)

        constraint[:rows] = -matrix[:, column]
        constraint[-1] = 1

        A_ub.append(constraint)
        b_ub.append(0)

    A_eq = [
        [1] * rows + [0]
    ]

    b_eq = [1]

    ids_result = linprog(
        objective,
        A_ub=np.asarray(A_ub),
        b_ub=np.asarray(b_ub),
        A_eq=np.asarray(A_eq),
        b_eq=np.asarray(b_eq),
        bounds=[
            (0, 1)
        ] * rows + [
            (None, None)
        ],
        method="highs"
    )

    if not ids_result.success:
        raise RuntimeError(
            f"IDS optimization failed: "
            f"{ids_result.message}"
        )

    ids_strategy = ids_result.x[:rows]
    game_value = ids_result.x[-1]

    # ========================================================
    # ATTACKER OPTIMIZATION
    # ========================================================

    # Minimize attacker expected IDS payoff
    attacker_objective = np.zeros(cols + 1)
    attacker_objective[-1] = 1

    A_ub_attacker = []
    b_ub_attacker = []

    for row in range(rows):

        constraint = np.zeros(cols + 1)

        constraint[:cols] = matrix[row, :]
        constraint[-1] = -1

        A_ub_attacker.append(
            constraint
        )

        b_ub_attacker.append(0)

    A_eq_attacker = [
        [1] * cols + [0]
    ]

    b_eq_attacker = [1]

    attacker_result = linprog(
        attacker_objective,
        A_ub=np.asarray(
            A_ub_attacker
        ),
        b_ub=np.asarray(
            b_ub_attacker
        ),
        A_eq=np.asarray(
            A_eq_attacker
        ),
        b_eq=np.asarray(
            b_eq_attacker
        ),
        bounds=[
            (0, 1)
        ] * cols + [
            (None, None)
        ],
        method="highs"
    )

    if not attacker_result.success:
        raise RuntimeError(
            f"Attacker optimization failed: "
            f"{attacker_result.message}"
        )

    attacker_strategy = (
        attacker_result.x[:cols]
    )

    # Numerical cleanup
    ids_strategy = np.clip(
        ids_strategy,
        0,
        1
    )

    attacker_strategy = np.clip(
        attacker_strategy,
        0,
        1
    )

    # Normalize
    ids_strategy /= ids_strategy.sum()
    attacker_strategy /= attacker_strategy.sum()

    return {
        "ids_strategy": ids_strategy,
        "attacker_strategy": attacker_strategy,
        "game_value": float(game_value)
    }


# ============================================================
# COMPLETE GAME ANALYSIS
# ============================================================

def analyze_game(
    detection_rate: float,
    false_positive_rate: float,
    attack_success_rate: float
) -> dict:

    # Validate input
    for value, name in [
        (detection_rate, "Detection rate"),
        (false_positive_rate, "False positive rate"),
        (attack_success_rate, "Attack success rate")
    ]:
        if not 0 <= value <= 1:
            raise ValueError(
                f"{name} must be between 0 and 1."
            )

    matrix = calculate_payoff_matrix(
        detection_rate,
        false_positive_rate,
        attack_success_rate
    )

    solution = solve_zero_sum_game(
        matrix
    )

    return {
        "payoff_matrix": matrix,
        "ids_strategy": solution[
            "ids_strategy"
        ],
        "attacker_strategy": solution[
            "attacker_strategy"
        ],
        "game_value": solution[
            "game_value"
        ]
    }
