"""
Reward engine for DataCleanEnv-X.

Provides trajectory-aware reward signal:
- Positive reward for reducing issues
- Negative penalty for no-op / destructive actions
- Bonus/penalty for finishing with few/many remaining issues
"""


def compute_reward(prev_issues: int, new_issues: int, action_type: str) -> float:
    """
    Compute reward based on issue reduction with trajectory signal.

    Args:
        prev_issues: Total issue count before the action.
        new_issues: Total issue count after the action.
        action_type: The type of action that was taken.

    Returns:
        Reward value in [-1.0, 1.0].
    """
    delta = prev_issues - new_issues  # positive = improvement

    # Base reward: proportional to issues fixed
    if prev_issues > 0:
        reward = delta / prev_issues
    else:
        reward = 0.0

    # Penalty for no-op actions that don't reduce issues
    if delta == 0 and action_type not in ("finish", "escalate"):
        reward = -0.05

    # Larger penalty for destructive actions that increase issues
    if delta < 0:
        reward = -0.2

    # Bonus/penalty for finishing
    if action_type == "finish":
        if prev_issues == 0:
            reward = 0.1  # perfect finish bonus
        elif new_issues <= 5:
            reward = 0.05  # near-clean finish bonus
        elif new_issues > 20:
            reward = -0.1  # premature finish penalty
        else:
            reward = 0.0

    # Clamp to valid range
    reward = max(-1.0, min(1.0, reward))

    return round(reward, 4)
