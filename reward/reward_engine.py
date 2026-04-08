

def compute_reward(prev_issues: int, new_issues: int, action_type: str):
    """
    Compute reward based on issue reduction
    """
    delta = prev_issues - new_issues
    reward = delta / max(prev_issues, 1)

    if delta == 0:
        reward -= 0.05

    if action_type == "drop_row" and delta < 0:
        reward -= 0.2

    reward = max(-1.0, min(1.0, reward))

    return round(reward, 4)


# test block
if __name__ == "__main__":
    reward = compute_reward(10, 5, "fill_value")
    print("Reward:", reward)
