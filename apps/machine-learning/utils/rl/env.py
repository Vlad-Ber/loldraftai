def create_solo_queue_draft_order():
    # Define the draft order as a list of dicts
    draft_order = []
    # Ban phase: 5 bans per team
    for _ in range(5):
        draft_order.append({"team": 0, "action_type": "ban"})  # Blue ban
        draft_order.append({"team": 1, "action_type": "ban"})  # Red ban

    # Pick phase
    draft_order.append({"team": 0, "action_type": "pick"})  # Blue pick 1
    draft_order.append({"team": 1, "action_type": "pick"})  # Red pick 1
    draft_order.append({"team": 1, "action_type": "pick"})  # Red pick 2
    draft_order.append({"team": 0, "action_type": "pick"})  # Blue pick 2
    draft_order.append({"team": 0, "action_type": "pick"})  # Blue pick 3
    draft_order.append({"team": 1, "action_type": "pick"})  # Red pick 3
    draft_order.append({"team": 1, "action_type": "pick"})  # Red pick 4
    draft_order.append({"team": 0, "action_type": "pick"})  # Blue pick 4
    draft_order.append({"team": 0, "action_type": "pick"})  # Blue pick 5
    draft_order.append({"team": 1, "action_type": "pick"})  # Red pick 5

    # Role selection phase: 5 picks per team
    for role_index in range(5):
        draft_order.append(
            {"team": 0, "action_type": "role_selection", "role_index": role_index}
        )
    for role_index in range(5):
        draft_order.append(
            {"team": 1, "action_type": "role_selection", "role_index": role_index}
        )

    return draft_order
