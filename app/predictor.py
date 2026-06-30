from collections import deque

people_history = deque(maxlen=20)
occupancy_history = deque(maxlen=20)


def update_history(people, occupancy):
    people_history.append(people)
    occupancy_history.append(occupancy)


def calculate_growth(history):
    if len(history) < 2:
        return 0

    growth = []

    for i in range(1, len(history)):
        growth.append(history[i] - history[i - 1])

    return sum(growth) / len(growth)


def predict_people():
    if len(people_history) == 0:
        return 0

    current = people_history[-1]
    growth = calculate_growth(people_history)

    return max(0, round(current + growth * 6))


def predict_occupancy():
    if len(occupancy_history) == 0:
        return 0

    current = occupancy_history[-1]
    growth = calculate_growth(occupancy_history)

    return max(0, round(current + growth * 6))


def risk_trend():
    growth = calculate_growth(occupancy_history)

    if growth > 0.3:
        return "Increasing"

    if growth < -0.3:
        return "Decreasing"

    return "Stable"