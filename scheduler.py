from ortools.sat.python import cp_model


def solve_schedule(
    num_agents,
    num_days,
    m,
    n,
    days_off,
    morning_required,
    afternoon_required,
    wants_double,
):
    model = cp_model.CpModel()

    shifts = [0, 1]  # 0 = morning, 1 = afternoon

    # ============================================================
    # Decision variables
    # ============================================================
    x = {}
    for a in range(num_agents):
        for d in range(num_days):
            for s in shifts:
                x[(a, d, s)] = model.NewBoolVar(f"x_a{a}_d{d}_s{s}")

    # ============================================================
    # HARD CONSTRAINTS
    # ============================================================

    # Coverage
    for d in range(num_days):
        model.Add(sum(x[(a, d, 0)] for a in range(num_agents)) == m)
        model.Add(sum(x[(a, d, 1)] for a in range(num_agents)) == n)

    # Days off
    for a in range(num_agents):
        for d in days_off[a]:
            for s in shifts:
                model.Add(x[(a, d, s)] == 0)

    # Daily limits
    for a in range(num_agents):
        for d in range(num_days):
            if not wants_double[a]:
                model.Add(x[(a, d, 0)] + x[(a, d, 1)] <= 1)
            else:
                model.Add(x[(a, d, 0)] + x[(a, d, 1)] <= 2)

    # ============================================================
    # ACTUAL SHIFT COUNTS
    # ============================================================
    actual_m = {}
    actual_a = {}

    for a in range(num_agents):
        actual_m[a] = sum(x[(a, d, 0)] for d in range(num_days))
        actual_a[a] = sum(x[(a, d, 1)] for d in range(num_days))

    # ============================================================
    # DEVIATIONS (fairness)
    # ============================================================
    dev_m = {}
    dev_a = {}
    max_dev = model.NewIntVar(0, num_days, "max_dev")

    for a in range(num_agents):
        dev_m[a] = model.NewIntVar(0, num_days, f"dev_m_a{a}")
        dev_a[a] = model.NewIntVar(0, num_days, f"dev_a_a{a}")

        model.Add(dev_m[a] >= actual_m[a] - morning_required[a])
        model.Add(dev_m[a] >= morning_required[a] - actual_m[a])

        model.Add(dev_a[a] >= actual_a[a] - afternoon_required[a])
        model.Add(dev_a[a] >= afternoon_required[a] - actual_a[a])

        model.Add(dev_m[a] <= max_dev)
        model.Add(dev_a[a] <= max_dev)

    total_deviation = sum(dev_m[a] + dev_a[a] for a in range(num_agents))

    # ============================================================
    # DOUBLE SHIFT VARIABLES
    # ============================================================
    double_shift = {}
    for a in range(num_agents):
        for d in range(num_days):
            double_shift[(a, d)] = model.NewBoolVar(f"double_a{a}_d{d}")

            model.Add(x[(a, d, 0)] + x[(a, d, 1)] == 2).OnlyEnforceIf(double_shift[(a, d)])
            model.Add(x[(a, d, 0)] + x[(a, d, 1)] <= 1).OnlyEnforceIf(double_shift[(a, d)].Not())

    # ============================================================
    # PREFERENCES
    # ============================================================
    penalty_terms = []
    reward_terms = []

    for a in range(num_agents):
        for d in range(num_days):
            if not wants_double[a]:
                penalty_terms.append(double_shift[(a, d)])
            else:
                reward_terms.append(double_shift[(a, d)])

    total_penalty = sum(penalty_terms)
    total_reward = sum(reward_terms)

    # ============================================================
    # WORK INDICATOR (for spread objective)
    # ============================================================
    work = {}
    for a in range(num_agents):
        for d in range(num_days):
            work[(a, d)] = model.NewBoolVar(f"work_a{a}_d{d}")

            model.Add(work[(a, d)] >= x[(a, d, 0)])
            model.Add(work[(a, d)] >= x[(a, d, 1)])
            model.Add(work[(a, d)] <= x[(a, d, 0)] + x[(a, d, 1)])

    consecutive_work = []
    for a in range(num_agents):
        for d in range(num_days - 1):
            c = model.NewBoolVar(f"consec_a{a}_d{d}")

            model.AddBoolAnd([work[(a, d)], work[(a, d + 1)]]).OnlyEnforceIf(c)
            model.AddBoolOr([work[(a, d)].Not(), work[(a, d + 1)].Not()]).OnlyEnforceIf(c.Not())

            consecutive_work.append(c)

    total_consecutive = sum(consecutive_work)

    # ============================================================
    # PHASE 1: minimize max deviation
    # ============================================================
    model.Minimize(max_dev)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10

    if solver.Solve(model) not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    best_max_dev = solver.Value(max_dev)

    # ============================================================
    # PHASE 2: minimize total deviation
    # ============================================================
    model.Add(max_dev == best_max_dev)
    model.Minimize(total_deviation)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10

    if solver.Solve(model) not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    best_total_dev = solver.Value(total_deviation)

    # ============================================================
    # PHASE 3: minimize unwanted double shifts
    # ============================================================
    model.Add(total_deviation == best_total_dev)
    model.Minimize(total_penalty)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10

    if solver.Solve(model) not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    best_penalty = solver.Value(total_penalty)

    # ============================================================
    # PHASE 4: maximize desired double shifts
    # ============================================================
    model.Add(total_penalty == best_penalty)
    model.Maximize(total_reward)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10

    if solver.Solve(model) not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    best_reward = solver.Value(total_reward)

    # ============================================================
    # PHASE 5: minimize consecutive working days (SPREAD OBJECTIVE)
    # ============================================================
    model.Add(total_reward == best_reward)
    model.Minimize(total_consecutive)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10

    if solver.Solve(model) not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    # ============================================================
    # EXTRACT SOLUTION
    # ============================================================
    schedule = {}
    for a in range(num_agents):
        schedule[a] = []
        for d in range(num_days):
            assigned = []
            for s in shifts:
                if solver.Value(x[(a, d, s)]):
                    assigned.append(s)
            schedule[a].append(assigned)

    return {
        "schedule": schedule,
        "max_deviation": solver.Value(max_dev),
        "total_deviation": solver.Value(total_deviation),
        "preference_penalty": solver.Value(total_penalty),
        "preferred_double_shifts": solver.Value(total_reward),
        "consecutive_work_penalty": solver.Value(total_consecutive),
    }

def print_schedule(
    result,
    num_agents,
    num_days,
    morning_required,
    afternoon_required
):
    if result is None:
        print("No feasible solution found.")
        return

    schedule = result["schedule"]

    # ------------------------
    # SCHEDULE TABLE
    # ------------------------
    headers = ["Day"] + [f"A{a}" for a in range(num_agents)] + ["M_count", "A_count"]
    print("\n## Schedule\n")
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")

    for d in range(num_days):
        row = [str(d)]
        morning_count = 0
        afternoon_count = 0

        for a in range(num_agents):
            shifts = schedule[a][d]

            if shifts == []:
                cell = "-"
            elif shifts == [0]:
                cell = "M"
                morning_count += 1
            elif shifts == [1]:
                cell = "A"
                afternoon_count += 1
            elif set(shifts) == {0, 1}:
                cell = "MA"
                morning_count += 1
                afternoon_count += 1
            else:
                cell = "?"

            row.append(cell)

        row.append(str(morning_count))
        row.append(str(afternoon_count))

        print("| " + " | ".join(row) + " |")

    # ------------------------
    # AGENT SUMMARY (NEW STRUCTURE)
    # ------------------------
    print("\n## Agent Summary\n")

    headers = [
        "Agent",
        "M_assigned",
        "M_required",
        "M_diff",
        "A_assigned",
        "A_required",
        "A_diff",
        "Total_assigned"
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")

    for a in range(num_agents):
        m_assigned = sum(1 for d in range(num_days) if 0 in schedule[a][d])
        a_assigned = sum(1 for d in range(num_days) if 1 in schedule[a][d])

        m_req = morning_required[a]
        a_req = afternoon_required[a]

        m_diff = m_assigned - m_req
        a_diff = a_assigned - a_req

        total = m_assigned + a_assigned

        print(
            f"| A{a} | {m_assigned} | {m_req} | {m_diff} | "
            f"{a_assigned} | {a_req} | {a_diff} | {total} |"
        )

    # ------------------------
    # METRICS
    # ------------------------
    print("\n## Metrics\n")
    print(f"- Max deviation: {result.get('max_deviation', '?')}")
    print(f"- Total deviation: {result.get('total_deviation', '?')}")
    print(f"- Preference penalty: {result.get('preference_penalty', '?')}")
    print(f"- Preference reward: {result.get('preferred_double_shifts', '?')}")
    print(f"- Consecutive work penalty: {result.get('consecutive_work_penalty', '?')}")


# Example usage
if __name__ == "__main__":
    num_agents = 38
    num_days = 37
    m = 4  # morning shifts per day
    n = 2  # afternoon shifts per day

    # Days off per agent
    days_off = {
        0: [1,5,6,7,16,27,28],
        1: [3,9,11,12,15,20,25,22,23,27,28,29,30,31,32,33,34,35,36],
        2: [3,11,14,15,20,25,30,35],
        3: [2,3,11,15,18,20,25,30,35],
        4: [3,11,15,20,25,30,35],
        5: [0,13,14,15,16,17,18,19,20,21],
        6: [],
        7: [4,12],
        8: [],
        9: [28,29],
        10: [22,23,24,25,26,27,28,29,30,31],
        11: [5,6,7,13,14,15,16,17,18,19,20,21,22,23,24,25,26],
        12: [],
        13: [2,8,19,22],
        14: [3,4],
        15: [15,16],
        16: [16,21],
        17: [0,1,2,3,4,31],
        18: [],
        19: [0,13,14,15,16,17,18,19,20,21],
        20: [5,6,7,16],
        21: [7,8,9,10,11,12,13,14,15,16],
        22: [8,9,10,11,27,28,29,30,31,32,33],
        23: [],
        24: [1,4,5,6,7,9,13,18,23,28,33],
        25: [8,9],
        26: [13,33],
        27: [2,14,21,32,33,34,35,36],
        28: [5,6,7,24,31,32],
        29: [13,14,29],
        30: [1,2,3,4,34,35,36],
        31: [8,9,10,11,12,13,14,15,16,17,18,19,20,21,29],
        32: [2,14,29],
        33: [14,29],
        34: [0,4,5,8,12,16,14,29],
        35: [4,8,9,10,11,12,13,14,15,16,29],
        36: [14,29],
        37: [1,5,6,8,9,11,12,13,14,29],
    }

    # Required number of shifts per agent
    morning_required = {
        0: 0,
        1: 5,
        2: 5,
        3: 5,
        4: 0,
        5: 6,
        6: 6,
        7: 5,
        8: 4,
        9: 4,
        10: 3,
        11: 4,
        12: 6,
        13: 6,
        14: 5,
        15: 5,
        16: 6,
        17: 4,
        18: 4,
        19: 2,
        20: 4,
        21: 1,
        22: 3,
        23: 2,
        24: 2,
        25: 2,
        26: 4,
        27: 1,
        28: 4,
        29: 4,
        30: 4,
        31: 6,
        32: 4,
        33: 4,
        34: 4,
        35: 4,
        36: 4,
        37: 6,
    }
    afternoon_required = {
        0: 6,
        1: 1,
        2: 1,
        3: 1,
        4: 6,
        5: 0,
        6: 0,
        7: 1,
        8: 2,
        9: 0,
        10: 3,
        11: 4,
        12: 2,
        13: 2,
        14: 3,
        15: 3,
        16: 2,
        17: 0,
        18: 0,
        19: 0,
        20: 4,
        21: 0,
        22: 0,
        23: 3,
        24: 2,
        25: 5,
        26: 0,
        27: 3,
        28: 0,
        29: 2,
        30: 2,
        31: 3,
        32: 2,
        33: 2,
        34: 2,
        35: 2,
        36: 2,
        37: 3,
    }

    # Wants double shifts
    wants_double = {
        0:False,
        1:True,
        2:False,
        3:False,
        4:False,
        5:False,
        6:True,
        7:True,
        8:True,
        9:False,
        10:True,
        11:True,
        12:False,
        13:True,
        14:True,
        15:True,
        16:False,
        17:True,
        18:True,
        19:False,
        20:True,
        21:False,
        22:False,
        23:True,
        24:True,
        25:True,
        26:False,
        27:True,
        28:True,
        29:False,
        30:True,
        31:False,
        32:True,
        33:True,
        34:False,
        35:True,
        36:False,
        37:False,
    }

    schedule = solve_schedule(
        num_agents,
        num_days,
        m,
        n,
        days_off,
        morning_required,
        afternoon_required,
        wants_double
    )

    print_schedule(schedule, num_agents, num_days, morning_required, afternoon_required)