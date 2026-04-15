from ortools.sat.python import cp_model


def solve_schedule(num_agents, num_days, m, n, days_off, shifts_required, wants_double):
    model = cp_model.CpModel()

    # Shifts: 0 = morning, 1 = afternoon
    shifts = [0, 1]

    # Decision variables
    x = {}
    for a in range(num_agents):
        for d in range(num_days):
            for s in shifts:
                x[(a, d, s)] = model.new_bool_var(f"x_a{a}_d{d}_t{s}")

    # Constraint 1: Each day has exactly m morning and n afternoon workers
    for d in range(num_days):
        # Morning
        model.Add(sum(x[(a, d, 0)] for a in range(num_agents)) == m)
        # Afternoon
        model.Add(sum(x[(a, d, 1)] for a in range(num_agents)) == n)

    # Constraint 2: Respect days off
    for a in range(num_agents):
        for d in days_off[a]:
            for t in shifts:
                model.Add(x[(a, d, t)] == 0)

    # Constraint 3: Each agent works exactly s(a) shifts
    for a in range(num_agents):
        model.Add(
            sum(x[(a, d, t)] for d in range(num_days) for t in shifts)
            == shifts_required[a]
        )

    # Constraint 4: At most one or two shifts per day per agent depending on preference
    for a in range(num_agents):
        for d in range(num_days):
            if not wants_double[a]:
                model.add(x[(a, d, 0)] + x[(a, d, 1)] <= 1)
            else:
                model.add(x[(a, d, 0)] + x[(a, d, 1)] <= 2)

    # Preference for double shifts
    double_shift = {}
    for a in range(num_agents):
        for d in range(num_days):
            double_shift[(a, d)] = model.new_bool_var(f"double_a{a}_d{d}")

            # If double_shift == 1 -> both shifts assigned
            model.add(x[(a, d, 0)] + x[(a, d, 1)] == 2).only_enforce_if(double_shift[(a, d)])

            # If double_shift == 0 -> NOT both shifts
            model.Add(x[(a, d, 0)] + x[(a, d, 1)] <= 1).only_enforce_if(double_shift[(a, d)].Not())

            # If agent is NOT allowed double shifts -> force variable to 0
            if not wants_double[a]:
                model.Add(double_shift[(a, d)] == 0)

    # Maximize preferred double shifts
    model.Maximize(
        sum(double_shift[(a, d)]
            for a in range(num_agents)
            for d in range(num_days)
            if wants_double[a])
    )

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10

    status = solver.Solve(model)

    # Output
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        schedule = {}
        for d in range(num_days):
            schedule[d] = {"morning": [], "afternoon": []}
            for a in range(num_agents):
                if solver.Value(x[(a, d, 0)]):
                    schedule[d]["morning"].append(a)
                if solver.Value(x[(a, d, 1)]):
                    schedule[d]["afternoon"].append(a)

        print("\nSchedule:\n")
        for d in range(num_days):
            print(f"Day {d}:")
            print(f"  Morning: {schedule[d]['morning']}")
            print(f"  Afternoon: {schedule[d]['afternoon']}")

            doubles = [a for a in range(num_agents)
                       if solver.Value(double_shift[(a, d)])]
            if doubles:
                print(f"  Double shifts: {doubles}")

        return schedule
    else:
        print("No feasible solution found.")
        return None


# Example usage
if __name__ == "__main__":
    num_agents = 8
    num_days = 5
    m = 4  # morning shifts per day
    n = 2  # afternoon shifts per day

    # Days off per agent
    days_off = {
        0: [2],
        1: [],
        2: [1, 4],
        3: [1],
        4: [3],
        5: [2, 3],
        6: [0],
        7: [0, 4]
    }

    # Required number of shifts per agent
    shifts_required = {
        0: 4,
        1: 5,
        2: 3,
        3: 4,
        4: 4,
        5: 3,
        6: 4,
        7: 3,
    }

    # Wants double shifts
    wants_double = {
        0: True,
        1: True,
        2: False,
        3: False,
        4: False,
        5: True,
        6: False,
        7: False,
    }

    schedule = solve_schedule(
        num_agents,
        num_days,
        m,
        n,
        days_off,
        shifts_required,
        wants_double
    )