# Constraint-Based Workforce Scheduling Solver

A flexible scheduling system built with **Google OR-Tools CP-SAT** that generates fair, preference-aware shift schedules with multiple optimization priorities.

---

## Overview

This solver assigns agents to daily morning and afternoon shifts under a rich set of constraints, including:

- Fixed staffing requirements per shift per day
- Agent availability (days off)
- Fair workload distribution
- Required morning/afternoon shift balances per agent
- Preferences for double shifts
- Penalties for undesired double shifts
- Preference for desired double shifts
- Workload spacing (burnout reduction, soft constraint)

It uses a **multi-phase lexicographic optimization strategy**, ensuring higher-priority objectives are never sacrificed for lower-priority ones.

---

## Optimization Hierarchy

The solver optimizes in strict priority order:

1. **Fairness (primary)**
   - Minimize maximum deviation from required morning/afternoon shifts per agent

2. **Total balance**
   - Minimize total deviation across all agents

3. **Avoid unwanted double shifts**
   - Penalize assigning double shifts to agents who did not request them

4. **Encourage desired double shifts**
   - Maximize double shifts for agents who prefer them

5. **Workload spreading (fatigue reduction)**
   - Minimize consecutive working days (soft constraint)