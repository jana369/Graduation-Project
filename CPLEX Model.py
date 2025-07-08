import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from docplex.mp.model import Model

# Initialize the model
model = Model(name="Smart_Home_Energy_Scheduling")

# Sets (Time periods and Controllable Electrical Loads)
T = range(24)  # 24-hour scheduling period, reduced to 12 for testing
CELs = ["Washing Machine", "Dryer", "Dishwasher", "Oven"]
T_i = 0  # Start time of the period
T_f = 23  # End time of the period, adjusted to match T

# Updated Renewable Energy Generation and Costs (more realistic profiles)
P_sun = [0.0, 0.0, 0.1, 0.3, 0.7, 1.0]*4  # Hourly solar power (kW)
P_wind = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]*4  # Hourly wind power (kW)
P_re = [sun + wind for sun, wind in zip(P_sun, P_wind)]  # Total renewable power (kW)
C_re = [0.15, 0.10, 0.08, 0.07, 0.09, 0.05]*4  # Renewable energy cost (cost units/kWh)
# Updated Grid Cost Over Time Slots (more realistic pricing with peak/off-peak)
C_grid = [0.25, 0.20, 0.18, 0.15, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.07, 0.10, 0.13]*2  # Hourly grid cost (cost units/kWh)

# Add selling price data (example)
R_sell = [0.20, 0.15, 0.25, 0.05, 0.20, 0.06, 0.08, 0.10, 0.20, 0.10, 0.08, 0.20, 0.15, 0.12 , 0.10]*2
# Updated Non-controllable Load Profile (more realistic variation)
Pncl = [1.5, 1.2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.2, 0.1, 0.5]*2  # Hourly non-controllable load (kW)

# Define start and end of allowed working period for each device (with intersections)
Tcli = {"Washing Machine": 1, "Dryer": 5, "Dishwasher": 7, "Oven": 9}  # Start time per device
Tclf = {"Washing Machine": 8, "Dryer": 11, "Dishwasher": 10, "Oven": 13}  # End time per device

# Load information (power and duration)
Pcl = {"Washing Machine": 1.0, "Dryer": 0.8, "Dishwasher": 0.6, "Oven": 1.2}
Duration = {"Washing Machine": 3, "Dryer": 1, "Dishwasher": 2, "Oven": 1}

# Updated Precedence (including Microwave after Oven)
Precedence = {"Dryer": "Washing Machine"}  # Dryer after Washing Machine, Microwave after Oven

# The precedence Matrix
N = 4  # Number of CEL devices
Phi = np.zeros((N, N))  # Initialize with zeros
Phi[1, 0] = 1  # Dryer depends on Washing Machine

# System Constants
lambdaa = 2  # Max CELs connected in the same Δt
gamma = 20  # Max load from CELs in the same Δt

# Home Battery Parameters
E_b_init = 5.0  # Initial battery energy (kWh)
P_br_max = 5.0  # Max charging power (kW)
P_bi_max = 5.0  # Max discharging power (kW)
E_b_max = 30.0  # Max battery capacity (kWh)

# EV Battery Parameters
Pev_max = 3.0  # Maximum EV charging power [kW]
Eev_max = 20.0  # Maximum EV battery capacity [kWh]
ρ = 1 # Required battery charge percentage

# Multi objective Parameters
w1 = 0.7  # Weight for energy cost
w2 = 0.3  # Weight for discomfort
Z_cost_max = 195.84  # Maximum estimated energy cost (adjust based on actual bounds)
D_max = 7.0  # Maximum discomfort (number of appliances)

# Constants
P_grid_max = 20
P_sell_max = 4.0

# Binary Variables
X = model.binary_var_matrix(CELs, T, name="X")  # X[j, t] = 1 if appliance j is ON at time t
Alpha = model.binary_var_matrix(CELs, T, name="alpha")  # α[j, t] = 1 if appliance j is Connected at time t
Beta = model.binary_var_matrix(CELs, T, name="beta")   # β[j, t] = 1 if appliance j is Disconnected at time t
Delta = model.binary_var_matrix(CELs, T, name="Delta")  # D[j, t] = 1 if appliance j is Finished at time t

# Binary variable to control grid interaction mode
grid_mode = model.binary_var_list(T, name="grid_mode")  # 1 = buy, 0 = sell

# Continuous Variables
P_grid = model.continuous_var_list(T, name="Pgrid")  # Power from the grid
Pclr = model.continuous_var_list(T, name="Pclr", lb=0)
eta = model.continuous_var_dict(CELs, name="eta", lb=0)  # Actual start time for each appliance

# Add selling variables
P_sell = model.continuous_var_list(T, name="P_sell", lb=0)

# Home Battery Variables
P_br = model.continuous_var_list(T, name="P_br")  # Charging power
P_bi = model.continuous_var_list(T, name="P_bi")  # Discharging power
E_b = model.continuous_var_list(T, name="E_b")      # Stored energy
mode = model.binary_var_list(T, name="mode")  # 1 for charging, 0 for discharging

#Electric Vehicle Battery
EevTf = model.continuous_var(name='EevTf')  # Final EV battery energy
Ptevb = model.continuous_var_dict(T, name='Ptevb')  # EV charging power

# Original Objectives
energy_cost_objective = model.sum(C_grid[t] * P_grid[t] for t in T) - model.sum(R_sell[t] * P_sell[t] for t in T)+ model.sum(C_re[t] * P_re[t] for t in T)

discomfort_objective = model.sum((eta[j] - Tcli[j]) / (Tclf[j] - Tcli[j]) for j in CELs)

# Normalized Objectives
E_cost_norm = energy_cost_objective / Z_cost_max
D_norm = discomfort_objective / D_max

# Combined Objective
objective = w1 * E_cost_norm + w2 * D_norm
model.minimize(objective)

# Constraint: Define eta[j] as the start time
for j in CELs:
    model.add_constraint(
        eta[j] == model.sum(t * Alpha[j, t] for t in T),
        ctname=f"Start_Time_{j}"
    )

# Energy Balance Constraint
for t in T:
    model.add_constraint(
        P_sun[t] + P_wind[t] + P_grid[t] + P_bi[t] == P_br[t] + Ptevb[t] + Pclr[t] +
        Pncl[t] + model.sum(Pcl[j] * X[j, t] for j in CELs),
        ctname=f"Energy_Balance_{t}"
    )

# Update in home battery
for t in range(len(T)):
  if t == 0:
    model.add_constraint(E_b[t] == E_b_init - P_bi[t], ctname="Initial_(E_b)_value")
  else :
    model.add_constraint(
        E_b[t] == E_b[t-1] + P_br[t] - P_bi[t] - P_sell[t],
        ctname=f"Battery_Energy_{t}"
    )

# P_bi cannot exceed stored energy
for t in range(len(T)):
    if t > 0:
        model.add_constraint(
            P_bi[t] <= E_b[t - 1],
            ctname=f"Battery_Discharge_Limit_{t}"
        )
    else:
        model.add_constraint(
            P_bi[t] <= E_b_init,  # Use E_b_init as a value within the constraint for t=0
            ctname=f"Battery_Discharge_Limit_{t}"
        )

# Prevent Charging at Max Capacity
for t in range(1, len(T)):
    model.add_constraint(
        P_br[t] <= (E_b_max - E_b[t-1]) ,
        ctname=f"Charge_Capacity_Limit_{t}"
    )

# Mutual Exclusivity for Charging and Discharging
for t in T:
    model.add_constraint(P_br[t] <= P_br_max * mode[t], ctname=f"Charge_Mode_{t}")
    model.add_constraint(P_bi[t] <= P_bi_max * (1 - mode[t]), ctname=f"Discharge_Mode_{t}")

# Allow selling only if profitable
sell_allowed = [1 if R_sell[t] > C_grid[t] + 0.08 else 0 for t in T]

for t in T:
    # Mutually exclusive grid buy or sell
    model.add_constraint(P_grid[t] <= grid_mode[t] * P_grid_max, f"grid_buy_limit_{t}")
    model.add_constraint(P_sell[t] <= (1 - grid_mode[t]), f"grid_sell_limit_{t}")

    # Selling power constraints
    model.add_constraint(P_sell[t] <= E_b[t], f"sell_less_than_Eb_{t}")
    model.add_constraint(P_sell[t] <= sell_allowed[t] * P_sell_max, f"sell_only_if_profitable_{t}")

#--- Constraint (4): Status change equals connection/disconnection actions ---
for j in CELs:
    for t in range(1, len(T)):
        model.add_constraint(
            X[j,t] - X[j,t-1] == Alpha[j ,t] - Beta[j ,t],
            f'status_change_{j}_{t}'
        )

#--- Constraint (5): Initial Condition ---
for j in CELs:
    model.add_constraint(X[j, 0] == Alpha[j, 0] - Beta[j, 0], ctname=f"Initial_{j}")

# --- Constraint (6): Each device is connected exactly once in its allowed window ---
for j in CELs:
    allowed_window = range(Tcli[j], min(Tclf[j] + 1, T_f))
    model.add_constraint(
        model.sum(Alpha[j, t] for t in allowed_window) == 1,
        ctname=f"Connect_once_{j}"
    )

# --- Constraint (7): Each device is disconnected exactly once in its allowed window ---
for j in CELs:
    allowed_window = range(Tcli[j], min(Tclf[j] + 1, T_f))
    model.add_constraint(
        model.sum(Beta[j, t] for t in allowed_window) == 1,
        ctname=f"Disconnect_once_{j}"
    )

# --- Constraint (8): Device operates for its required duration ---
for j in CELs:
    allowed_window = range(Tcli[j], min(Tclf[j] + 1, T_f))
    model.add_constraint(
        model.sum(X[j, t] for t in allowed_window) == Duration[j],
        ctname=f"Duration_{j}"
    )

# --- Constraint (9): Device cannot operate outside its allowed window ---
for j in CELs:
    model.add_constraint(
        model.sum(X[j, t] for t in T if t < Tcli[j] or t > Tclf[j]) == 0,
        ctname=f"OutsideWindow_{j}"
    )

# --- Constraint (11): Completion status cannot be active if device is running ---
for j in CELs:
    for t in T:
        model.add_constraint(
            Delta[j, t] <= 1 - X[j, t],
            ctname=f"Delta_Limit_{j}_{t}"
        )

# --- Constraint (12): Completion status propagates forward unless disconnected ---
for j in CELs:
    for t in T:
        if t >= 1:  # Avoid t=0 (no t-1)
            model.add_constraint(
                Delta[j, t] >= Delta[j, t-1] + Beta[j, t],
                ctname=f"Delta_Propagation_{j}_{t}"
            )

# --- Constraint (13): Completion status persists unless disconnected ---
for j in CELs:
    for t in T:
        if t < T_f - 1:  # Ensure t+1 is within T
            model.add_constraint(
                Delta[j, t+1] <= Delta[j, t] + (1 - Beta[j, t]),
                ctname=f"Delta_Persistence_{j}_{t}"
            )

# --- Constraint (14): Precedence constraints (device j can start only after i completes) ---
# Map Phi matrix indices to device names (e.g., Phi[1,0] = Dryer depends on Washing Machine)
for j in CELs:
    for i in CELs:
        for t in T:  # Iterate over time periods
            if Phi[CELs.index(j), CELs.index(i)] == 1:  # Apply constraint only when Phi[j, i] is 1
                model.add_constraint(
                    Alpha[j, t] <= Delta[i, t] + (1 - Phi[CELs.index(j), CELs.index(i)]),  # Use correct indices
                    ctname=f'connection_precedence_{j}_{i}_{t}'  # More descriptive constraint name
            )

# --- Constraint (15): Max CELs connected simultaneously ---
for t in T:
    model.add_constraint(
        model.sum(X[j, t] for j in CELs) <= lambdaa,
        ctname=f"Max_Simultaneous_CELs_{t}"
    )

# --- Constraint (16): Max total load from CELs ---
for t in T:
    model.add_constraint(
        model.sum(X[j, t] * Pcl[j] for j in CELs) <= gamma,
        ctname=f"Max_Total_Load_{t}"
    )

for t in T:
  model.add_constraint(Ptevb[t] <= Pev_max, f'EV_powerlimit{t}')
model.add_constraint(EevTf == model.sum(Ptevb[t] for t in T), "EV_final_energy")
model.add_constraint(EevTf == ρ * Eev_max, "EV_battery_limit")

# Solve Model
solution = model.solve()

if solution:
    # Display the full scheduling DataFrame
    print("\n📊 **Complete Decision Variable & Parameter Table** 📊")
    # Display Results
    print("\n📈 **Optimization Results** 📈")
    print(f"Combined Objective Value (Weighted Sum): {model.objective_value:.4f}")
    print("\n💰 **Energy Cost Breakdown** 💰")
    # Evaluate the linear expression to get its numerical value
    energy_cost_value = energy_cost_objective.solution_value
    print(f"Total Energy Cost (Before Normalization): {energy_cost_value:.2f} cost units")
    E_cost_norm_value = E_cost_norm.solution_value  # Get the numerical value
    print(f"Normalized Energy Cost: {E_cost_norm_value:.4f}")  # Format the numerical value
    print("\n⏳ **Discomfort (Waiting Time) Breakdown** ⏳")
    # Evaluate the linear expression to get its numerical value
    discomfort_objective_value = discomfort_objective.solution_value
    print(f"Total Discomfort (Before Normalization): {discomfort_objective_value:.2f}")
    D_norm_value = D_norm.solution_value  # Get the numerical value
    print(f"Normalized Discomfort: {D_norm_value:.2f}")  # Format the numerical value
    print("\n⚖️ **Contribution to Objective** ⚖️")
    print(f"Energy Cost Contribution (w1 * Z_cost_norm): {w1 * E_cost_norm_value:.4f}")
    print(f"Discomfort Contribution (w2 * D_norm): {w2 * D_norm_value:.2f}")
else:
    print("❌ No feasible solution found.")

# Create DataFrame for Appliance Schedule
appliances_schedule = []

for t in T:
    row = {"Time Period": t}

    # Add Binary Decision Variables for each appliance
    for j in CELs:
        row[f"X_{j}"] = X[j, t].solution_value
        row[f"Alpha_{j}"] = Alpha[j, t].solution_value
        row[f"Beta_{j}"] = Beta[j, t].solution_value
        row[f"Delta_{j}"] = Delta[j, t].solution_value

    appliances_schedule.append(row)

# Convert to DataFrame and drop index column
df_schedule = pd.DataFrame(appliances_schedule).reset_index(drop=True)

from docplex.mp.conflict_refiner import ConflictRefiner

refiner = ConflictRefiner()
conflicts = refiner.refine_conflict(model)
conflicts.display()

# Create DataFrame for Energy and Cost Data
energy_cost_data = []

for t in T:
    row = {"Time Period": t}

    # Supply
    row["C_re"] = C_re[t]
    row["C_grid"] = C_grid[t]
    row["R_sell"] = R_sell[t]
    row["P_re"] = P_re[t]
    row["Pgrid"] = P_grid[t].solution_value
    row["P_bi"] = P_bi[t].solution_value
    row["=="] = "=="

    # Demand
    row["P_br"] = P_br[t].solution_value
    row["P_sell"] = P_sell[t].solution_value
    row["E_b"] = round(E_b[t].solution_value, 2)
    row["Pncl"] = Pncl[t]
    row["P_evb"] = Ptevb[t].solution_value
    row["Pclr"] = Pclr[t].solution_value
    # Add Total CEL Consumption
    row["Pcl_total"] = sum(Pcl[j] * X[j, t].solution_value for j in CELs)

    energy_cost_data.append(row)

# Convert to DataFrame and drop index column
df_energy_cost = pd.DataFrame(energy_cost_data).reset_index(drop=True)

pd.set_option('display.max_columns', None)  # Display all columns
df_schedule

pd.set_option('display.max_columns', None)  # Display all columns
df_energy_cost

print(EevTf.solution_value)

from docplex.mp.conflict_refiner import ConflictRefiner

refiner = ConflictRefiner()
conflicts = refiner.refine_conflict(model)
conflicts.display()

df_schedule.to_csv('Appaliances_Scheduling.csv', index=False)

df_energy_cost.to_csv('Energy_Cost.csv', index=False)

# # Create a DataFrame to store all variables
# schedule_data = []

# for t in T:
#   row = {"Time Period": t}

#   # Supply
#   row["C_re"]= C_re[t]
#   row["C_grid"]= C_grid[t]
#   row["P_re"] = P_re[t]
#   row["Pgrid"] = P_grid[t].solution_value
#   row["P_bi"] = P_bi[t].solution_value
#   row["=="] = "=="

#   # Demand
#   row["P_br"] = P_br[t].solution_value
#   row["E_b"] = round(E_b[t].solution_value,2)
#   row["Pncl"] = Pncl[t]
#   row["P_evb"] = Ptevb[t].solution_value
#   # Add Total CEL Consumption
#   row["Pcl_total"] = sum(Pcl[j] * X[j, t].solution_value for j in CELs)


#   # Add Binary Decision Variables for each appliance
#   row["||"] = "||"
#   for j in CELs:
#       row[f"X_{j}"] = X[j, t].solution_value
#       row[f"Alpha_{j}"] = Alpha[j, t].solution_value
#       row[f"Beta_{j}"] = Beta[j, t].solution_value
#       row[f"Delta_{j}"] = Delta[j, t].solution_value

#   schedule_data.append(row)

# # Convert to DataFrame
# df_schedule = pd.DataFrame(schedule_data)

