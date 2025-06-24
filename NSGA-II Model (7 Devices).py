# Import libraries
import numpy as np
import pandas as pd
import random
from deap import algorithms, base, creator, tools
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Problem Parameters
T = range(24)  # 24-hour scheduling period

# MODIFIED: Reduced to 8 devices only
CELs = ["Washing Machine", "Dryer", "Dishwasher", "Oven", "Rice Cooker", "Kettle", "Water Heater", "AC"]

T_i = 0
T_f = 23

# Renewable Energy Generation and Costs
P_sun = np.array([0.0, 0.0, 0.0,0.0,0.0,0.0,0.1,0.2,0.3,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.3,0.2,0.1,0.0,0.0,0.0,0.0,0.0])  # 24 hours
P_wind = np.array([0.3, 0.3, 0.3,0.3,0.3,0.3,0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.3,0.3,0.3,0.3,0.3,0.3])  # 24 hours
P_re = P_sun + P_wind
C_re = np.array([0.15, 0.20] * 12)  # 24 hours

# Grid Cost and Selling Price
C_grid = np.array([0.25, 0.20, 0.18, 0.15, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.07, 0.10, 0.13, 0.16, 0.18, 0.20, 0.22, 0.25, 0.23, 0.21, 0.19, 0.17])
R_sell = np.array([0.20, 0.15, 0.25, 0.05, 0.20, 0.06, 0.08, 0.10, 0.20, 0.10, 0.08, 0.20, 0.15, 0.12, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.18, 0.16, 0.14, 0.12])

# Non-controllable Load
Pncl = np.array([1.5, 1.2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.2, 0.1, 0.5, 0.8, 1.0, 1.3, 1.5, 1.7, 1.6, 1.4, 1.2, 1.0])

# Device parameters
Tcli = {
    "Washing Machine": 1, "Dryer": 5, 
    "Dishwasher": 7, "Oven": 9 ,
    "Rice Cooker": 2, "Kettle": 3,
    "Water Heater": 4, "AC": 6
    }

Tclf = {
    "Washing Machine": 8, "Dryer": 11, 
    "Dishwasher": 10, "Oven": 13,
    "Rice Cooker": 7, "Kettle": 9,
    "Water Heater": 12, "AC": 15
    }

Pcl = {
    "Washing Machine": 1.0, "Dryer": 0.8, 
    "Dishwasher": 0.6, "Oven": 1.2 ,
    "Rice Cooker": 0.5, "Kettle": 0.3,
    "Water Heater": 1.5, "AC": 2.0
    }

Duration = {
    "Washing Machine": 3, "Dryer": 1, 
    "Dishwasher": 2, "Oven": 1 ,
    "Rice Cooker": 1, "Kettle": 1,
    "Water Heater": 2, "AC": 3
    }

Precedence = {
    "Dryer": "Washing Machine",
    "Dishwasher": "Oven"
    }


# System Constants
lambdaa = 2  # MODIFIED: Max CELs connected simultaneously (reduced from 5 to 4)
gamma = 20   # Max load from CELs

# Battery Parameters
E_b_init = 5.0
P_br_max = 5.0
P_bi_max = 5.0
E_b_max = 30.0

# EV Parameters
Eev_max = 20.0
rho = 1  # Required battery charge percentage
EV_CHARGING_RATE = 2.0  # Fixed charging rate in kW

# Grid limits
P_grid_max = 20
P_sell_max = 4.0

# SmartHomeScheduler Class
class SmartHomeScheduler:
    def __init__(self):
        self.setup_deap()

    def setup_deap(self):
        """Setup DEAP framework for NSGA-II"""
        # Create fitness class with feasibility flag
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0), crowding_dist=2.0, feasible=True)
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        self.toolbox = base.Toolbox()

        # Device start times
        for device in CELs:
            self.toolbox.register(f"start_time_{device}", random.randint, Tcli[device], Tclf[device] - Duration[device] + 1)

        # Battery mode: discrete (0=discharge, 0.5=idle, 1=charge)
        self.toolbox.register("battery_mode", random.choice, [0.0, 0.5, 1.0])

        # EV charging start time
        self.toolbox.register("ev_start", random.randint, 0, 23)

        # MODIFIED: Create individual: 4 device starts + 24 battery modes + 1 EV start = 29 genes
        individual_components = tuple(self.toolbox.__getattribute__(f"start_time_{device}") for device in CELs) + \
                               tuple([self.toolbox.battery_mode] * 24) + \
                               (self.toolbox.ev_start,)
        self.toolbox.register("individual", tools.initCycle, creator.Individual, individual_components, n=1)

        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate_individual)
        self.toolbox.register("mate", self.custom_crossover)
        self.toolbox.register("mutate", self.custom_mutation)
        self.toolbox.register("select", tools.selNSGA2)  # Use default NSGA-II selection

        # Disable multiprocessing, use default map
        self.toolbox.register("map", map)

    def decode_individual(self, individual):
        """Decode individual chromosome into schedule variables"""
        device_starts = {device: int(individual[i]) for i, device in enumerate(CELs)}

        # Repair precedence
        for dependent, prerequisite in Precedence.items():
            prereq_end = device_starts[prerequisite] + Duration[prerequisite]
            device_starts[dependent] = max(prereq_end, min(device_starts[dependent], Tclf[dependent] - Duration[dependent] + 1))

        # Battery modes
        battery_modes = np.array(individual[len(CELs):len(CELs) + 24])

        # EV start time
        ev_start = int(individual[-1])

        return device_starts, battery_modes, ev_start

    def create_device_schedule(self, device_starts):
        """Create binary schedule matrix for devices"""
        X = {j: np.zeros(24, dtype=int) for j in CELs}

        for device in CELs:
            start_time = device_starts[device]
            duration = Duration[device]

            for t in range(start_time, min(start_time + duration, 24)):
                if Tcli[device] <= t <= Tclf[device]:
                    X[device][t] = 1

        return X

    def create_ev_schedule(self, ev_start):
        """Create EV charging schedule with fixed rate until target energy is reached"""
        Ptevb = np.zeros(24)
        required_energy = rho * Eev_max  # 20 kWh
        charging_rate = EV_CHARGING_RATE  # 2.0 kW

        total_charged = 0.0
        for t in range(ev_start, 24):
            if total_charged < required_energy:
                charge = min(charging_rate, required_energy - total_charged)
                Ptevb[t] = charge
                total_charged += charge

        # Normalize to exactly required_energy if total_charged is close
        if total_charged > 0 and abs(total_charged - required_energy) > 0.01:
            Ptevb[ev_start:24] *= required_energy / total_charged

        return Ptevb

    def solve_energy_balance(self, X, battery_modes, ev_start):
        """Solve energy balance without battery efficiency"""
        P_br = np.zeros(24)  # Battery charging
        P_bi = np.zeros(24)  # Battery discharging
        P_grid = np.zeros(24)  # Grid power
        P_sell = np.zeros(24)  # Selling power
        E_b = np.zeros(24)   # Battery energy
        Ptevb = self.create_ev_schedule(ev_start)

        E_b[0] = E_b_init

        for t in range(24):
            # Controllable load
            Pclr = sum(Pcl[j] * X[j][t] for j in CELs)

            # Total demand
            demand = Ptevb[t] + Pclr + Pncl[t]

            # Renewable energy
            renewable_available = P_re[t]
            net_demand = demand - renewable_available

            # Battery action
            if battery_modes[t] == 1.0:  # Charge
                if net_demand < 0:  # Excess renewable
                    charge_amount = min(-net_demand, P_br_max, E_b_max - E_b[t])
                    P_br[t] = charge_amount
                    net_demand += charge_amount
                elif C_grid[t] < 0.1:  # Cheap grid
                    charge_amount = min(P_br_max, E_b_max - E_b[t])
                    P_br[t] = charge_amount
                    net_demand += charge_amount

            elif battery_modes[t] == 0.0:  # Discharge
                if net_demand > 0 and C_grid[t] > 0.15:  # Expensive grid
                    discharge_amount = min(net_demand, P_bi_max, E_b[t])
                    P_bi[t] = discharge_amount
                    net_demand -= discharge_amount

            # Grid interaction
            if net_demand > 0:
                P_grid[t] = min(net_demand, P_grid_max)
            elif net_demand < 0 and R_sell[t] > C_grid[t] * 0.8:
                P_sell[t] = min(-net_demand, P_sell_max, E_b[t])

            # Update battery
            if t < 23:
                E_b[t+1] = E_b[t] + P_br[t] - P_bi[t] - P_sell[t]
                E_b[t+1] = np.clip(E_b[t+1], 0, E_b_max)

        return P_br, P_bi, P_grid, P_sell, E_b, Ptevb

    def check_constraints(self, X, device_starts, P_br, P_bi, P_grid, P_sell, E_b, Ptevb):
        """Check constraints and return penalty and feasibility"""
        penalty = 0
        feasible = True

        # Device scheduling
        for device in CELs:
            actual_duration = np.sum(X[device])
            if actual_duration != Duration[device]:
                penalty += abs(actual_duration - Duration[device]) * 1000
                feasible = False

            for t in range(24):
                if X[device][t] == 1 and (t < Tcli[device] or t > Tclf[device]):
                    penalty += 1000
                    feasible = False

        # Precedence
        for dependent, prerequisite in Precedence.items():
            if device_starts[dependent] < device_starts[prerequisite] + Duration[prerequisite]:
                penalty += 500
                feasible = False

        # System constraints
        for t in range(24):
            active_devices = sum(X[j][t] for j in CELs)
            if active_devices > lambdaa:
                penalty += (active_devices - lambdaa) * 200
                feasible = False

            total_cel_load = sum(Pcl[j] * X[j][t] for j in CELs)
            if total_cel_load > gamma:
                penalty += (total_cel_load - gamma) * 100
                feasible = False

            if E_b[t] < 0 or E_b[t] > E_b_max:
                penalty += abs(E_b[t] - np.clip(E_b[t], 0, E_b_max)) * 500
                feasible = False

            if P_br[t] > P_br_max or P_bi[t] > P_bi_max:
                penalty += (max(P_br[t] - P_br_max, 0) + max(P_bi[t] - P_bi_max, 0)) * 200
                feasible = False

            if P_grid[t] > P_grid_max or P_sell[t] > P_sell_max:
                penalty += (max(P_grid[t] - P_grid_max, 0) + max(P_sell[t] - P_sell_max, 0)) * 200
                feasible = False

        # EV constraint
        total_ev_energy = np.sum(Ptevb)
        required_ev_energy = rho * Eev_max
        if abs(total_ev_energy - required_ev_energy) > 0.01:
            penalty += abs(total_ev_energy - required_ev_energy) * 100
            feasible = False

        # Energy balance
        for t in range(24):
            supply = P_re[t] + P_grid[t] + P_bi[t]
            demand = P_br[t] + Ptevb[t] + sum(Pcl[j] * X[j][t] for j in CELs) + Pncl[t] + P_sell[t]
            balance_error = abs(supply - demand)
            if balance_error > 0.01:
                penalty += balance_error * 1000
                feasible = False

        return penalty, feasible

    def calculate_objectives(self, P_grid, P_sell, device_starts):
        """Calculate objectives"""
        grid_cost = np.sum(C_grid * P_grid)
        renewable_cost = np.sum(C_re * P_re)
        selling_revenue = np.sum(R_sell * P_sell)
        energy_cost = grid_cost + renewable_cost - selling_revenue

        discomfort = sum((device_starts[j] - Tcli[j]) / max(1, Tclf[j] - Tcli[j]) for j in CELs)

        return energy_cost, discomfort

    def evaluate_individual(self, individual):
        """Evaluate individual"""
        try:
            device_starts, battery_modes, ev_start = self.decode_individual(individual)
            X = self.create_device_schedule(device_starts)
            P_br, P_bi, P_grid, P_sell, E_b, Ptevb = self.solve_energy_balance(X, battery_modes, ev_start)

            penalty, feasible = self.check_constraints(X, device_starts, P_br, P_bi, P_grid, P_sell, E_b, Ptevb)
            energy_cost, discomfort = self.calculate_objectives(P_grid, P_sell, device_starts)

            return (energy_cost + penalty / 1000, discomfort + penalty / 1000, feasible)

        except Exception as e:
            print(f"Error: Evaluation error: {e}")
            return (10000, 10000, False)

    def custom_crossover(self, ind1, ind2):
        """Custom crossover"""
        # Device start times
        for i in range(len(CELs)):
            if random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]

        # Battery modes
        for i in range(len(CELs), len(CELs) + 24):
            if random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]

        # EV start
        if random.random() < 0.5:
            ind1[-1], ind2[-1] = ind2[-1], ind1[-1]

        return ind1, ind2

    def custom_mutation(self, individual, indpb):
        """Custom mutation with repair"""
        for i, device in enumerate(CELs):
            if random.random() < indpb:
                max_start = Tclf[device] - Duration[device] + 1
                individual[i] = random.randint(Tcli[device], max_start)

        # Repair precedence
        for dependent, prerequisite in Precedence.items():
            prereq_idx = CELs.index(prerequisite)
            dep_idx = CELs.index(dependent)
            prereq_end = individual[prereq_idx] + Duration[prerequisite]
            individual[dep_idx] = max(prereq_end, min(individual[dep_idx], Tclf[dependent] - Duration[dependent] + 1))

        # Battery modes
        for i in range(len(CELs), len(CELs) + 24):
            if random.random() < indpb:
                individual[i] = random.choice([0.0, 0.5, 1.0])

        # EV start
        if random.random() < indpb:
            individual[-1] = random.randint(0, 23)

        return individual,

    def run_optimization(self, pop_size=100, generations=50):
        """Run NSGA-II optimization"""
        print(f"Starting NSGA-II Optimization: pop_size={pop_size}, generations={generations}")

        pop = self.toolbox.population(n=pop_size)

        print("Evaluating initial population...")
        fitnesses = self.toolbox.map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = (fit[0], fit[1])
            ind.fitness.feasible = fit[2]

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("min", np.min, axis=0)

        print("Starting evolution...")

        for gen in range(generations):
            print(f"Generation {gen+1}/{generations}")
            # Adaptive mutation probability
            indpb = 0.3 * (1 - gen / generations) + 0.05

            offspring = self.toolbox.select(pop, len(pop))
            offspring = [self.toolbox.clone(ind) for ind in offspring]

            for i in range(1, len(offspring), 2):
                if random.random() < 0.7:
                    self.toolbox.mate(offspring[i-1], offspring[i])
                    del offspring[i-1].fitness.values
                    del offspring[i].fitness.values

            for i in range(len(offspring)):
                if random.random() < 0.5:
                    self.toolbox.mutate(offspring[i], indpb=indpb)
                    del offspring[i].fitness.values

            invalid_ind = [ind for ind in offspring if not hasattr(ind.fitness, 'values') or not ind.fitness.values]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = (fit[0], fit[1])
                ind.fitness.feasible = fit[2]

            pop = self.toolbox.select(pop + offspring, pop_size)

            fits = [ind.fitness.values for ind in pop]
            if fits:
                avg_cost, avg_discomfort = np.mean(fits, axis=0)
                min_cost, min_discomfort = np.min(fits, axis=0)
                print(f"  Avg Cost: {avg_cost:.2f}, Min Cost: {min_cost:.2f}")
                print(f"  Avg Discomfort: {avg_discomfort:.2f}, Min Discomfort: {min_discomfort:.2f}")

        pareto_front = tools.selNSGA2(pop, len(pop))
        pareto_front = [ind for ind in pareto_front if ind.fitness.feasible]

        return pareto_front

    def analyze_solution(self, individual, solution_num):
        """Analyze solution"""
        device_starts, battery_modes, ev_start = self.decode_individual(individual)
        X = self.create_device_schedule(device_starts)
        P_br, P_bi, P_grid, P_sell, E_b, Ptevb = self.solve_energy_balance(X, battery_modes, ev_start)

        print(f"\n--- Solution {solution_num} Analysis ---")
        print("Device Schedule:")
        for device in CELs:
            start_time = device_starts[device]
            duration = Duration[device]
            end_time = start_time + duration - 1
            print(f"  {device}: Hours {start_time}-{end_time} (Duration: {duration}h)")

        energy_cost, discomfort = self.calculate_objectives(P_grid, P_sell, device_starts)
        total_ev_energy = np.sum(Ptevb)
        print(f"EV Start Hour: {ev_start}")
        print(f"EV Energy Charged: {total_ev_energy:.2f} kWh")
        print(f"Final Battery Level: {E_b[-1]:.2f} kWh")
        print(f"Energy Cost: {energy_cost:.2f}")
        print(f"Discomfort: {discomfort:.2f}")

        return energy_cost, discomfort

def plot_pareto_front(costs, discomforts):
        """Plot Pareto front with solution labels."""
        plt.figure(figsize=(10, 8))
        plt.scatter(costs, discomforts, c='red', s=100, alpha=0.7, edgecolors='black', linewidth=1)
        
        # Add solution numbers as labels
        for i, (cost, discomfort) in enumerate(zip(costs, discomforts), 1):
            plt.annotate(f'S{i}', (cost, discomfort), xytext=(5, 5), 
                        textcoords='offset points', fontsize=9, alpha=0.8)
        
        plt.xlabel('Energy Cost', fontsize=12)
        plt.ylabel('User Discomfort', fontsize=12)
        plt.title('Pareto Front: Energy Cost vs User Discomfort\n(7 Devices)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Add some styling
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        plt.show()

def plot_solution_distributions(energy_costs, discomforts, solutions, CELs):
    """Plot distributions for energy cost, discomfort, and appliance start times."""
    plt.figure(figsize=(16, 10))

    # Energy cost distribution
    plt.subplot(2, 2, 2)
    plt.hist(energy_costs, bins=min(10, len(energy_costs)), alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Energy Cost')
    plt.ylabel('Frequency')
    plt.title('Energy Cost Distribution')
    plt.grid(True, alpha=0.3)

    # Discomfort distribution
    plt.subplot(2, 2, 3)
    plt.hist(discomforts, bins=min(10, len(discomforts)), alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Discomfort')
    plt.ylabel('Frequency')
    plt.title('Discomfort Distribution')
    plt.grid(True, alpha=0.3)

    # Start times distribution
    plt.subplot(2, 2, 4)
    start_times_data = np.array(solutions)
    for i, device in enumerate(CELs):
        plt.hist(start_times_data[:, i], bins=range(0, 25), alpha=0.5, 
                 label=device, edgecolor='black')
    plt.xlabel('Start Time (Hour)')
    plt.ylabel('Frequency')
    plt.title('Appliance Start Times Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    

# Main execution
if __name__ == "__main__":
    scheduler = SmartHomeScheduler()
    pareto_front = scheduler.run_optimization(pop_size=100, generations=200)

    # MODIFIED: Print ALL Pareto front solutions
    print(f"\n**All Pareto Front Solutions ({len(pareto_front)} solutions):**\n")
    
    costs = []
    discomforts = []
    
    for i, ind in enumerate(pareto_front, 1):
        energy_cost, discomfort = scheduler.analyze_solution(ind, i)
        costs.append(energy_cost)
        discomforts.append(discomfort)
        print("-" * 50)

    # Call plot functions
    plot_pareto_front(costs, discomforts)
    #plot_solution_distributions(costs, discomforts, pareto_front, CELs)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame({
        'Solution': [f'S{i}' for i in range(1, len(pareto_front) + 1)],
        'Energy_Cost': costs,
        'User_Discomfort': discomforts
    })
    
    print(f"\n**Pareto Front Summary:**")
    print(summary_df.to_string(index=False))
    
    # Save summary to CSV
    # summary_df.to_csv('pareto_front_solutions.csv', index=False)
    # print(f"\nPareto front solutions saved to 'pareto_front_solutions.csv'")
    
    # Get the best solution (lowest energy cost) for detailed analysis
    best_idx = costs.index(min(costs))
    best_ind = pareto_front[best_idx]
    
    device_starts, battery_modes, ev_start = scheduler.decode_individual(best_ind)
    X = scheduler.create_device_schedule(device_starts)
    P_br, P_bi, P_grid, P_sell, E_b, Ptevb = scheduler.solve_energy_balance(X, battery_modes, ev_start)

    # Create detailed analysis DataFrame for best solution
    analysis_data = {
        'Hour': range(24),
        'Psun': P_sun,
        'Pwind': P_wind,
        'C_re': C_re,
        'P_grid': P_grid,
        'C_grid': C_grid,
        'P_sell': P_sell,
        'R_sell': R_sell,
        'P_bi': P_bi,
        'P_br': P_br,
        'E_battery': E_b,
        'Ptevb': Ptevb,
        'Pncl': Pncl
    }
    
    # Add total device consumption (P_cl) for each hour
    analysis_data['P_cl'] = [sum(Pcl[device] * X[device][t] for device in CELs) for t in range(24)]
    
    # Add devices running at each hour
    analysis_data['Devices_On'] = [
        ', '.join([device for device in CELs if X[device][t] == 1]) for t in range(24)
    ]
    
    analysis_df = pd.DataFrame(analysis_data)
    # analysis_df.to_csv('best_solution_detailed_analysis.csv', index=False)
    # print(f"Best solution detailed analysis saved to 'best_solution_detailed_analysis.csv'")