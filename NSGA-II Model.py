import numpy as np
import pandas as pd
import random
from deap import base, creator, tools
import matplotlib.pyplot as plt
import warnings
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

class Device:
    def __init__(self, name: str, start_time: int, end_time: int, power: float, duration: int):
        self.name = name
        self.start_time = start_time
        self.end_time = end_time
        self.power = power
        self.duration = duration

class Scenario:
    def __init__(self, num_devices: int, precedences: Dict[str, str]):
        self.num_devices = num_devices
        self.precedences = precedences
        self.devices = self._generate_devices()

    def _generate_devices(self) -> List[Device]:
        """Generate devices based on the number of devices in the scenario."""
        base_devices = [
            Device("Washing Machine", 1, 8, 3.5, 3),
            Device("Dryer", 5, 11, 1.3, 1),
            Device("Dishwasher", 17, 23, 0.9, 2),
            Device("Oven", 15, 20, 3.0, 2),
            Device("Rice Cooker", 12, 16, 0.7, 1),
            Device("Kettle", 6, 12, 0.9, 1),
            Device("Heater", 2, 7, 2.0, 3),
            Device("Air Conditioner", 12, 18, 4.0, 4),
            Device("Vacuum Cleaner", 18, 23, 1.0, 1),
            Device("Smart fan", 12, 23, 0.8, 6),
            Device("Pool Pump", 6, 12, 2.5, 4),
            Device("Water Heater", 7, 12, 1.25, 2)
        ]
        return base_devices[:self.num_devices]

class EnergySystem:
    def __init__(self):
        self.T = range(24)
        self.T_i = 0
        self.T_f = 23
        self.lambdaa = 4
        self.gamma = 20
        self.P_grid_max = 20
        self.P_sell_max = 4.0
        self.initialize_energy_data()

    def initialize_energy_data(self):
        P_sun_base = np.array([0.0] * 6 + [0.1, 0.2, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.3, 0.2, 0.1] + [0.0] * 5)
        P_wind_base = np.array([0.3] * 6 + [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3] + [0.3] * 5)
        C_re_base = np.array([0.18] * 6 + [0.10, 0.08, 0.07, 0.06, 0.06, 0.06, 0.06, 0.06, 0.07, 0.08, 0.10, 0.12, 0.14, 0.16, 0.17, 0.18, 0.18, 0.18])
        C_grid_base = np.array([0.25, 0.20, 0.18, 0.15, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.07, 0.10, 0.13, 0.16, 0.18, 0.20, 0.22, 0.25, 0.23, 0.21, 0.19, 0.17])
        R_sell_base = np.array([0.20, 0.15, 0.25, 0.05, 0.20, 0.06, 0.08, 0.10, 0.20, 0.10, 0.08, 0.20, 0.15, 0.12, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.18, 0.16, 0.14, 0.12])
        Pncl_base = np.array([1.5, 1.2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.2, 0.1, 0.5, 0.8, 1.0, 1.3, 1.5, 1.7, 1.6, 1.4, 1.2, 1.0])

        USE_STOCHASTIC_INPUTS = False
        if USE_STOCHASTIC_INPUTS:
            self.P_sun = np.clip(P_sun_base + np.random.normal(0, 0.03, size=24), 0, 1)
            self.P_wind = np.clip(P_wind_base + np.random.normal(0, 0.03, size=24), 0, 1)
            self.C_grid = np.clip(C_grid_base * (1 + np.random.uniform(-0.1, 0.1, size=24)), 0.01, 1.0)
            self.R_sell = np.clip(R_sell_base * (1 + np.random.uniform(-0.1, 0.1, size=24)), 0.01, 1.0)
            self.Pncl = np.clip(Pncl_base * (1 + np.random.normal(0, 0.1, size=24)), 0.1, 5.0)
            self.C_re = np.clip(C_re_base * (1 + np.random.uniform(-0.1, 0.1, size=24)), 0.01, 1.0)
        else:
            self.P_sun = P_sun_base
            self.P_wind = P_wind_base
            self.C_grid = C_grid_base
            self.R_sell = R_sell_base
            self.Pncl = Pncl_base
            self.C_re = C_re_base

        self.P_re = self.P_sun + self.P_wind

class Battery:
    def __init__(self, initial_energy: float = 5.0, max_energy: float = 30.0, max_charge_rate: float = 5.0, max_discharge_rate: float = 5.0):
        self.E_b_init = initial_energy
        self.E_b_max = max_energy
        self.P_br_max = max_charge_rate
        self.P_bi_max = max_discharge_rate

class ElectricVehicle:
    def __init__(self, max_energy: float = 20.0, charging_rate: float = 2.0, required_charge_percentage: float = 1.0):
        self.Eev_max = max_energy
        self.charging_rate = charging_rate
        self.rho = required_charge_percentage

class SmartHomeScheduler:
    def __init__(self, scenario: Scenario, energy_system: EnergySystem, battery: Battery, ev: ElectricVehicle):
        self.devices = scenario.devices
        self.precedence = scenario.precedences
        self.energy_system = energy_system
        self.battery = battery
        self.ev = ev
        self.toolbox = base.Toolbox()
        self.setup_deap()

    def setup_deap(self):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0), crowding_dist=2.0, feasible=True)
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        for device in self.devices:
            self.toolbox.register(
                f"start_time_{device.name}",
                random.randint,
                device.start_time,
                device.end_time - device.duration + 1
            )

        self.toolbox.register("battery_mode", random.choice, [0.0, 0.5, 1.0])
        self.toolbox.register("ev_start", random.randint, 0, 23)

        individual_components = (
            tuple(self.toolbox.__getattribute__(f"start_time_{device.name}") for device in self.devices) +
            tuple([self.toolbox.battery_mode] * 24) +
            (self.toolbox.ev_start,)
        )

        self.toolbox.register("individual", tools.initCycle, creator.Individual, individual_components, n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate_individual)
        self.toolbox.register("mate", self.custom_crossover)
        self.toolbox.register("mutate", self.custom_mutation)
        self.toolbox.register("select", tools.selNSGA2)
        self.toolbox.register("map", map)

    def decode_individual(self, individual: List[float]) -> Tuple[Dict[str, int], np.ndarray, int]:
        device_starts = {device.name: int(individual[i]) for i, device in enumerate(self.devices)}
        for dependent, prerequisite in self.precedence.items():
            prereq_end = device_starts[prerequisite] + next(d.duration for d in self.devices if d.name == prerequisite)
            device_starts[dependent] = max(prereq_end, min(device_starts[dependent], next(d.end_time for d in self.devices if d.name == dependent) - next(d.duration for d in self.devices if d.name == dependent) + 1))
        battery_modes = np.array(individual[len(self.devices):len(self.devices) + 24])
        ev_start = int(individual[-1])
        return device_starts, battery_modes, ev_start

    def create_device_schedule(self, device_starts: Dict[str, int]) -> Dict[str, np.ndarray]:
        X = {device.name: np.zeros(24, dtype=int) for device in self.devices}
        for device in self.devices:
            start_time = device_starts[device.name]
            duration = device.duration
            for t in range(start_time, min(start_time + duration, 24)):
                if device.start_time <= t <= device.end_time:
                    X[device.name][t] = 1
        return X

    def create_ev_schedule(self, ev_start: int) -> np.ndarray:
        Ptevb = np.zeros(24)
        required_energy = self.ev.rho * self.ev.Eev_max
        total_charged = 0.0
        for t in range(ev_start, 24):
            if total_charged < required_energy:
                charge = min(self.ev.charging_rate, required_energy - total_charged)
                Ptevb[t] = charge
                total_charged += charge
        if total_charged > 0 and abs(total_charged - required_energy) > 0.01:
            Ptevb[ev_start:24] *= required_energy / total_charged
        return Ptevb

    def solve_energy_balance(self, X: Dict[str, np.ndarray], battery_modes: np.ndarray, ev_start: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        P_br = np.zeros(24)
        P_bi = np.zeros(24)
        P_grid = np.zeros(24)
        P_sell = np.zeros(24)
        E_b = np.zeros(24)
        Ptevb = self.create_ev_schedule(ev_start)
        E_b[0] = self.battery.E_b_init

        for t in range(24):
            Pclr = sum(device.power * X[device.name][t] for device in self.devices)
            demand = Ptevb[t] + Pclr + self.energy_system.Pncl[t]
            renewable_available = self.energy_system.P_re[t]
            net_demand = demand - renewable_available

            if battery_modes[t] == 1.0:
                if net_demand < 0:
                    charge_amount = min(-net_demand, self.battery.P_br_max, self.battery.E_b_max - E_b[t])
                    P_br[t] = charge_amount
                    net_demand += charge_amount
                elif self.energy_system.C_grid[t] < 0.1:
                    charge_amount = min(self.battery.P_br_max, self.battery.E_b_max - E_b[t])
                    P_br[t] = charge_amount
                    net_demand += charge_amount
            elif battery_modes[t] == 0.0:
                if net_demand > 0 and self.energy_system.C_grid[t] > 0.15:
                    discharge_amount = min(net_demand, self.battery.P_bi_max, E_b[t])
                    P_bi[t] = discharge_amount
                    net_demand -= discharge_amount

            if net_demand > 0:
                P_grid[t] = min(net_demand, self.energy_system.P_grid_max)
            elif net_demand < 0 and self.energy_system.R_sell[t] > self.energy_system.C_grid[t] * 0.8:
                P_sell[t] = min(-net_demand, self.energy_system.P_sell_max, E_b[t])

            if t < 23:
                E_b[t + 1] = E_b[t] + P_br[t] - P_bi[t] - P_sell[t]
                E_b[t + 1] = np.clip(E_b[t + 1], 0, self.battery.E_b_max)

        return P_br, P_bi, P_grid, P_sell, E_b, Ptevb

    def check_constraints(self, X: Dict[str, np.ndarray], device_starts: Dict[str, int], P_br: np.ndarray, P_bi: np.ndarray, P_grid: np.ndarray, P_sell: np.ndarray, E_b: np.ndarray, Ptevb: np.ndarray) -> Tuple[float, bool]:
        penalty = 0
        feasible = True

        for device in self.devices:
            actual_duration = np.sum(X[device.name])
            if actual_duration != device.duration:
                penalty += abs(actual_duration - device.duration) * 1000
                feasible = False
            for t in range(24):
                if X[device.name][t] == 1 and (t < device.start_time or t > device.end_time):
                    penalty += 1000
                    feasible = False

        for dependent, prerequisite in self.precedence.items():
            if device_starts[dependent] < device_starts[prerequisite] + next(d.duration for d in self.devices if d.name == prerequisite):
                penalty += 500
                feasible = False

        for t in range(24):
            active_devices = sum(X[device.name][t] for device in self.devices)
            if active_devices > self.energy_system.lambdaa:
                penalty += (active_devices - self.energy_system.lambdaa) * 200
                feasible = False
            total_cel_load = sum(device.power * X[device.name][t] for device in self.devices)
            if total_cel_load > self.energy_system.gamma:
                penalty += (total_cel_load - self.energy_system.gamma) * 100
                feasible = False
            if E_b[t] < 0 or E_b[t] > self.battery.E_b_max:
                penalty += abs(E_b[t] - np.clip(E_b[t], 0, self.battery.E_b_max)) * 500
                feasible = False
            if P_br[t] > self.battery.P_br_max or P_bi[t] > self.battery.P_bi_max:
                penalty += (max(P_br[t] - self.battery.P_br_max, 0) + max(P_bi[t] - self.battery.P_bi_max, 0)) * 200
                feasible = False
            if P_grid[t] > self.energy_system.P_grid_max or P_sell[t] > self.energy_system.P_sell_max:
                penalty += (max(P_grid[t] - self.energy_system.P_grid_max, 0) + max(P_sell[t] - self.energy_system.P_sell_max, 0)) * 200
                feasible = False

        total_ev_energy = np.sum(Ptevb)
        required_ev_energy = self.ev.rho * self.ev.Eev_max
        if abs(total_ev_energy - required_ev_energy) > 0.01:
            penalty += abs(total_ev_energy - required_ev_energy) * 100
            feasible = False

        for t in range(24):
            supply = self.energy_system.P_re[t] + P_grid[t] + P_bi[t]
            demand = P_br[t] + Ptevb[t] + sum(device.power * X[device.name][t] for device in self.devices) + self.energy_system.Pncl[t] + P_sell[t]
            balance_error = abs(supply - demand)
            if balance_error > 0.01:
                penalty += balance_error * 1000
                feasible = False

        return penalty, feasible

    def calculate_objectives(self, P_grid: np.ndarray, P_sell: np.ndarray, device_starts: Dict[str, int]) -> Tuple[float, float]:
        grid_cost = np.sum(self.energy_system.C_grid * P_grid)
        renewable_cost = np.sum(self.energy_system.C_re * self.energy_system.P_re)
        selling_revenue = np.sum(self.energy_system.R_sell * P_sell)
        energy_cost = grid_cost + renewable_cost - selling_revenue
        discomfort = sum((device_starts[device.name] - device.start_time) / max(1, device.end_time - device.start_time) for device in self.devices)
        return energy_cost, discomfort

    def evaluate_individual(self, individual: List[float]) -> Tuple[float, float, bool]:
        try:
            device_starts, battery_modes, ev_start = self.decode_individual(individual)
            X = self.create_device_schedule(device_starts)
            P_br, P_bi, P_grid, P_sell, E_b, Ptevb = self.solve_energy_balance(X, battery_modes, ev_start)
            penalty, feasible = self.check_constraints(X, device_starts, P_br, P_bi, P_grid, P_sell, E_b, Ptevb)
            energy_cost, discomfort = self.calculate_objectives(P_grid, P_sell, device_starts)
            return energy_cost + penalty / 1000, discomfort + penalty / 1000, feasible
        except Exception as e:
            print(f"Error: Evaluation error: {e}")
            return 10000, 10000, False

    def custom_crossover(self, ind1: List[float], ind2: List[float]) -> Tuple[List[float], List[float]]:
        for i in range(len(self.devices)):
            if random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]
        for i in range(len(self.devices), len(self.devices) + 24):
            if random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]
        if random.random() < 0.5:
            ind1[-1], ind2[-1] = ind2[-1], ind1[-1]
        return ind1, ind2

    def custom_mutation(self, individual: List[float], indpb: float) -> Tuple[List[float]]:
        for i, device in enumerate(self.devices):
            if random.random() < indpb:
                max_start = device.end_time - device.duration + 1
                individual[i] = random.randint(device.start_time, max_start)
        for dependent, prerequisite in self.precedence.items():
            prereq_idx = next(i for i, d in enumerate(self.devices) if d.name == prerequisite)
            dep_idx = next(i for i, d in enumerate(self.devices) if d.name == dependent)
            prereq_end = individual[prereq_idx] + next(d.duration for d in self.devices if d.name == prerequisite)
            individual[dep_idx] = max(prereq_end, min(individual[dep_idx], next(d.end_time for d in self.devices if d.name == dependent) - next(d.duration for d in self.devices if d.name == dependent) + 1))
        for i in range(len(self.devices), len(self.devices) + 24):
            if random.random() < indpb:
                individual[i] = random.choice([0.0, 0.5, 1.0])
        if random.random() < indpb:
            individual[-1] = random.randint(0, 23)
        return (individual,)

    def run_optimization(self, pop_size: int = 100, generations: int = 50) -> List[List[float]]:
        print(f"Starting NSGA-II Optimization: pop_size={pop_size}, generations={generations}")
        pop = self.toolbox.population(n=pop_size)
        print("Evaluating initial population...")
        fitnesses = self.toolbox.map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = (fit[0], fit[1])
            ind.fitness.feasible = fit[2]
        print("Starting evolution...")
        for gen in range(generations):
            print(f"Generation {gen+1}/{generations}")
            indpb = 0.3 * (1 - gen / generations) + 0.05
            offspring = self.toolbox.select(pop, len(pop))
            offspring = [self.toolbox.clone(ind) for ind in offspring]
            for i in range(1, len(offspring), 2):
                if random.random() < 0.7:
                    self.toolbox.mate(offspring[i - 1], offspring[i])
                    del offspring[i - 1].fitness.values
                    del offspring[i].fitness.values
            for i in range(len(offspring)):
                if random.random() < 0.5:
                    self.toolbox.mutate(offspring[i], indpb=indpb)
                    del offspring[i].fitness.values
            invalid_ind = [ind for ind in offspring if not hasattr(ind.fitness, "values") or not ind.fitness.values]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = (fit[0], fit[1])
                ind.fitness.feasible = fit[2]
            pop = self.toolbox.select(pop + offspring, pop_size)
        pareto_front = tools.selNSGA2(pop, len(pop))
        return [ind for ind in pareto_front if ind.fitness.feasible]

    def analyze_solution(self, individual: List[float], solution_num: int) -> Tuple[float, float]:
        device_starts, battery_modes, ev_start = self.decode_individual(individual)
        X = self.create_device_schedule(device_starts)
        P_br, P_bi, P_grid, P_sell, E_b, Ptevb = self.solve_energy_balance(X, battery_modes, ev_start)
        print(f"\n--- Solution {solution_num} Analysis ---")
        print("Device Schedule:")
        for device in self.devices:
            start_time = device_starts[device.name]
            duration = device.duration
            end_time = start_time + duration - 1
            print(f"  {device.name}: Hours {start_time}-{end_time} (Duration: {duration}h)")
        energy_cost, discomfort = self.calculate_objectives(P_grid, P_sell, device_starts)
        total_ev_energy = np.sum(Ptevb)
        print(f"EV Start Hour: {ev_start}")
        print(f"EV Energy Charged: {total_ev_energy:.2f} kWh")
        print(f"Final Battery Level: {E_b[-1]:.2f} kWh")
        print(f"Energy Cost: {energy_cost:.2f}")
        print(f"Discomfort: {discomfort:.2f}")
        return energy_cost, discomfort

def plot_solution_distributions(energy_costs: List[float], discomforts: List[float], solutions: List[List[float]], devices: List[Device]):
    plt.figure(figsize=(16, 10))
    plt.subplot(2, 2, 2)
    plt.hist(energy_costs, bins=min(10, len(energy_costs)), alpha=0.7, color="blue", edgecolor="black")
    plt.xlabel("Energy Cost")
    plt.ylabel("Frequency")
    plt.title("Energy Cost Distribution")
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 2, 3)
    plt.hist(discomforts, bins=min(10, len(discomforts)), alpha=0.7, color="green", edgecolor="black")
    plt.xlabel("Discomfort")
    plt.ylabel("Frequency")
    plt.title("Discomfort Distribution")
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 2, 4)
    start_times_data = np.array(solutions)
    for i, device in enumerate(devices):
        plt.hist(start_times_data[:, i], bins=range(0, 25), alpha=0.5, label=device.name, edgecolor="black")
    plt.xlabel("Start Time (Hour)")
    plt.ylabel("Frequency")
    plt.title("Appliance Start Times Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def remove_duplicate_solutions(pareto_front: List[List[float]]) -> List[List[float]]:
    seen = set()
    unique_solutions = []
    for solution in pareto_front:
        solution_tuple = tuple(solution)
        if solution_tuple not in seen:
            seen.add(solution_tuple)
            unique_solutions.append(solution)
    return unique_solutions

def remove_duplicates_across_pareto_fronts(all_pareto_fronts: List[List[List[float]]]) -> List[List[List[float]]]:
    if not all_pareto_fronts:
        return []
    all_solutions = [ind for pf in all_pareto_fronts for ind in pf]
    all_solution_tuples = {tuple(ind): ind for ind in all_solutions}
    unique_pareto_fronts = []
    for i, pf in enumerate(all_pareto_fronts):
        unique_pf = []
        seen_in_other_fronts = set()
        for j, other_pf in enumerate(all_pareto_fronts):
            if i != j:
                seen_in_other_fronts.update(tuple(ind) for ind in other_pf)
        for ind in pf:
            ind_tuple = tuple(ind)
            if ind_tuple not in seen_in_other_fronts:
                unique_pf.append(ind)
        if unique_pf:
            unique_pareto_fronts.append(unique_pf)
    return unique_pareto_fronts

def remove_dominated_solutions_across_all_pareto_fronts(all_pareto_fronts: List[List[List[float]]]) -> List[List[float]]:
    if not all_pareto_fronts:
        return []
    all_solutions = [ind for pf in all_pareto_fronts for ind in pf]
    non_dominated_solutions = []
    for ind in all_solutions:
        is_dominated = False
        for other in all_solutions:
            if other != ind:
                if (other.fitness.values[0] <= ind.fitness.values[0] and 
                    other.fitness.values[1] <= ind.fitness.values[1] and 
                    (other.fitness.values[0] < ind.fitness.values[0] or 
                    other.fitness.values[1] < ind.fitness.values[1])):
                    is_dominated = True
                    break
        if not is_dominated:
            non_dominated_solutions.append(ind)
    return non_dominated_solutions

def plot_pareto_front(non_dominated_solutions: List[List[float]], scenario_name: str):
    costs = [ind.fitness.values[0] for ind in non_dominated_solutions]
    discomforts = [ind.fitness.values[1] for ind in non_dominated_solutions]
    plt.figure(figsize=(10, 8))
    plt.scatter(costs, discomforts, c='red', s=100, alpha=0.7, edgecolors='black', linewidth=1)
    for i, (cost, discomfort) in enumerate(zip(costs, discomforts), 1):
        plt.annotate(f'S{i}', (cost, discomfort), xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.8)
    plt.xlabel('Energy Cost', fontsize=12)
    plt.ylabel('User Discomfort', fontsize=12)
    plt.title(f'Pareto Front: Energy Cost vs User Discomfort\n({scenario_name})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.show()

def print_non_dominated_solutions(non_dominated_solutions: List[List[float]]):
    costs = [ind.fitness.values[0] for ind in non_dominated_solutions]
    discomforts = [ind.fitness.values[1] for ind in non_dominated_solutions]
    print("\n**Non-Dominated Solutions Costs and Discomforts:**")
    for i, (cost, discomfort) in enumerate(zip(costs, discomforts), 1):
        print(f"Solution {i}: Cost = {cost:.2f}, Discomfort = {discomfort:.2f}")

#function to plot a schedule of devices over 24 hours
def plot_gantt_chart(device_starts: Dict[str, int], devices: List[Device], solution_num: int, scenario_name: str):
    """Plot a Gantt chart showing the schedule of each device over a 24-hour period."""
    fig, ax = plt.subplots(figsize=(12, len(devices) * 0.5 + 2))
    colors = plt.cm.tab20(np.linspace(0, 1, len(devices)))
    
    for i, device in enumerate(devices):
        start = device_starts[device.name]
        duration = device.duration
        ax.barh(device.name, duration, left=start, height=0.4, color=colors[i], edgecolor='black')
    
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Devices', fontsize=12)
    ax.set_title(f'Gantt Chart: Device Schedule for Solution {solution_num} ({scenario_name})', fontsize=14)
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define scenarios
    scenarios = {
        "Scenario_1": Scenario(4, {"Dryer": "Washing Machine"}),
        "Scenario_2": Scenario(8, {"Dryer": "Washing Machine", "Dishwasher": "Oven"}),
        "Scenario_3": Scenario(12, {
            "Dryer": "Washing Machine",
            "Dishwasher": "Oven",
            "Oven": "Rice Cooker",
            "Smart fan": "Air Conditioner",
        })
    }

    # Select scenario to run (change this to switch scenarios)
    SCENARIO = "Scenario_1"  # Options: "Scenario_1", "Scenario_2", "Scenario_3"

    scenario = scenarios[SCENARIO]
    energy_system = EnergySystem()
    battery = Battery()
    ev = ElectricVehicle()

    all_pareto_fronts = []
    num_runs = 30
    num_solutions = {}
    
    for run in range(num_runs):
        print(f"\n=== Run {run+1}/{num_runs} ===")
        scheduler = SmartHomeScheduler(scenario, energy_system, battery, ev)
        pareto_front = scheduler.run_optimization(pop_size=200, generations=500)
        removed = remove_duplicate_solutions(pareto_front)
        num_solutions[run] = len(removed)
        all_pareto_fronts.append(removed)
    
    # Print summary of all runs
    print(f"\n=== Completed {num_runs} Runs for {SCENARIO} ===")
    print(f"Size of all Pareto fronts: {len(all_pareto_fronts)}")
    
    # Print number of unique solutions for each run
    for run, size in num_solutions.items():
        print(f"Run {run+1}: {size} unique solutions")
        
    # Remove duplicates across all Pareto fronts and analyze results
    unique_pareto_fronts = remove_duplicates_across_pareto_fronts(all_pareto_fronts)
    total_unique_solutions = sum(len(pf) for pf in unique_pareto_fronts)
    print(f"\n**Number of Unique Pareto Fronts (after cross-run deduplication):** {len(unique_pareto_fronts)}")
    print(f"**Total Unique Solutions Across All Pareto Fronts:** {total_unique_solutions}")
    
    # Remove dominated solutions across all Pareto fronts
    non_dominated_solutions = remove_dominated_solutions_across_all_pareto_fronts(unique_pareto_fronts)
    print(f"\n**Number of Non-Dominated Solutions Across All Runs:** {len(non_dominated_solutions)}")
    
    # Plot Pareto front and print non-dominated solutions
    plot_pareto_front(non_dominated_solutions, SCENARIO)
    
    # Print non-dominated solutions
    print("\n**Non-Dominated Solutions:**")
    print_non_dominated_solutions(non_dominated_solutions)
    
    # Analyze and print details of the first 5 non-dominated solutions
    print("\n**Analyzing First 5 Non-Dominated Solutions:**")
    for i, solution in enumerate(non_dominated_solutions[:5], 1):
        energy_cost, discomfort = scheduler.analyze_solution(solution, i)
        print(f"Solution {i}: Energy Cost = {energy_cost:.2f}, Discomfort = {discomfort:.2f}")
    
    # Plot Gantt chart for the first non-dominated solution
    if non_dominated_solutions:
        device_starts, _, _ = scheduler.decode_individual(non_dominated_solutions[0])
        energy_cost, discomfort= scheduler.analyze_solution(non_dominated_solutions[0], 1)
        print(f"Solution 1: Energy Cost = {energy_cost:.2f}, Discomfort = {discomfort:.2f}")
        plot_gantt_chart(device_starts, scenario.devices, 1, SCENARIO)
        
    # Plot solution distributions
    energy_costs = [ind.fitness.values[0] for ind in non_dominated_solutions]
    discomforts = [ind.fitness.values[1] for ind in non_dominated_solutions]
    plot_solution_distributions(energy_costs, discomforts, non_dominated_solutions, scenario.devices)