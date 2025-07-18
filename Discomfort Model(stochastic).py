import numpy as np
import pandas as pd
import random
from deap import base, creator, tools
import matplotlib.pyplot as plt
import warnings
from typing import Dict, List, Tuple
import seaborn as sns

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

class StochasticEnergySystem:
    def __init__(self):
        self.T = range(24)
        self.T_i = 0
        self.T_f = 23
        self.lambdaa = 4
        self.gamma = 20
        self.P_grid_max = 20
        self.P_sell_max = 4.0
        
        # Base parameters (deterministic values)
        self.P_sun_base = np.array([0.0] * 6 + [0.1, 0.2, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.3, 0.2, 0.1] + [0.0] * 5)
        self.P_wind_base = np.array([0.3] * 6 + [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3] + [0.3] * 5)
        self.C_re_base = np.array([0.18] * 6 + [0.10, 0.08, 0.07, 0.06, 0.06, 0.06, 0.06, 0.06, 0.07, 0.08, 0.10, 0.12, 0.14, 0.16, 0.17, 0.18, 0.18, 0.18])
        self.C_grid_base = np.array([0.25, 0.20, 0.18, 0.15, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.07, 0.10, 0.13, 0.16, 0.18, 0.20, 0.22, 0.25, 0.23, 0.21, 0.19, 0.17])
        self.R_sell_base = np.array([0.20, 0.15, 0.25, 0.05, 0.20, 0.06, 0.08, 0.10, 0.20, 0.10, 0.08, 0.20, 0.15, 0.12, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.18, 0.16, 0.14, 0.12])
        self.Pncl_base = np.array([1.5, 1.2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.2, 0.1, 0.5, 0.8, 1.0, 1.3, 1.5, 1.7, 1.6, 1.4, 1.2, 1.0])
        
        # Initialize with deterministic values first
        self.sample_stochastic_parameters()

    def sample_stochastic_parameters(self):
        """Sample stochastic parameters for one realization"""
        self.P_sun = np.clip(self.P_sun_base + np.random.normal(0, 0.03, size=24), 0, 1)
        self.P_wind = np.clip(self.P_wind_base + np.random.normal(0, 0.03, size=24), 0, 1)
        self.C_grid = np.clip(self.C_grid_base * (1 + np.random.uniform(-0.1, 0.1, size=24)), 0.01, 1.0)
        self.R_sell = np.clip(self.R_sell_base * (1 + np.random.uniform(-0.1, 0.1, size=24)), 0.01, 1.0)
        self.Pncl = np.clip(self.Pncl_base * (1 + np.random.normal(0, 0.1, size=24)), 0.1, 5.0)
        self.C_re = np.clip(self.C_re_base * (1 + np.random.uniform(-0.1, 0.1, size=24)), 0.01, 1.0)
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

class StochasticSmartHomeScheduler:
    def __init__(self, scenario: Scenario, energy_system: StochasticEnergySystem, battery: Battery, ev: ElectricVehicle):
        self.devices = scenario.devices
        self.precedence = scenario.precedences
        self.energy_system = energy_system
        self.battery = battery
        self.ev = ev
        self.toolbox = base.Toolbox()
        self.setup_deap()

    def setup_deap(self):
        # Single-objective optimization (minimize discomfort only)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

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
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def decode_individual(self, individual: List[float]) -> Tuple[Dict[str, int], np.ndarray, int]:
        device_starts = {device.name: int(individual[i]) for i, device in enumerate(self.devices)}
        
        for dependent, prerequisite in self.precedence.items():
            prereq_end = device_starts[prerequisite] + next(d.duration for d in self.devices if d.name == prerequisite)
            max_start = next(d.end_time for d in self.devices if d.name == dependent) - next(d.duration for d in self.devices if d.name == dependent) + 1
            device_starts[dependent] = max(prereq_end, min(device_starts[dependent], max_start))
        
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

            if battery_modes[t] == 1.0:  # Charge mode
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
                P_sell[t] = min(-net_demand, self.energy_system.P_sell_max)

            if t < 23:
                E_b[t + 1] = E_b[t] + P_br[t] - P_bi[t]
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
            prereq_end = device_starts[prerequisite] + next(d.duration for d in self.devices if d.name == prerequisite)
            if device_starts[dependent] < prereq_end:
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

        total_ev_energy = np.sum(Ptevb)
        required_ev_energy = self.ev.rho * self.ev.Eev_max
        if abs(total_ev_energy - required_ev_energy) > 0.01:
            penalty += abs(total_ev_energy - required_ev_energy) * 100
            feasible = False

        return penalty, feasible

    def calculate_discomfort(self, device_starts: Dict[str, int]) -> float:
        """Calculate discomfort as sum of delays from earliest start time"""
        discomfort = 0
        for device in self.devices:
            delay = device_starts[device.name] - device.start_time
            discomfort += delay
        return discomfort

    def evaluate_individual(self, individual: List[float]) -> Tuple[float]:
        """Evaluate individual for single objective: discomfort"""
        try:
            device_starts, battery_modes, ev_start = self.decode_individual(individual)
            X = self.create_device_schedule(device_starts)
            P_br, P_bi, P_grid, P_sell, E_b, Ptevb = self.solve_energy_balance(X, battery_modes, ev_start)
            penalty, feasible = self.check_constraints(X, device_starts, P_br, P_bi, P_grid, P_sell, E_b, Ptevb)
            discomfort = self.calculate_discomfort(device_starts)
            return (discomfort + penalty,)
        except Exception as e:
            return (10000.0,)

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
            max_start = next(d.end_time for d in self.devices if d.name == dependent) - next(d.duration for d in self.devices if d.name == dependent) + 1
            individual[dep_idx] = max(prereq_end, min(individual[dep_idx], max_start))
        
        for i in range(len(self.devices), len(self.devices) + 24):
            if random.random() < indpb:
                individual[i] = random.choice([0.0, 0.5, 1.0])
        if random.random() < indpb:
            individual[-1] = random.randint(0, 23)
        return (individual,)

    def run_single_optimization(self, pop_size: int = 200, generations: int = 500) -> List[float]:
        """Run single optimization and return best solution"""
        pop = self.toolbox.population(n=pop_size)
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for gen in range(generations):
            offspring = self.toolbox.select(pop, len(pop))
            offspring = [self.toolbox.clone(ind) for ind in offspring]
            
            for i in range(1, len(offspring), 2):
                if random.random() < 0.7:
                    self.toolbox.mate(offspring[i - 1], offspring[i])
                    del offspring[i - 1].fitness.values
                    del offspring[i].fitness.values
            
            for i in range(len(offspring)):
                if random.random() < 0.3:
                    self.toolbox.mutate(offspring[i], indpb=0.1)
                    del offspring[i].fitness.values
            
            invalid_ind = [ind for ind in offspring if not hasattr(ind.fitness, "values") or not ind.fitness.values]
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            pop[:] = offspring
        
        best_ind = tools.selBest(pop, 1)[0]
        return best_ind

class StochasticAnalyzer:
    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.devices = scenario.devices
        self.battery = Battery()
        self.ev = ElectricVehicle()
        
        self.all_solutions = []
        self.all_discomforts = []
        self.device_start_probabilities = {}
        self.battery_mode_probabilities = np.zeros((24, 3))
        self.ev_start_probabilities = np.zeros(24)
        
        for device in self.devices:
            possible_starts = range(device.start_time, device.end_time - device.duration + 2)
            self.device_start_probabilities[device.name] = {start: 0 for start in possible_starts}

    def run_stochastic_analysis(self, num_iterations: int = 10000):
        """Run stochastic analysis with specified number of iterations"""
        print(f"Starting stochastic analysis with {num_iterations} iterations...")
        print(f"Scenario: {len(self.devices)} devices with precedences: {self.scenario.precedences}")
        
        valid_solutions = 0
        
        for iteration in range(num_iterations):
            if (iteration + 1) % 1000 == 0:
                print(f"Completed {iteration + 1}/{num_iterations} iterations...")
            
            try:
                energy_system = StochasticEnergySystem()
                energy_system.sample_stochastic_parameters()
                scheduler = StochasticSmartHomeScheduler(self.scenario, energy_system, self.battery, self.ev)
                best_solution = scheduler.run_single_optimization(pop_size=30, generations=50)
                device_starts, battery_modes, ev_start = scheduler.decode_individual(best_solution)
                
                self.all_solutions.append(best_solution.copy())
                self.all_discomforts.append(best_solution.fitness.values[0])
                self._update_probabilities(device_starts, battery_modes, ev_start)
                valid_solutions += 1
                
            except Exception as e:
                print(f"Error in iteration {iteration + 1}: {e}")
                continue
        
        print(f"Completed analysis. Valid solutions: {valid_solutions}/{num_iterations}")
        self._normalize_probabilities(valid_solutions)
    
    def _update_probabilities(self, device_starts: Dict[str, int], battery_modes: np.ndarray, ev_start: int):
        """Update probability counters"""
        for device_name, start_time in device_starts.items():
            if start_time in self.device_start_probabilities[device_name]:
                self.device_start_probabilities[device_name][start_time] += 1
        
        for t, mode in enumerate(battery_modes):
            if mode == 0.0:
                self.battery_mode_probabilities[t, 0] += 1
            elif mode == 0.5:
                self.battery_mode_probabilities[t, 1] += 1
            elif mode == 1.0:
                self.battery_mode_probabilities[t, 2] += 1
        
        self.ev_start_probabilities[ev_start] += 1
    
    def _normalize_probabilities(self, num_valid_solutions: int):
        """Convert counts to probabilities"""
        for device_name in self.device_start_probabilities:
            for start_time in self.device_start_probabilities[device_name]:
                self.device_start_probabilities[device_name][start_time] /= num_valid_solutions
        
        self.battery_mode_probabilities /= num_valid_solutions
        self.ev_start_probabilities /= num_valid_solutions
    
    def print_results(self):
        """Print probability results for discomfort"""
        print("\n" + "="*80)
        print("STOCHASTIC OPTIMIZATION RESULTS (DISCOMFORT)")
        print("="*80)
        
        print(f"\nAverage Discomfort: {np.mean(self.all_discomforts):.4f} ± {np.std(self.all_discomforts):.4f}")
        print(f"Best Discomfort: {np.min(self.all_discomforts):.4f}")
        print(f"Worst Discomfort: {np.max(self.all_discomforts):.4f}")
        
        print("\n" + "-"*60)
        print("DEVICE START TIME PROBABILITIES")
        print("-"*60)
        
        for device in self.devices:
            print(f"\n{device.name} (Duration: {device.duration}h, Window: {device.start_time}-{device.end_time}):")
            probs = self.device_start_probabilities[device.name]
            for start_time, prob in sorted(probs.items()):
                if prob > 0.01:
                    print(f"  Start at hour {start_time}: {prob:.3f} ({prob*100:.1f}%)")
        
        print("\n" + "-"*60)
        print("BATTERY MODE PROBABILITIES")
        print("-"*60)
        print("Hour | Discharge | Idle     | Charge")
        print("-"*40)
        for t in range(24):
            discharge_prob = self.battery_mode_probabilities[t, 0]
            idle_prob = self.battery_mode_probabilities[t, 1]
            charge_prob = self.battery_mode_probabilities[t, 2]
            print(f"{t:2d}   | {discharge_prob:.3f}     | {idle_prob:.3f}    | {charge_prob:.3f}")
        
        print("\n" + "-"*60)
        print("EV CHARGING START TIME PROBABILITIES")
        print("-"*60)
        for t in range(24):
            if self.ev_start_probabilities[t] > 0.01:
                print(f"Start at hour {t}: {self.ev_start_probabilities[t]:.3f} ({self.ev_start_probabilities[t]*100:.1f}%)")
    
    def plot_device_probabilities(self):
        """Plot device start time probabilities"""
        num_devices = len(self.devices)
        rows = (num_devices + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))
        axes = axes.flatten() if num_devices > 1 else [axes]
        
        for i, device in enumerate(self.devices):
            start_times = []
            probabilities = []
            for start_time, prob in sorted(self.device_start_probabilities[device.name].items()):
                start_times.append(start_time)
                probabilities.append(prob)
            
            axes[i].bar(start_times, probabilities, alpha=0.7, color=f'C{i}')
            axes[i].set_title(f'{device.name}\n(Duration: {device.duration}h)')
            axes[i].set_xlabel('Start Time (hour)')
            axes[i].set_ylabel('Probability')
            axes[i].set_xticks(start_times)
            axes[i].grid(True, alpha=0.3)
        
        for i in range(num_devices, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('Device Start Time Probabilities', fontsize=16, y=1.02)
        plt.show()
    
    def plot_battery_probabilities(self):
        """Plot battery mode probabilities as a stacked bar chart"""
        fig, ax = plt.subplots(figsize=(15, 6))
        
        hours = np.arange(24)
        discharge_probs = self.battery_mode_probabilities[:, 0]
        idle_probs = self.battery_mode_probabilities[:, 1]
        charge_probs = self.battery_mode_probabilities[:, 2]
        
        ax.bar(hours, discharge_probs, label='Discharge', color='red', alpha=0.7)
        ax.bar(hours, idle_probs, bottom=discharge_probs, label='Idle', color='blue', alpha=0.7)
        ax.bar(hours, charge_probs, bottom=discharge_probs + idle_probs, label='Charge', color='green', alpha=0.7)
        
        ax.set_title('Battery Mode Probabilities Over 24 Hours', fontsize=16)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Probability')
        ax.set_xticks(hours)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function to run stochastic smart home scheduling analysis"""
    precedences = {
        "Dryer": "Washing Machine",
        "Dishwasher": "Oven",
        "Oven": "Rice Cooker",
        "Smart fan": "Air Conditioner",
    }
    scenario = Scenario(num_devices=12, precedences=precedences)
    
    analyzer = StochasticAnalyzer(scenario)
    analyzer.run_stochastic_analysis(num_iterations=10000)
    analyzer.print_results()
    analyzer.plot_device_probabilities()
    analyzer.plot_battery_probabilities()

if __name__ == "__main__":
    main()