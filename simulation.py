# simulation_runner.py

from typing import Dict, Any, Tuple
from simulation_environment import CRSNEnvironment
from energy_management import EnergyManager, EnergyParameters
from k_sacb_ec import KSABEC
from k_sacb_wec import KSABWEC
from visualization import plot_network_energy, plot_first_node_death, visualize_clusters

class SimulationRunner:
    """Runs simulations and collects data for different algorithms"""
    
    def __init__(self, num_nodes: int = 100, num_channels: int = 10):
        self.num_nodes = num_nodes
        self.num_channels = num_channels
        self.energy_params = EnergyParameters()
        
    def setup_environment(self, alpha: float, beta: float) -> Tuple[CRSNEnvironment, EnergyManager]:
        """Setup simulation environment with given parameters"""
        params = {
            "NUM_SUS": self.num_nodes,
            "NUM_CHANNELS": self.num_channels,
            "PU_ALPHA_RANGE": (alpha, alpha),  # Fixed alpha
            "PU_BETA_RANGE": (beta, beta),     # Fixed beta
        }
        
        env = CRSNEnvironment(params)
        energy_manager = EnergyManager(env.nodes, self.energy_params)
        return env, energy_manager
    
    def run_single_algorithm(self, 
                           algo_class: Any, 
                           env: CRSNEnvironment,
                           energy_manager: EnergyManager,
                           max_rounds: int = 900) -> Dict:
        """
        Run simulation for a single algorithm
        
        Returns:
            Dictionary containing:
            - energy_history: List of network energy values
            - alive_nodes_history: List of alive node counts
            - first_death_round: Round when first node died
            - clusters: Final cluster configuration
        """
        algorithm = algo_class(env)
        clusters = algorithm.form_clusters()
        
        energy_history = []
        alive_nodes_history = []
        
        for _ in range(max_rounds):
            # Record current state
            energy_history.append(energy_manager.get_network_energy())
            alive_nodes_history.append(energy_manager.get_alive_nodes_count())
            
            # Run one round
            energy_manager.consume_energy_for_round(clusters)
            
            # Check if all nodes are dead
            if energy_manager.get_alive_nodes_count() == 0:
                break
        
        return {
            'energy_history': energy_history,
            'alive_nodes_history': alive_nodes_history,
            'first_death_round': energy_manager.first_node_death_round,
            'clusters': clusters
        }
    
    def run_comparison(self, 
                      alpha: float, 
                      beta: float,
                      max_rounds: int = 900) -> Dict:
        """
        Run comparison of all algorithms
        
        Returns:
            Dictionary containing results for each algorithm
        """
        # Setup environment
        env, energy_manager = self.setup_environment(alpha, beta)
        
        algorithms = {
            'k-SACB-EC': KSABEC,
            'k-SACB-WEC': KSABWEC,
            # Add other algorithms here when implemented
        }
        
        results = {}
        
        for algo_name, algo_class in algorithms.items():
            # Reset environment for each algorithm
            energy_manager.reset()
            
            # Run algorithm
            results[algo_name] = self.run_single_algorithm(
                algo_class, env, energy_manager, max_rounds
            )
            
            # Visualize clusters for this algorithm
            visualize_clusters(
                env.nodes,
                results[algo_name]['clusters'],
                f'Cluster Formation - {algo_name} (α={alpha}, β={beta})'
            )
        
        # Generate energy comparison plot
        plot_network_energy(
            {name: res['energy_history'] for name, res in results.items()},
            alpha, beta
        )
        
        # Generate first node death comparison plot
        plot_first_node_death(
            {name: res['first_death_round'] for name, res in results.items()},
            alpha, beta
        )
        
        return results

def run_simulations():
    """Run all required simulations for paper results"""
    runner = SimulationRunner(num_nodes=100, num_channels=10)
    
    # Scenarios from paper
    scenarios = [
        (2.0, 2.0),  # Figures 7,11
        (2.0, 0.5),  # Figures 8,12
        (0.5, 2.0),  # Figures 9,13
        (0.5, 0.5)   # Figures 10,14
    ]
    
    for alpha, beta in scenarios:
        print(f"\nRunning simulation for α={alpha}, β={beta}")
        results = runner.run_comparison(alpha, beta)
        print(f"Results for α={alpha}, β={beta}:")
        for algo_name, algo_results in results.items():
            print(f"\n{algo_name}:")
            print(f"First node death round: {algo_results['first_death_round']}")
            print(f"Final alive nodes: {algo_results['alive_nodes_history'][-1]}")

if __name__ == "__main__":
    run_simulations()