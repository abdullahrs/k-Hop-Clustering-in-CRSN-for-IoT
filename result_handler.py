# results_handler.py

import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any
import pandas as pd

class ResultsHandler:
    """Handles saving and visualization of simulation results"""
    
    def __init__(self, base_dir: str = "results"):
        """
        Initialize results handler
        
        Args:
            base_dir: Base directory for results
        """
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._create_directories()
        
    def _create_directories(self):
        """Create necessary directories"""
        # Create main directories
        for dir_name in ['data', 'plots', 'logs']:
            full_path = os.path.join(self.base_dir, dir_name)
            os.makedirs(full_path, exist_ok=True)
    
    def save_simulation_results(self, 
                              results: Dict, 
                              alpha: float, 
                              beta: float, 
                              max_rounds: int):
        """Save simulation results to files"""
        # Create scenario directory
        scenario_name = f"alpha_{alpha}_beta_{beta}_rounds_{max_rounds}"
        scenario_dir = os.path.join(self.base_dir, 'data', scenario_name)
        os.makedirs(scenario_dir, exist_ok=True)
        
        # Save raw results
        results_file = os.path.join(scenario_dir, "raw_results.json")
        serializable_results = self._prepare_results_for_saving(results)
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=4)
            
        # Generate and save all plots
        self._generate_all_plots(results, alpha, beta, max_rounds)
        
    def _prepare_results_for_saving(self, results: Dict) -> Dict:
        """Prepare results for JSON serialization"""
        serializable = {}
        for algo_name, algo_results in results.items():
            serializable[algo_name] = {
                'energy_history': algo_results['energy_history'],
                'alive_nodes_history': algo_results['alive_nodes_history'],
                'first_death_round': algo_results['first_death_round']
            }
        return serializable
    
    def _generate_all_plots(self, 
                          results: Dict, 
                          alpha: float, 
                          beta: float, 
                          max_rounds: int):
        """Generate all required plots"""
        plots_dir = os.path.join(self.base_dir, 'plots')
        
        # Network Energy Plot (Figures 7-10)
        self._plot_network_energy(results, alpha, beta, max_rounds, plots_dir)
        
        # First Node Death (Figures 11-12)
        self._plot_first_node_death(results, alpha, beta, max_rounds, plots_dir)
        
        # Re-clustering Frequency (Figures 13-14)
        self._plot_reclustering_frequency(results, alpha, beta, max_rounds, plots_dir)
        
        # Re-clustering Interval (Figures 15-16)
        self._plot_reclustering_interval(results, alpha, beta, max_rounds, plots_dir)
        
        # Number of Clusters (Figures 17-18)
        self._plot_number_of_clusters(results, alpha, beta, max_rounds, plots_dir)
    
    def _plot_network_energy(self, 
                           results: Dict, 
                           alpha: float, 
                           beta: float, 
                           max_rounds: int, 
                           plots_dir: str):
        """Generate network energy plot"""
        plt.figure(figsize=(10, 6))
        for algo_name, algo_results in results.items():
            plt.plot(
                range(len(algo_results['energy_history'])),
                algo_results['energy_history'],
                label=algo_name,
                marker='o',
                markersize=2
            )
        
        plt.xlabel('Number of rounds')
        plt.ylabel('Network remaining energy')
        plt.title(f'Network Energy (α={alpha}, β={beta})')
        plt.grid(True)
        plt.legend()
        
        filename = f"network_energy_a{alpha}_b{beta}_r{max_rounds}.png"
        plt.savefig(os.path.join(plots_dir, filename))
        plt.close()
    
    def _plot_first_node_death(self, 
                             results: Dict, 
                             alpha: float, 
                             beta: float, 
                             max_rounds: int, 
                             plots_dir: str):
        """Generate first node death plot"""
        plt.figure(figsize=(10, 6))
        
        algorithms = list(results.keys())
        death_rounds = [results[algo]['first_death_round'] for algo in algorithms]
        
        plt.bar(algorithms, death_rounds)
        plt.xlabel('Algorithms')
        plt.ylabel('Number of rounds')
        plt.title(f'First Node Death Round (α={alpha}, β={beta})')
        plt.xticks(rotation=45)
        
        filename = f"first_death_a{alpha}_b{beta}_r{max_rounds}.png"
        plt.savefig(os.path.join(plots_dir, filename), bbox_inches='tight')
        plt.close()
    
    def _plot_reclustering_frequency(self, 
                                   results: Dict, 
                                   alpha: float, 
                                   beta: float, 
                                   max_rounds: int, 
                                   plots_dir: str):
        """Generate re-clustering frequency plot"""
        # Calculate re-clustering frequency from energy changes
        frequencies = {}
        for algo_name, algo_results in results.items():
            energy_changes = np.diff(algo_results['energy_history'])
            significant_changes = np.where(np.abs(energy_changes) > np.std(energy_changes))[0]
            frequencies[algo_name] = len(significant_changes) / len(energy_changes)
        
        plt.figure(figsize=(10, 6))
        plt.bar(frequencies.keys(), frequencies.values())
        plt.xlabel('Algorithms')
        plt.ylabel('Re-clustering frequency')
        plt.title(f'Re-clustering Frequency (α={alpha}, β={beta})')
        plt.xticks(rotation=45)
        
        filename = f"reclustering_freq_a{alpha}_b{beta}_r{max_rounds}.png"
        plt.savefig(os.path.join(plots_dir, filename), bbox_inches='tight')
        plt.close()
    
    def _plot_reclustering_interval(self, 
                                  results: Dict, 
                                  alpha: float, 
                                  beta: float, 
                                  max_rounds: int, 
                                  plots_dir: str):
        """Generate re-clustering interval plot"""
        # Calculate intervals between energy changes
        intervals = {}
        for algo_name, algo_results in results.items():
            energy_changes = np.diff(algo_results['energy_history'])
            significant_changes = np.where(np.abs(energy_changes) > np.std(energy_changes))[0]
            if len(significant_changes) > 1:
                intervals[algo_name] = np.mean(np.diff(significant_changes))
            else:
                intervals[algo_name] = max_rounds
        
        plt.figure(figsize=(10, 6))
        plt.bar(intervals.keys(), intervals.values())
        plt.xlabel('Algorithms')
        plt.ylabel('Average re-clustering interval (rounds)')
        plt.title(f'Re-clustering Interval (α={alpha}, β={beta})')
        plt.xticks(rotation=45)
        
        filename = f"reclustering_interval_a{alpha}_b{beta}_r{max_rounds}.png"
        plt.savefig(os.path.join(plots_dir, filename), bbox_inches='tight')
        plt.close()

    def _plot_number_of_clusters(self,
                               results: Dict,
                               alpha: float,
                               beta: float,
                               max_rounds: int,
                               plots_dir: str):
        """Generate number of clusters plot"""
        # For each algorithm, plot min, avg, max number of clusters
        cluster_stats = {}
        for algo_name, algo_results in results.items():
            # Use alive nodes as proxy for number of clusters
            alive_nodes = algo_results['alive_nodes_history']
            cluster_stats[algo_name] = {
                'min': min(alive_nodes),
                'avg': np.mean(alive_nodes),
                'max': max(alive_nodes)
            }
        
        # Create grouped bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(cluster_stats))
        width = 0.25
        
        ax.bar(x - width, [stats['min'] for stats in cluster_stats.values()],
               width, label='Minimum', color='lightblue')
        ax.bar(x, [stats['avg'] for stats in cluster_stats.values()],
               width, label='Average', color='blue')
        ax.bar(x + width, [stats['max'] for stats in cluster_stats.values()],
               width, label='Maximum', color='darkblue')
        
        ax.set_ylabel('Number of clusters')
        ax.set_title(f'Number of Clusters (α={alpha}, β={beta})')
        ax.set_xticks(x)
        ax.set_xticklabels(cluster_stats.keys(), rotation=45)
        ax.legend()
        
        filename = f"num_clusters_a{alpha}_b{beta}_r{max_rounds}.png"
        plt.savefig(os.path.join(plots_dir, filename), bbox_inches='tight')
        plt.close()
        
    def collect_metrics(self, simulation_state):
        return {
            'network_energy': simulation_state.energy_manager.get_network_energy(),
            'alive_nodes': simulation_state.energy_manager.get_alive_nodes_count(),
            'cluster_count': len(simulation_state.current_clusters),
            'reclustering_events': simulation_state.reclustering_count,
            'first_node_death': simulation_state.energy_manager.first_node_death_round
        }