from result_handler import ResultsHandler
from simulation import SimulationRunner


def run_simulations():
    """Run all required simulations for paper results"""
    runner = SimulationRunner(num_nodes=100, num_channels=10)
    results_handler = ResultsHandler()

    # Scenarios from paper
    scenarios = [
        (2.0, 2.0),
        (2.0, 0.5),
    ]

    max_rounds = 2000  # Reduced for faster execution

    for alpha, beta in scenarios:
        print(f"\nRunning simulation for α={alpha}, β={beta}")
        results = runner.run_comparison(alpha, beta, max_rounds)

        # Save results
        results_handler.save_simulation_results(results, alpha, beta, max_rounds)

        # Print summary
        print(f"\nResults for α={alpha}, β={beta}:")
        for algo_name, algo_results in results.items():
            print(f"\n{algo_name}:")
            print(f"First node death round: {algo_results['first_death_round']}")
            print(f"Final alive nodes: {algo_results['alive_nodes_history'][-1]}")


if __name__ == "__main__":
    run_simulations()
