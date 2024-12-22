# test_mwcbg.py

from mwcbg import find_MWCBG
from simulation_environment import Node

def create_test_scenarios():
    """
    Create two test scenarios: 
    1. Basic scenario from Figure 4
    2. Complex scenario with more nodes and challenging channel distribution
    """
    scenarios = {}
    
    # Scenario 1: Basic test from Figure 4
    scenarios['basic'] = {
        'x': Node(id=1, x=150, y=150, 
                 available_channels={2,3,4,5},
                 initial_energy=0.2,
                 transmission_range=40),
        'y': Node(id=2, x=150, y=200, 
                 available_channels={1,2,3,4,5},
                 initial_energy=0.2,
                 transmission_range=40),
        'z': Node(id=3, x=200, y=150, 
                 available_channels={2,3,5,6},
                 initial_energy=0.2,
                 transmission_range=40),
        'a': Node(id=4, x=100, y=150, 
                 available_channels={2,3,5,6},
                 initial_energy=0.2,
                 transmission_range=40),
        'b': Node(id=5, x=150, y=100, 
                 available_channels={1,2,3,4,5},
                 initial_energy=0.2,
                 transmission_range=40)
    }
    
    # Scenario 2: Complex test case
    scenarios['complex'] = {
        # Central node with limited channels
        'center': Node(id=1, x=150, y=150, 
                      available_channels={1,2,3},
                      initial_energy=0.2,
                      transmission_range=40),
        
        # Nodes with many channels but far from center
        'far1': Node(id=2, x=180, y=180, 
                    available_channels={1,2,3,4,5,6},
                    initial_energy=0.2,
                    transmission_range=40),
        'far2': Node(id=3, x=120, y=180, 
                    available_channels={1,2,3,4,5},
                    initial_energy=0.2,
                    transmission_range=40),
        
        # Nodes with few channels but close to center
        'near1': Node(id=4, x=140, y=160, 
                     available_channels={1,2},
                     initial_energy=0.2,
                     transmission_range=40),
        'near2': Node(id=5, x=160, y=140, 
                     available_channels={2,3},
                     initial_energy=0.2,
                     transmission_range=40),
        
        # Nodes with partially overlapping channels
        'overlap1': Node(id=6, x=150, y=170, 
                        available_channels={1,2,4},
                        initial_energy=0.2,
                        transmission_range=40),
        'overlap2': Node(id=7, x=170, y=150, 
                        available_channels={2,3,5},
                        initial_energy=0.2,
                        transmission_range=40),
        
        # Node with unique channels
        'unique': Node(id=8, x=130, y=130, 
                      available_channels={1,2,6,7},
                      initial_energy=0.2,
                      transmission_range=40),
        
        # Node with all common channels but very far
        'distant': Node(id=9, x=200, y=200, 
                       available_channels={1,2,3},
                       initial_energy=0.2,
                       transmission_range=40)
    }
    
    return scenarios

def test_mwcbg():
    """Test MWCBG procedure with both scenarios"""
    
    # Get test scenarios
    scenarios = create_test_scenarios()
    
    # Test each scenario
    for scenario_name, nodes in scenarios.items():
        print(f"\n\nTesting {scenario_name} scenario:")
        print("=" * (len(scenario_name) + 8))
        
        # Use first node as source node
        source_node = next(iter(nodes.values()))
        
        # Get neighbors (all nodes in this scenario except source)
        neighbors = set(nodes.values()) - {source_node}
        
        # Run MWCBG
        result = find_MWCBG(
            node=source_node,
            channels=source_node.available_channels,
            neighbors=neighbors
        )
        
        # Print results
        print(f"Source Node: {source_node.id} (channels {source_node.available_channels})")
        print(f"\nSelected Nodes: {[n.id for n in result.nodes]}")
        print(f"Common Channels: {result.channels}")
        print(f"Weight: {result.weight:.2f}")
        
        # Verify results
        print("\nVerification:")
        print("=============")
        print(f"Number of selected nodes: {len(result.nodes)}")
        print(f"Number of common channels: {len(result.channels)}")
        print(f"Bi-channel connectivity requirement met: {len(result.channels) >= 2}")
        
        # Verify that selected channels are actually common to all selected nodes
        verification_passed = True
        for node in result.nodes:
            if not result.channels.issubset(node.available_channels):
                verification_passed = False
                print(f"Error: Node {node.id} doesn't have all common channels!")
        
        print(f"All selected nodes have common channels: {verification_passed}")

if __name__ == "__main__":
    test_mwcbg()