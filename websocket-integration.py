#!/usr/bin/env python3
"""
Integration example showing how to connect your existing Octopus AI code
with the WebSocket visualization server.

This file shows you how to modify your existing simulation to work with
the WebSocket server.
"""

# Example of how to modify your existing Octopus class to work with WebSocket
class WebSocketCompatibleOctopus:
    """
    Example of how to modify your existing Octopus class to be WebSocket compatible
    """
    
    def __init__(self, x, y, num_arms, limb_rows, limb_cols):
        # Your existing initialization code
        self.x = x
        self.y = y
        self.num_arms = num_arms
        # ... your existing code ...
        
        # Make sure limbs and suckers are JSON-serializable
        self.limbs = []  # List of lists of {x, y} dicts
        self.suckers = []  # List of {x, y, color, target_color} dicts
        
        self._initialize_limbs_and_suckers(limb_rows, limb_cols)
    
    def _initialize_limbs_and_suckers(self, limb_rows, limb_cols):
        """Initialize limbs and suckers in WebSocket-compatible format"""
        import math
        
        for arm_idx in range(self.num_arms):
            angle = (arm_idx / self.num_arms) * 2 * math.pi
            limb = []
            
            for row in range(limb_rows):
                distance = 1 + (row / limb_rows) * 4
                limb_x = self.x + math.cos(angle) * distance
                limb_y = self.y + math.sin(angle) * distance
                
                limb.append({'x': float(limb_x), 'y': float(limb_y)})
                
                # Add suckers along this limb segment
                for col in range(limb_cols):
                    offset = (col - limb_cols / 2 + 0.5) * 0.3
                    sucker_x = limb_x + math.cos(angle + math.pi/2) * offset
                    sucker_y = limb_y + math.sin(angle + math.pi/2) * offset
                    
                    self.suckers.append({
                        'x': float(sucker_x),
                        'y': float(sucker_y),
                        'color': 0.5,  # Initial gray color
                        'target_color': 0.0  # Will be updated based on surface
                    })
            
            self.limbs.append(limb)
    
    def get_serializable_state(self):
        """Return state in format compatible with WebSocket JSON serialization"""
        return {
            'head': {'x': float(self.x), 'y': float(self.y)},
            'limbs': self.limbs,
            'suckers': self.suckers
        }


# Example of how to modify your RandomSurface class
class WebSocketCompatibleSurface:
    """
    Example of how to modify your RandomSurface class to be WebSocket compatible
    """
    
    def __init__(self, x_len, y_len, seed=None):
        import numpy as np
        if seed is not None:
            np.random.seed(seed)
        
        self.x_len = x_len
        self.y_len = y_len
        self.grid = np.random.randint(2, size=(y_len, x_len))
    
    def get_color_at(self, x, y):
        """Get color at specific coordinates"""
        try:
            return float(self.grid[int(y)][int(x)])
        except (IndexError, ValueError):
            return 0.0
    
    def get_serializable_grid(self):
        """Return grid in JSON-serializable format"""
        return self.grid.tolist()


# Example setup script
def setup_websocket_server():
    """
    Example of how to set up the WebSocket server with your existing code
    """
    
    # Install required packages
    install_instructions = """
    To run the WebSocket server, you need to install:
    
    pip install websockets asyncio
    
    If you don't have them already:
    pip install numpy matplotlib
    """
    
    print(install_instructions)


# Example of how to run your simulation with WebSocket support
def run_simulation_with_websocket():
    """
    Example of how to integrate WebSocket support into your existing simulation loop
    """
    
    # This is how you might modify your existing main simulation function
    import asyncio
    import json
    from websocket_server import OctopusSimulationServer
    
    # Create server instance
    server = OctopusSimulationServer(port=8765)
    
    # If you want to use your existing trained models, you can load them here:
    # server.load_model('path/to/your/model.keras')
    
    # Start the server
    try:
        print("Starting WebSocket server...")
        print("Open your web browser and navigate to the visualization interface")
        print("The server will run on ws://localhost:8765")
        
        asyncio.run(server.start_server())
        
    except KeyboardInterrupt:
        print("\nShutting down server...")


# File structure example for your project
file_structure_example = """
Your project structure should look like this:

octopus_ai/
├── OctoConfig.py                 # Your existing config file
├── websocket_server.py           # The WebSocket server (from above)
├── integration_example.py       # This file
├── simulator/
│   ├── __init__.py
│   ├── Octopus.py               # Your existing octopus class
│   ├── RandomSurface.py         # Your existing surface class
│   ├── AgentGenerator.py        # Your existing agent generator
│   └── simutil.py               # Your existing utilities
├── training/
│   ├── models/
│   │   ├── sucker.keras         # Your trained models
│   │   └── limb.keras
│   └── ...
└── frontend/
    └── octopus_visualizer.html  # Save the React component as HTML

To run:
1. Save the WebSocket server as 'websocket_server.py'
2. Save the React visualizer as an HTML file or run it in a React app
3. Run: python websocket_server.py
4. Open the visualizer in your browser
"""

# Configuration adapter
def adapt_config_for_websocket(your_existing_config):
    """
    Example of how to adapt your existing OctoConfig for WebSocket use
    """
    
    # The WebSocket server expects certain key names
    # This function shows how to map your config to the expected format
    
    websocket_config = {
        # Map your config keys to WebSocket expected keys
        'x_len': your_existing_config.get('x_len', 15),
        'y_len': your_existing_config.get('y_len', 15),
        'num_iterations': your_existing_config.get('num_iterations', 120),
        'rand_seed': your_existing_config.get('rand_seed', 0),
        'debug_mode': your_existing_config.get('debug_mode', False),
        
        # Agent parameters
        'agent_number_of_agents': your_existing_config.get('agent_number_of_agents', 5),
        'agent_max_velocity': your_existing_config.get('agent_max_velocity', 0.2),
        'agent_max_theta': your_existing_config.get('agent_max_theta', 0.1),
        'agent_range_radius': your_existing_config.get('agent_range_radius', 5),
        
        # Octopus parameters
        'octo_max_body_velocity': your_existing_config.get('octo_max_body_velocity', 0.25),
        'octo_max_arm_theta': your_existing_config.get('octo_max_arm_theta', 0.1),
        'octo_max_limb_offset': your_existing_config.get('octo_max_limb_offset', 0.5),
        'octo_num_arms': your_existing_config.get('octo_num_arms', 8),
        'octo_max_sucker_distance': your_existing_config.get('octo_max_sucker_distance', 0.3),
        'octo_min_sucker_distance': your_existing_config.get('octo_min_sucker_distance', 0.1),
        
        # Limb parameters
        'limb_rows': your_existing_config.get('limb_rows', 16),
        'limb_cols': your_existing_config.get('limb_cols', 2),
        
        # Sucker parameters
        'octo_max_hue_change': your_existing_config.get('octo_max_hue_change', 0.25),
    }
    
    return websocket_config


if __name__ == "__main__":
    print("Octopus AI WebSocket Integration Example")
    print("="*50)
    print(file_structure_example)
    print("\nTo get started:")
    print("1. Install dependencies: pip install websockets")
    print("2. Save websocket_server.py in your project root")
    print("3. Run: python websocket_server.py")
    print("4. Open the React visualizer in your browser")
    print("5. Connect to ws://localhost:8765")
    
    # Uncomment this line to actually run the server
    # run_simulation_with_websocket()