import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Settings, Play, Pause, RotateCcw, Download, Upload, Wifi, WifiOff } from 'lucide-react';

const OctopusAIVisualizer = () => {
  // WebSocket connection state
  const [wsConnected, setWsConnected] = useState(false);
  const [wsUrl, setWsUrl] = useState('ws://localhost:8765');
  const wsRef = useRef(null);

  // Configuration state based on your OctoConfig.py
  const [config, setConfig] = useState({
    // General game parameters
    num_iterations: 120,
    x_len: 15,
    y_len: 15,
    rand_seed: 0,
    debug_mode: false,
    save_images: false,
    adjacency_radius: 1.0,
    
    // Agent parameters
    agent_number_of_agents: 5,
    agent_max_velocity: 0.2,
    agent_max_theta: 0.1,
    agent_range_radius: 5,
    
    // Octopus parameters
    octo_max_body_velocity: 0.25,
    octo_max_arm_theta: 0.1,
    octo_max_limb_offset: 0.5,
    octo_num_arms: 8,
    octo_max_sucker_distance: 0.3,
    octo_min_sucker_distance: 0.1,
    
    // Limb parameters
    limb_rows: 16,
    limb_cols: 2,
    
    // Sucker parameters
    octo_max_hue_change: 0.25
  });

  // Simulation state
  const [isPlaying, setIsPlaying] = useState(false);
  const [showConfig, setShowConfig] = useState(false);
  const [iteration, setIteration] = useState(0);
  const [visibility_score, setVisibilityScore] = useState(0);
  
  // Simulation data state - updated via WebSocket
  const [simulationData, setSimulationData] = useState({
    background: [],
    octopus: {
      head: { x: 7, y: 7 },
      limbs: [],
      suckers: []
    },
    agents: [],
    metadata: {
      iteration: 0,
      visibility_score: 0,
      fps: 0
    }
  });

  const canvasRef = useRef(null);

  // WebSocket connection management
  const connectWebSocket = useCallback(() => {
    try {
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        setWsConnected(true);
        wsRef.current = ws;
        
        // Send initial config
        ws.send(JSON.stringify({
          type: 'config_update',
          data: config
        }));
      };
      
      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          
          switch (message.type) {
            case 'simulation_state':
              setSimulationData(message.data);
              setIteration(message.data.metadata?.iteration || 0);
              setVisibilityScore(message.data.metadata?.visibility_score || 0);
              break;
              
            case 'config_response':
              console.log('Config updated on server');
              break;
              
            case 'simulation_control_response':
              console.log('Simulation control:', message.data);
              break;
              
            default:
              console.log('Unknown message type:', message.type);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
      
      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setWsConnected(false);
        wsRef.current = null;
        
        // Auto-reconnect after 3 seconds
        setTimeout(() => {
          if (!wsConnected) {
            connectWebSocket();
          }
        }, 3000);
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setWsConnected(false);
      };
      
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
      setWsConnected(false);
    }
  }, [wsUrl, config, wsConnected]);

  const disconnectWebSocket = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
      setWsConnected(false);
    }
  }, []);

  // Send WebSocket messages
  const sendWebSocketMessage = useCallback((type, data) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type, data }));
    } else {
      console.warn('WebSocket not connected');
    }
  }, []);

  // Initialize WebSocket connection
  useEffect(() => {
    connectWebSocket();
    return () => disconnectWebSocket();
  }, []);

  // Generate initial fallback data
  useEffect(() => {
    if (!wsConnected) {
      generateFallbackData();
    }
  }, [config.x_len, config.y_len, config.octo_num_arms, config.agent_number_of_agents, wsConnected]);

  const generateFallbackData = () => {
    // Generate random background (checkerboard pattern)
    const background = Array(config.y_len).fill().map(() => 
      Array(config.x_len).fill().map(() => Math.random() > 0.5 ? 1 : 0)
    );

    // Generate octopus limbs and suckers
    const octopus = {
      head: { x: config.x_len / 2, y: config.y_len / 2 },
      limbs: [],
      suckers: []
    };

    // Generate limbs radiating from center
    for (let i = 0; i < config.octo_num_arms; i++) {
      const angle = (i / config.octo_num_arms) * 2 * Math.PI;
      const limb = [];
      const suckers = [];
      
      for (let j = 0; j < config.limb_rows; j++) {
        const distance = 1 + (j / config.limb_rows) * 4;
        const x = octopus.head.x + Math.cos(angle) * distance;
        const y = octopus.head.y + Math.sin(angle) * distance;
        
        limb.push({ x, y });
        
        // Add suckers along the limb
        for (let k = 0; k < config.limb_cols; k++) {
          const offset = (k - config.limb_cols / 2 + 0.5) * 0.3;
          const suckerX = x + Math.cos(angle + Math.PI/2) * offset;
          const suckerY = y + Math.sin(angle + Math.PI/2) * offset;
          suckers.push({ 
            x: suckerX, 
            y: suckerY, 
            color: Math.random(),
            targetColor: background[Math.floor(suckerY)]?.[Math.floor(suckerX)] || 0
          });
        }
      }
      
      octopus.limbs.push(limb);
      octopus.suckers.push(...suckers);
    }

    // Generate agents (predators and prey)
    const agents = [];
    for (let i = 0; i < config.agent_number_of_agents; i++) {
      agents.push({
        x: Math.random() * config.x_len,
        y: Math.random() * config.y_len,
        type: Math.random() > 0.5 ? 'predator' : 'prey',
        velocity: Math.random() * config.agent_max_velocity,
        angle: Math.random() * 2 * Math.PI
      });
    }

    setSimulationData({ 
      background, 
      octopus, 
      agents,
      metadata: { iteration: 0, visibility_score: 0, fps: 0 }
    });
  };

  // Canvas drawing function
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const cellSize = 30;
    
    canvas.width = config.x_len * cellSize;
    canvas.height = config.y_len * cellSize;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw background checkerboard
    simulationData.background.forEach((row, y) => {
      row.forEach((cell, x) => {
        ctx.fillStyle = cell ? '#ffffff' : '#000000';
        ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
      });
    });
    
    // Draw grid lines
    ctx.strokeStyle = '#444444';
    ctx.lineWidth = 0.5;
    for (let x = 0; x <= config.x_len; x++) {
      ctx.beginPath();
      ctx.moveTo(x * cellSize, 0);
      ctx.lineTo(x * cellSize, canvas.height);
      ctx.stroke();
    }
    for (let y = 0; y <= config.y_len; y++) {
      ctx.beginPath();
      ctx.moveTo(0, y * cellSize);
      ctx.lineTo(canvas.width, y * cellSize);
      ctx.stroke();
    }
    
    // Draw octopus limbs
    ctx.strokeStyle = '#ff4444';
    ctx.lineWidth = 3;
    simulationData.octopus.limbs.forEach(limb => {
      if (limb.length > 1) {
        ctx.beginPath();
        ctx.moveTo(limb[0].x * cellSize, limb[0].y * cellSize);
        for (let i = 1; i < limb.length; i++) {
          ctx.lineTo(limb[i].x * cellSize, limb[i].y * cellSize);
        }
        ctx.stroke();
      }
    });
    
    // Draw octopus head
    ctx.fillStyle = '#ff6666';
    ctx.beginPath();
    ctx.arc(
      simulationData.octopus.head.x * cellSize, 
      simulationData.octopus.head.y * cellSize, 
      cellSize * 0.4, 
      0, 
      2 * Math.PI
    );
    ctx.fill();
    
    // Draw suckers
    simulationData.octopus.suckers.forEach(sucker => {
      const gray = Math.floor(sucker.color * 255);
      ctx.fillStyle = `rgb(${gray}, ${gray}, ${gray})`;
      ctx.strokeStyle = '#333333';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(sucker.x * cellSize, sucker.y * cellSize, cellSize * 0.15, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
    });
    
    // Draw agents
    simulationData.agents.forEach(agent => {
      ctx.fillStyle = agent.type === 'predator' ? '#ff69b4' : '#32cd32';
      ctx.beginPath();
      ctx.arc(agent.x * cellSize, agent.y * cellSize, cellSize * 0.2, 0, 2 * Math.PI);
      ctx.fill();
      
      // Draw range circle if debug mode
      if (config.debug_mode) {
        ctx.strokeStyle = agent.type === 'predator' ? '#ff69b4' : '#32cd32';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.arc(
          agent.x * cellSize, 
          agent.y * cellSize, 
          config.agent_range_radius * cellSize, 
          0, 
          2 * Math.PI
        );
        ctx.stroke();
        ctx.setLineDash([]);
      }
    });
    
  }, [simulationData, config]);

  const handleConfigChange = (key, value) => {
    const newConfig = { ...config, [key]: value };
    setConfig(newConfig);
    
    // Send config update via WebSocket
    sendWebSocketMessage('config_update', newConfig);
  };

  const handlePlayPause = () => {
    const newPlayState = !isPlaying;
    setIsPlaying(newPlayState);
    sendWebSocketMessage('simulation_control', { 
      action: newPlayState ? 'play' : 'pause' 
    });
  };

  const handleReset = () => {
    setIsPlaying(false);
    setIteration(0);
    sendWebSocketMessage('simulation_control', { action: 'reset' });
  };

  const exportConfig = () => {
    const dataStr = JSON.stringify(config, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'octopus_config.json';
    link.click();
    URL.revokeObjectURL(url);
  };

  const importConfig = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const newConfig = JSON.parse(e.target.result);
          setConfig(newConfig);
          sendWebSocketMessage('config_update', newConfig);
        } catch (error) {
          alert('Error loading configuration file');
        }
      };
      reader.readAsText(file);
    }
  };

  return (
    <div className="w-full h-screen bg-gray-900 text-white flex">
      {/* Main visualization area */}
      <div className="flex-1 flex flex-col">
        {/* Control bar */}
        <div className="bg-gray-800 p-4 flex items-center justify-between">
          <div className="flex items-center space-x-4">
            {/* WebSocket connection status */}
            <div className="flex items-center space-x-2">
              <input
                type="text"
                value={wsUrl}
                onChange={(e) => setWsUrl(e.target.value)}
                placeholder="WebSocket URL"
                className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm w-40"
                disabled={wsConnected}
              />
              <button
                onClick={wsConnected ? disconnectWebSocket : connectWebSocket}
                className={`flex items-center space-x-2 px-3 py-2 rounded-lg transition-colors ${
                  wsConnected 
                    ? 'bg-green-600 hover:bg-green-700' 
                    : 'bg-red-600 hover:bg-red-700'
                }`}
              >
                {wsConnected ? <Wifi size={16} /> : <WifiOff size={16} />}
                <span className="text-sm">{wsConnected ? 'Connected' : 'Disconnected'}</span>
              </button>
            </div>
            
            <div className="h-6 w-px bg-gray-600"></div>
            
            <button
              onClick={handlePlayPause}
              className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
              disabled={!wsConnected}
            >
              {isPlaying ? <Pause size={16} /> : <Play size={16} />}
              <span>{isPlaying ? 'Pause' : 'Play'}</span>
            </button>
            
            <button
              onClick={handleReset}
              className="flex items-center space-x-2 px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded-lg transition-colors"
              disabled={!wsConnected}
            >
              <RotateCcw size={16} />
              <span>Reset</span>
            </button>
            
            <button
              onClick={() => setShowConfig(!showConfig)}
              className="flex items-center space-x-2 px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors"
            >
              <Settings size={16} />
              <span>Config</span>
            </button>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="text-sm space-x-4">
              <span>Iteration: {iteration}</span>
              <span>Visibility: {visibility_score.toFixed(3)}</span>
              {simulationData.metadata?.fps && (
                <span>FPS: {simulationData.metadata.fps.toFixed(1)}</span>
              )}
            </div>
            <div className="flex space-x-2">
              <button
                onClick={exportConfig}
                className="p-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors"
                title="Export Config"
              >
                <Download size={16} />
              </button>
              <label className="p-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors cursor-pointer" title="Import Config">
                <Upload size={16} />
                <input
                  type="file"
                  accept=".json"
                  onChange={importConfig}
                  className="hidden"
                />
              </label>
            </div>
          </div>
        </div>
        
        {/* Canvas area */}
        <div className="flex-1 flex items-center justify-center bg-gray-800 p-4">
          <canvas
            ref={canvasRef}
            className="border border-gray-600 rounded-lg"
            style={{ maxWidth: '100%', maxHeight: '100%' }}
          />
        </div>
      </div>
      
      {/* Configuration panel */}
      {showConfig && (
        <div className="w-80 bg-gray-800 border-l border-gray-700 overflow-y-auto">
          <div className="p-4">
            <h2 className="text-xl font-bold mb-4">Configuration</h2>
            
            <div className="space-y-6">
              {/* Connection Status */}
              <div>
                <h3 className="text-lg font-semibold mb-3 text-yellow-400">Connection</h3>
                <div className="space-y-2">
                  <div className={`text-sm p-2 rounded ${wsConnected ? 'bg-green-900 text-green-200' : 'bg-red-900 text-red-200'}`}>
                    Status: {wsConnected ? 'Connected' : 'Disconnected'}
                  </div>
                  {!wsConnected && (
                    <div className="text-xs text-gray-400">
                      Make sure your Python WebSocket server is running on {wsUrl}
                    </div>
                  )}
                </div>
              </div>
              
              {/* General Parameters */}
              <div>
                <h3 className="text-lg font-semibold mb-3 text-blue-400">General</h3>
                <div className="space-y-3">
                  <div>
                    <label className="block text-sm font-medium mb-1">Grid Size X</label>
                    <input
                      type="number"
                      value={config.x_len}
                      onChange={(e) => handleConfigChange('x_len', parseInt(e.target.value))}
                      className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                      min="5"
                      max="30"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">Grid Size Y</label>
                    <input
                      type="number"
                      value={config.y_len}
                      onChange={(e) => handleConfigChange('y_len', parseInt(e.target.value))}
                      className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                      min="5"
                      max="30"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">Iterations</label>
                    <input
                      type="number"
                      value={config.num_iterations}
                      onChange={(e) => handleConfigChange('num_iterations', parseInt(e.target.value))}
                      className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                      min="-1"
                    />
                  </div>
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      checked={config.debug_mode}
                      onChange={(e) => handleConfigChange('debug_mode', e.target.checked)}
                      className="mr-2"
                    />
                    <label className="text-sm">Debug Mode</label>
                  </div>
                </div>
              </div>
              
              {/* Agent Parameters */}
              <div>
                <h3 className="text-lg font-semibold mb-3 text-green-400">Agents</h3>
                <div className="space-y-3">
                  <div>
                    <label className="block text-sm font-medium mb-1">Number of Agents</label>
                    <input
                      type="number"
                      value={config.agent_number_of_agents}
                      onChange={(e) => handleConfigChange('agent_number_of_agents', parseInt(e.target.value))}
                      className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                      min="0"
                      max="20"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">Max Velocity</label>
                    <input
                      type="number"
                      step="0.1"
                      value={config.agent_max_velocity}
                      onChange={(e) => handleConfigChange('agent_max_velocity', parseFloat(e.target.value))}
                      className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                      min="0"
                      max="1"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">Range Radius</label>
                    <input
                      type="number"
                      step="0.5"
                      value={config.agent_range_radius}
                      onChange={(e) => handleConfigChange('agent_range_radius', parseFloat(e.target.value))}
                      className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                      min="1"
                      max="10"
                    />
                  </div>
                </div>
              </div>
              
              {/* Octopus Parameters */}
              <div>
                <h3 className="text-lg font-semibold mb-3 text-red-400">Octopus</h3>
                <div className="space-y-3">
                  <div>
                    <label className="block text-sm font-medium mb-1">Number of Arms</label>
                    <input
                      type="number"
                      value={config.octo_num_arms}
                      onChange={(e) => handleConfigChange('octo_num_arms', parseInt(e.target.value))}
                      className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                      min="4"
                      max="12"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">Max Body Velocity</label>
                    <input
                      type="number"
                      step="0.05"
                      value={config.octo_max_body_velocity}
                      onChange={(e) => handleConfigChange('octo_max_body_velocity', parseFloat(e.target.value))}
                      className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                      min="0"
                      max="1"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">Max Hue Change</label>
                    <input
                      type="number"
                      step="0.05"
                      value={config.octo_max_hue_change}
                      onChange={(e) => handleConfigChange('octo_max_hue_change', parseFloat(e.target.value))}
                      className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                      min="0.05"
                      max="1"
                    />
                  </div>
                </div>
              </div>
              
              {/* Limb Parameters */}
              <div>
                <h3 className="text-lg font-semibold mb-3 text-purple-400">Limbs</h3>
                <div className="space-y-3">
                  <div>
                    <label className="block text-sm font-medium mb-1">Limb Rows</label>
                    <input
                      type="number"
                      value={config.limb_rows}
                      onChange={(e) => handleConfigChange('limb_rows', parseInt(e.target.value))}
                      className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                      min="5"
                      max="30"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">Limb Columns</label>
                    <input
                      type="number"
                      value={config.limb_cols}
                      onChange={(e) => handleConfigChange('limb_cols', parseInt(e.target.value))}
                      className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                      min="1"
                      max="4"
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default OctopusAIVisualizer;