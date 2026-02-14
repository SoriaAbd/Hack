// AdaptNav Web Dashboard JavaScript

class AdaptNavDashboard {
    constructor() {
        this.websocket = null;
        this.canvas = document.getElementById('main-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.isConnected = false;
        
        // Data storage
        this.robotPose = null;
        this.globalPlan = null;
        this.obstacles = [];
        this.navigationState = null;
        this.safetyStatus = null;
        this.lidarScan = null;
        
        // Visualization parameters
        this.scale = 50; // pixels per meter
        this.centerX = 0;
        this.centerY = 0;
        
        this.initializeCanvas();
        this.connectWebSocket();
        this.startRenderLoop();
    }
    
    initializeCanvas() {
        // Set canvas size
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
        
        // Set initial view center
        this.centerX = this.canvas.width / 2;
        this.centerY = this.canvas.height / 2;
    }
    
    resizeCanvas() {
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
    }
    
    connectWebSocket() {
        const wsUrl = 'ws://localhost:8765';
        
        try {
            this.websocket = new WebSocket(wsUrl);
            
            this.websocket.onopen = () => {
                console.log('WebSocket connected');
                this.setConnectionStatus(true);
            };
            
            this.websocket.onmessage = (event) => {
                this.handleWebSocketMessage(event.data);
            };
            
            this.websocket.onclose = () => {
                console.log('WebSocket disconnected');
                this.setConnectionStatus(false);
                // Attempt to reconnect after 3 seconds
                setTimeout(() => this.connectWebSocket(), 3000);
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.setConnectionStatus(false);
            };
            
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
            this.setConnectionStatus(false);
            // Attempt to reconnect after 3 seconds
            setTimeout(() => this.connectWebSocket(), 3000);
        }
    }
    
    setConnectionStatus(connected) {
        this.isConnected = connected;
        const indicator = document.getElementById('status-indicator');
        const text = document.getElementById('connection-text');
        
        if (connected) {
            indicator.classList.add('connected');
            text.textContent = 'Connected';
        } else {
            indicator.classList.remove('connected');
            text.textContent = 'Disconnected';
        }
    }
    
    handleWebSocketMessage(data) {
        try {
            const message = JSON.parse(data);
            
            switch (message.type) {
                case 'initial_data':
                case 'navigation_update':
                    this.updateData(message.data);
                    break;
                case 'command_response':
                    this.handleCommandResponse(message);
                    break;
                case 'error':
                    console.error('Server error:', message.message);
                    break;
                default:
                    console.warn('Unknown message type:', message.type);
            }
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    }
    
    updateData(data) {
        // Update stored data
        this.robotPose = data.robot_pose;
        this.globalPlan = data.global_plan;
        this.obstacles = data.obstacles || [];
        this.navigationState = data.navigation_state;
        this.safetyStatus = data.safety_status;
        this.lidarScan = data.lidar_scan;
        
        // Update UI elements
        this.updateNavigationStateUI();
        this.updateSafetyStatusUI();
        this.updatePerformanceMetricsUI(data.performance_metrics);
        this.updateObstacleListUI();
        this.updateRobotStatusUI();
    }
    
    updateNavigationStateUI() {
        if (!this.navigationState) return;
        
        document.getElementById('nav-state').textContent = this.navigationState.state;
        document.getElementById('distance-to-goal').textContent = 
            this.navigationState.distance_to_goal ? 
            `${this.navigationState.distance_to_goal.toFixed(2)}m` : '--';
        document.getElementById('progress').textContent = 
            this.navigationState.progress_percentage ? 
            `${this.navigationState.progress_percentage.toFixed(1)}%` : '--';
        document.getElementById('reasoning-text').textContent = 
            this.navigationState.reasoning || 'No reasoning available';
    }
    
    updateSafetyStatusUI() {
        if (!this.safetyStatus) return;
        
        const safetyStateElement = document.getElementById('safety-state');
        safetyStateElement.textContent = this.safetyStatus.state;
        
        // Update color based on state
        safetyStateElement.className = 'status-value';
        if (this.safetyStatus.state === 'SAFE') {
            safetyStateElement.classList.add('state-safe');
        } else if (this.safetyStatus.state === 'CAUTION') {
            safetyStateElement.classList.add('state-caution');
        } else if (this.safetyStatus.state === 'EMERGENCY_STOP') {
            safetyStateElement.classList.add('state-emergency');
        }
        
        document.getElementById('closest-obstacle').textContent = 
            this.safetyStatus.closest_obstacle_distance !== undefined ? 
            `${this.safetyStatus.closest_obstacle_distance.toFixed(2)}m` : '--';
        document.getElementById('velocity-scale').textContent = 
            this.safetyStatus.velocity_scale !== undefined ? 
            `${(this.safetyStatus.velocity_scale * 100).toFixed(0)}%` : '--';
        document.getElementById('override-active').textContent = 
            this.safetyStatus.override_active !== undefined ? 
            (this.safetyStatus.override_active ? 'Yes' : 'No') : '--';
    }
    
    updatePerformanceMetricsUI(metrics) {
        if (!metrics) return;
        
        document.getElementById('control-freq').textContent = 
            metrics.control_frequency ? metrics.control_frequency.toFixed(1) : '--';
        document.getElementById('detection-latency').textContent = 
            metrics.detection_latency ? metrics.detection_latency.toFixed(1) : '--';
    }
    
    updateObstacleListUI() {
        const obstacleList = document.getElementById('obstacle-list');
        
        if (this.obstacles.length === 0) {
            obstacleList.innerHTML = `
                <div style="color: #aaa; text-align: center; padding: 20px;">
                    No obstacles detected
                </div>
            `;
            return;
        }
        
        let html = '';
        this.obstacles.forEach(obstacle => {
            const distance = this.robotPose ? 
                Math.sqrt(
                    Math.pow(obstacle.position.x - this.robotPose.position.x, 2) +
                    Math.pow(obstacle.position.y - this.robotPose.position.y, 2)
                ).toFixed(2) : '--';
            
            const velocity = Math.sqrt(
                obstacle.velocity.x * obstacle.velocity.x +
                obstacle.velocity.y * obstacle.velocity.y
            ).toFixed(2);
            
            html += `
                <div class="obstacle-item obstacle-${obstacle.classification}">
                    <div><strong>${obstacle.classification} #${obstacle.id}</strong></div>
                    <div>Distance: ${distance}m</div>
                    <div>Velocity: ${velocity}m/s</div>
                    <div>Confidence: ${(obstacle.confidence * 100).toFixed(0)}%</div>
                </div>
            `;
        });
        
        obstacleList.innerHTML = html;
    }
    
    updateRobotStatusUI() {
        if (!this.robotPose) return;
        
        document.getElementById('robot-position').textContent = 
            `(${this.robotPose.position.x.toFixed(2)}, ${this.robotPose.position.y.toFixed(2)})`;
        
        if (this.robotPose.velocity) {
            document.getElementById('linear-velocity').textContent = 
                `${this.robotPose.velocity.linear.x.toFixed(2)} m/s`;
            document.getElementById('angular-velocity').textContent = 
                `${this.robotPose.velocity.angular.z.toFixed(2)} rad/s`;
        }
    }
    
    handleCommandResponse(message) {
        console.log(`Command ${message.command}: ${message.success ? 'Success' : 'Failed'}`);
        console.log(`Message: ${message.message}`);
        
        // You could show a toast notification here
        // For now, just log to console
    }
    
    startRenderLoop() {
        const render = () => {
            this.renderVisualization();
            requestAnimationFrame(render);
        };
        render();
    }
    
    renderVisualization() {
        // Clear canvas
        this.ctx.fillStyle = '#1e3c72';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw grid
        this.drawGrid();
        
        // Draw global plan
        if (this.globalPlan) {
            this.drawGlobalPlan();
        }
        
        // Draw obstacles
        this.drawObstacles();
        
        // Draw LiDAR scan
        if (this.lidarScan) {
            this.drawLidarScan();
        }
        
        // Draw robot
        if (this.robotPose) {
            this.drawRobot();
        }
        
        // Draw safety zones
        if (this.robotPose && this.safetyStatus) {
            this.drawSafetyZones();
        }
    }
    
    drawGrid() {
        this.ctx.strokeStyle = '#333';
        this.ctx.lineWidth = 1;
        
        const gridSize = this.scale; // 1 meter grid
        
        // Vertical lines
        for (let x = this.centerX % gridSize; x < this.canvas.width; x += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.canvas.height);
            this.ctx.stroke();
        }
        
        // Horizontal lines
        for (let y = this.centerY % gridSize; y < this.canvas.height; y += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.canvas.width, y);
            this.ctx.stroke();
        }
    }
    
    drawGlobalPlan() {
        if (!this.globalPlan.waypoints || this.globalPlan.waypoints.length === 0) return;
        
        this.ctx.strokeStyle = '#4caf50';
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();
        
        this.globalPlan.waypoints.forEach((waypoint, index) => {
            const screenPos = this.worldToScreen(waypoint.position.x, waypoint.position.y);
            
            if (index === 0) {
                this.ctx.moveTo(screenPos.x, screenPos.y);
            } else {
                this.ctx.lineTo(screenPos.x, screenPos.y);
            }
        });
        
        this.ctx.stroke();
        
        // Draw waypoint markers
        this.ctx.fillStyle = '#4caf50';
        this.globalPlan.waypoints.forEach(waypoint => {
            const screenPos = this.worldToScreen(waypoint.position.x, waypoint.position.y);
            this.ctx.beginPath();
            this.ctx.arc(screenPos.x, screenPos.y, 4, 0, 2 * Math.PI);
            this.ctx.fill();
        });
    }
    
    drawObstacles() {
        this.obstacles.forEach(obstacle => {
            const screenPos = this.worldToScreen(obstacle.position.x, obstacle.position.y);
            
            // Choose color based on classification
            let color;
            switch (obstacle.classification) {
                case 'worker':
                    color = '#2196f3';
                    break;
                case 'forklift':
                    color = '#ff9800';
                    break;
                default:
                    color = '#9e9e9e';
            }
            
            // Draw obstacle
            this.ctx.fillStyle = color;
            this.ctx.beginPath();
            this.ctx.arc(screenPos.x, screenPos.y, 15, 0, 2 * Math.PI);
            this.ctx.fill();
            
            // Draw velocity vector
            const velMagnitude = Math.sqrt(
                obstacle.velocity.x * obstacle.velocity.x +
                obstacle.velocity.y * obstacle.velocity.y
            );
            
            if (velMagnitude > 0.1) {
                const velScale = 50; // Scale factor for velocity visualization
                const endX = screenPos.x + obstacle.velocity.x * velScale;
                const endY = screenPos.y - obstacle.velocity.y * velScale; // Flip Y
                
                this.ctx.strokeStyle = color;
                this.ctx.lineWidth = 2;
                this.ctx.beginPath();
                this.ctx.moveTo(screenPos.x, screenPos.y);
                this.ctx.lineTo(endX, endY);
                this.ctx.stroke();
                
                // Arrow head
                const angle = Math.atan2(endY - screenPos.y, endX - screenPos.x);
                const arrowLength = 8;
                this.ctx.beginPath();
                this.ctx.moveTo(endX, endY);
                this.ctx.lineTo(
                    endX - arrowLength * Math.cos(angle - Math.PI / 6),
                    endY - arrowLength * Math.sin(angle - Math.PI / 6)
                );
                this.ctx.moveTo(endX, endY);
                this.ctx.lineTo(
                    endX - arrowLength * Math.cos(angle + Math.PI / 6),
                    endY - arrowLength * Math.sin(angle + Math.PI / 6)
                );
                this.ctx.stroke();
            }
            
            // Draw ID label
            this.ctx.fillStyle = '#ffffff';
            this.ctx.font = '12px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(obstacle.id.toString(), screenPos.x, screenPos.y - 20);
        });
    }
    
    drawLidarScan() {
        if (!this.lidarScan.ranges || !this.robotPose) return;
        
        const robotScreenPos = this.worldToScreen(
            this.robotPose.position.x, 
            this.robotPose.position.y
        );
        
        this.ctx.strokeStyle = '#00ffff';
        this.ctx.lineWidth = 1;
        
        this.lidarScan.ranges.forEach((range, index) => {
            if (range < this.lidarScan.range_min || range > this.lidarScan.range_max) return;
            
            const angle = this.lidarScan.angle_min + index * this.lidarScan.angle_increment;
            const endX = robotScreenPos.x + Math.cos(angle) * range * this.scale;
            const endY = robotScreenPos.y - Math.sin(angle) * range * this.scale; // Flip Y
            
            this.ctx.beginPath();
            this.ctx.moveTo(robotScreenPos.x, robotScreenPos.y);
            this.ctx.lineTo(endX, endY);
            this.ctx.stroke();
        });
    }
    
    drawRobot() {
        const screenPos = this.worldToScreen(
            this.robotPose.position.x, 
            this.robotPose.position.y
        );
        
        // Draw robot body
        this.ctx.fillStyle = '#ffffff';
        this.ctx.beginPath();
        this.ctx.arc(screenPos.x, screenPos.y, 12, 0, 2 * Math.PI);
        this.ctx.fill();
        
        // Draw orientation indicator
        // For simplicity, assume robot is facing along x-axis
        // In a real implementation, you'd extract orientation from quaternion
        this.ctx.strokeStyle = '#ffffff';
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();
        this.ctx.moveTo(screenPos.x, screenPos.y);
        this.ctx.lineTo(screenPos.x + 15, screenPos.y);
        this.ctx.stroke();
    }
    
    drawSafetyZones() {
        const screenPos = this.worldToScreen(
            this.robotPose.position.x, 
            this.robotPose.position.y
        );
        
        // Collision zone
        this.ctx.strokeStyle = this.getSafetyZoneColor();
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([5, 5]);
        this.ctx.beginPath();
        this.ctx.arc(screenPos.x, screenPos.y, 0.5 * this.scale, 0, 2 * Math.PI);
        this.ctx.stroke();
        
        // Emergency stop zone
        this.ctx.strokeStyle = '#ff4444';
        this.ctx.beginPath();
        this.ctx.arc(screenPos.x, screenPos.y, 0.3 * this.scale, 0, 2 * Math.PI);
        this.ctx.stroke();
        
        this.ctx.setLineDash([]); // Reset line dash
    }
    
    getSafetyZoneColor() {
        if (!this.safetyStatus) return '#44ff44';
        
        switch (this.safetyStatus.state) {
            case 'SAFE':
                return '#44ff44';
            case 'CAUTION':
                return '#ffaa44';
            case 'EMERGENCY_STOP':
                return '#ff4444';
            default:
                return '#44ff44';
        }
    }
    
    worldToScreen(worldX, worldY) {
        // Simple transformation - in a real implementation you'd want
        // proper coordinate frame handling
        return {
            x: this.centerX + worldX * this.scale,
            y: this.centerY - worldY * this.scale // Flip Y axis
        };
    }
}

// Command sending function
function sendCommand(command) {
    if (dashboard.websocket && dashboard.websocket.readyState === WebSocket.OPEN) {
        const message = {
            command: command
        };
        dashboard.websocket.send(JSON.stringify(message));
    } else {
        console.error('WebSocket not connected');
    }
}

// Initialize dashboard when page loads
let dashboard;
window.addEventListener('load', () => {
    dashboard = new AdaptNavDashboard();
});