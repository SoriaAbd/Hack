#!/usr/bin/env python3
"""
Simple test script to verify navigation components can be imported and instantiated.
"""

def test_imports():
    """Test that all navigation components can be imported."""
    try:
        # Test core imports
        from adaptnav.core.warehouse_map import WarehouseMap
        from adaptnav.core.path import Path, Waypoint
        print("✓ Core modules imported successfully")
        
        # Test planning imports
        from adaptnav.planning.astar_planner import AStarPlanner
        from adaptnav.planning.path_smoother import PathSmoother
        print("✓ Planning modules imported successfully")
        
        # Test navigation imports  
        from adaptnav.navigation.navigation_state_machine import NavigationStateMachine, NavigationState
        from adaptnav.navigation.hybrid_navigator import HybridNavigator
        print("✓ Navigation modules imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of key components."""
    try:
        from adaptnav.core.warehouse_map import WarehouseMap
        from adaptnav.core.path import Path, Waypoint
        from adaptnav.planning.astar_planner import AStarPlanner
        from adaptnav.navigation.navigation_state_machine import NavigationStateMachine, NavigationState
        from adaptnav.navigation.hybrid_navigator import HybridNavigator
        
        # Test warehouse map
        warehouse_map = WarehouseMap(10.0, 10.0, 0.1)
        print("✓ WarehouseMap created successfully")
        
        # Test path
        waypoints = [Waypoint(0, 0), Waypoint(1, 1), Waypoint(2, 2)]
        path = Path(waypoints)
        print(f"✓ Path created with {len(path.waypoints)} waypoints")
        
        # Test A* planner
        planner = AStarPlanner(warehouse_map)
        print("✓ AStarPlanner created successfully")
        
        # Test state machine
        state_machine = NavigationStateMachine()
        print(f"✓ NavigationStateMachine created, initial state: {state_machine.current_state}")
        
        # Test hybrid navigator
        navigator = HybridNavigator()
        print("✓ HybridNavigator created successfully")
        
        return True
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing AdaptNav Navigation Components")
    print("=" * 50)
    
    # Test imports
    if test_imports():
        print("\n" + "=" * 50)
        # Test basic functionality
        test_basic_functionality()
    
    print("\n" + "=" * 50)
    print("Test completed")