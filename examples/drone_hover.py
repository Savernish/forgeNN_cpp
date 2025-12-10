"""
Drone Hover Test - Tests motor thrust system
A drone with 2 motors tries to hover by balancing thrust
"""
import sys
import os
import math

script_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(os.path.dirname(script_dir), 'diff_sim_core')
os.add_dll_directory(core_dir)
sys.path.insert(0, core_dir)

import rigidRL as rigid

def run():
    print("=== Drone Hover Test ===")
    engine = rigid.Engine(800, 600, 40, 0.016, 30)
    engine.set_gravity(0, -9.81)
    
    # Floor
    engine.add_collider(0, -5, 20, 1, 0)
    
    # Create drone body (wide rectangle)
    drone = rigid.Body(0, 3, 1.0, 2.0, 0.3)
    
    # Create motors with MORE separation
    motor_left = rigid.Motor(-0.9, 0.2, 0.2, 0.3, 0.1, 20.0)
    motor_right = rigid.Motor(0.9, 0.2, 0.2, 0.3, 0.1, 20.0)
    
    drone.add_motor(motor_left)
    drone.add_motor(motor_right)
    engine.add_body(drone)
    
    total_mass = 1.0 + 0.2
    hover_thrust = (total_mass * 9.81) / 2
    print(f"Hover thrust per motor: {hover_thrust:.2f} N")
    
    # Test 1: Hover
    print("\nPhase 1: Hover (3 seconds)")
    motor_left.thrust = hover_thrust
    motor_right.thrust = hover_thrust
    
    frame = 0
    while frame < 180:
        if not engine.step():
            break
        frame += 1
    
    # Test 2: Hard right turn
    print("Phase 2: FULL LEFT MOTOR (2 seconds) - should spin right!")
    motor_left.thrust = hover_thrust * 1.5
    motor_right.thrust = hover_thrust * 0.5
    
    while frame < 300:
        if not engine.step():
            break
        if frame % 30 == 0:
            rot = math.degrees(drone.get_rotation())
            print(f"Frame {frame}: rot={rot:.1f}°")
        frame += 1
    
    # Test 3: Fall and crash
    print("Phase 3: NO THRUST - fall!")
    motor_left.thrust = 0
    motor_right.thrust = 0
    
    while frame < 450:
        if not engine.step():
            break
        if frame % 30 == 0:
            y = drone.get_y()
            rot = math.degrees(drone.get_rotation())
            print(f"Frame {frame}: y={y:.1f}, rot={rot:.1f}°")
        frame += 1
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    run()
