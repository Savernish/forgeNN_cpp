"""
Drone Liftoff Demo - Realistic motor thrust demonstration
1. Drone on ground, gradually increases thrust
2. Lifts off and stabilizes at hover height
3. Payload drops onto it, causing slow descent
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
    print("=== Drone Liftoff Demo ===")
    engine = rigid.Engine(800, 600, 40, 0.016, 30)
    engine.set_gravity(0, -9.81)
    running = True
    
    # Ground
    engine.add_collider(0, -1, 20, 1, 0)
    
    # Drone starting on ground
    drone = rigid.Body(0, -0.35, 1.0, 2.0, 0.3)
    
    # Motors on drone
    motor_left = rigid.Motor(-0.9, 0.2, 0.2, 0.3, 0.1, 25.0)
    motor_right = rigid.Motor(0.9, 0.2, 0.2, 0.3, 0.1, 25.0)
    drone.add_motor(motor_left)
    drone.add_motor(motor_right)
    engine.add_body(drone)
    
    # Calculate hover thrust
    drone_mass = 1.0 + 0.2
    hover_thrust = (drone_mass * 9.81) / 2
    target_height = 3.0
    
    print(f"Drone mass: {drone_mass:.1f} kg")
    print(f"Hover thrust: {hover_thrust:.2f} N per motor")
    print(f"Target height: {target_height} m")
    
    # Phase 1: Liftoff with altitude control
    print("\nPhase 1: Liftoff...")
    frame = 0
    while running and frame < 240:
        y = drone.get_y()
        vy = drone.vel.get(1, 0)
        
        # Simple altitude controller
        height_error = target_height - y
        vertical_speed_target = height_error * 2.0
        speed_error = vertical_speed_target - vy
        
        thrust = hover_thrust + speed_error * 0.5
        thrust = max(0, min(thrust, 15))
        
        motor_left.thrust = thrust
        motor_right.thrust = thrust
        
        if not engine.step():
            running = False
            break
        
        if frame % 40 == 0:
            print(f"Frame {frame}: y={y:.2f}, thrust={thrust:.1f}N")
        frame += 1
    
    if not running:
        return
    
    # Phase 2: Stable hover
    print("\nPhase 2: Hovering at target height...")
    for _ in range(60):
        y = drone.get_y()
        height_error = target_height - y
        thrust = hover_thrust + height_error * 1.0
        motor_left.thrust = thrust
        motor_right.thrust = thrust
        
        if not engine.step():
            running = False
            break
        frame += 1
    
    if not running:
        return
    
    print(f"Stable at y={drone.get_y():.2f}")
    
    # Phase 3: Drop payload ABOVE drone
    print("\nPhase 3: Dropping payload...")
    payload = rigid.Body(0, drone.get_y() + 4, 0.3, 0.5, 0.5)
    engine.add_body(payload)
    
    while running and frame < 480:
        y = drone.get_y()
        height_error = target_height - y
        vel_y = drone.vel.get(1, 0)
        thrust = hover_thrust + height_error * 0.5 - vel_y * 0.2
        thrust = max(0, min(thrust, 15))
        motor_left.thrust = thrust
        motor_right.thrust = thrust
        
        if not engine.step():
            running = False
            break
        
        if frame % 30 == 0:
            py = payload.get_y()
            print(f"Frame {frame}: drone_y={y:.2f}, payload_y={py:.2f}")
        frame += 1
    
    if not running:
        return
    
    # Phase 4: Payload landed - same thrust but heavier now
    print("\nPhase 4: Payload on drone - descending with same thrust...")
    combined_mass = drone_mass + 0.3
    required = (combined_mass * 9.81) / 2
    print(f"Combined mass: {combined_mass:.1f} kg")
    print(f"Need {required:.2f} N, have {hover_thrust:.2f} N - will descend!")
    
    motor_left.thrust = hover_thrust
    motor_right.thrust = hover_thrust
    
    while running and frame < 660:
        if not engine.step():
            running = False
            break
        
        if frame % 30 == 0:
            y = drone.get_y()
            print(f"Frame {frame}: y={y:.2f} (descending)")
        frame += 1
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    run()
