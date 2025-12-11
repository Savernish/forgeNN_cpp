"""
Drone Instability Demo - Asymmetric payload causes tipping
Box falls on left side of drone, causing rotation and crash
"""
import sys
import os
import math
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(os.path.dirname(script_dir), 'diff_sim_core')
os.add_dll_directory(core_dir)
sys.path.insert(0, core_dir)

import rigidRL as rigid

def run():
    print("=== Drone Instability Demo ===")
    engine = rigid.Engine(800, 600, 40, 0.016, 30)
    engine.set_gravity(0, -9.81)
    running = True
    renderer = engine.get_renderer()
    
    # Ground
    engine.add_collider(0, -1, 20, 1, 0)
    
    # Drone
    drone = rigid.Body(0, -0.35, 1.0, 2.0, 0.3)
    motor_left = rigid.Motor(-0.9, 0.2, 0.2, 0.3, 0.1, 25.0)
    motor_right = rigid.Motor(0.9, 0.2, 0.2, 0.3, 0.1, 25.0)
    drone.add_motor(motor_left)
    drone.add_motor(motor_right)
    engine.add_body(drone)

    
    drone_mass = 1.0 + 0.2
    hover_thrust = (drone_mass * 9.81) / 2
    target_height = 3.0
    
    print(f"Hover thrust: {hover_thrust:.2f} N per motor")
    
    # Phase 1: Rise and hover
    print("\nPhase 1: Rising to hover...")
    frame = 0
    while running and frame < 180:
        y = drone.get_y()
        height_error = target_height - y
        vel_y = drone.vel.get(1, 0)
        thrust = hover_thrust + height_error * 1.0 - vel_y * 0.3
        thrust = max(0, min(thrust, 15))
        motor_left.thrust = thrust
        motor_right.thrust = thrust
        
        # Manual render loop for custom drawing, in the future there will be better integration.
        if not renderer.process_events():
            running = False
            break
        engine.update()  # Physics only
        renderer.clear()
        engine.render_bodies()
        renderer.draw_circle(motor_left.local_x + drone.get_x(), motor_left.local_y + drone.get_y() - motor_left.height / 2, 0.4, 1.0, 1.0, 1.0)  # White circle around drone
        renderer.draw_circle(motor_right.local_x + drone.get_x(), motor_right.local_y + drone.get_y() - motor_right.height / 2, 0.4, 1.0, 1.0, 1.0)  # White circle around drone
        renderer.present()
        # Since we are manually updating the renderer, we need to sleep to maintain frame rate.
        time.sleep(0.016)
        frame += 1
    
    if not running:
        return
    
    print(f"Hovering at y={drone.get_y():.2f}")
    
    # Phase 2: Drop box to the LEFT of drone
    print("\nPhase 2: Dropping box on LEFT side...")
    box = rigid.Body(-0.7, drone.get_y() + 3, 0.4, 0.5, 0.5)
    engine.add_body(box)
    
    while running and frame < 360:
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
            rot = math.degrees(drone.get_rotation())
            by = box.get_y()
            print(f"Frame {frame}: drone_rot={rot:.1f}°, box_y={by:.2f}")
        frame += 1
    
    if not running:
        return
    
    # Phase 3: Watch it become unstable
    print("\nPhase 3: Asymmetric weight causing instability...")
    while running and frame < 600:
        motor_left.thrust = hover_thrust
        motor_right.thrust = hover_thrust
        
        if not engine.step():
            running = False
            break
        
        if frame % 30 == 0:
            y = drone.get_y()
            rot = math.degrees(drone.get_rotation())
            print(f"Frame {frame}: y={y:.2f}, rot={rot:.1f}°")
        frame += 1
    
    if not running:
        return
    
    final_rot = math.degrees(drone.get_rotation())
    print(f"\nFinal rotation: {final_rot:.1f}°")
    if abs(final_rot) > 30:
        print("DRONE TIPPED OVER!")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    run()
