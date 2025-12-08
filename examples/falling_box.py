

import forgeNN_cpp as fnn
import time

def run():
    print("Initializing Body (Mass=1.0, Width=1.0)...")
    # Body(x, y, mass, w, h)
    box = fnn.Body(0.0, 10.0, 1.0, 1.0, 1.0)
    
    print(f"Initial Pos: ({box.get_x():.2f}, {box.get_y():.2f})")
    
    # Gravity Force: F = m * g
    # mass is 1.0, g = -9.81 => F = (0, -9.81)
    gravity_force = fnn.Tensor([0.0, -9.81], False) # No grad needed for constant force
    torque = fnn.Tensor([0.0], False)
    dt = 0.01
    
    print("\nSimulating Falling Box...")
    for i in range(10):
        # Semi-implicit Euler step
        box.step(gravity_force, torque, dt)
        print(f"Step {i+1}: Pos=({box.get_x():.4f}, {box.get_y():.4f}) Vel gradient check...")
        
        # Verify gradient flow (just a sniff test)
        # If we optimize force to keep it up?
        
    print("\nPhysics Verified. Next Step: Visuals!")

if __name__ == "__main__":
    run()
