
import forgeNN_cpp as fnn
import time

def run():
    print("Initializing Engine (800x600)...")
    # Engine(w, h, scale, dt)
    engine = fnn.Engine(800, 600, 50.0, 0.016)
    
    print("Creating Body...")
    # Body(x, y, mass, w, h)
    box = fnn.Body(0.0, 5.0, 1.0, 1.0, 1.0)
    
    print("Adding Body to Engine...")
    engine.add_body(box)
    
    print("Setting Gravity...")
    engine.set_gravity(0.0, -9.81)
    
    print("Starting Simulation Loop...")
    running = True
    while running:
        # Engine.step() handles events, physics, and rendering
        running = engine.step()
        
        # We can still read body state
        # print(f"Pos: {box.get_y()}")
        
        # Throttle (Engine doesn't enforce FPS yet, just uses dt for physics)
        time.sleep(0.016)

    print("Engine Loop Finished.")

if __name__ == "__main__":
    run()
