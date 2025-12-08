import forgeNN_cpp as fnn
import time

width = 1600
height = 900
scale = 100

def run():
    print("Initializing Engine (and Window)...")
    # Engine creates the window internally
    engine = fnn.Engine(width, height, scale, 0.016, 10) # 10 substeps
    
    print("Initializing Body...")
    # Body (x, y, w, h, mass)
    # Spawn HIGHER (y=8.0) to avoid spawning inside the peak (y=6.0)
    # Slight offset (x=0.1) to ensure it picks a side
    box = fnn.Body(-3, 7.0, 1.0, 1.0, 1.0) 
    engine.add_body(box)
    engine.set_gravity(7.81, -9.81)
    
    # Get the renderer from the engine for manual drawing
    renderer = engine.get_renderer()
    
    print("Adding Ground Segments...")
    # 1. The "Inverted V" (Splitter/Peak) on top
    # Left face of peak
    engine.add_ground_segment(-4, 2, 0, 6, 0.1) # Icy Peak
    # Right face of peak
    #engine.add_ground_segment(0, 6, 4, 2, 0.1)  # Icy Peak
    
    # 2. Flat Bottom (Ground)
    # Sticky Floor
    engine.add_ground_segment(-20, 0, 20, 0, 1.0)
    
    # Get the renderer from the engine for manual drawing
    renderer = engine.get_renderer()
    
    print("Starting Loop...")
    running = True
    while running:
        if not renderer.process_events():
            break
            
        # 2. Physics
        engine.update()
        
        # 3. Rendering
        renderer.clear()
        
        # Draw Peak
        renderer.draw_line(-4, 2, 0, 6, 0.0, 1.0, 1.0) # Cyan
        renderer.draw_line(0, 6, 4, 2, 0.0, 1.0, 1.0)
        
        # Draw Floor
        renderer.draw_line(-20, 0, 20, 0, 0.0, 1.0, 0.0) # Green
        
        # Render Simulation Objects
        engine.render_bodies()
        
        # Present
        renderer.present()
        
        time.sleep(0.016)

    print("Engine Loop Finished.")

if __name__ == "__main__":
    run()
