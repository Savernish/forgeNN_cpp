"""
Text Rendering Demo for rigidRL
Demonstrates the new DrawText and LoadFont capabilities using stb_truetype.
"""
import rigidRL as rigid
import time

def main():
    # Create engine with window
    sim = rigid.Engine(width=800, height=600, scale=50.0, dt=0.016, substeps=10)
    sim.set_gravity(0.0, -9.81)
    
    # Get renderer
    renderer = sim.get_renderer()
    
    # Try to load a system font
    font_paths = [
        "C:/Windows/Fonts/consola.ttf",   # Consolas (good monospace)
        "C:/Windows/Fonts/arial.ttf",      # Arial
        "C:/Windows/Fonts/segoeui.ttf",    # Segoe UI
    ]
    
    font_loaded = False
    for font_path in font_paths:
        if renderer.load_font(font_path, 20):
            print(f"Loaded font: {font_path}")
            font_loaded = True
            break
    
    if not font_loaded:
        print("Warning: Could not load any system font!")
    
    # Add ground
    sim.Collider(0, 1.5, 20, 1, 0.0, 0.5)
    
    # Create physics objects
    bodies = []
    for i in range(5):
        box = rigid.Body.Rect(x=-3 + i * 1.5, y=3 + i * 0.5, mass=1.0, width=0.6, height=0.6)
        sim.add_body(box)
        bodies.append(box)
    
    circle = rigid.Body.Circle(x=0.56, y=16, mass=1.5, radius=0.4, restitution=1.7)
    sim.add_body(circle)
    bodies.append(circle)
    
    # Fixed timestep loop
    target_dt = 0.016  # 60 Hz physics
    last_time = time.time()
    accumulator = 0.0
    frame_count = 0
    fps_timer = time.time()
    fps = 0.0
    
    running = True
    while running:
        # Calculate time since last frame
        current_time = time.time()
        frame_time = current_time - last_time
        last_time = current_time
        accumulator += frame_time
        
        # Process events
        running = renderer.process_events()
        if not running:
            break
        
        # Fixed timestep physics updates
        while accumulator >= target_dt:
            sim.update()
            accumulator -= target_dt
        
        # Render (as fast as possible, VSync should cap it)
        renderer.clear()
        sim.render_bodies()
        
        if font_loaded:
            renderer.draw_text(10, 570, f"FPS: {fps:.1f}", 0.0, 1.0, 0.0)
            renderer.draw_text(10, 545, f"Objects: {len(bodies)}", 1.0, 1.0, 0.0)
            renderer.draw_text(280, 570, "rigidRL Text Demo", 1.0, 1.0, 1.0)
            renderer.draw_text(10, 30, "Press ESC to exit", 0.7, 0.7, 0.7)
        
        renderer.present()
        
        # FPS calculation
        frame_count += 1
        if current_time - fps_timer >= 1.0:
            fps = frame_count
            frame_count = 0
            fps_timer = current_time
    
    print(f"Final FPS: {fps:.0f}")

if __name__ == "__main__":
    main()
