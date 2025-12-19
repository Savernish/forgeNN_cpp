"""
Test script for new rendering primitives: Triangle and CircleFilled
"""
import rigidRL as rigid
import time

# Create engine with rendering
engine = rigid.Engine(800, 600, 50.0, 0.016, 1, False)

# Get renderer
renderer = engine.get_renderer()

print("Testing new rendering primitives...")
print("Press window close to exit")

running = True
while running:
    renderer.clear()
    
    # Draw filled circle (red)
    renderer.draw_circle_filled(0, 3, 0.5, 1.0, 0.2, 0.2)
    
    # Draw outline circle (green)
    renderer.draw_circle(-2, 3, 0.5, 0.2, 1.0, 0.2)
    
    # Draw filled triangle (blue)
    renderer.draw_triangle_filled(2, 2, 3, 4, 4, 2, 0.2, 0.2, 1.0)
    
    # Draw outline triangle (yellow)
    renderer.draw_triangle(-4, 2, -3, 4, -2, 2, 1.0, 1.0, 0.2)
    
    # Draw some boxes for comparison
    renderer.draw_box_filled(0, 1, 1, 1, 0, 0.5, 0.5, 0.5)
    
    renderer.present()
    running = renderer.process_events()

print("Test complete!")
