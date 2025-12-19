"""
Test script for Phase 2: Shape Abstraction
Tests Body shape management with BOX, CIRCLE, and TRIANGLE types
"""
import rigidRL as rigid
import time

# Create engine with rendering
engine = rigid.Engine(800, 600, 50.0, 0.016, 10, False)
engine.set_gravity(0, -9.81)

# Add ground
engine.add_collider(0, -0.5, 15, 0.5, 0)

# Create a body with circle shape
circle_body = rigid.Body(0, 4, 1.0, 0.5, 0.5)  # Default box shape
circle_body.clear_shapes()  # Remove default
circle_body.add_circle_shape(0.5)  # Add circle with radius 0.5
engine.add_body(circle_body)

# Create a body with triangle shape
triangle_body = rigid.Body(2, 4, 1.0, 1.0, 1.0)
triangle_body.clear_shapes()
triangle_body.add_triangle_shape(-0.4, -0.3, 0.4, -0.3, 0, 0.4)  # Pointing up
engine.add_body(triangle_body)

# Create a body with box shape (default)
box_body = rigid.Body(-2, 4, 1.0, 0.8, 0.6)
engine.add_body(box_body)

# Create a body with multiple shapes
multi_body = rigid.Body(4, 4, 2.0, 0.5, 0.5)
multi_body.clear_shapes()
multi_body.add_box_shape(0.6, 0.3, 0, 0)  # Center box
multi_body.add_circle_shape(0.2, -0.4, 0)  # Left wheel
multi_body.add_circle_shape(0.2, 0.4, 0)   # Right wheel
engine.add_body(multi_body)

print("Testing shape abstraction...")
print("- Circle body (center)")
print("- Triangle body (right)")
print("- Box body (left)")
print("- Multi-shape body (far right)")
print("Press window close to exit")

while engine.step():
    pass

print("Test complete!")
