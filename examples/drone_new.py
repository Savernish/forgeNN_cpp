"""
Drone RL Training Demo - Shows learning progress with SDL rendering
Renders tensor simulation trajectories using SDL (not re-simulating)
Shows INITIAL (untrained) vs FINAL (trained) trajectory
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

# Simulation constants
g = 9.81
dt = 0.04
steps = 100
L = 0.25
m = 1.0
I = 0.5


def simulate_step(state, tl, tr):
    """Tensor-based simulation for training"""
    x, vx, y, vy, theta, omega = state[0], state[1], state[2], state[3], state[4], state[5]
    
    F = tl + tr
    tau = (tr - tl) * rigid.Tensor([L])
    
    sin_th = theta.sin()
    cos_th = theta.cos()
    
    ax = (rigid.Tensor([-1.0]) * F * sin_th) / rigid.Tensor([m])
    ay = (F * cos_th) / rigid.Tensor([m]) - rigid.Tensor([g])
    alpha = tau / rigid.Tensor([I])
    
    return rigid.Tensor.stack([
        x + vx * dt, vx + ax * dt,
        y + vy * dt, vy + ay * dt,
        theta + omega * dt, omega + alpha * dt
    ])


def get_trajectory(state_init, thrust_left, thrust_right):
    """Run simulation and return trajectory as list of (x, y, theta)"""
    trajectory = []
    curr_state = state_init
    
    for i in range(steps):
        tl = thrust_left[i]
        tr = thrust_right[i]
        curr_state = simulate_step(curr_state, tl, tr)
        
        trajectory.append({
            'x': curr_state[0].data[0, 0],
            'y': curr_state[2].data[0, 0],
            'theta': curr_state[4].data[0, 0]
        })
    
    return trajectory


def render_trajectory(title, trajectory):
    """Render a pre-computed trajectory using SDL"""
    print(f"\n=== Rendering: {title} ===")
    
    renderer = rigid.SDLRenderer(800, 600, 50)
    
    for i, state in enumerate(trajectory):
        if not renderer.process_events():
            return False
        
        renderer.clear()
        
        # Ground
        renderer.draw_box(0, -5.5, 25, 0.5, 0, 0.4, 0.4, 0.4)
        
        # Target (0, 0.5)
        renderer.draw_box(0, 0.5, 0.4, 0.4, 0, 0.0, 1.0, 0.0)
        
        # Drone
        cx, cy, th = state['x'], state['y'], state['theta']
        
        # Arm endpoints
        lx = cx - L * math.cos(th)
        ly = cy - L * math.sin(th)
        rx = cx + L * math.cos(th)
        ry = cy + L * math.sin(th)
        
        # Draw arm
        renderer.draw_line(lx, ly, rx, ry, 1.0, 1.0, 1.0)
        
        # Draw motors
        renderer.draw_box(lx, ly, 0.12, 0.12, th, 1.0, 0.3, 0.3)  # Left - red
        renderer.draw_box(rx, ry, 0.12, 0.12, th, 0.3, 0.3, 1.0)  # Right - blue
        
        # Center
        renderer.draw_box(cx, cy, 0.08, 0.08, th, 1.0, 1.0, 0.0)
        
        renderer.present()
        time.sleep(dt)
        
        if i % 25 == 0:
            print(f"Frame {i}: pos=({cx:.2f}, {cy:.2f})")
    
    print(f"Final: ({trajectory[-1]['x']:.2f}, {trajectory[-1]['y']:.2f})")
    time.sleep(0.5)
    return True


def run():
    print("=== Drone RL Training Demo ===")
    print("Shows INITIAL vs FINAL trained trajectory\n")
    
    state_init = rigid.Tensor([-5.0, 0.0, 5.0, 0.0, 0.0, 0.0], requires_grad=False)
    hover = (m * g) / 2  # 4.9N
    
    # Initial thrusts (just hover)
    initial_thrust = [hover for _ in range(steps)]
    thrust_left = rigid.Tensor(initial_thrust, requires_grad=True)
    thrust_right = rigid.Tensor(initial_thrust, requires_grad=True)
    
    # Get INITIAL trajectory
    initial_tl = rigid.Tensor(initial_thrust, requires_grad=False)
    initial_tr = rigid.Tensor(initial_thrust, requires_grad=False)
    initial_trajectory = get_trajectory(state_init, initial_tl, initial_tr)
    
    # Render INITIAL
    if not render_trajectory("INITIAL (Untrained - Hover Only)", initial_trajectory):
        return
    
    # Training
    print("\n=== Training via Backpropagation... ===")
    optimizer = rigid.AdamW([thrust_left, thrust_right], lr=0.1, weight_decay=0.0)
    
    for epoch in range(200):
        optimizer.zero_grad()
        
        curr_state = state_init
        total_loss = rigid.Tensor([0.0])
        
        for i in range(steps):
            tl = thrust_left[i]
            tr = thrust_right[i]
            curr_state = simulate_step(curr_state, tl, tr)
            
            x, y = curr_state[0], curr_state[2]
            vx, vy = curr_state[1], curr_state[3]
            theta = curr_state[4]
            
            dist_sq = x.pow(2.0) + (y - rigid.Tensor([0.5])).pow(2.0)
            vel_sq = vx.pow(2.0) + vy.pow(2.0)
            angle_sq = theta.pow(2.0)
            
            floor_dist = y + rigid.Tensor([5.0])
            floor_penalty = rigid.Tensor([1.0]) / floor_dist
            
            effort = (tl - rigid.Tensor([hover])).pow(2.0) + (tr - rigid.Tensor([hover])).pow(2.0)
            
            step_loss = dist_sq * 10.0 + vel_sq * 1.0 + angle_sq * 10.0 + effort * 0.001 + floor_penalty * 20.0
            total_loss = total_loss + step_loss

        total_loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            pos_x = curr_state[0].data[0, 0]
            pos_y = curr_state[2].data[0, 0]
            print(f"Epoch {epoch}: Loss={total_loss.data[0,0]:.1f} | Pos=({pos_x:.2f}, {pos_y:.2f})")
    
    print("Training complete!")
    
    # Get FINAL trajectory with trained thrusts
    final_trajectory = get_trajectory(state_init, thrust_left, thrust_right)
    
    # Render FINAL
    if not render_trajectory("FINAL (Trained)", final_trajectory):
        return
    
    print("\n=== Demo Complete ===")
    print("Trained drone should reach target (0, 0.5)!")


if __name__ == "__main__":
    run()
