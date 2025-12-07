import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../diff_sim_core'))

import forgeNN_cpp as fnn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

start_time = time.time()

# --- 1. Simulation Constants ---
g = 9.81
m = 1.0   # Mass (kg)
I = 0.5   # Moment of Inertia
L = 0.25  # Arm length (center to motor)
dt = 0.04
steps = 100 # 4 seconds

# --- 2. Setup ---
# Initial State
x_init = -5.0
y_init = 5.0

# Controls: Left and Right Motor Thrusts
thrust_left = []
thrust_right = []
for i in range(steps):
    # Init at hover thrust (mg/2) to prevent immediate falling
    tl = fnn.Tensor([4.9], requires_grad=True)
    tr = fnn.Tensor([4.9], requires_grad=True)
    thrust_left.append(tl)
    thrust_right.append(tr)

# Optimizer: AdamW with weight_decay=0 (Decay would push thrust to 0 -> crash)
params = thrust_left + thrust_right
optimizer = fnn.AdamW(params, lr=0.1, weight_decay=0.0)

print("Starting Drone Optimization...")

def simulate_step(curr_x, curr_vx, curr_y, curr_vy, curr_theta, curr_dtheta, tl, tr):
    # Physics
    F = tl + tr
    tau = (tr - tl) * fnn.Tensor([L])
    
    sin_th = curr_theta.sin()
    cos_th = curr_theta.cos()
    
    # Accelerations
    ax = (fnn.Tensor([-1.0]) * F * sin_th) / fnn.Tensor([m])
    ay = (F * cos_th) / fnn.Tensor([m]) - fnn.Tensor([g])
    alpha = tau / fnn.Tensor([I])
    
    # Integration
    new_x = curr_x + curr_vx * dt
    new_vx = curr_vx + ax * dt
    
    new_y = curr_y + curr_vy * dt
    new_vy = curr_vy + ay * dt
    
    new_theta = curr_theta + curr_dtheta * dt
    new_dtheta = curr_dtheta + alpha * dt
    
    return new_x, new_vx, new_y, new_vy, new_theta, new_dtheta

# --- 3. Optimization Loop ---
for epoch in range(200):
    optimizer.zero_grad()
    
    curr_x = fnn.Tensor([x_init])
    curr_vx = fnn.Tensor([0.0])
    curr_y = fnn.Tensor([y_init])
    curr_vy = fnn.Tensor([0.0])
    curr_theta = fnn.Tensor([0.0])
    curr_dtheta = fnn.Tensor([0.0])
    
    total_loss = fnn.Tensor([0.0])
    
    # Tracking for printing
    min_y = 100.0
    
    for i in range(steps):
        tl = thrust_left[i]
        tr = thrust_right[i]
        
        curr_x, curr_vx, curr_y, curr_vy, curr_theta, curr_dtheta = simulate_step(
            curr_x, curr_vx, curr_y, curr_vy, curr_theta, curr_dtheta, tl, tr
        )
        
        # Track stats (using data access for float value)
        if curr_y.data[0,0] < min_y:
            min_y = curr_y.data[0,0]

        # Loss Terms
        # Target: x=0, y=0.5 (Hover 0.5m above ground)
        dist_sq = curr_x*curr_x + (curr_y - fnn.Tensor([0.5]))*(curr_y - fnn.Tensor([0.5]))
        vel_sq = curr_vx*curr_vx + curr_vy*curr_vy
        angle_sq = curr_theta*curr_theta
        
        # Floor Barrier: 1.0 / (y + 5.0)
        # Singularity at -5.0.
        floor_penalty = fnn.Tensor([1.0]) / (curr_y + fnn.Tensor([5.0]))

        # Effort relative to hover thrust
        effort = (tl-fnn.Tensor([4.9]))*(tl-fnn.Tensor([4.9])) + \
                 (tr-fnn.Tensor([4.9]))*(tr-fnn.Tensor([4.9]))
        
        # Reduce barrier weight to 20.0 to allow getting closer to ground
        step_loss = dist_sq * 10.0 + vel_sq * 1.0 + angle_sq * 10.0 + effort * 0.001 + floor_penalty * 20.0
        total_loss = total_loss + step_loss
        
    total_loss = total_loss.sum()
    total_loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss={total_loss.data[0,0]:.2f} | MinY={min_y:.2f} | FinalPos=({curr_x.data[0,0]:.2f}, {curr_y.data[0,0]:.2f})")

opt_time = time.time() - start_time
print(f"Optimization Complete (Time: {opt_time:.4f}s)")

# --- 4. Animation ---
print("\nGenerating Animation...")

history_x = []
history_y = []
history_theta = []

curr_x = fnn.Tensor([x_init])
curr_vx = fnn.Tensor([0.0])
curr_y = fnn.Tensor([y_init])
curr_vy = fnn.Tensor([0.0])
curr_theta = fnn.Tensor([0.0])
curr_dtheta = fnn.Tensor([0.0])

for i in range(steps):
    tl = thrust_left[i]
    tr = thrust_right[i]
    
    curr_x, curr_vx, curr_y, curr_vy, curr_theta, curr_dtheta = simulate_step(
        curr_x, curr_vx, curr_y, curr_vy, curr_theta, curr_dtheta, tl, tr
    )
    
    history_x.append(curr_x.data[0,0])
    history_y.append(curr_y.data[0,0])
    history_theta.append(curr_theta.data[0,0])

# Animate
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-6, 6)
ax.set_ylim(-1, 8)
ax.grid()

line, = ax.plot([], [], 'k-', linewidth=3)
l_motor, = ax.plot([], [], 'ro', markersize=8)
r_motor, = ax.plot([], [], 'bo', markersize=8)
target_marker = ax.plot(0, 0, 'gx', markersize=10, label='Target')[0]

def animate(i):
    cx = history_x[i]
    cy = history_y[i]
    th = history_theta[i]
    
    lx = cx - L * np.cos(th)
    ly = cy - L * np.sin(th)
    rx = cx + L * np.cos(th)
    ry = cy + L * np.sin(th)
    
    line.set_data([lx, rx], [ly, ry])
    l_motor.set_data([lx], [ly])
    r_motor.set_data([rx], [ry])
    return line, l_motor, r_motor

ani = animation.FuncAnimation(fig, animate, frames=steps, interval=10, blit=True)
ani.save('drone.gif', writer='pillow', fps=25)
print(f"Saved drone.gif (Total Time: {time.time() - start_time:.4f}s)")