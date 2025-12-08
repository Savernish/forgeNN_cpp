#include "engine/engine.h"
#include "renderer/sdl_renderer.h"
#include "engine/activations.h"
#include <iostream>
#include <cmath>
#include <algorithm>

Engine::Engine(int width, int height, float scale, float dt_val, int substeps_val) 
    : renderer(nullptr), dt(dt_val), substeps(substeps_val), paused(false) 
{
    // Default gravity (0, -9.81)
    std::vector<float> grav_vec = {0.0f, -9.81f};
    gravity = Tensor(grav_vec, false);

    renderer = new SDLRenderer(width, height, scale);
}

Engine::~Engine() {
    if (renderer) {
        delete renderer;
    }
    // We do not delete bodies as they are owned by Python/External
}

void Engine::add_body(Body* b) {
    bodies.push_back(b);
}

void Engine::set_gravity(float x, float y) {
    std::vector<float> grav_vec = {x, y};
    gravity = Tensor(grav_vec, false);
}

bool Engine::step() {
    // 1. Process Events
    if (!renderer->process_events()) {
        return false;
    }

    // 2. Physics
    update();

    // 3. Render
    renderer->clear();
    // Draw Ground Line due to Engine knowledge (y=0) -> Visual only, segments drawn manually in python now ideally
    // But keeping it just in case won't hurt, though we rely on render_bodies usually.
    // renderer->draw_line(-1000.0f, 0.0f, 1000.0f, 0.0f, 0.0f, 1.0f, 0.0f); // Wide Green Line

    render_bodies();
    
    renderer->present();

    return true;
}

// Helper for AABB overlap
bool aabb_overlap(float min_x1, float max_x1, float min_y1, float max_y1, 
                  float min_x2, float max_x2, float min_y2, float max_y2) {
    return (min_x1 <= max_x2 && max_x1 >= min_x2 &&
            min_y1 <= max_y2 && max_y1 >= min_y2);
}

void Engine::add_ground_segment(float x1, float y1, float x2, float y2, float friction) {
    GroundSegment s;
    s.x1 = x1; s.y1 = y1;
    s.x2 = x2; s.y2 = y2;
    s.k = 20000.0f; // Hardcoded High Stiffness
    s.damping = 100.0f; // Hardcoded Stable Damping
    s.friction = friction;
    
    // Calculate Normal (Standard: Left Normal)
    float dx = x2 - x1;
    float dy = y2 - y1;
    float len = std::sqrt(dx*dx + dy*dy);
    if (len > 0) {
        s.nx = -dy / len;
        s.ny = dx / len;
    } else {
        s.nx = 0; s.ny = 1;
    }

    // AABB
    s.min_x = std::min(x1, x2);
    s.max_x = std::max(x1, x2);
    s.min_y = std::min(y1, y2);
    s.max_y = std::max(y1, y2);
    
    // Expand AABB slightly for safety (e.g. penetration)
    float margin = 1.0f;
    s.min_x -= margin; s.max_x += margin;
    s.min_y -= margin; s.max_y += margin;
    
    static_geometry.push_back(s);
}

void Engine::clear_geometry() {
    static_geometry.clear();
}

void Engine::update() {
    float sub_dt = dt / (float)substeps;

    for (int step_i = 0; step_i < substeps; ++step_i) {
        for (Body* b : bodies) {
            // Apply Gravity: F = m * g
            Tensor force_gravity = gravity * b->mass; 
            b->apply_force(force_gravity);

            // --- Ground Segment Collision ---
            std::vector<Tensor> corners = b->get_corners();
            AABB b_aabb = b->get_aabb();

            // Broadphase: Filter candidate segments
            std::vector<int> candidates;
            for (int i = 0; i < static_geometry.size(); ++i) {
                const auto& seg = static_geometry[i];
                if (aabb_overlap(b_aabb.min_x, b_aabb.max_x, b_aabb.min_y, b_aabb.max_y,
                                  seg.min_x, seg.max_x, seg.min_y, seg.max_y)) {
                    candidates.push_back(i);
                }
            }

            // Narrowphase: Weighted Average Contact per Corner
            for (size_t i = 0; i < corners.size(); i += 2) {
                Tensor cx = corners[i];
                Tensor cy = corners[i+1];
                
                float px = cx.get(0,0);
                float py = cy.get(0,0);

                // Accumulators for Weighted Average
                // We need to initialize them. Tensors of 0.0f.
                std::vector<float> zero_vec = {0.0f};
                Tensor sum_fx = b->keep(Tensor(zero_vec, false));
                Tensor sum_fy = b->keep(Tensor(zero_vec, false));
                Tensor sum_weight = b->keep(Tensor(zero_vec, false)); // Weight by |dist|
                
                bool has_contact = false;

                // Check all candidates
                for (int seg_idx : candidates) {
                    const auto& seg = static_geometry[seg_idx];

                    // Signed Distance to Line
                    float dx = px - seg.x1;
                    float dy = py - seg.y1;
                    float dist = dx * seg.nx + dy * seg.ny;

                    // Check Bounds (Projection on Segment)
                    float seg_dx = seg.x2 - seg.x1;
                    float seg_dy = seg.y2 - seg.y1;
                    float seg_len_sq = seg_dx*seg_dx + seg_dy*seg_dy;
                    float t = (dx * seg_dx + dy * seg_dy) / seg_len_sq;

                    // Relaxed Bounds for Vertex Overlap (Prevents tunneling at seams)
                    float t_epsilon = 0.05f; // 5% extension to ensure overlap
                    if (dist < 0.0f && t >= -t_epsilon && t <= 1.0f + t_epsilon) {
                         has_contact = true;
                         
                         // --- 1. Calculate Per-Segment Force (Normal + Friction) ---
                         
                         // Penetration Depth & Normal Force
                         std::vector<float> vec_x1 = {seg.x1};
                         std::vector<float> vec_y1 = {seg.y1};
                         
                         Tensor& x1_t = b->keep(Tensor(vec_x1, false));
                         Tensor& y1_t = b->keep(Tensor(vec_y1, false));
                         
                         Tensor& diff_x = b->keep(cx - x1_t);
                         Tensor& diff_y = b->keep(cy - y1_t);
                         
                         Tensor& term_x = b->keep(diff_x * seg.nx);
                         Tensor& term_y = b->keep(diff_y * seg.ny);
                         
                         Tensor& dist_t = b->keep(term_x + term_y); // Negative value

                         Tensor& spring_force_mag = b->keep(dist_t * (-1.0f * seg.k));
                         
                         // Velocity Calculation (Point Velocity)
                         Tensor pos_x = const_cast<Tensor&>(b->pos).select(0);
                         Tensor pos_y = const_cast<Tensor&>(b->pos).select(1);
                         b->keep(pos_x); b->keep(pos_y);
                         
                         Tensor& rx = b->keep(cx - pos_x);
                         Tensor& ry = b->keep(cy - pos_y);
                         
                         Tensor& omega = b->keep(b->ang_vel); 
                         
                         Tensor& v_rot_x = b->keep(omega * ry * -1.0f);
                         Tensor& v_rot_y = b->keep(omega * rx);
                         
                         Tensor vx = const_cast<Tensor&>(b->vel).select(0);
                         Tensor vy = const_cast<Tensor&>(b->vel).select(1);
                         b->keep(vx); b->keep(vy);
                         
                         Tensor& vp_x = b->keep(vx + v_rot_x);
                         Tensor& vp_y = b->keep(vy + v_rot_y);
                         
                         // Normal Damping
                         Tensor& vp_proj_x = b->keep(vp_x * seg.nx);
                         Tensor& vp_proj_y = b->keep(vp_y * seg.ny);
                         Tensor& v_proj = b->keep(vp_proj_x + vp_proj_y);
                         
                         Tensor& damp_force_mag = b->keep(v_proj * (-1.0f * seg.damping));
                         
                         Tensor& total_normal_mag = b->keep(spring_force_mag + damp_force_mag);
                         
                         // Normal Force Vector
                         std::vector<float> n_vec = {seg.nx, seg.ny};
                         Tensor& n_tensor = b->keep(Tensor(n_vec, false));
                         Tensor& f_normal = b->keep(n_tensor * total_normal_mag);
                         
                         // Friction
                         float tx = -seg.ny;
                         float ty = seg.nx;
                         
                         Tensor& vt_proj_x = b->keep(vp_x * tx);
                         Tensor& vt_proj_y = b->keep(vp_y * ty);
                         Tensor& v_tan = b->keep(vt_proj_x + vt_proj_y);
                         
                         Tensor& friction_coeff = b->keep(total_normal_mag * (-1.0f * seg.friction));
                         Tensor& tan_dir = b->keep(v_tan * 2.0f); 
                         Tensor& friction_dir = b->keep(tanh(tan_dir)); 
                         Tensor& f_friction_mag = b->keep(friction_coeff * friction_dir);
                         
                         std::vector<float> t_vec = {tx, ty};
                         Tensor& t_tensor = b->keep(Tensor(t_vec, false));
                         Tensor& f_friction = b->keep(t_tensor * f_friction_mag);
                         
                         // Sub-Total Force for this Segment
                         Tensor& f_seg = b->keep(f_normal + f_friction);
                         
                         Tensor& f_seg_x = b->keep(f_seg.select(0));
                         Tensor& f_seg_y = b->keep(f_seg.select(1));
                         
                         // --- 2. Accumulate Weighted ---
                         // Weight = |dist_t| = -dist_t
                         Tensor& weight = b->keep(dist_t * -1.0f);
                         
                         Tensor& w_fx = b->keep(f_seg_x * weight);
                         Tensor& w_fy = b->keep(f_seg_y * weight);
                         
                         // Update Sums (New tensors)
                         sum_fx = b->keep(sum_fx + w_fx);
                         sum_fy = b->keep(sum_fy + w_fy);
                         sum_weight = b->keep(sum_weight + weight);
                    }
                }

                // Apply Weighted Average Force
                if (has_contact) {
                     // F_final = Sum_Weighted_F / Sum_Weights
                     // Note: If weight is super small, this could be unstable? 
                     // But we only enter here if dist < 0, so weight > 0.
                     
                     Tensor& final_fx = b->keep(sum_fx / sum_weight);
                     Tensor& final_fy = b->keep(sum_fy / sum_weight);
                     
                     // Combined Vector (2,1)
                     // Does Tensor have a constructor for (2,1) from two scalars? No.
                     // stack? bindings has static stack?
                     // Let's use std::vector<float> approach is hard because values are Tensors.
                     // Engine doesn't have a 'stack' helper easily accessible on Body?
                     // Body::keep returns Tensor&.
                     // Tensor::stack(std::vector<Tensor*>) is static.
                     // We need to call Tensor::stack.
                     
                     std::vector<Tensor*> components;
                     components.push_back(&final_fx);
                     components.push_back(&final_fy);
                     Tensor& total_force = b->keep(Tensor::stack(components));

                     // Apply at Point
                     // Which point? The corner itself (cx, cy).
                     std::vector<float> x_axis = {1.0f, 0.0f};
                     std::vector<float> y_axis = {0.0f, 1.0f};
                     Tensor& ax_x = b->keep(Tensor(x_axis, false));
                     Tensor& ax_y = b->keep(Tensor(y_axis, false));
                     
                     Tensor& pc1 = b->keep(ax_x * cx);
                     Tensor& pc2 = b->keep(ax_y * cy);
                     Tensor& p_corner = b->keep(pc1 + pc2);

                     b->apply_force_at_point(total_force, p_corner);
                }
            }
            // Integrate
            b->step(sub_dt); 
        }
    }
}

void Engine::render_bodies() {
     for (Body* b : bodies) {
        // Simple shape rendering (BOX only for now)
        for (const auto& s : b->shapes) {
            if (s.type == Shape::BOX) {
                renderer->draw_box(b->get_x(), b->get_y(), s.width, s.height, b->get_rotation());
            }
        }
    }
}
