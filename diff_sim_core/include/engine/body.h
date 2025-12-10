#ifndef BODY_H
#define BODY_H

#include "engine/tensor.h"
#include "engine/motor.h"
#include <vector>
#include <list>
#include <string>
#include <stdexcept>

// Simple shape definition for now
struct Shape {
    enum Type { BOX, CIRCLE };
    Type type;
    float width;  // or radius
    float height;
    // Relative offset from body center
    float offset_x; 
    float offset_y;
};

struct AABB {
    float min_x, min_y;
    float max_x, max_y;

};

class Body {
public:
    // State Tensors (Differentiable)
    Tensor pos;      // (2, 1) [x, y]
    Tensor vel;      // (2, 1) [vx, vy]
    Tensor rotation; // (1, 1) [theta]
    Tensor ang_vel;  // (1, 1) [omega]

    // Properties (Potentially differentiable!)
    Tensor mass;     // (1, 1)
    Tensor inertia;  // (1, 1)
    
    // Force Accumulators
    Tensor force_accumulator; // (2, 1)
    Tensor torque_accumulator; // (1, 1)
    
    // Geometry (Static for now)
    std::vector<Shape> shapes;
    std::string name;
    
    // Motors attached to this body
    std::vector<Motor*> motors;
    
    // Physics properties
    bool is_static;     // Static bodies don't move (infinite mass for collision)
    float friction;     // Friction coefficient [0, 1]
    float restitution;  // Bounciness [0 = no bounce, 1 = full bounce]

    Body(float x, float y, float mass_val, float width, float height);
    
    // Static body factory (for ground/walls)
    static Body* create_static(float x, float y, float width, float height, float rotation = 0.0f);
    
    // Motor management
    void add_motor(Motor* m) {
        // Check for overlap with existing motors
        for (Motor* existing : motors) {
            if (existing->overlaps(*m)) {
                throw std::runtime_error("Motor overlap detected! Cannot attach motor - it collides with an existing motor.");
            }
        }
        m->parent = this;
        motors.push_back(m);
        
        // Update mass (motor mass adds to body mass)
        float new_mass = mass.get(0, 0) + m->mass;
        mass = Tensor(std::vector<float>{new_mass}, true);
        
        // Update inertia (I = I + m*r^2 for point mass at distance r)
        float r_sq = m->local_x * m->local_x + m->local_y * m->local_y;
        float new_inertia = inertia.get(0, 0) + m->mass * r_sq;
        inertia = Tensor(std::vector<float>{new_inertia}, true);
    }
    
    // Apply all motor forces
    void apply_motor_forces();
    
    // Physics integration step
    // Old method (Manual):
    void step(const Tensor& forces, const Tensor& torque, float dt);

    // New method (Automatic): Uses accumulators and clears them
    void step(float dt);

    void apply_force(const Tensor& f);
    void apply_force_at_point(const Tensor& force, const Tensor& point);
    void apply_torque(const Tensor& t);
    void reset_forces();
    
    // Getters for rendering
    float get_x() const;
    float get_y() const;
    float get_rotation() const;

    std::vector<Tensor> get_corners();

    AABB get_aabb() const;

    // Internal memory management for C++ variables to survive autograd
    // Must be std::list to prevent pointer invalidation on push_back!
    std::list<Tensor> garbage_collector; 
    
    // Helper to keep a tensor alive and return a stable reference
    Tensor& keep(const Tensor& t);
};

#endif // BODY_H

