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
    float offsetX; 
    float offsetY;
};

struct AABB {
    float minX, minY;
    float maxX, maxY;
};

class Body {
public:
    // State Tensors (Differentiable)
    Tensor pos;      // (2, 1) [x, y]
    Tensor vel;      // (2, 1) [vx, vy]
    Tensor rotation; // (1, 1) [theta]
    Tensor ang_vel;  // (1, 1) [omega] - kept as ang_vel for Python API compatibility

    // Properties (Potentially differentiable!)
    Tensor mass;     // (1, 1)
    Tensor inertia;  // (1, 1)
    
    // Force Accumulators
    Tensor m_ForceAccumulator;  // (2, 1)
    Tensor m_TorqueAccumulator; // (1, 1)
    
    // Geometry (Static for now)
    std::vector<Shape> shapes;
    std::string m_Name;
    
    // Motors attached to this body
    std::vector<Motor*> motors;
    
    // Physics properties
    bool is_static;     // Static bodies don't move (infinite mass for collision)
    float friction;     // Friction coefficient [0, 1]
    float restitution;  // Bounciness [0 = no bounce, 1 = full bounce]

    // Constructor
    Body(float x, float y, float massVal, float width, float height);
    
    // Static body factory (for ground/walls)
    static Body* CreateStatic(float x, float y, float width, float height, float rotation = 0.0f);
    
    // Motor management
    void AddMotor(Motor* pMotor) {
        // Check for overlap with existing motors
        for (Motor* pExisting : motors) {
            if (pExisting->Overlaps(*pMotor)) {
                throw std::runtime_error("Motor overlap detected! Cannot attach motor - it collides with an existing motor.");
            }
        }
        pMotor->parent = this;
        motors.push_back(pMotor);
        
        // Update mass (motor mass adds to body mass)
        float newMass = mass.Get(0, 0) + pMotor->mass;
        mass = Tensor(std::vector<float>{newMass}, true);
        
        // Update inertia (I = I + m*r^2 for point mass at distance r)
        float rSq = pMotor->local_x * pMotor->local_x + pMotor->local_y * pMotor->local_y;
        float newInertia = inertia.Get(0, 0) + pMotor->mass * rSq;
        inertia = Tensor(std::vector<float>{newInertia}, true);
    }
    
    // Apply all motor forces
    void ApplyMotorForces();
    
    // Physics integration step
    // Old method (Manual):
    void Step(const Tensor& forces, const Tensor& torque, float dt);

    // New method (Automatic): Uses accumulators and clears them
    void Step(float dt);

    void ApplyForce(const Tensor& f);
    void ApplyForceAtPoint(const Tensor& force, const Tensor& point);
    void ApplyTorque(const Tensor& t);
    void ResetForces();
    
    // Getters for rendering
    float GetX() const;
    float GetY() const;
    float GetRotation() const;

    std::vector<Tensor> GetCorners();

    AABB GetAABB() const;

    // Internal memory management for C++ variables to survive autograd
    // Must be std::list to prevent pointer invalidation on push_back!
    std::list<Tensor> garbage_collector; 
    
    // Helper to keep a tensor alive and return a stable reference
    Tensor& Keep(const Tensor& t);
};

#endif // BODY_H
