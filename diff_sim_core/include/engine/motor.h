#ifndef MOTOR_H
#define MOTOR_H

#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class Body;  // Forward declaration

class Motor {
public:
    // Position relative to body center
    float local_x = 0;
    float local_y = 0;
    
    // Physical dimensions
    float width = 0.1f;
    float height = 0.1f;
    float mass = 0.1f;
    
    // Thrust properties
    float thrust = 0;        // Current thrust (0 to max_thrust)
    float max_thrust = 10.0f;
    float angle = M_PI / 2;  // Thrust direction (default: up)
    
    // Parent body (set when attached)
    Body* parent = nullptr;
    
    // Constructors
    Motor() = default;
    
    Motor(float lx, float ly) 
        : local_x(lx), local_y(ly) {}
    
    Motor(float lx, float ly, float w, float h, float m, float maxT)
        : local_x(lx), local_y(ly), width(w), height(h), mass(m), max_thrust(maxT) {}
    
    // Set thrust (clamped to 0..max_thrust)
    void SetThrust(float t) {
        thrust = std::max(0.0f, std::min(t, max_thrust));
    }
    
    // Check if this motor overlaps with another (in local body space)
    bool Overlaps(const Motor& other) const {
        float left1 = local_x - width/2, right1 = local_x + width/2;
        float bottom1 = local_y - height/2, top1 = local_y + height/2;
        
        float left2 = other.local_x - other.width/2, right2 = other.local_x + other.width/2;
        float bottom2 = other.local_y - other.height/2, top2 = other.local_y + other.height/2;
        
        return !(right1 < left2 || right2 < left1 || top1 < bottom2 || top2 < bottom1);
    }
};

#endif // MOTOR_H
