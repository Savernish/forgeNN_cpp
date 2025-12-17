#include "engine/body.h"
#include <iostream>
#include <cmath>

Body::Body(float x, float y, float massVal, float width, float height) 
    : m_Name("Body"), is_static(false), friction(0.5f), restitution(0.0f)  // No bounce by default
{
    // Initialize State (requires_grad=false by default for state, but can be turned on)
    std::vector<float> posVec = {x, y};
    pos = Tensor(posVec, true); // Allow gradients to flow through pos
    
    std::vector<float> velVec = {0.0f, 0.0f};
    vel = Tensor(velVec, true);
    
    std::vector<float> rotVec = {0.0f};
    rotation = Tensor(rotVec, true);
    
    std::vector<float> angVelVec = {0.0f};
    ang_vel = Tensor(angVelVec, true);

    // Properties
    std::vector<float> massVec = {massVal};
    mass = Tensor(massVec, false); 
    
    // Box inertia: m * (w^2 + h^2) / 12
    float I = massVal * (width*width + height*height) / 12.0f;
    std::vector<float> inertiaVec = {I};
    inertia = Tensor(inertiaVec, false);

    // Shape
    Shape s;
    s.type = Shape::BOX;
    s.width = width;
    s.height = height;
    s.offsetX = 0;
    s.offsetY = 0;
    shapes.push_back(s);

    // Initialize accumulators
    ResetForces();
}

// Factory for static colliders (ground, walls, platforms)
Body* Body::CreateStatic(float x, float y, float width, float height, float rotationVal) {
    // Use mass=1 internally but mark as static  
    Body* pBody = new Body(x, y, 1.0f, width, height);
    pBody->is_static = true;
    pBody->friction = 0.8f;  // Static objects typically have higher friction
    pBody->restitution = 0.0f;  // No bounce for ground
    
    // Set rotation if specified
    if (rotationVal != 0.0f) {
        std::vector<float> rotVec = {rotationVal};
        pBody->rotation = Tensor(rotVec, false);
    }
    
    return pBody;
}

void Body::Step(const Tensor& forces, const Tensor& torque, float dt) {
    // 1. Linear Acceleration: a = F / m
    std::vector<float> one = {1.0f};
    Tensor invMass = Tensor(one, false) / mass; 
    Tensor acc = forces * invMass;

    // 2. Angular Acceleration: alpha = tau / I
    Tensor invI = Tensor(one, false) / inertia;
    Tensor alpha = torque * invI;

    // 3. Integration (Semi-Implicit Euler)
    std::vector<float> dtVec = {dt};
    Tensor dtTensor(dtVec, false);

    // v_new = v + a * dt
    vel = vel + acc * dtTensor;
    
    // pos_new = pos + v_new * dt
    pos = pos + vel * dtTensor;

    // omega_new = omega + alpha * dt
    ang_vel = ang_vel + alpha * dtTensor;

    // theta_new = theta + omega_new * dt
    rotation = rotation + ang_vel * dtTensor;
}

void Body::Step(float dt) {
    Step(m_ForceAccumulator, m_TorqueAccumulator, dt);
    ResetForces();
}

void Body::ApplyForce(const Tensor& f) {
    // If accumulator is zero (no grad), and f has grad, result has grad.
    m_ForceAccumulator = m_ForceAccumulator + f;
}

void Body::ApplyForceAtPoint(const Tensor& force, const Tensor& point) {
    // 1. Apply Linear Force
    ApplyForce(force);

    // 2. Calculate Torque = (point - pos) x force
    // r = point - pos
    // Note: 'point' should be world coordinates.

    // Access components (select is now const-qualified)
    Tensor px = pos.Select(0); 
    Tensor py = pos.Select(1);
    garbage_collector.push_back(px);
    garbage_collector.push_back(py);

    Tensor pointX = point.Select(0);
    Tensor pointY = point.Select(1);
    garbage_collector.push_back(pointX);
    garbage_collector.push_back(pointY);

    Tensor dx = pointX - px;
    Tensor dy = pointY - py;
    garbage_collector.push_back(dx);
    garbage_collector.push_back(dy);

    Tensor fx = force.Select(0);
    Tensor fy = force.Select(1);
    garbage_collector.push_back(fx);
    garbage_collector.push_back(fy);

    // Cross product 2D: rx * fy - ry * fx
    Tensor t1 = dx * fy;
    Tensor t2 = dy * fx;
    garbage_collector.push_back(t1);
    garbage_collector.push_back(t2);

    Tensor torque = t1 - t2;
    garbage_collector.push_back(torque);

    ApplyTorque(torque);
}

void Body::ApplyTorque(const Tensor& t) {
    m_TorqueAccumulator = m_TorqueAccumulator + t;
}

void Body::ResetForces() {
    std::vector<float> zeroVec = {0.0f, 0.0f};
    m_ForceAccumulator = Tensor(zeroVec, false);
    
    std::vector<float> zeroRot = {0.0f};
    m_TorqueAccumulator = Tensor(zeroRot, false);
}

float Body::GetX() const {
    return const_cast<Tensor*>(&pos)->Get(0,0);
}

float Body::GetY() const {
    return const_cast<Tensor*>(&pos)->Get(1,0); 
}

float Body::GetRotation() const {
    return const_cast<Tensor*>(&rotation)->Get(0,0);
}

std::vector<Tensor> Body::GetCorners() {
    std::vector<Tensor> corners;
    // Clear old graph nodes (assume single step usage)
    garbage_collector.clear();

    float w = shapes[0].width;
    float h = shapes[0].height;
    float hw = w / 2.0f;
    float hh = h / 2.0f;

    struct Point { float x, y; };
    // Corners: TR, TL, BL, BR
    std::vector<Point> offsets = {
        {hw, hh}, {-hw, hh}, {-hw, -hh}, {hw, -hh}
    };

    Tensor cosT = rotation.Cos();
    Tensor sinT = rotation.Sin();
    
    // Save to GC to keep alive
    garbage_collector.push_back(cosT);
    garbage_collector.push_back(sinT);

    Tensor px = pos.Select(0);
    Tensor py = pos.Select(1);
    garbage_collector.push_back(px);
    garbage_collector.push_back(py);

    for (const auto& off : offsets) {
        // rotX = off.x * cos - off.y * sin
        Tensor rotX = cosT * off.x - sinT * off.y;
        garbage_collector.push_back(rotX);

        // rotY = off.x * sin + off.y * cos
        Tensor rotY = sinT * off.x + cosT * off.y;
        garbage_collector.push_back(rotY);

        Tensor finalX = px + rotX;
        Tensor finalY = py + rotY;
        
        garbage_collector.push_back(finalX);
        garbage_collector.push_back(finalY);

        corners.push_back(finalX);
        corners.push_back(finalY);
    }
    return corners;
}

AABB Body::GetAABB() const {
    float w = shapes[0].width;
    float h = shapes[0].height;
    // Radius = distance from center to corner
    float radius = std::sqrt(w*w + h*h) / 2.0f;

    AABB aabb;
    float x = GetX();
    float y = GetY();
    
    aabb.minX = x - radius;
    aabb.maxX = x + radius;
    aabb.minY = y - radius;
    aabb.maxY = y + radius;
    return aabb;
}

Tensor& Body::Keep(const Tensor& t) {
    garbage_collector.push_back(t);
    return garbage_collector.back();
}

void Body::ApplyMotorForces() {
    float bodyRot = rotation.Get(0, 0);
    float cosR = std::cos(bodyRot);
    float sinR = std::sin(bodyRot);
    
    for (Motor* pMotor : motors) {
        if (pMotor->thrust <= 0) continue;
        
        // Motor thrust direction in local space (default: up = +y)
        float localFx = std::cos(pMotor->angle) * pMotor->thrust;
        float localFy = std::sin(pMotor->angle) * pMotor->thrust;
        
        // Transform force to world space
        float worldFx = cosR * localFx - sinR * localFy;
        float worldFy = sinR * localFx + cosR * localFy;
        
        // Apply linear force
        std::vector<float> forceVec = {worldFx, worldFy};
        Tensor force(forceVec, false);
        ApplyForce(force);
        
        // Calculate torque: tau = r x F (cross product in 2D)
        float rx = cosR * pMotor->local_x - sinR * pMotor->local_y;
        float ry = sinR * pMotor->local_x + cosR * pMotor->local_y;
        
        float torque = rx * worldFy - ry * worldFx;
        std::vector<float> torqueVec = {torque};
        ApplyTorque(Tensor(torqueVec, false));
    }
}
