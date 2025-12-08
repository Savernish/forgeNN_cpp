#ifndef ENGINE_H
#define ENGINE_H

#include <vector>
#include "renderer/renderer.h"
#include "engine/body.h"
#include "engine/tensor.h"

// Ground Segment for arbitrary terrain
struct GroundSegment {
    float x1, y1;
    float x2, y2;
    float nx, ny; // Normal vector (normalized)
    float k;
    float damping;
    float friction; // Friction coefficient
    
    // AABB for Optimization
    float min_x, max_x, min_y, max_y;
};

class Engine {
    Renderer* renderer;
    std::vector<Body*> bodies;
    std::vector<GroundSegment> static_geometry;
    float dt;
    int substeps; // Number of physics steps per frame
    Tensor gravity;
    bool paused;

public:
    Engine(int width, int height, float scale=50.0f, float dt=0.016f, int substeps=10);
    ~Engine();

    void add_body(Body* b);
    void set_gravity(float x, float y);
    
    // Static Geometry
    // k=20000, damping=100 hardcoded internally for stability
    void add_ground_segment(float x1, float y1, float x2, float y2, float friction=0.5f);
    void clear_geometry();
    
    // Fine-grained control
    void update();
    void render_bodies();

    // Simulation Loop (Wrapper)
    // Returns false if Quit event received
    bool step(); 

    // Accessor for Python visualization
    Renderer* get_renderer() { return renderer; }
};

#endif // ENGINE_H
