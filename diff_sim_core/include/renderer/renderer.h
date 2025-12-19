#ifndef RENDERER_H
#define RENDERER_H

class Renderer {
public:
    Renderer(int width, int height, float scale) : m_Width(width), m_Height(height), m_Scale(scale) {}
    virtual ~Renderer() = default;

    // Window properties
    int GetWidth() const { return m_Width; }
    int GetHeight() const { return m_Height; }
    float GetScale() const { return m_Scale; }

    // Window management
    virtual void Clear() = 0;
    virtual void Present() = 0;
    virtual bool ProcessEvents() = 0; // Return false if user requested close

    // Drawing primitives
    // Coordinates are in simulation space (meters), renderer handles scaling
    virtual void DrawBox(float x, float y, float w, float h, float rotation, 
                         float r=1.0f, float g=1.0f, float b=1.0f) = 0;
    
    virtual void DrawBoxFilled(float x, float y, float w, float h, float rotation,
                               float r=1.0f, float g=1.0f, float b=1.0f) = 0;

    virtual void DrawLine(float x1, float y1, float x2, float y2, 
                          float r=1.0f, float g=1.0f, float b=1.0f) = 0;

    virtual void DrawCircle(float centerX, float centerY, float radius, 
                            float r=1.0f, float g=1.0f, float b=1.0f) = 0;

    virtual void DrawCircleFilled(float centerX, float centerY, float radius,
                                  float r=1.0f, float g=1.0f, float b=1.0f) = 0;

    // Triangle: 3 vertices in simulation coordinates
    virtual void DrawTriangle(float x1, float y1, float x2, float y2, float x3, float y3,
                              float r=1.0f, float g=1.0f, float b=1.0f) = 0;

    virtual void DrawTriangleFilled(float x1, float y1, float x2, float y2, float x3, float y3,
                                    float r=1.0f, float g=1.0f, float b=1.0f) = 0;
    
protected:
    float m_Scale;
    int m_Width, m_Height;
};

#endif // RENDERER_H
