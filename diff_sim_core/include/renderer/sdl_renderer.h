#ifndef SDL_RENDERER_H
#define SDL_RENDERER_H

#include "renderer.h"
#include <SDL2/SDL.h>

class SDLRenderer : public Renderer {
private:
    SDL_Window* m_pWindow;
    SDL_Renderer* m_pRenderer;

    // Coordinate conversion
    int ToScreenX(float simX);
    int ToScreenY(float simY);

public:
    SDLRenderer(int width=800, int height=600, float scaleFactor=50.0f);
    ~SDLRenderer();

    void Clear() override;
    void Present() override;
    bool ProcessEvents() override;
    void DrawBox(float x, float y, float w, float h, float rotation,
                 float r, float g, float b) override;
    void DrawBoxFilled(float x, float y, float w, float h, float rotation,
                       float r, float g, float b) override;
    void DrawCircle(float centerX, float centerY, float radius,
                    float r, float g, float b) override;
    void DrawCircleFilled(float centerX, float centerY, float radius,
                          float r, float g, float b) override;
    void DrawTriangle(float x1, float y1, float x2, float y2, float x3, float y3,
                      float r, float g, float b) override;
    void DrawTriangleFilled(float x1, float y1, float x2, float y2, float x3, float y3,
                            float r, float g, float b) override;
    void DrawLine(float x1, float y1, float x2, float y2, 
                  float r, float g, float b) override;
};

#endif // SDL_RENDERER_H
