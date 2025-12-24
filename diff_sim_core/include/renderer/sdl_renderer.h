#ifndef SDL_RENDERER_H
#define SDL_RENDERER_H

#include "renderer.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <string>
#include <vector>

class SDLRenderer : public Renderer {
private:
    SDL_Window* m_pWindow;
    SDL_GLContext m_GLContext;
    
    // Font data for stb_truetype
    std::vector<unsigned char> m_FontBuffer;
    void* m_FontInfo;  // stbtt_fontinfo*
    float m_FontScale;
    int m_FontAscent;
    GLuint m_FontTexture;
    static const int ATLAS_WIDTH = 512;
    static const int ATLAS_HEIGHT = 512;
    std::vector<unsigned char> m_FontAtlas;
    float m_CharData[96][8];  // For ASCII 32-127: x0,y0,x1,y1,xoff,yoff,xadvance,unused

    // Coordinate conversion (to OpenGL screen space)
    float ToScreenX(float simX);
    float ToScreenY(float simY);

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
    
    // Text rendering (uses stb_truetype)
    void DrawText(int screenX, int screenY, const std::string& text,
                  float r=1.0f, float g=1.0f, float b=1.0f);
    bool LoadFont(const std::string& fontPath, int fontSize=16);
};

#endif // SDL_RENDERER_H
