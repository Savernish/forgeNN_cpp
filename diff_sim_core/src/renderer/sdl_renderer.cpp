#include "renderer/sdl_renderer.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <climits>

SDLRenderer::SDLRenderer(int width, int height, float scaleFactor) 
    : Renderer(width, height, scaleFactor), m_pWindow(nullptr), m_pRenderer(nullptr)
{
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
        return;
    }

    m_pWindow = SDL_CreateWindow("rigidRL Physics Engine", 
                              SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 
                              width, height, SDL_WINDOW_SHOWN);
    if (!m_pWindow) {
        std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return;
    }

    m_pRenderer = SDL_CreateRenderer(m_pWindow, -1, SDL_RENDERER_ACCELERATED);
    if (!m_pRenderer) {
        std::cerr << "Renderer could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        // Try software renderer as fallback
        m_pRenderer = SDL_CreateRenderer(m_pWindow, -1, SDL_RENDERER_SOFTWARE);
        if (!m_pRenderer) {
            std::cerr << "Software renderer also failed! SDL_Error: " << SDL_GetError() << std::endl;
            SDL_DestroyWindow(m_pWindow);
            m_pWindow = nullptr;
            SDL_Quit();
            return;
        }
        std::cout << "Using software renderer as fallback" << std::endl;
    }
}

SDLRenderer::~SDLRenderer() {
    if (m_pRenderer) {
        SDL_DestroyRenderer(m_pRenderer);
    }
    if (m_pWindow) {
        SDL_DestroyWindow(m_pWindow);
    }
    SDL_Quit();
}

void SDLRenderer::Clear() {
    if (!m_pRenderer) return;
    SDL_SetRenderDrawColor(m_pRenderer, 30, 30, 30, 255); // Dark Gray
    SDL_RenderClear(m_pRenderer);
}

void SDLRenderer::Present() {
    if (!m_pRenderer) return;
    SDL_RenderPresent(m_pRenderer);
}

bool SDLRenderer::ProcessEvents() {
    if (!m_pWindow) return false;  // No window means we should quit
    SDL_Event e;
    while (SDL_PollEvent(&e) != 0) {
        if (e.type == SDL_QUIT) {
            return false;
        }
    }
    return true;
}

int SDLRenderer::ToScreenX(float simX) {
    return m_Width / 2 + (int)(simX * m_Scale);
}

int SDLRenderer::ToScreenY(float simY) {
    return m_Height - 50 - (int)(simY * m_Scale); // Ground at bottom, Y-up
}

void SDLRenderer::DrawBox(float x, float y, float w, float h, float rotation, float r, float g, float b) {
    if (!m_pRenderer) return;
    SDL_SetRenderDrawColor(m_pRenderer, (Uint8)(r*255), (Uint8)(g*255), (Uint8)(b*255), 255);

    // Calculate corners
    float cosT = std::cos(rotation);
    float sinT = std::sin(rotation);

    float hw = w / 2.0f;
    float hh = h / 2.0f;

    float localX[] = {-hw, hw, hw, -hw};
    float localY[] = {-hh, -hh, hh, hh};

    SDL_Point points[5]; // 5 points to close loop

    for (int i = 0; i < 4; ++i) {
        float rotX = localX[i] * cosT - localY[i] * sinT;
        float rotY = localX[i] * sinT + localY[i] * cosT;
        
        points[i].x = ToScreenX(x + rotX);
        points[i].y = ToScreenY(y + rotY);
    }
    points[4] = points[0];

    SDL_RenderDrawLines(m_pRenderer, points, 5);
}

void SDLRenderer::DrawBoxFilled(float x, float y, float w, float h, float rotation, float r, float g, float b) {
    if (!m_pRenderer) return;
    SDL_SetRenderDrawColor(m_pRenderer, (Uint8)(r*255), (Uint8)(g*255), (Uint8)(b*255), 255);

    float cosT = std::cos(rotation);
    float sinT = std::sin(rotation);
    float hw = w / 2.0f;
    float hh = h / 2.0f;

    float localX[] = {-hw, hw, hw, -hw};
    float localY[] = {-hh, -hh, hh, hh};

    // Get screen coordinates of corners
    int screenX[4], screenY[4];
    int minY = INT_MAX, maxY = INT_MIN;
    
    for (int i = 0; i < 4; ++i) {
        float rotX = localX[i] * cosT - localY[i] * sinT;
        float rotY = localX[i] * sinT + localY[i] * cosT;
        screenX[i] = ToScreenX(x + rotX);
        screenY[i] = ToScreenY(y + rotY);
        minY = std::min(minY, screenY[i]);
        maxY = std::max(maxY, screenY[i]);
    }

    // Scanline fill: for each row, find intersections with polygon edges
    for (int scanY = minY; scanY <= maxY; ++scanY) {
        std::vector<int> intersections;
        
        for (int i = 0; i < 4; ++i) {
            int j = (i + 1) % 4;
            int y1 = screenY[i], y2 = screenY[j];
            int x1 = screenX[i], x2 = screenX[j];
            
            if ((y1 <= scanY && y2 > scanY) || (y2 <= scanY && y1 > scanY)) {
                int intersectX = x1 + (scanY - y1) * (x2 - x1) / (y2 - y1);
                intersections.push_back(intersectX);
            }
        }
        
        std::sort(intersections.begin(), intersections.end());
        
        for (size_t i = 0; i + 1 < intersections.size(); i += 2) {
            SDL_RenderDrawLine(m_pRenderer, intersections[i], scanY, intersections[i+1], scanY);
        }
    }
}

void SDLRenderer::DrawLine(float x1, float y1, float x2, float y2, float r, float g, float b) {
    if (!m_pRenderer) return;
    SDL_SetRenderDrawColor(m_pRenderer, (Uint8)(r*255), (Uint8)(g*255), (Uint8)(b*255), 255);
    SDL_RenderDrawLine(m_pRenderer, ToScreenX(x1), ToScreenY(y1), ToScreenX(x2), ToScreenY(y2));
}

void SDLRenderer::DrawCircle(float centreX, float centreY, float radius, float r, float g, float b) {
    if (!m_pRenderer) return;
    // Convert world center to screen coordinates
    int screenCenterX = ToScreenX(centreX);
    int screenCenterY = ToScreenY(centreY);
    int screenRadius = static_cast<int>(radius * m_Scale);
    
    const int32_t diameter = (screenRadius * 2);

    int32_t x = (screenRadius - 1);
    int32_t y = 0;
    int32_t tx = 1;
    int32_t ty = 1;
    int32_t error = (tx - diameter);

    SDL_SetRenderDrawColor(m_pRenderer, (Uint8)(r*255), (Uint8)(g*255), (Uint8)(b*255), 255);

    while (x >= y)
    {
        //  Each of the following renders an octant of the circle
        SDL_RenderDrawPoint(m_pRenderer, screenCenterX + x, screenCenterY - y);
        SDL_RenderDrawPoint(m_pRenderer, screenCenterX + x, screenCenterY + y);
        SDL_RenderDrawPoint(m_pRenderer, screenCenterX - x, screenCenterY - y);
        SDL_RenderDrawPoint(m_pRenderer, screenCenterX - x, screenCenterY + y);
        SDL_RenderDrawPoint(m_pRenderer, screenCenterX + y, screenCenterY - x);
        SDL_RenderDrawPoint(m_pRenderer, screenCenterX + y, screenCenterY + x);
        SDL_RenderDrawPoint(m_pRenderer, screenCenterX - y, screenCenterY - x);
        SDL_RenderDrawPoint(m_pRenderer, screenCenterX - y, screenCenterY + x);

        if (error <= 0)
        {
            ++y;
            error += ty;
            ty += 2;
        }

        if (error > 0)
        {
            --x;
            tx += 2;
            error += (tx - diameter);
        }
    }
}

void SDLRenderer::DrawCircleFilled(float centreX, float centreY, float radius, float r, float g, float b) {
    if (!m_pRenderer) return;
    
    int cx = ToScreenX(centreX);
    int cy = ToScreenY(centreY);
    int rad = static_cast<int>(radius * m_Scale);
    
    SDL_SetRenderDrawColor(m_pRenderer, (Uint8)(r*255), (Uint8)(g*255), (Uint8)(b*255), 255);
    
    // Bresenham-based filled circle (no sqrt, integer only)
    int x = rad;
    int y = 0;
    int radiusError = 1 - x;
    
    while (x >= y) {
        // Draw horizontal lines for each octant pair
        SDL_RenderDrawLine(m_pRenderer, cx - x, cy + y, cx + x, cy + y);
        SDL_RenderDrawLine(m_pRenderer, cx - x, cy - y, cx + x, cy - y);
        SDL_RenderDrawLine(m_pRenderer, cx - y, cy + x, cx + y, cy + x);
        SDL_RenderDrawLine(m_pRenderer, cx - y, cy - x, cx + y, cy - x);
        
        y++;
        if (radiusError < 0) {
            radiusError += 2 * y + 1;
        } else {
            x--;
            radiusError += 2 * (y - x + 1);
        }
    }
}

void SDLRenderer::DrawTriangle(float x1, float y1, float x2, float y2, float x3, float y3,
                               float r, float g, float b) {
    if (!m_pRenderer) return;
    
    SDL_SetRenderDrawColor(m_pRenderer, (Uint8)(r*255), (Uint8)(g*255), (Uint8)(b*255), 255);
    
    int sx1 = ToScreenX(x1), sy1 = ToScreenY(y1);
    int sx2 = ToScreenX(x2), sy2 = ToScreenY(y2);
    int sx3 = ToScreenX(x3), sy3 = ToScreenY(y3);
    
    SDL_RenderDrawLine(m_pRenderer, sx1, sy1, sx2, sy2);
    SDL_RenderDrawLine(m_pRenderer, sx2, sy2, sx3, sy3);
    SDL_RenderDrawLine(m_pRenderer, sx3, sy3, sx1, sy1);
}

void SDLRenderer::DrawTriangleFilled(float x1, float y1, float x2, float y2, float x3, float y3,
                                     float r, float g, float b) {
    if (!m_pRenderer) return;
    
    // Convert to screen coordinates
    float sx1 = static_cast<float>(ToScreenX(x1)), sy1 = static_cast<float>(ToScreenY(y1));
    float sx2 = static_cast<float>(ToScreenX(x2)), sy2 = static_cast<float>(ToScreenY(y2));
    float sx3 = static_cast<float>(ToScreenX(x3)), sy3 = static_cast<float>(ToScreenY(y3));
    
    // Use SDL_RenderGeometry for GPU-accelerated triangle rendering
    SDL_Vertex vertices[3];
    SDL_Color color = {(Uint8)(r*255), (Uint8)(g*255), (Uint8)(b*255), 255};
    
    vertices[0].position = {sx1, sy1};
    vertices[0].color = color;
    vertices[0].tex_coord = {0, 0};
    
    vertices[1].position = {sx2, sy2};
    vertices[1].color = color;
    vertices[1].tex_coord = {0, 0};
    
    vertices[2].position = {sx3, sy3};
    vertices[2].color = color;
    vertices[2].tex_coord = {0, 0};
    
    SDL_RenderGeometry(m_pRenderer, nullptr, vertices, 3, nullptr, 0);
}


