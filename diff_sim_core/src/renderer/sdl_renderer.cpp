#include "renderer/sdl_renderer.h"
#include <iostream>
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

SDLRenderer::SDLRenderer(int width, int height, float scaleFactor) 
    : Renderer(width, height, scaleFactor), m_pWindow(nullptr), m_GLContext(nullptr)
{
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
        return;
    }

    // Set OpenGL attributes
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

    // Create window with OpenGL flag
    m_pWindow = SDL_CreateWindow("rigidRL Physics Engine", 
                              SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 
                              width, height, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
    if (!m_pWindow) {
        std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return;
    }

    // Create OpenGL context
    m_GLContext = SDL_GL_CreateContext(m_pWindow);
    if (!m_GLContext) {
        std::cerr << "OpenGL context could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(m_pWindow);
        m_pWindow = nullptr;
        SDL_Quit();
        return;
    }

    // Enable VSync
    SDL_GL_SetSwapInterval(1);

    // Initialize OpenGL
    float halfW = width / 2.0f;
    float halfH = height / 2.0f;

    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    // X: -halfW to halfW (0 is Center)
    // Y: 0 to height      (0 is Bottom, Y goes Up)
    glOrtho(-halfW, halfW, 0.0f, height, -1.0f, 1.0f);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Enable blending for smooth edges
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // Enable line smoothing
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
}

SDLRenderer::~SDLRenderer() {
    if (m_GLContext) {
        SDL_GL_DeleteContext(m_GLContext);
    }
    if (m_pWindow) {
        SDL_DestroyWindow(m_pWindow);
    }
    SDL_Quit();
}

void SDLRenderer::Clear() {
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);  // Dark blue-gray background
    glClear(GL_COLOR_BUFFER_BIT);
}

void SDLRenderer::Present() {
    SDL_GL_SwapWindow(m_pWindow);
}

bool SDLRenderer::ProcessEvents() {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT) return false;
        if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE) return false;
    }
    return true;
}

float SDLRenderer::ToScreenX(float simX) {
    return simX * m_Scale;
}

float SDLRenderer::ToScreenY(float simY) {
    return simY * m_Scale;
}

void SDLRenderer::DrawBox(float x, float y, float w, float h, float rotation,
                          float r, float g, float b) {
    float hw = w / 2.0f * m_Scale;
    float hh = h / 2.0f * m_Scale;
    float cx = ToScreenX(x);
    float cy = ToScreenY(y);
    
    float cosR = std::cos(rotation);
    float sinR = std::sin(rotation);
    
    // Four corners relative to center
    float corners[4][2] = {
        {-hw, -hh}, {hw, -hh}, {hw, hh}, {-hw, hh}
    };
    
    glColor3f(r, g, b);
    glBegin(GL_LINE_LOOP);
    for (int i = 0; i < 4; i++) {
        float rx = corners[i][0] * cosR - corners[i][1] * sinR;
        float ry = corners[i][0] * sinR + corners[i][1] * cosR;
        glVertex2f(cx + rx, cy + ry);
    }
    glEnd();
}

void SDLRenderer::DrawBoxFilled(float x, float y, float w, float h, float rotation,
                                float r, float g, float b) {
    float hw = w / 2.0f * m_Scale;
    float hh = h / 2.0f * m_Scale;
    float cx = ToScreenX(x);
    float cy = ToScreenY(y);
    
    float cosR = std::cos(rotation);
    float sinR = std::sin(rotation);
    
    float corners[4][2] = {
        {-hw, -hh}, {hw, -hh}, {hw, hh}, {-hw, hh}
    };
        
    glColor3f(r, g, b);
    glBegin(GL_QUADS);
    for (int i = 0; i < 4; i++) {
        float rx = corners[i][0] * cosR - corners[i][1] * sinR;
        float ry = corners[i][0] * sinR + corners[i][1] * cosR;
        glVertex2f(cx + rx, cy + ry);
    }
    glEnd();
}

void SDLRenderer::DrawLine(float x1, float y1, float x2, float y2,
                           float r, float g, float b) {
    glColor3f(r, g, b);
    glBegin(GL_LINES);
    glVertex2f(ToScreenX(x1), ToScreenY(y1));
    glVertex2f(ToScreenX(x2), ToScreenY(y2));
    glEnd();
}

void SDLRenderer::DrawCircle(float centreX, float centreY, float radius,
                             float r, float g, float b) {
    float cx = ToScreenX(centreX);
    float cy = ToScreenY(centreY);
    float rad = radius * m_Scale;
    
    const int segments = 32;
    
    glColor3f(r, g, b);
    glBegin(GL_LINE_LOOP);
    for (int i = 0; i < segments; i++) {
        float angle = 2.0f * M_PI * i / segments;
        glVertex2f(cx + rad * std::cos(angle), cy + rad * std::sin(angle));
    }
    glEnd();
}

void SDLRenderer::DrawCircleFilled(float centreX, float centreY, float radius,
                                   float r, float g, float b) {
    float cx = ToScreenX(centreX);
    float cy = ToScreenY(centreY);
    float rad = radius * m_Scale;
    
    const int segments = 32;
    
    glColor3f(r, g, b);
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(cx, cy);  // Center
    for (int i = 0; i <= segments; i++) {
        float angle = 2.0f * M_PI * i / segments;
        glVertex2f(cx + rad * std::cos(angle), cy + rad * std::sin(angle));
    }
    glEnd();
}

void SDLRenderer::DrawTriangle(float x1, float y1, float x2, float y2, float x3, float y3,
                               float r, float g, float b) {
    glColor3f(r, g, b);
    glBegin(GL_LINE_LOOP);
    glVertex2f(ToScreenX(x1), ToScreenY(y1));
    glVertex2f(ToScreenX(x2), ToScreenY(y2));
    glVertex2f(ToScreenX(x3), ToScreenY(y3));
    glEnd();
}

void SDLRenderer::DrawTriangleFilled(float x1, float y1, float x2, float y2, float x3, float y3,
                                     float r, float g, float b) {
    glColor3f(r, g, b);
    glBegin(GL_TRIANGLES);
    glVertex2f(ToScreenX(x1), ToScreenY(y1));
    glVertex2f(ToScreenX(x2), ToScreenY(y2));
    glVertex2f(ToScreenX(x3), ToScreenY(y3));
    glEnd();
}
