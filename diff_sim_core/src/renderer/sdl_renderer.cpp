#include "renderer/sdl_renderer.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>

#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

SDLRenderer::SDLRenderer(int width, int height, float scaleFactor) 
    : Renderer(width, height, scaleFactor), m_pWindow(nullptr), m_GLContext(nullptr),
      m_FontInfo(nullptr), m_FontScale(0), m_FontAscent(0), m_FontTexture(0)
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
    
    // Initialize font atlas
    m_FontAtlas.resize(ATLAS_WIDTH * ATLAS_HEIGHT, 0);
    memset(m_CharData, 0, sizeof(m_CharData));
}

SDLRenderer::~SDLRenderer() {
    if (m_FontTexture) {
        glDeleteTextures(1, &m_FontTexture);
    }
    if (m_FontInfo) {
        delete static_cast<stbtt_fontinfo*>(m_FontInfo);
    }
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

bool SDLRenderer::LoadFont(const std::string& fontPath, int fontSize) {
    // Read font file
    std::ifstream file(fontPath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open font file: " << fontPath << std::endl;
        return false;
    }
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    m_FontBuffer.resize(static_cast<size_t>(size));
    if (!file.read(reinterpret_cast<char*>(m_FontBuffer.data()), size)) {
        std::cerr << "Failed to read font file: " << fontPath << std::endl;
        return false;
    }
    file.close();
    
    // Initialize font
    if (m_FontInfo) {
        delete static_cast<stbtt_fontinfo*>(m_FontInfo);
    }
    m_FontInfo = new stbtt_fontinfo();
    stbtt_fontinfo* info = static_cast<stbtt_fontinfo*>(m_FontInfo);
    
    if (!stbtt_InitFont(info, m_FontBuffer.data(), 0)) {
        std::cerr << "stbtt_InitFont failed" << std::endl;
        delete info;
        m_FontInfo = nullptr;
        return false;
    }
    
    m_FontScale = stbtt_ScaleForPixelHeight(info, static_cast<float>(fontSize));
    
    int ascent, descent, lineGap;
    stbtt_GetFontVMetrics(info, &ascent, &descent, &lineGap);
    m_FontAscent = static_cast<int>(ascent * m_FontScale);
    
    // Build font atlas for ASCII 32-127
    std::fill(m_FontAtlas.begin(), m_FontAtlas.end(), 0);
    
    int atlasX = 0, atlasY = 0;
    int rowHeight = 0;
    
    for (int c = 32; c < 128; c++) {
        int w, h, xoff, yoff;
        unsigned char* bitmap = stbtt_GetCodepointBitmap(info, 0, m_FontScale, c, &w, &h, &xoff, &yoff);
        
        if (atlasX + w >= ATLAS_WIDTH) {
            atlasX = 0;
            atlasY += rowHeight + 1;
            rowHeight = 0;
        }
        
        if (atlasY + h >= ATLAS_HEIGHT) {
            stbtt_FreeBitmap(bitmap, nullptr);
            break;  // Atlas full
        }
        
        // Copy glyph to atlas
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                m_FontAtlas[(atlasY + y) * ATLAS_WIDTH + (atlasX + x)] = bitmap[y * w + x];
            }
        }
        
        // Store character data
        int idx = c - 32;
        m_CharData[idx][0] = static_cast<float>(atlasX) / ATLAS_WIDTH;  // u0
        m_CharData[idx][1] = static_cast<float>(atlasY) / ATLAS_HEIGHT; // v0
        m_CharData[idx][2] = static_cast<float>(atlasX + w) / ATLAS_WIDTH;  // u1
        m_CharData[idx][3] = static_cast<float>(atlasY + h) / ATLAS_HEIGHT; // v1
        m_CharData[idx][4] = static_cast<float>(xoff);
        m_CharData[idx][5] = static_cast<float>(yoff);
        m_CharData[idx][6] = static_cast<float>(w);
        m_CharData[idx][7] = static_cast<float>(h);
        
        int advanceWidth, leftBearing;
        stbtt_GetCodepointHMetrics(info, c, &advanceWidth, &leftBearing);
        m_CharData[idx][6] = advanceWidth * m_FontScale;  // xadvance
        
        atlasX += w + 1;
        rowHeight = std::max(rowHeight, h);
        
        stbtt_FreeBitmap(bitmap, nullptr);
    }
    
    // Convert alpha atlas to RGBA (white with alpha)
    std::vector<unsigned char> rgbaAtlas(ATLAS_WIDTH * ATLAS_HEIGHT * 4);
    for (int i = 0; i < ATLAS_WIDTH * ATLAS_HEIGHT; i++) {
        rgbaAtlas[i * 4 + 0] = 255;  // R
        rgbaAtlas[i * 4 + 1] = 255;  // G
        rgbaAtlas[i * 4 + 2] = 255;  // B
        rgbaAtlas[i * 4 + 3] = m_FontAtlas[i];  // A
    }
    
    // Create OpenGL texture
    if (m_FontTexture) {
        glDeleteTextures(1, &m_FontTexture);
    }
    glGenTextures(1, &m_FontTexture);
    glBindTexture(GL_TEXTURE_2D, m_FontTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ATLAS_WIDTH, ATLAS_HEIGHT, 0, 
                 GL_RGBA, GL_UNSIGNED_BYTE, rgbaAtlas.data());
    
    return true;
}

void SDLRenderer::DrawText(int screenX, int screenY, const std::string& text,
                           float r, float g, float b) {
    if (!m_FontInfo || !m_FontTexture || text.empty()) return;
    
    // Convert screen coordinates to OpenGL coordinates
    // screenX is from left edge, screenY is from bottom edge
    float x = static_cast<float>(screenX) - m_Width / 2.0f;
    float baselineY = static_cast<float>(screenY);
    
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, m_FontTexture);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    glColor4f(r, g, b, 1.0f);
    
    glBegin(GL_QUADS);
    for (char c : text) {
        if (c < 32 || c >= 128) continue;  // Skip non-printable
        
        int idx = c - 32;
        float u0 = m_CharData[idx][0];
        float v0 = m_CharData[idx][1];
        float u1 = m_CharData[idx][2];
        float v1 = m_CharData[idx][3];
        float xoff = m_CharData[idx][4];
        float yoff = m_CharData[idx][5];  // stb_truetype: negative = above baseline
        float xadvance = m_CharData[idx][6];
        
        float charW = (u1 - u0) * ATLAS_WIDTH;
        float charH = (v1 - v0) * ATLAS_HEIGHT;
        
        // Position glyph: x0,y0 is bottom-left of quad
        // yoff from stb is negative for chars above baseline
        float x0 = x + xoff;
        float y0 = baselineY - yoff - charH;  // Bottom of glyph
        float x1 = x0 + charW;
        float y1 = y0 + charH;  // Top of glyph
        
        // Draw quad with correct texture coords (v0=top, v1=bottom in texture)
        glTexCoord2f(u0, v1); glVertex2f(x0, y0);  // bottom-left
        glTexCoord2f(u1, v1); glVertex2f(x1, y0);  // bottom-right
        glTexCoord2f(u1, v0); glVertex2f(x1, y1);  // top-right
        glTexCoord2f(u0, v0); glVertex2f(x0, y1);  // top-left
        
        x += xadvance;
    }
    glEnd();
    
    glDisable(GL_TEXTURE_2D);
}

