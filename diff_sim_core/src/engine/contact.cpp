#include "engine/contact.h"
#include "engine/body.h"
#include <cmath>

void ContactManifold::compute_mass() {
    if (!body_a || !body_b) return;
    
    // Get inverse masses and inertias
    float invMassA = body_a->is_static ? 0.0f : 1.0f / body_a->mass.Get(0, 0);
    float invMassB = body_b->is_static ? 0.0f : 1.0f / body_b->mass.Get(0, 0);
    float invInertiaA = body_a->is_static ? 0.0f : 1.0f / body_a->inertia.Get(0, 0);
    float invInertiaB = body_b->is_static ? 0.0f : 1.0f / body_b->inertia.Get(0, 0);
    
    float ax = body_a->pos.Get(0, 0);
    float ay = body_a->pos.Get(1, 0);
    float bx = body_b->pos.Get(0, 0);
    float by = body_b->pos.Get(1, 0);
    
    for (int i = 0; i < point_count; ++i) {
        ContactPoint& p = points[i];
        
        // Vector from body center to contact point
        float raX = p.position[0] - ax;
        float raY = p.position[1] - ay;
        float rbX = p.position[0] - bx;
        float rbY = p.position[1] - by;
        
        // Cross products for normal direction
        float raCrossN = raX * normal[1] - raY * normal[0];
        float rbCrossN = rbX * normal[1] - rbY * normal[0];
        
        // Effective mass for normal constraint
        float kNormal = invMassA + invMassB + 
                        raCrossN * raCrossN * invInertiaA +
                        rbCrossN * rbCrossN * invInertiaB;
        
        normal_mass[i] = (kNormal > 0) ? 1.0f / kNormal : 0.0f;
        
        // Cross products for tangent direction
        float raCrossT = raX * tangent[1] - raY * tangent[0];
        float rbCrossT = rbX * tangent[1] - rbY * tangent[0];
        
        // Effective mass for friction constraint
        float kTangent = invMassA + invMassB + 
                         raCrossT * raCrossT * invInertiaA +
                         rbCrossT * rbCrossT * invInertiaB;
        
        tangent_mass[i] = (kTangent > 0) ? 1.0f / kTangent : 0.0f;
    }
}

// ============================================================================
// ContactManager Implementation
// ============================================================================

ContactManifold* ContactManager::GetOrCreate(Body* pBodyA, Body* pBodyB) {
    ContactKey key{pBodyA, pBodyB};
    
    auto it = m_ManifoldCache.find(key);
    if (it != m_ManifoldCache.end()) {
        // Existing manifold - mark as still active
        it->second.was_touching = it->second.touching;
        return &it->second;
    }
    
    // Create new manifold
    ContactManifold manifold;
    manifold.body_a = pBodyA;
    manifold.body_b = pBodyB;
    
    // Combine material properties
    manifold.friction = std::sqrt(pBodyA->friction * pBodyB->friction);
    manifold.restitution = std::max(pBodyA->restitution, pBodyB->restitution);
    
    auto result = m_ManifoldCache.insert({key, manifold});
    return &result.first->second;
}

ContactManifold* ContactManager::Find(Body* pBodyA, Body* pBodyB) {
    ContactKey key{pBodyA, pBodyB};
    auto it = m_ManifoldCache.find(key);
    return (it != m_ManifoldCache.end()) ? &it->second : nullptr;
}

void ContactManager::BeginFrame() {
    // Mark all manifolds as not touching (will be updated during collision detection)
    for (auto& pair : m_ManifoldCache) {
        pair.second.was_touching = pair.second.touching;
        pair.second.touching = false;
    }
    m_ActiveManifolds.clear();
}

void ContactManager::EndFrame() {
    // Remove manifolds that are no longer touching
    for (auto it = m_ManifoldCache.begin(); it != m_ManifoldCache.end();) {
        if (!it->second.touching) {
            it = m_ManifoldCache.erase(it);
        } else {
            // Add to active list for solving
            m_ActiveManifolds.push_back(&it->second);
            ++it;
        }
    }
}

void ContactManager::Clear() {
    m_ManifoldCache.clear();
    m_ActiveManifolds.clear();
}
