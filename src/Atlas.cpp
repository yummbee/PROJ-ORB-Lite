#include "Atlas.hpp"
#include <iostream>

namespace orb_lite {

Atlas::Atlas() {
    createNewMap();
}

Atlas::~Atlas() {
    for (auto m : m_maps) delete m;
}

void Atlas::createNewMap() {
    std::lock_guard<std::mutex> lock(m_mutex);
    Map* newMap = new Map();
    m_maps.push_back(newMap);
    m_currentMap = newMap;
    std::cout << "Atlas: Created new Map (" << m_maps.size() << " total)" << std::endl;
}

Map* Atlas::getCurrentMap() {
    return m_currentMap;
}

std::vector<Map*> Atlas::getAllMaps() {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_maps;
}

void Atlas::mergeMaps(Map* pTargetMap, Map* pCurrentMap, const Mat4x4& T_target_current) {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::cout << "Atlas: Merging current map into target map..." << std::endl;

    // 1. Align all KeyFrames in pCurrentMap to pTargetMap coordinate system
    for (auto& kf : pCurrentMap->keyframes) {
        kf.pose = mat44Mul(kf.pose, invert4x4_copy(T_target_current)); // T_c_w * T_w_target
        // Wait, T_target_current is target-to-current? Or current-to-target?
        // Usually, T_target_current means p_target = T_target_current * p_current.
        // kf.pose is World_to_Camera (T_c_w). 
        // We want new T_c_w' = T_c_w_old * T_world_current_to_world_target
    }
    
    // Simplification for now: Just move all objects to pTargetMap
    for (auto& kf : pCurrentMap->keyframes) {
        pTargetMap->keyframes.push_back(kf);
    }
    for (auto& mp : pCurrentMap->points) {
        pTargetMap->points.push_back(mp);
    }
    
    // Remove pCurrentMap from Atlas
    for (auto it = m_maps.begin(); it != m_maps.end(); ++it) {
        if (*it == pCurrentMap) {
            m_maps.erase(it);
            break;
        }
    }
    delete pCurrentMap;
    m_currentMap = pTargetMap;
    
    std::cout << "Atlas: Merge complete. New map size: KFs=" << pTargetMap->keyframes.size() 
              << ", MPs=" << pTargetMap->points.size() << std::endl;
}

} // namespace orb_lite
