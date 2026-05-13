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

void Atlas::mergeMaps(Map* pTargetMap, Map* pCurrentMap, const Mat4x4& T_Wt_Wc) {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::cout << "Atlas: Merging Current Map (" << pCurrentMap->points.size() 
              << " pts) into Target Map (" << pTargetMap->points.size() << " pts)..." << std::endl;

    // 1. Calculate the inverse matrix for the KeyFrames
    Mat4x4 T_Wc_Wt;
    invert4x4(T_Wt_Wc, T_Wc_Wt); // You'll need an invert4x4 function

    // 2. Calculate the Index Offset for the arrays
    size_t pointIndexOffset = pTargetMap->points.size();
    size_t kfIndexOffset = pTargetMap->keyframes.size();

    // 3. Transform and Migrate MapPoints
    for (auto& mp : pCurrentMap->points) {
        Vec3 old_pos = mp.pos;
        
        // P_target = T_Wt_Wc * P_current
        mp.pos.x = T_Wt_Wc.m[0]*old_pos.x + T_Wt_Wc.m[1]*old_pos.y + T_Wt_Wc.m[2]*old_pos.z + T_Wt_Wc.m[3];
        mp.pos.y = T_Wt_Wc.m[4]*old_pos.x + T_Wt_Wc.m[5]*old_pos.y + T_Wt_Wc.m[6]*old_pos.z + T_Wt_Wc.m[7];
        mp.pos.z = T_Wt_Wc.m[8]*old_pos.x + T_Wt_Wc.m[9]*old_pos.y + T_Wt_Wc.m[10]*old_pos.z + T_Wt_Wc.m[11];
        
        // Update the pointer so this point knows it lives in a new map
        mp.pMap = pTargetMap;
        pTargetMap->points.push_back(mp);
    }

    // 4. Transform and Migrate KeyFrames
    for (auto& kf : pCurrentMap->keyframes) {
        // T_C_Wtarget = T_C_Wcurrent * T_Wcurrent_Wtarget
        kf.pose = mat44Mul(kf.pose, T_Wc_Wt); 
        
        // CRITICAL: Shift the MapPoint IDs so the KeyFrame looks at the correct memory locations!
        for (int i = 0; i < (int)kf.mapPointIds.size(); i++) {
            if (kf.mapPointIds[i] >= 0) {
                kf.mapPointIds[i] += pointIndexOffset;
            }
        }
        
        // Shift Keyframe relationships
        kf.id += kfIndexOffset;
        if (kf.parentId >= 0) kf.parentId += kfIndexOffset;
        for (int i = 0; i < (int)kf.neighbors.size(); i++) {
            kf.neighbors[i] += kfIndexOffset;
        }

        kf.pMap = pTargetMap;
        pTargetMap->keyframes.push_back(kf);
    }
    
    // 5. Erase the old map from the Atlas memory
    for (auto it = m_maps.begin(); it != m_maps.end(); ++it) {
        if (*it == pCurrentMap) {
            m_maps.erase(it);
            break;
        }
    }
    delete pCurrentMap;
    
    // 6. Set the Target Map as the active map
    m_currentMap = pTargetMap;
    
    std::cout << "Atlas: Merge complete. New map size: KFs=" << pTargetMap->keyframes.size() 
              << ", MPs=" << pTargetMap->points.size() << std::endl;
}

} // namespace orb_lite
