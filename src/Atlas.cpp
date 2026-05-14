#include "Atlas.hpp"
#include "KeyFrameDatabase.hpp"
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

void Atlas::mergeMaps(Map* pTargetMap, Map* pCurrentMap, const Sim3& S_Wt_Wc, KeyFrameDatabase* pKFDB) {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::cout << "Atlas: Merging Current Map (" << pCurrentMap->points.size() 
              << " pts) into Target Map (" << pTargetMap->points.size() << " pts)..." << std::endl;

    // 0. Remove from database before pointers become invalid
    if (pKFDB) {
        for (auto& kf : pCurrentMap->keyframes) {
            pKFDB->erase(&kf);
        }
    }

    // 2. Calculate the Index Offset for the arrays
    size_t pointIndexOffset = pTargetMap->points.size();
    size_t kfIndexOffset = pTargetMap->keyframes.size();

    // 3. Transform and Migrate MapPoints
    for (auto& mp : pCurrentMap->points) {
        mp.pos = S_Wt_Wc.transform(mp.pos);
        mp.pMap = pTargetMap;
        pTargetMap->points.push_back(mp);
    }

    // 4. Transform and Migrate KeyFrames
    Mat3x3 R_Wt_Wc = qToMat33(S_Wt_Wc.q);
    for (auto& kf : pCurrentMap->keyframes) {
        // T_Wt_c = S_Wt_Wc * T_Wc_c
        Mat3x3 R_Wc_c = mat33ExtractRotation(kf.pose);
        Vec3 t_Wc_c = {kf.pose.m[3], kf.pose.m[7], kf.pose.m[11]};
        
        Mat3x3 R_Wt_c = mat33Mul(R_Wt_Wc, R_Wc_c);
        Vec3 t_Wt_c = {
            S_Wt_Wc.s * (R_Wt_Wc.m[0]*t_Wc_c.x + R_Wt_Wc.m[1]*t_Wc_c.y + R_Wt_Wc.m[2]*t_Wc_c.z) + S_Wt_Wc.t.x,
            S_Wt_Wc.s * (R_Wt_Wc.m[3]*t_Wc_c.x + R_Wt_Wc.m[4]*t_Wc_c.y + R_Wt_Wc.m[5]*t_Wc_c.z) + S_Wt_Wc.t.y,
            S_Wt_Wc.s * (R_Wt_Wc.m[6]*t_Wc_c.x + R_Wt_Wc.m[7]*t_Wc_c.y + R_Wt_Wc.m[8]*t_Wc_c.z) + S_Wt_Wc.t.z
        };
        
        kf.pose = {
            R_Wt_c.m[0], R_Wt_c.m[1], R_Wt_c.m[2], t_Wt_c.x,
            R_Wt_c.m[3], R_Wt_c.m[4], R_Wt_c.m[5], t_Wt_c.y,
            R_Wt_c.m[6], R_Wt_c.m[7], R_Wt_c.m[8], t_Wt_c.z,
            0, 0, 0, 1
        };
        
        // Shift the MapPoint IDs
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
        
        if (pKFDB) {
            pKFDB->add(&pTargetMap->keyframes.back());
        }
    }
    
    // 5. Erase the old map
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
