#include "LoopClosing.hpp"
#include "Vision.hpp"
#include "Optimizer.hpp"
#include <iostream>
#include <map>

namespace orb_lite {

LoopClosing::LoopClosing(Atlas* pAtlas, KeyFrameDatabase* pDB, Vocabulary* pVoc)
    : mpAtlas(pAtlas), mpDB(pDB), mpVoc(pVoc) {}

bool LoopClosing::detectLoop(KeyFrame* pCurrentKF, KeyFrame*& pLoopKF) {
    if(!mpVoc || !mpDB) return false;

    // Detect loop candidates from all maps
    auto candidates = mpDB->detectLoopCandidates(pCurrentKF, 0.01);
    
    for(KeyFrame* pKF : candidates) {
        // Skip neighbors in the same map
        if (pKF->pMap == pCurrentKF->pMap) {
            bool isNeighbor = false;
            for(int nid : pCurrentKF->neighbors) {
                if(nid == pKF->id) { isNeighbor = true; break; }
            }
            if(isNeighbor) continue;
        }

        std::vector<std::pair<int, int>> bow_matches;
        // Returns pairs: (Index in pKF, Index in pCurrentKF)
        SearchByBoW(pKF, pCurrentKF->descriptors, pCurrentKF->featVec, bow_matches);
        
        // --- BUG FIX 1: Translate BoW Descriptor matches into Global 3D MapPoint matches ---
        std::vector<std::pair<int, int>> opt_matches;
        for (const auto& match : bow_matches) {
            int idx_candidate = match.first;
            int idx_current = match.second;
            
            // Look up the actual 3D MapPoint ID associated with this old feature
            int global_mp_id = pKF->mapPointIds[idx_candidate];
            
            if (global_mp_id >= 0 && !pKF->pMap->points[global_mp_id].isBad) {
                // Optimizer expects: <Index of 2D pixel in CurrentKF, Index of 3D MapPoint>
                opt_matches.push_back({idx_current, global_mp_id});
            }
        }
        
        if (opt_matches.size() >= 20) {
            NavState state;
            
            // --- BUG FIX 2: Correct Global Camera Center Calculation ---
            // If pKF->pose is T_cw (World to Camera), the camera's location in the world is -R^T * t
            Mat3x3 R_cw = mat33ExtractRotation(pKF->pose);
            Mat3x3 R_wc = mat33Transpose(R_cw);
            state.q = mat33ToQuat(R_wc);
            
            Vec3 t_cw = {pKF->pose.m[3], pKF->pose.m[7], pKF->pose.m[11]};
            state.p.x = -(R_wc.m[0]*t_cw.x + R_wc.m[1]*t_cw.y + R_wc.m[2]*t_cw.z);
            state.p.y = -(R_wc.m[3]*t_cw.x + R_wc.m[4]*t_cw.y + R_wc.m[5]*t_cw.z);
            state.p.z = -(R_wc.m[6]*t_cw.x + R_wc.m[7]*t_cw.y + R_wc.m[8]*t_cw.z);
            
            Camera cam = {700.0, 700.0, 320.0, 240.0};
            
            // The optimizer will refine the state and remove outliers from opt_matches
            Optimizer::poseOptimization(opt_matches, pCurrentKF->kps, cam, state, *pKF->pMap);
            
            // If the geometry is physically valid, a minimum number of inliers will survive
            if (opt_matches.size() >= 15) {
                std::cout << "Loop Detection: CONFIRMED loop with KeyFrame " << pKF->id 
                          << " in map " << pKF->pMap << std::endl;
                pLoopKF = pKF;
                return true;
            }
        }
    }
    
    return false;
}

void LoopClosing::correctLoop(KeyFrame* pCurrentKF, KeyFrame* pLoopKF) {
    std::cout << "Loop Closing: Correcting KF " << pCurrentKF->id << " using KF " << pLoopKF->id << std::endl;
    
    // Compute relative correction T_loop_curr
    // We want P_loop = T_loop_curr * P_curr
    // Since we optimized 'state' in detectLoop to match pLoopKF, we can use that.
    // However, for simplicity, we compute it from the confirmed poses.
    
    Mat4x4 T_curr_world; invert4x4(pCurrentKF->pose, T_curr_world);
    Mat4x4 T_loop_curr = mat44Mul(pLoopKF->pose, T_curr_world);

    if (pCurrentKF->pMap != pLoopKF->pMap) {
        std::cout << "Loop Closing: Cross-map loop detected. Merging maps..." << std::endl;
        mpAtlas->mergeMaps(pLoopKF->pMap, pCurrentKF->pMap, T_loop_curr);
    } else {
        std::cout << "Loop Closing: Intra-map loop detected. Running Essential Graph PGO..." << std::endl;
        Optimizer::optimizeEssentialGraph(*pCurrentKF->pMap, pCurrentKF, pLoopKF, T_loop_curr);
    }
}

} // namespace orb_lite
