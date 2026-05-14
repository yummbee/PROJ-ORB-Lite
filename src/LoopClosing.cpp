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
    auto candidates = mpDB->detectLoopCandidates(pCurrentKF, 0.015); // Slightly relaxed BoW score
    
    if (candidates.empty()) {
        std::cout << "Loop Detection: No candidates found for KF " << pCurrentKF->id << std::endl;
        return false;
    }
    std::cout << "Loop Detection: Found " << candidates.size() << " raw candidate(s)." << std::endl;

    for(KeyFrame* pKF : candidates) {
        // Skip neighbors in the same map (Prevents matching against the frame from 0.1 seconds ago)
        if (pKF->pMap == pCurrentKF->pMap) {
            bool isNeighbor = false;
            for(int nid : pCurrentKF->neighbors) {
                if(nid == pKF->id) { isNeighbor = true; break; }
            }
            if(isNeighbor) {
                std::cout << " -> Candidate " << pKF->id << " rejected (Is a local neighbor)." << std::endl;
                continue;
            }
        }

        std::vector<std::pair<int, int>> bow_matches;
        SearchByBoW(pKF, pCurrentKF->descriptors, pCurrentKF->featVec, bow_matches);
        
        std::vector<std::pair<int, int>> opt_matches;
        for (const auto& match : bow_matches) {
            int idx_current = match.first;
            int global_mp_id = match.second;
            
            if (global_mp_id >= 0 && global_mp_id < (int)pKF->pMap->points.size() && !pKF->pMap->points[global_mp_id].isBad) {
                opt_matches.push_back({idx_current, global_mp_id});
            }
        }
        
        std::cout << " -> Candidate KF " << pKF->id << " (Map " << pKF->pMap << ") yields " << opt_matches.size() << " 3D matches." << std::endl;

        // --- FIX 1: Lower the entrance threshold for FAST features ---
        if (opt_matches.size() >= 12) { 
            NavState state;
            Mat3x3 R_cw = mat33ExtractRotation(pKF->pose);
            Mat3x3 R_wc = mat33Transpose(R_cw);
            state.q = mat33ToQuat(R_wc);
            
            Vec3 t_cw = {pKF->pose.m[3], pKF->pose.m[7], pKF->pose.m[11]};
            state.p.x = -(R_wc.m[0]*t_cw.x + R_wc.m[1]*t_cw.y + R_wc.m[2]*t_cw.z);
            state.p.y = -(R_wc.m[3]*t_cw.x + R_wc.m[4]*t_cw.y + R_wc.m[5]*t_cw.z);
            state.p.z = -(R_wc.m[6]*t_cw.x + R_wc.m[7]*t_cw.y + R_wc.m[8]*t_cw.z);
            
            Camera cam = {700.0, 700.0, 320.0, 240.0};
            
            // --- FIX 2: Capture the inliers returned by Gauss-Newton ---
            int inliers = Optimizer::poseOptimization(opt_matches, pCurrentKF->kps, cam, state, *pKF->pMap, mExtractor);
            
            std::cout << " -> Candidate " << pKF->id << " Gauss-Newton Inliers: " << inliers << std::endl;

            // --- FIX 3: Check the INLIERS, not the original array size ---
            if (inliers >= 10) {
                std::cout << "Loop Detection: CONFIRMED loop with KeyFrame " << pKF->id 
                          << " in map " << pKF->pMap << std::endl;
                pLoopKF = pKF;
                return true;
            }
        }
    }

    std::cout << "Loop Detection: No valid loops found after geometric verification." << std::endl;
    
    return false;
}

void LoopClosing::correctLoop(KeyFrame* pCurrentKF, KeyFrame* pLoopKF) {
    std::cout << "Loop Closing: Correcting KF " << pCurrentKF->id << " using KF " << pLoopKF->id << std::endl;
    
    // 1. Find 3D correspondences for Sim3 calculation
    std::vector<std::pair<int, int>> bow_matches;
    SearchByBoW(pLoopKF, pCurrentKF->descriptors, pCurrentKF->featVec, bow_matches);
    
    std::vector<Vec3> pts_loop, pts_curr;
    for (const auto& match : bow_matches) {
        int idx_curr = match.first;
        int mp_id_loop = match.second;
        int mp_id_curr = pCurrentKF->mapPointIds[idx_curr];
        
        if (mp_id_loop >= 0 && mp_id_curr >= 0 && 
            mp_id_loop < (int)pLoopKF->pMap->points.size() &&
            mp_id_curr < (int)pCurrentKF->pMap->points.size()) {
            pts_loop.push_back(pLoopKF->pMap->points[mp_id_loop].pos);
            pts_curr.push_back(pCurrentKF->pMap->points[mp_id_curr].pos);
        }
    }
    
    if (pts_loop.size() < 7) {
        std::cout << "Loop Closing: Insufficient correspondences (" << pts_loop.size() << ") for Sim3 calculation." << std::endl;
        return;
    }

    // 2. Compute Sim3 (Horn's Method)
    Sim3 S_loop_curr = Optimizer::computeSim3(pts_loop, pts_curr);
    std::cout << "Loop Closing: Sim3 calculated. Scale factor: " << S_loop_curr.s << std::endl;

    if (pCurrentKF->pMap != pLoopKF->pMap) {
        std::cout << "Loop Closing: Cross-map loop detected. Merging maps..." << std::endl;
        mpAtlas->mergeMaps(pLoopKF->pMap, pCurrentKF->pMap, S_loop_curr, mpDB);
        Optimizer::bundleAdjustment(*pLoopKF->pMap, mExtractor, 5); // Global BA after merge
    } else {
        std::cout << "Loop Closing: Intra-map loop detected. Running Essential Graph PGO..." << std::endl;
        
        // Convert Sim3 to Mat4x4 for the legacy PGO (assumes scale ~ 1.0 for intra-map loops)
        Mat3x3 R = qToMat33(S_loop_curr.q);
        Mat4x4 T_loop_curr = {
            R.m[0], R.m[1], R.m[2], S_loop_curr.t.x,
            R.m[3], R.m[4], R.m[5], S_loop_curr.t.y,
            R.m[6], R.m[7], R.m[8], S_loop_curr.t.z,
            0, 0, 0, 1
        };
        
        Optimizer::optimizeEssentialGraph(*pCurrentKF->pMap, pCurrentKF, pLoopKF, T_loop_curr);
        Optimizer::bundleAdjustment(*pCurrentKF->pMap, mExtractor, 5); // Global BA after PGO
    }
}

} // namespace orb_lite
