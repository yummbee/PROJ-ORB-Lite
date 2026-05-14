#include "Optimizer.hpp"
#include <cmath>
#include <iostream>
#include <unordered_set>
#include <vector>

namespace orb_lite {

int Optimizer::poseOptimization(std::vector<std::pair<int, int>> &matches,
                                 const std::vector<KeyPoint> &kps,
                                 const Camera &cam, NavState &state, Map &map,
                                 const FeatureExtractor &extractor) {
  /*if (matches.size() < 10)
    return 0;*/

  // 1. INITIALIZATION: Snapshot the absolute starting state
  Mat3x3 R_wc_init = qToMat33(state.q);
  double x = state.p.x, y = state.p.y, z = state.p.z;
  
  // The delta correction MUST start at exactly 0!
  double yaw = 0.0; 

  const double thHuber2D = 36.0;
  const double delta = std::sqrt(thHuber2D);
  std::vector<bool> bOutlier(matches.size(), false);

  for (int iter = 0; iter < 10; iter++) {
    Mat4x4 JTJ = Mat4x4::zeros(); 
    Vec4 JTr = {0, 0, 0, 0};

    Mat3x3 R_yaw = {
        (float)std::cos(yaw), (float)-std::sin(yaw), 0,
        (float)std::sin(yaw), (float)std::cos(yaw),  0,
        0,                    0,                     1
    };
    
    // Apply the delta to the initial state
    Mat3x3 R_wc = mat33Mul(R_wc_init, R_yaw);
    Mat3x3 R_cw = mat33Transpose(R_wc);

    for (size_t i = 0; i < matches.size(); i++) {
      if (bOutlier[i]) continue;

      const auto &m = matches[i];
      const auto &mp = map.points[m.second];
      const auto &kp = kps[m.first];
      double invSigma2 = extractor.mvInvLevelSigma2[kp.octave];

      Vec3 rel_p = {mp.pos.x - x, mp.pos.y - y, mp.pos.z - z};
      Vec3 P_c = {
          R_cw.m[0] * rel_p.x + R_cw.m[1] * rel_p.y + R_cw.m[2] * rel_p.z,
          R_cw.m[3] * rel_p.x + R_cw.m[4] * rel_p.y + R_cw.m[5] * rel_p.z,
          R_cw.m[6] * rel_p.x + R_cw.m[7] * rel_p.y + R_cw.m[8] * rel_p.z};

      if (P_c.z <= 0.1) continue;

      double invz = 1.0 / P_c.z;
      double invz2 = invz * invz;
      Vec2 uv = cam.project(P_c);

      double ex = kp.x - uv.x;
      double ey = kp.y - uv.y;
      double chi2 = (ex * ex + ey * ey) * invSigma2;

      double weight = 1.0;
      if (chi2 > thHuber2D) {
          weight = delta / std::sqrt(chi2);
      }
      weight *= invSigma2; 

      double du_dx_c = cam.fx * invz;
      double du_dz_c = -cam.fx * P_c.x * invz2;
      double dv_dy_c = cam.fy * invz;
      double dv_dz_c = -cam.fy * P_c.y * invz2;

      double J[2][4]; 
      for (int j = 0; j < 3; j++) {
        J[0][j] = -(du_dx_c * R_cw.m[j] + du_dz_c * R_cw.m[6 + j]);
        J[1][j] = -(dv_dy_c * R_cw.m[3 + j] + dv_dz_c * R_cw.m[6 + j]);
      }

      double sy = std::sin(yaw);
      double cy = std::cos(yaw);
      Mat3x3 dR_yaw_T = {(float)-sy, (float)cy,  0,
                         (float)-cy, (float)-sy, 0,
                         0,          0,          0};

      Mat3x3 R_wc_init_T = mat33Transpose(R_wc_init);
      Mat3x3 dR_cw_dyaw = mat33Mul(dR_yaw_T, R_wc_init_T);
      Vec3 dP_dyaw = mat33MulVec3(dR_cw_dyaw, rel_p);

      J[0][3] = (du_dx_c * dP_dyaw.x + du_dz_c * dP_dyaw.z);
      J[1][3] = (dv_dy_c * dP_dyaw.y + dv_dz_c * dP_dyaw.z);

      for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
          JTJ.at(r, c) += (float)(weight * (J[0][r] * J[0][c] + J[1][r] * J[1][c]));
        }
        JTr[r] += weight * (J[0][r] * ex + J[1][r] * ey);
      }
    }

    // TIKHONOV REGULARIZATION (The IMU Anchor)
    // Multiply translation diagonals by a penalty factor to damp lateral sliding
    float imu_trust_factor = 1.0f; // This is a hyperparameter that controls how strongly we anchor to the IMU prediction.
    float penalty = 2.0f; 
    JTJ.at(0, 0) *= imu_trust_factor;
    JTJ.at(1, 1) *= imu_trust_factor;
    JTJ.at(2, 2) *= imu_trust_factor;

    for (int r = 0; r < 4; r++) JTJ.at(r, r) += 1e-2;

    Vec4 update = solve4x4(JTJ, JTr);

    // --- DIAGNOSTIC X-RAY ---
    if (iter == 0) {
        std::cout << "\n[OPTIMIZER X-RAY] Iteration 0" << std::endl;
        std::cout << "Calculated Update -> X:" << update.x << " Y:" << update.y << " Z:" << update.z 
                  << " Yaw(deg):" << (update.w * 180.0 / 3.1415926535) << std::endl;
        
        for (size_t i = 0; i < std::min((size_t)3, matches.size()); i++) {
            const auto &mp = map.points[matches[i].second];
            const auto &kp = kps[matches[i].first];
            Vec3 rel_p = {mp.pos.x - x, mp.pos.y - y, mp.pos.z - z};
            Vec3 P_c = {
                R_cw.m[0]*rel_p.x + R_cw.m[1]*rel_p.y + R_cw.m[2]*rel_p.z,
                R_cw.m[3]*rel_p.x + R_cw.m[4]*rel_p.y + R_cw.m[5]*rel_p.z,
                R_cw.m[6]*rel_p.x + R_cw.m[7]*rel_p.y + R_cw.m[8]*rel_p.z};
            Vec2 uv = cam.project(P_c);
            std::cout << "  Point " << i << ": Reality=[" << kp.x << "," << kp.y << "] Math=[" << uv.x << "," << uv.y << "] Err=[" << (kp.x-uv.x) << "," << (kp.y-uv.y) << "]" << std::endl;
        }
    }

    // Limit translation update
    double u_mag = std::sqrt(update.x * update.x + update.y * update.y + update.z * update.z);
    if (u_mag > 0.2) { update.x *= 0.2 / u_mag; update.y *= 0.2 / u_mag; update.z *= 0.2 / u_mag; }
    
    // Limit yaw update to 0.01 radians
    if (std::abs(update.w) > 0.01) { update.w = (update.w > 0) ? 0.01 : -0.01; }

    x += update.x; y += update.y; z += update.z; yaw += update.w;
    if (u_mag < 1e-4 && std::abs(update.w) < 1e-4) break;
  }

  // 2. STATE SAVE: Write the combined matrices back to the global state
  Mat3x3 R_yaw_final = {
      (float)std::cos(yaw), (float)-std::sin(yaw), 0,
      (float)std::sin(yaw), (float)std::cos(yaw),  0,
      0,                    0,                     1
  };
  
  state.p = { (float)x, (float)y, (float)z };
  state.q = mat33ToQuat(mat33Mul(R_wc_init, R_yaw_final));

  return (int)matches.size();
}

// --- HELPER FUNCTION: Optimize a single 3D MapPoint (3-DoF) ---
void Optimizer::optimizeSingleMapPoint(
    MapPoint &mp, Map &map, const Camera &cam, int mp_idx,
    const FeatureExtractor &extractor,
    const std::unordered_set<int> &fixedKfs) {
  Mat3x3 H = Mat3x3::zeros();
  Vec3 b = {0, 0, 0};
  double total_chi2 = 0;
  int obs_count = 0;

  // Find all KeyFrames that observe this point
  for (const auto &kf : map.keyframes) {
    if (!fixedKfs.empty() && fixedKfs.find(kf.id) == fixedKfs.end())
      continue;

    // Find if this KF sees this map point
    int kp_idx = -1;
    for (size_t i = 0; i < kf.mapPointIds.size(); i++) {
      if (kf.mapPointIds[i] == mp_idx) {
        kp_idx = (int)i;
        break;
      }
    }
    if (kp_idx < 0)
      continue;

    // Get camera pose T_cw
    Mat3x3 R_cw;
    Vec3 t_cw = {kf.pose.m[3], kf.pose.m[7], kf.pose.m[11]};
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        R_cw.m[i * 3 + j] = kf.pose.at(i, j);

    // Project point
    Vec3 P_c = {R_cw.m[0] * mp.pos.x + R_cw.m[1] * mp.pos.y +
                    R_cw.m[2] * mp.pos.z + t_cw.x,
                R_cw.m[3] * mp.pos.x + R_cw.m[4] * mp.pos.y +
                    R_cw.m[5] * mp.pos.z + t_cw.y,
                R_cw.m[6] * mp.pos.x + R_cw.m[7] * mp.pos.y +
                    R_cw.m[8] * mp.pos.z + t_cw.z};

    if (P_c.z < 0.1)
      continue;

    double invz = 1.0 / P_c.z;
    double invz2 = invz * invz;
    Vec2 uv = cam.project(P_c);

    const auto &kp = kf.kps[kp_idx];
    double invSigma2 = extractor.mvInvLevelSigma2[kp.octave];

    double ex = kp.x - uv.x;
    double ey = kp.y - uv.y;
    double chi2 = (ex * ex + ey * ey) * invSigma2;

    double weight = 1.0;
    const double huber2D = 36.0;
    const double delta = std::sqrt(huber2D);
    if (chi2 > huber2D)
      weight = delta / std::sqrt(chi2);

    weight *= invSigma2;

    // Jacobian of projection w.r.t P_c
    double du_dx = cam.fx * invz;
    double du_dz = -cam.fx * P_c.x * invz2;
    double dv_dy = cam.fy * invz;
    double dv_dz = -cam.fy * P_c.y * invz2;

    // Jacobian of error w.r.t P_w (Chain Rule: J_proj * R_cw)
    // Since error = measured - projected, Jacobian is negative
    double Jx[3], Jy[3];
    for (int i = 0; i < 3; i++) {
      Jx[i] = -(du_dx * R_cw.m[i] + du_dz * R_cw.m[6 + i]);
      Jy[i] = -(dv_dy * R_cw.m[3 + i] + dv_dz * R_cw.m[6 + i]);
    }

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        H.m[i * 3 + j] += (float)(weight * (Jx[i] * Jx[j] + Jy[i] * Jy[j]));
      }
      b[i] += weight * (Jx[i] * ex + Jy[i] * ey);
    }
    obs_count++;
  }

  if (obs_count < 2)
    return; // Needs at least 2 views to optimize

  // Damping (Levenberg-Marquardt)
  for (int i = 0; i < 3; i++)
    H.m[i * 3 + i] += H.m[i * 3 + i] * 0.01f + 1e-4f;

  // Solve 3x3
  Vec3 update =
      solve3x3(H, b); // Assumes you have a 3x3 solver next to your 4x4 solver

  mp.pos.x += update.x;
  mp.pos.y += update.y;
  mp.pos.z += update.z;
}

// --- GLOBAL BUNDLE ADJUSTMENT ---
void Optimizer::bundleAdjustment(Map &map, const FeatureExtractor &extractor,
                                 int numIterations) {
  if (map.keyframes.size() < 2)
    return;
  std::cout << "Starting Global BA. KFs: " << map.keyframes.size()
            << " Points: " << map.points.size() << std::endl;

  for (int iter = 0; iter < numIterations; iter++) {

    // 1. Optimize MapPoints (Geometry)
    for (size_t i = 0; i < map.points.size(); i++) {
      if (!map.points[i].isBad) {
        Camera cam = {700.0, 700.0, 320.0, 240.0};
        std::unordered_set<int> emptySet;
        Optimizer::optimizeSingleMapPoint(map.points[i], map, cam, (int)i,
                                          extractor, emptySet);
      }
    }

    // 2. Optimize KeyFrames (Poses)
    for (size_t i = 1; i < map.keyframes.size(); i++) {
      KeyFrame &kf = map.keyframes[i];

      std::vector<std::pair<int, int>> kf_matches;
      for (size_t j = 0; j < kf.mapPointIds.size(); j++) {
        if (kf.mapPointIds[j] >= 0 &&
            kf.mapPointIds[j] < (int)map.points.size() &&
            !map.points[kf.mapPointIds[j]].isBad) {
          kf_matches.push_back({(int)j, kf.mapPointIds[j]});
        }
      }

      NavState kf_state;
      Mat3x3 R_cw = mat33ExtractRotation(kf.pose);
      Mat3x3 R_wc = mat33Transpose(R_cw);
      kf_state.q = mat33ToQuat(R_wc);
      Vec3 t_cw = {kf.pose.m[3], kf.pose.m[7], kf.pose.m[11]};
      kf_state.p.x =
          -(R_wc.m[0] * t_cw.x + R_wc.m[1] * t_cw.y + R_wc.m[2] * t_cw.z);
      kf_state.p.y =
          -(R_wc.m[3] * t_cw.x + R_wc.m[4] * t_cw.y + R_wc.m[5] * t_cw.z);
      kf_state.p.z =
          -(R_wc.m[6] * t_cw.x + R_wc.m[7] * t_cw.y + R_wc.m[8] * t_cw.z);

      Camera cam = {700.0, 700.0, 320.0, 240.0};
      Optimizer::poseOptimization(kf_matches, kf.kps, cam, kf_state, map,
                                  extractor);

      // Convert back
      R_wc = qToMat33(kf_state.q);
      R_cw = mat33Transpose(R_wc);
      for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
          kf.pose.at(r, c) = R_cw.m[r * 3 + c];
      kf.pose.m[3] = -(R_cw.m[0] * kf_state.p.x + R_cw.m[1] * kf_state.p.y +
                       R_cw.m[2] * kf_state.p.z);
      kf.pose.m[7] = -(R_cw.m[3] * kf_state.p.x + R_cw.m[4] * kf_state.p.y +
                       R_cw.m[5] * kf_state.p.z);
      kf.pose.m[11] = -(R_cw.m[6] * kf_state.p.x + R_cw.m[7] * kf_state.p.y +
                        R_cw.m[8] * kf_state.p.z);
    }
  }
}

// --- LOCAL BUNDLE ADJUSTMENT ---
void Optimizer::localBundleAdjustment(Map &map,
                                      const FeatureExtractor &extractor,
                                      int currentKfId, int windowSize) {
  if (currentKfId < 0 || currentKfId >= (int)map.keyframes.size()) {
    std::cout << "localBundleAdjustment: Invalid currentKfId " << currentKfId
              << std::endl;
    return;
  }

  KeyFrame &currentKF = map.keyframes[currentKfId];

  // Rule 1: Local KeyFrames
  std::unordered_set<int> localKFs;
  localKFs.insert(currentKF.id);
  for (int neighbor_id : currentKF.neighbors) {
    if (neighbor_id >= 0 && neighbor_id < (int)map.keyframes.size()) {
      localKFs.insert(neighbor_id);
    }
  }

  // Ensure KeyFrame 0 is NEVER in localKFs (it must be anchored)
  if (localKFs.find(0) != localKFs.end()) {
    std::cout
        << "localBundleAdjustment: KeyFrame 0 is in localKFs (should be fixed)"
        << std::endl;
    localKFs.erase(0);
  }

  // Rule 2: Local MapPoints
  std::unordered_set<int> localMPs;
  for (int kf_id : localKFs) {
    KeyFrame &localKF = map.keyframes[kf_id];
    for (int mp_id : localKF.mapPointIds) {
      if (mp_id >= 0 && mp_id < (int)map.points.size() &&
          !map.points[mp_id].isBad) {
        localMPs.insert(mp_id);
      }
    }
  }

  // Rule 3: Fixed KeyFrames (Anchors)
  std::unordered_set<int> fixedKFs;
  for (const auto &kf : map.keyframes) {
    if (localKFs.find(kf.id) != localKFs.end())
      continue;

    bool sees_local_mp = false;
    for (int mp_id : kf.mapPointIds) {
      if (localMPs.find(mp_id) != localMPs.end()) {
        sees_local_mp = true;
        break;
      }
    }
    if (sees_local_mp) {
      fixedKFs.insert(kf.id);
    }
  }
  // Always include K0 as fixed if not already
  fixedKFs.insert(0);

  // Combined active KFs
  std::unordered_set<int> activeKFs;
  activeKFs.insert(localKFs.begin(), localKFs.end());
  activeKFs.insert(fixedKFs.begin(), fixedKFs.end());

  // Run Alternating Optimization 5 times
  for (int iter = 0; iter < 5; iter++) {

    // 1. Optimize Local MapPoints
    for (int mp_id : localMPs) {
      Camera cam = {700.0, 700.0, 320.0, 240.0};
      Optimizer::optimizeSingleMapPoint(map.points[mp_id], map, cam, mp_id,
                                        extractor, activeKFs);
    }

    // 2. Optimize Local KeyFrames ONLY
    for (int i : localKFs) {
      if (i >= (int)map.keyframes.size())
        continue;

      KeyFrame &kf = map.keyframes[i];

      // Build pseudo-matches
      std::vector<std::pair<int, int>> kf_matches;
      for (size_t j = 0; j < kf.mapPointIds.size(); j++) {
        int mp_id = kf.mapPointIds[j];
        if (mp_id >= 0 && mp_id < (int)map.points.size() &&
            !map.points[mp_id].isBad) {
          kf_matches.push_back({j, mp_id});
        }
      }

      // Convert to NavState, optimize, convert back
      NavState kf_state;
      Mat3x3 R_cw;
      for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
          R_cw.m[r * 3 + c] = kf.pose.at(r, c);
      Mat3x3 R_wc = mat33Transpose(R_cw);

      kf_state.q = mat33ToQuat(R_wc);
      Vec3 t_cw = {kf.pose.m[3], kf.pose.m[7], kf.pose.m[11]};
      kf_state.p.x =
          -(R_wc.m[0] * t_cw.x + R_wc.m[1] * t_cw.y + R_wc.m[2] * t_cw.z);
      kf_state.p.y =
          -(R_wc.m[3] * t_cw.x + R_wc.m[4] * t_cw.y + R_wc.m[5] * t_cw.z);
      kf_state.p.z =
          -(R_wc.m[6] * t_cw.x + R_wc.m[7] * t_cw.y + R_wc.m[8] * t_cw.z);

      Camera cam = {700.0, 700.0, 320.0, 240.0};
      Optimizer::poseOptimization(kf_matches, kf.kps, cam, kf_state, map,
                                  extractor);

      R_wc = qToMat33(kf_state.q);
      R_cw = mat33Transpose(R_wc);
      for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
          kf.pose.at(r, c) = R_cw.m[r * 3 + c];

      Vec3 new_t_cw = {-(R_cw.m[0] * kf_state.p.x + R_cw.m[1] * kf_state.p.y +
                         R_cw.m[2] * kf_state.p.z),
                       -(R_cw.m[3] * kf_state.p.x + R_cw.m[4] * kf_state.p.y +
                         R_cw.m[5] * kf_state.p.z),
                       -(R_cw.m[6] * kf_state.p.x + R_cw.m[7] * kf_state.p.y +
                         R_cw.m[8] * kf_state.p.z)};
      kf.pose.m[3] = new_t_cw.x;
      kf.pose.m[7] = new_t_cw.y;
      kf.pose.m[11] = new_t_cw.z;
    }
  }

  std::cout << "Local BA finished. Local KFs: " << localKFs.size()
            << " Local MPs: " << localMPs.size() << std::endl;
}

void Optimizer::optimizeEssentialGraph(Map &map, KeyFrame *pCurrentKF,
                                       KeyFrame *pLoopKF, const Mat4x4 &Tloop) {
  std::vector<int> path;
  int currId = pCurrentKF->id;

  int safety = 0;
  while (currId != -1 && currId != pLoopKF->id && safety < 1000) {
    path.push_back(currId);
    if (currId >= (int)map.keyframes.size())
      break;
    currId = map.keyframes[currId].parentId;
    safety++;
  }

  if (currId == pLoopKF->id) {
    std::cout << "PGO: Optimizing path of length " << path.size() << std::endl;

    // Tloop represents the error of CurrentKF to LoopKF
    double yaw_error = std::atan2(Tloop.m[4], Tloop.m[0]);
    Vec3 t_error = {Tloop.m[3], Tloop.m[7], Tloop.m[11]};

    std::unordered_set<int> shifted_points;
    int total_edges = path.size();

    for (int i = 0; i < total_edges; i++) {
      int kfId = path[i];
      if (kfId < 0 || kfId >= (int)map.keyframes.size())
        continue;
      KeyFrame &kf = map.keyframes[kfId];

      double weight = (double)(total_edges - i) / (double)total_edges;

      // Distribute yaw error
      double dYaw = yaw_error * weight;
      Mat3x3 R_cw;
      for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
          R_cw.m[r * 3 + c] = kf.pose.at(r, c);

      Mat3x3 dRy = {(float)std::cos(dYaw),
                    (float)-std::sin(dYaw),
                    0.0f,
                    (float)std::sin(dYaw),
                    (float)std::cos(dYaw),
                    0.0f,
                    0.0f,
                    0.0f,
                    1.0f};

      Mat3x3 R_wc = mat33Transpose(R_cw);
      R_wc = mat33Mul(dRy, R_wc); // Rotate around world Z
      R_cw = mat33Transpose(R_wc);
      for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
          kf.pose.at(r, c) = R_cw.m[r * 3 + c];

      // Distribute translation error
      Vec3 dt = {t_error.x * weight, t_error.y * weight, t_error.z * weight};

      // Since we updated R_cw we must apply translation to C_w then calculate
      // t_cw But simple translation addition in T_cw is not exactly world
      // space. t_cw = -R_cw * C_w. If C_w shifts by dt, then t_cw_new = -R_cw *
      // (C_w + dt) = t_cw_old - R_cw*dt Actually, Tloop provided is likely
      // world to world or delta. Assuming dt is in world coordinates:
      Vec3 dt_cam = {R_cw.m[0] * dt.x + R_cw.m[1] * dt.y + R_cw.m[2] * dt.z,
                     R_cw.m[3] * dt.x + R_cw.m[4] * dt.y + R_cw.m[5] * dt.z,
                     R_cw.m[6] * dt.x + R_cw.m[7] * dt.y + R_cw.m[8] * dt.z};

      kf.pose.m[3] -= dt_cam.x;
      kf.pose.m[7] -= dt_cam.y;
      kf.pose.m[11] -= dt_cam.z;

      // CRITICAL: Move the associated MapPoints
      for (int mp_id : kf.mapPointIds) {
        if (mp_id >= 0 && mp_id < (int)map.points.size() &&
            !map.points[mp_id].isBad) {
          if (shifted_points.find(mp_id) == shifted_points.end()) {
            // Rotate and translate MapPoint in world space
            Vec3 p = map.points[mp_id].pos;
            Vec3 p_rot = {dRy.m[0] * p.x + dRy.m[1] * p.y + dRy.m[2] * p.z,
                          dRy.m[3] * p.x + dRy.m[4] * p.y + dRy.m[5] * p.z,
                          dRy.m[6] * p.x + dRy.m[7] * p.y + dRy.m[8] * p.z};
            map.points[mp_id].pos.x = p_rot.x + dt.x;
            map.points[mp_id].pos.y = p_rot.y + dt.y;
            map.points[mp_id].pos.z = p_rot.z + dt.z;
            shifted_points.insert(mp_id);
          }
        }
      }
    }
  } else {
    pCurrentKF->pose = mat44Mul(Tloop, pCurrentKF->pose);
  }
}

Sim3 Optimizer::computeSim3(const std::vector<Vec3>& pts1, const std::vector<Vec3>& pts2) {
    if (pts1.size() != pts2.size() || pts1.empty()) return { {1,0,0,0}, {0,0,0}, 1.0 };

    size_t N = pts1.size();
    Vec3 c1 = {0,0,0}, c2 = {0,0,0};
    for (size_t i = 0; i < N; i++) {
        c1.x += pts1[i].x; c1.y += pts1[i].y; c1.z += pts1[i].z;
        c2.x += pts2[i].x; c2.y += pts2[i].y; c2.z += pts2[i].z;
    }
    c1.x /= N; c1.y /= N; c1.z /= N;
    c2.x /= N; c2.y /= N; c2.z /= N;

    std::vector<Vec3> d1(N), d2(N);
    double ss1 = 0, ss2 = 0;
    for (size_t i = 0; i < N; i++) {
        d1[i] = {pts1[i].x - c1.x, pts1[i].y - c1.y, pts1[i].z - c1.z};
        d2[i] = {pts2[i].x - c2.x, pts2[i].y - c2.y, pts2[i].z - c2.z};
        ss1 += d1[i].x*d1[i].x + d1[i].y*d1[i].y + d1[i].z*d1[i].z;
        ss2 += d2[i].x*d2[i].x + d2[i].y*d2[i].y + d2[i].z*d2[i].z;
    }

    double scale = (ss2 > 1e-12) ? std::sqrt(ss1 / ss2) : 1.0;

    double M[9] = {0};
    for (size_t i = 0; i < N; i++) {
        M[0] += d1[i].x * d2[i].x; M[1] += d1[i].x * d2[i].y; M[2] += d1[i].x * d2[i].z;
        M[3] += d1[i].y * d2[i].x; M[4] += d1[i].y * d2[i].y; M[5] += d1[i].y * d2[i].z;
        M[6] += d1[i].z * d2[i].x; M[7] += d1[i].z * d2[i].y; M[8] += d1[i].z * d2[i].z;
    }

    double Q[16];
    Q[0] = M[0] + M[4] + M[8];
    Q[1] = M[5] - M[7];
    Q[2] = M[6] - M[2];
    Q[3] = M[1] - M[3];

    Q[4] = Q[1];
    Q[5] = M[0] - M[4] - M[8];
    Q[6] = M[3] + M[1];
    Q[7] = M[6] + M[2];

    Q[8] = Q[2];
    Q[9] = Q[6];
    Q[10] = -M[0] + M[4] - M[8];
    Q[11] = M[7] + M[5];

    Q[12] = Q[3];
    Q[13] = Q[7];
    Q[14] = Q[11];
    Q[15] = -M[0] - M[4] + M[8];

    double q_vec[4] = {1, 0, 0, 0};
    for (int it = 0; it < 50; it++) {
        double nq[4] = {0};
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                nq[i] += Q[i * 4 + j] * q_vec[j];
            }
        }
        double norm = std::sqrt(nq[0]*nq[0] + nq[1]*nq[1] + nq[2]*nq[2] + nq[3]*nq[3]);
        if (norm < 1e-15) break;
        for (int i = 0; i < 4; i++) q_vec[i] = nq[i] / norm;
    }

    Sim3 result;
    result.q = {q_vec[0], q_vec[1], q_vec[2], q_vec[3]};
    result.s = scale;
    
    Mat3x3 R = qToMat33(result.q);
    Vec3 Rc2 = mat33MulVec3(R, c2);
    result.t = {c1.x - scale * Rc2.x, c1.y - scale * Rc2.y, c1.z - scale * Rc2.z};

    return result;
}

} // namespace orb_lite
