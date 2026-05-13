#include "Optimizer.hpp"
#include <cmath>
#include <iostream>
#include <vector>

namespace orb_lite {

void Optimizer::poseOptimization(
    const std::vector<std::pair<int, int>> &matches,
    const std::vector<KeyPoint> &kps, const Camera &cam, NavState &state,
    Map &map) {
  if (matches.size() < 10)
    return;

  // We optimize X, Y, Z, Yaw (4-DoF)
  // Pitch/Roll are locked to IMU gravity alignment
  double x = state.p.x, y = state.p.y, z = state.p.z;
  double yaw = 0.0; // Initial delta yaw

  Mat3x3 R_wc_init = qToMat33(state.q);

  // Huber threshold (in pixels)
  const double delta = 1.5;
  const double chi2_threshold = 5.991; // 2 DoF at 95%

  for (int iter = 0; iter < 20; iter++) {
    Mat4x4 JTJ = Mat4x4::zeros(); // 4x4 for X,Y,Z,Yaw
    Vec4 JTr = {0, 0, 0, 0};
    double total_chi2 = 0;
    int inliers = 0;

    Mat3x3 R_yaw = {(float)std::cos(yaw),
                    (float)-std::sin(yaw),
                    0,
                    (float)std::sin(yaw),
                    (float)std::cos(yaw),
                    0,
                    0,
                    0,
                    1};
    Mat3x3 R_wc = mat33Mul(R_yaw, R_wc_init);
    Mat3x3 R_cw = mat33Transpose(R_wc);

    for (const auto &m : matches) {
      const auto &mp = map.points[m.second];
      const auto &kp = kps[m.first];

      Vec3 rel_p = {mp.pos.x - x, mp.pos.y - y, mp.pos.z - z};
      Vec3 P_c = {
          R_cw.m[0] * rel_p.x + R_cw.m[1] * rel_p.y + R_cw.m[2] * rel_p.z,
          R_cw.m[3] * rel_p.x + R_cw.m[4] * rel_p.y + R_cw.m[5] * rel_p.z,
          R_cw.m[6] * rel_p.x + R_cw.m[7] * rel_p.y + R_cw.m[8] * rel_p.z};

      if (P_c.z <= 0.1)
        continue;

      double invz = 1.0 / P_c.z;
      double invz2 = invz * invz;
      Vec2 uv = cam.project(P_c);

      double ex = kp.x - uv.x;
      double ey = kp.y - uv.y;
      double chi2 = ex * ex + ey * ey;

      // Robust Huber weighting
      double weight = 1.0;
      if (chi2 > delta * delta) {
        weight = delta / std::sqrt(chi2);
      }

      if (iter > 10 && chi2 > chi2_threshold)
        continue; // Outlier rejection

      total_chi2 += chi2 * weight;
      inliers++;

      // Jacobian w.r.t. Camera Position (in world frame)
      // d(uv)/dp_w = d(uv)/dP_c * dP_c/dp_w
      // d(uv)/dP_c = [fx/z, 0, -fx*x/z^2; 0, fy/z, -fy*y/z^2]
      // dP_c/dp_w = -R_cw

      double du_dx_c = cam.fx * invz;
      double du_dz_c = -cam.fx * P_c.x * invz2;
      double dv_dy_c = cam.fy * invz;
      double dv_dz_c = -cam.fy * P_c.y * invz2;

      // J_pos (2x3) = d(uv)/dP_c * (-R_cw)
      double J[2][4]; // J for X, Y, Z, Yaw
      for (int i = 0; i < 3; i++) {
        J[0][i] = -(du_dx_c * R_cw.m[i] + du_dz_c * R_cw.m[6 + i]);
        J[1][i] = -(dv_dy_c * R_cw.m[3 + i] + dv_dz_c * R_cw.m[6 + i]);
      }

      // dP_c/dyaw = d(R_cw)/dyaw * (P_w - p_w)
      // R_cw = R_wc_init^T * R_yaw^T
      double sy = std::sin(yaw);
      double cy = std::cos(yaw);
      Mat3x3 dR_yaw_T = {-sy, cy, 0, -cy, -sy, 0, 0, 0, 0};
      Mat3x3 R_wc_init_T = mat33Transpose(R_wc_init);
      Mat3x3 dR_cw_dyaw = mat33Mul(R_wc_init_T, dR_yaw_T);
      Vec3 dP_dyaw = mat33MulVec3(dR_cw_dyaw, rel_p);

      J[0][3] = du_dx_c * dP_dyaw.x + du_dz_c * dP_dyaw.z;
      J[1][3] = dv_dy_c * dP_dyaw.y + dv_dz_c * dP_dyaw.z;

      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          JTJ.at(i, j) +=
              (float)(weight * (J[0][i] * J[0][j] + J[1][i] * J[1][j]));
        }
        JTr[i] += weight * (J[0][i] * ex + J[1][i] * ey);
      }
    }

    // Add Levenberg-Marquardt style damping (scaled by diagonal) to prevent the
    // solver from exploding
    for (int i = 0; i < 4; i++) {
      JTJ.at(i, i) += JTJ.at(i, i) * 0.01 + 1e-2;
    }

    // Solve 4x4 system
    Vec4 update = solve4x4(JTJ, JTr);

    // Trust region: cap the maximum positional update per iteration
    double u_mag = std::sqrt(update.x * update.x + update.y * update.y +
                             update.z * update.z);
    if (u_mag > 0.2) {
      update.x *= 0.2 / u_mag;
      update.y *= 0.2 / u_mag;
      update.z *= 0.2 / u_mag;
    }

    x += update.x;
    y += update.y;
    z += update.z;
    yaw += update.w;

    if (u_mag < 1e-4)
      break;
  }

  // Update state
  state.p = {x, y, z};
  Mat3x3 R_yaw = {(float)std::cos(yaw),
                  (float)-std::sin(yaw),
                  0,
                  (float)std::sin(yaw),
                  (float)std::cos(yaw),
                  0,
                  0,
                  0,
                  1};
  state.q = qNormalize(mat33ToQuat(mat33Mul(R_yaw, R_wc_init)));
}

void Optimizer::bundleAdjustment(Map &map, int numIterations) {
  (void)map;
  (void)numIterations;
}

void Optimizer::localBundleAdjustment(Map &map, int currentKfId,
                                      int windowSize) {
  (void)map;
  (void)currentKfId;
  (void)windowSize;
}

void Optimizer::optimizeEssentialGraph(Map &map, KeyFrame *pCurrentKF,
                                       KeyFrame *pLoopKF, const Mat4x4 &Tloop) {
  // Tloop is the correction matrix (delta) to be applied to the current KF
  // pose. We distribute this error back along the sequence of keyframes (the
  // parent chain) to smoothly correct the trajectory drift.

  std::vector<int> path;
  int currId = pCurrentKF->id;

  // Safety break for cycles or very long paths
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

    // Extract translation and rotation components from Tloop for interpolation
    Vec3 t_err = {Tloop.m[3], Tloop.m[7], Tloop.m[11]};

    // We propagate the correction. Simple linear interpolation of translation
    // is often sufficient for small local drifts, but for SE(3) we should
    // ideally interpolate in Lie algebra. For ORB-Lite, we'll do a simplified
    // linear distribution.

    for (size_t i = 0; i < path.size(); i++) {
      int kfId = path[i];
      KeyFrame &kf = map.keyframes[kfId];

      // Weight decreases as we get further from the current KF (closer to loop
      // KF)
      double weight = (double)(path.size() - i) / (double)path.size();

      // Apply weighted translation correction
      kf.pose.m[3] += t_err.x * weight;
      kf.pose.m[7] += t_err.y * weight;
      kf.pose.m[11] += t_err.z * weight;

      // For rotation, we should ideally use slerp, but for small drifts
      // the translation is the dominant error component in VIO.
    }
  } else {
    std::cout << "PGO: Could not find path to Loop KF. Applying local "
                 "correction only."
              << std::endl;
    pCurrentKF->pose = mat44Mul(Tloop, pCurrentKF->pose);
  }
}

} // namespace orb_lite
