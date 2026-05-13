#include "Tracking.hpp"
#include "BoW.hpp"
#include "LoopClosing.hpp"
#include "Optimizer.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace orb_lite {

// ORB-SLAM3 PredictStateIMU equivalent
// The IMU measures: a_body = R_wb^T * (a_world - g_world) + ba
// At rest: a_body ≈ R_wb^T * (-g_world) + ba
// Preintegration integrates (a_body - ba) in body frame
// Which gives: dP = integral of R_i * (a_i - ba) dt  (body frame)
// World update: p2 = p1 + v1*dt + 0.5*g*dt^2 + R_wb1 * dP
// This cancels gravity IF ba ≈ 0 and we set ba to absorb the measured gravity
// offset
void Tracking::predictPose(const std::vector<ImuSample> &imu,
                           double timestamp) {
  if (last_t < 0) {
    last_t = timestamp;
    return;
  }

  double dt = timestamp - last_t;
  if (dt <= 0 || dt > 1.0) {
    last_t = timestamp;
    return;
  }

  // ORB-SLAM3 PredictStateIMU: preintegrate in body frame
  Preintegrated pre;
  double t_ptr = last_t;
  for (const auto &s : imu) {
    double ds = s.timestamp - t_ptr;
    if (ds > 0 && ds < 0.5) {
      pre.update(s.acc, s.gyro, ds, state.ba, state.bg);
    }
    t_ptr = s.timestamp;
  }

  // World-frame gravity (Z-up convention, same as ORB-SLAM3 Gz)
  const double Gz = -9.81;
  Mat3x3 R_wb1 = qToMat33(state.q);

  // p2 = p1 + v1*dt + 0.5*g*dt^2 + R_wb1 * dP
  double idt = pre.dt > 0 ? pre.dt : dt;
  state.p.x +=
      state.v.x * idt + 0 +
      (R_wb1.m[0] * pre.dP.x + R_wb1.m[1] * pre.dP.y + R_wb1.m[2] * pre.dP.z);
  state.p.y +=
      state.v.y * idt + 0 +
      (R_wb1.m[3] * pre.dP.x + R_wb1.m[4] * pre.dP.y + R_wb1.m[5] * pre.dP.z);
  state.p.z +=
      state.v.z * idt + 0.5 * Gz * idt * idt +
      (R_wb1.m[6] * pre.dP.x + R_wb1.m[7] * pre.dP.y + R_wb1.m[8] * pre.dP.z);

  // v2 = v1 + g*dt + R_wb1 * dV
  state.v.x += 0 + (R_wb1.m[0] * pre.dV.x + R_wb1.m[1] * pre.dV.y +
                    R_wb1.m[2] * pre.dV.z);
  state.v.y += 0 + (R_wb1.m[3] * pre.dV.x + R_wb1.m[4] * pre.dV.y +
                    R_wb1.m[5] * pre.dV.z);
  state.v.z += Gz * idt + (R_wb1.m[6] * pre.dV.x + R_wb1.m[7] * pre.dV.y +
                           R_wb1.m[8] * pre.dV.z);

  // q2 = q1 * dR
  state.q = qNormalize(qMul(state.q, pre.dR));

  // Safety velocity cap (phone motion rarely exceeds 2 m/s)
  double v_mag = std::sqrt(state.v.x * state.v.x + state.v.y * state.v.y +
                           state.v.z * state.v.z);
  if (v_mag > 2.0) {
    state.v.x *= 2.0 / v_mag;
    state.v.y *= 2.0 / v_mag;
    state.v.z *= 2.0 / v_mag;
  }
  
  if (!is_initialized) {
    // Light damping during bootstrapping to prevent runaway, but still allow movement to accumulate baseline
    state.v.x *= 0.95; state.v.y *= 0.95; state.v.z *= 0.95;
  }

  last_t = t_ptr > 0 ? t_ptr : timestamp;
}

void Tracking::searchMapPoints(const std::vector<Descriptor> &descriptors,
                               const std::vector<KeyPoint> &kps,
                               std::vector<std::pair<int, int>> &matches) {
  matches.clear();
  Mat3x3 R_wc = qToMat33(state.q);
  Mat3x3 R_cw = mat33Transpose(R_wc);

  const float radius = is_initialized ? 50.0f : 100.0f;
  const int grid_radius = (int)(radius / 10.0) + 1;

  for (size_t j = 0; j < map->points.size(); j++) {
    const auto &mp = map->points[j];
    Vec3 rel_p = {mp.pos.x - state.p.x, mp.pos.y - state.p.y,
                  mp.pos.z - state.p.z};
    Vec3 P_c = {R_cw.m[0] * rel_p.x + R_cw.m[1] * rel_p.y + R_cw.m[2] * rel_p.z,
                R_cw.m[3] * rel_p.x + R_cw.m[4] * rel_p.y + R_cw.m[5] * rel_p.z,
                R_cw.m[6] * rel_p.x + R_cw.m[7] * rel_p.y +
                    R_cw.m[8] * rel_p.z};
    if (P_c.z < 0.1)
      continue;

    Vec2 uv = cam.project(P_c);
    if (std::isnan(uv.x) || std::isnan(uv.y) || uv.x < 0 || uv.x >= 640 ||
        uv.y < 0 || uv.y >= 480)
      continue;

    int best_dist = 64;
    int best_idx = -1;

    int gx_center = (int)(uv.x / 10.0);
    int gy_center = (int)(uv.y / 10.0);

    for (int gy = gy_center - grid_radius; gy <= gy_center + grid_radius;
         gy++) {
      if (gy < 0 || gy >= GRID_ROWS)
        continue;
      for (int gx = gx_center - grid_radius; gx <= gx_center + grid_radius;
           gx++) {
        if (gx < 0 || gx >= GRID_COLS)
          continue;

        for (int i : feature_grid[gy][gx]) {
          if (std::abs(kps[i].x - uv.x) < radius &&
              std::abs(kps[i].y - uv.y) < radius) {
            int dist;
            calculateHammingDistance(descriptors[i], mp.descriptor, dist);
            if (dist < best_dist) {
              best_dist = dist;
              best_idx = i;
            }
          }
        }
      }
    }

    if (best_idx >= 0) {
      matches.push_back({best_idx, (int)j});
    }
  }
  std::cout << "searchMapPoints: map_size=" << map->points.size()
            << " matches=" << matches.size() << std::endl;
}


bool Tracking::track(const Image &img, const std::vector<ImuSample> &imu,
                     double timestamp) {
  if (!imu_calibrated) {
    if (first_imu_time < 0 && !imu.empty())
      first_imu_time = imu.front().timestamp;
    if (first_imu_time < 0)
      first_imu_time = timestamp;

    imu_calibration_buffer.insert(imu_calibration_buffer.end(), imu.begin(),
                                  imu.end());

    if (timestamp - first_imu_time > 2.0 &&
        imu_calibration_buffer.size() > 50) {
      state.q = alignGravity(imu_calibration_buffer);
      state.p = {0, 0, 0};
      state.v = {0, 0, 0};

      Vec3 mean_gyro = {0, 0, 0};
      Vec3 mean_acc = {0, 0, 0};
      for (const auto &s : imu_calibration_buffer) {
        mean_acc.x += s.acc.x;
        mean_acc.y += s.acc.y;
        mean_acc.z += s.acc.z;
        mean_gyro.x += s.gyro.x;
        mean_gyro.y += s.gyro.y;
        mean_gyro.z += s.gyro.z;
      }
      mean_acc.x /= imu_calibration_buffer.size();
      mean_acc.y /= imu_calibration_buffer.size();
      mean_acc.z /= imu_calibration_buffer.size();

      mean_gyro.x /= imu_calibration_buffer.size();
      mean_gyro.y /= imu_calibration_buffer.size();
      mean_gyro.z /= imu_calibration_buffer.size();
      state.bg = mean_gyro;

      Mat3x3 R_wc = qToMat33(state.q);
      Mat3x3 R_cw = mat33Transpose(R_wc);
      Vec3 g_body = {R_cw.m[0] * 0 + R_cw.m[1] * 0 + R_cw.m[2] * 9.81,
                     R_cw.m[3] * 0 + R_cw.m[4] * 0 + R_cw.m[5] * 9.81,
                     R_cw.m[6] * 0 + R_cw.m[7] * 0 + R_cw.m[8] * 9.81};
      state.ba = {mean_acc.x - g_body.x, mean_acc.y - g_body.y,
                  mean_acc.z - g_body.z};

      std::cout << "Tracking IMU Calibrated over "
                << imu_calibration_buffer.size() << " samples. \n"
                << " ba=[" << state.ba.x << "," << state.ba.y << ","
                << state.ba.z << "]\n"
                << " bg=[" << state.bg.x << "," << state.bg.y << ","
                << state.bg.z << "]" << std::endl;

      imu_calibrated = true;
      last_t = timestamp;
      last_frame_time = timestamp;
    } else {
      std::cout << "Tracking IMU Calibrating... "
                << (timestamp - first_imu_time) << "s" << std::endl;
      return false;
    }
  }

  frame_count++;

  // Save position before IMU integration and optimization for velocity update
  // later
  Vec3 p_before_track = state.p;

  predictPose(imu, timestamp);

  std::vector<KeyPoint> kps;
  std::vector<Descriptor> descriptors;
  detectFAST(img, kps, 20, true);
  computeORB(img, kps, descriptors);
  current_kps = kps; // Save for visualizer testing

  for (int r = 0; r < GRID_ROWS; r++)
    for (int c = 0; c < GRID_COLS; c++)
      feature_grid[r][c].clear();
  for (int i = 0; i < (int)kps.size(); i++) {
    int gx = (int)(kps[i].x / 10.0);
    int gy = (int)(kps[i].y / 10.0);
    if (gx >= 0 && gx < GRID_COLS && gy >= 0 && gy < GRID_ROWS) {
      feature_grid[gy][gx].push_back(i);
    }
  }

  Vec3 p_prev_kf = {last_T_world_cam.m[3], last_T_world_cam.m[7],
                    last_T_world_cam.m[11]};
  double dist_from_last_kf = std::sqrt(std::pow(state.p.x - p_prev_kf.x, 2) +
                                       std::pow(state.p.y - p_prev_kf.y, 2) +
                                       std::pow(state.p.z - p_prev_kf.z, 2));

  // Conditional BoW computation (only for relocalization or keyframe insertion)
  bool is_lost = (tracking_state == LOST);
  bool potentially_kf = (dist_from_last_kf > 0.04) || (frame_count % 5 == 0);

  if (voc && (is_lost || potentially_kf)) {
    voc->transform(descriptors, current_bow, current_feat_vec, 4);
  }

  if (tracking_state == LOST) {
    consecutive_lost_frames++;
    if (relocalize(descriptors, kps)) {
      tracking_state = OK; /* NOT REALLY OK YET */
      consecutive_lost_frames = 0;
      std::cout << "Tracking: RELOCALIZED!" << std::endl;
    } else {
      if (consecutive_lost_frames > 30) {
          std::cout << "Tracking: Lost for " << consecutive_lost_frames << " frames. Starting new map." << std::endl;
          atlas->createNewMap();
          map = atlas->getCurrentMap();
          is_initialized = false;
          tracking_state = NOT_INITIALIZED;
          consecutive_lost_frames = 0;
          
          // Reset bootstrapping state
          last_kps.clear();
          last_descriptors.clear();
          tracked_map_indices.clear();
          
          // Hard reset for the new map's local origin
          state.p = {0,0,0};
          state.v = {0,0,0};
      }
      return false;
    }
  }

  if (!is_initialized) {
    if (last_kps.empty()) {
      last_kps = kps;
      last_descriptors = descriptors;
      Mat3x3 R = qToMat33(state.q);
      last_T_world_cam = Mat4x4::identity();
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++)
          last_T_world_cam.at(i, j) = R.at(i, j);
        last_T_world_cam.at(i, 3) = state.p[i];
      }
      return false;
    }

    // Try initialization every frame (grid matcher is now fast enough)
    if (true) {
      std::vector<std::pair<int, int>> matches;
      matchDescriptorsGrid(last_kps, last_descriptors, kps, descriptors, matches, 100.0f);

      Vec3 p_start = {last_T_world_cam.m[3], last_T_world_cam.m[7],
                      last_T_world_cam.m[11]};
      // Do not arbitrarily set tracking_state = OK here if not initialized yet

      double d_baseline = std::sqrt(std::pow(state.p.x - p_start.x, 2) +
                                    std::pow(state.p.y - p_start.y, 2) +
                                    std::pow(state.p.z - p_start.z, 2));

      if (frame_count % 10 == 0) {
          std::cout << "Tracking: Initializing... baseline=" << d_baseline << "m matches=" << matches.size() << std::endl;
      }

      if ((int)matches.size() > 20 && d_baseline > 0.005) {
        Mat4x4 T_curr = Mat4x4::identity();
        Mat3x3 R = qToMat33(state.q);
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++)
            T_curr.at(i, j) = R.at(i, j);
          T_curr.at(i, 3) = state.p[i];
        }
        Mat4x4 T1w, T2w;
        invert4x4(last_T_world_cam, T1w);
        invert4x4(T_curr, T2w);

        int valid_count = 0;
        for (const auto &m : matches) {
          Vec3 p1u = cam.unproject(last_kps[m.first].x, last_kps[m.first].y);
          Vec3 p2u = cam.unproject(kps[m.second].x, kps[m.second].y);
          Vec3 pw = triangulate(p1u, p2u, T1w, T2w);
          if (pw.x == 0 && pw.y == 0 && pw.z == 0)
            continue;

          // Check depth in current camera frame
          Vec3 pc = {
              T2w.m[0] * pw.x + T2w.m[1] * pw.y + T2w.m[2] * pw.z + T2w.m[3],
              T2w.m[4] * pw.x + T2w.m[5] * pw.y + T2w.m[6] * pw.z + T2w.m[7],
              T2w.m[8] * pw.x + T2w.m[9] * pw.y + T2w.m[10] * pw.z + T2w.m[11]};
          if (pc.z >= 0.1 && pc.z <= 15.0) {
            MapPoint mp_pt;
            mp_pt.pos = pw;
            mp_pt.descriptor = descriptors[m.second];
            mp_pt.pMap = map;
            map->points.push_back(mp_pt);
            valid_count++;
          }
        }

        if (valid_count > 10) {
          is_initialized = true;
          tracking_state = OK;
          std::cout << "Tracking: INITIALIZED! Map=" << map->points.size()
                    << " baseline=" << d_baseline << "m" << std::endl;
        } else {
          map->points.clear();
        }
      }

      // Update the reference frame if we've lost visibility or rotated too much without translating
      if (!is_initialized && (matches.size() < 40)) {
        last_kps = kps;
        last_descriptors = descriptors;
        
        Mat3x3 R = qToMat33(state.q);
        last_T_world_cam = Mat4x4::identity();
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            last_T_world_cam.at(i, j) = R.at(i, j);
          }
          last_T_world_cam.at(i, 3) = state.p[i];
        }
      }
    }

  } else {
    // TRACKING MODE
    tracked_map_indices.clear();
    searchMapPoints(descriptors, kps, tracked_map_indices);

    if ((int)tracked_map_indices.size() >= 12) {
      Optimizer::poseOptimization(tracked_map_indices, kps, cam, state, *map);
      std::cout << "Tracking: Optimized pos=[" << state.p.x << "," << state.p.y
                << "," << state.p.z
                << "] matches=" << tracked_map_indices.size() << std::endl;
      tracking_state = OK; /* NOT REALLY OK YET */
    } else {
      std::cout << "Tracking weak: " << tracked_map_indices.size()
                << " matches pos=[" << state.p.x << "," << state.p.y << ","
                << state.p.z << "]" << std::endl;
      if (tracked_map_indices.size() < 8) {
        tracking_state = LOST;
        std::cout << "Tracking LOST!" << std::endl;
      }
    }

    // Tightly couple back the velocity estimation from the optimizers new
    // position!
    if (last_frame_time > 0) {
      double dt_opt = timestamp - last_frame_time;
      if (dt_opt > 0 && dt_opt < 1.0) {
        state.v.x = (state.p.x - p_before_track.x) / dt_opt;
        state.v.y = (state.p.y - p_before_track.y) / dt_opt;
        state.v.z = (state.p.z - p_before_track.z) / dt_opt;

        // Keep bounded
        double v_mag = std::sqrt(state.v.x * state.v.x + state.v.y * state.v.y +
                                 state.v.z * state.v.z);
        if (v_mag > 2.0) {
          state.v.x *= 2.0 / v_mag;
          state.v.y *= 2.0 / v_mag;
          state.v.z *= 2.0 / v_mag;
        }
      }
    }

    if (dist_from_last_kf > 0.1 ||
        (frame_count % 10 == 0 && (int)tracked_map_indices.size() < 100)) {
      std::vector<std::pair<int, int>> frame_matches;
      matchDescriptorsGrid(last_kps, last_descriptors, kps, descriptors, frame_matches, 50.0f);

      Mat4x4 T_curr = Mat4x4::identity();
      Mat3x3 R = qToMat33(state.q);
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++)
          T_curr.at(i, j) = R.at(i, j);
        T_curr.at(i, 3) = state.p[i];
      }
      Mat4x4 T1w, T2w;
      invert4x4(last_T_world_cam, T1w);
      invert4x4(T_curr, T2w);

      // Save KeyFrame for trajectory mapping
      KeyFrame kf;
      kf.id = map->keyframes.size();
      kf.pose = T_curr;
      kf.kps = kps;
      kf.descriptors = descriptors;
      kf.bowVec = current_bow;
      kf.featVec = current_feat_vec;

      // Fill mapPointIds from tracked_map_indices
      kf.mapPointIds.assign(kps.size(), -1);
      for (auto &m : tracked_map_indices) {
        kf.mapPointIds[m.first] = m.second;
      }

      int new_pts = 0;
      for (const auto &m : frame_matches) {
        Vec3 p1u = cam.unproject(last_kps[m.first].x, last_kps[m.first].y);
        Vec3 p2u = cam.unproject(kps[m.second].x, kps[m.second].y);
        Vec3 pw = triangulate(p1u, p2u, T1w, T2w);
        if (pw.x == 0 && pw.y == 0 && pw.z == 0)
          continue;

        Vec3 pc = {
            T2w.m[0] * pw.x + T2w.m[1] * pw.y + T2w.m[2] * pw.z + T2w.m[3],
            T2w.m[4] * pw.x + T2w.m[5] * pw.y + T2w.m[6] * pw.z + T2w.m[7],
            T2w.m[8] * pw.x + T2w.m[9] * pw.y + T2w.m[10] * pw.z + T2w.m[11]};
        // User notes item bounds can be expanded now! Extending to 30 meters to
        // collect more of the scene.
        if (pc.z >= 0.20 && pc.z <= 30.0 && (int)map->points.size() < 100000) {
          int pt_id = map->addPoint(pw, descriptors[m.second], kf.id);
          kf.mapPointIds[m.second] = pt_id;
          new_pts++;
        }
      }

      if (!map->keyframes.empty()) {
        kf.parentId = map->keyframes.back().id;
        kf.neighbors.push_back(kf.parentId);
        map->keyframes.back().neighbors.push_back(kf.id);
      }

      map->addKeyframe(kf);
      if (kf_db)
        kf_db->add(&map->keyframes.back());

      // Loop Detection (every 5 KeyFrames)
      if (loop_closer && map->keyframes.size() % 5 == 0) {
        KeyFrame *pLoopKF = nullptr;
        if (loop_closer->detectLoop(&map->keyframes.back(), pLoopKF)) {
          loop_closer->correctLoop(&map->keyframes.back(), pLoopKF);
          // Update current state to the corrected pose
          state.p = {pLoopKF->pose.at(0, 3), pLoopKF->pose.at(1, 3),
                     pLoopKF->pose.at(2, 3)};

          Mat3x3 R_loop;
          for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
              R_loop.m[i * 3 + j] = pLoopKF->pose.at(i, j);
            }
          }
                    state.q = mat33ToQuat(R_loop);
          state.v = {
              0, 0,
              0}; // Reset velocity on loop closure to prevent runaway drift
        }
      }

      std::cout << "Tracking: KeyFrame inserted. Matches="
                << frame_matches.size() << " Triangulated=" << new_pts
                << " Total Map=" << map->points.size() << std::endl;
      last_T_world_cam = T_curr;
      last_kps = kps;
      last_descriptors = descriptors;
    }
  }

  last_frame_time = timestamp;
  return (tracking_state == OK);
}

bool Tracking::relocalize(const std::vector<Descriptor> &descriptors,
                          const std::vector<KeyPoint> &kps) {
  if (!voc || !kf_db)
    return false;

  auto candidates = kf_db->detectRelocalizationCandidates(current_bow);
  if (candidates.empty()) {
    std::cout << "Relocalize: No candidates found or BoW mismatch" << std::endl;
    return false;
  }

  for (KeyFrame *pKF : candidates) {
    std::vector<std::pair<int, int>> matches;
    SearchByBoW(pKF, descriptors, current_feat_vec, matches);

    std::cout << "Relocalize candidate " << pKF->id
              << " matches=" << matches.size() << std::endl;
    if (matches.size() >= 15) {
      // Initial pose from KeyFrame
      Mat4x4 T_kf = pKF->pose;

      // Pre-emptively reset the current drifting state to the KeyFrame's state
      // to give the optimizer a solid starting guess. Since kf.pose built from
      // T_curr in track(): T_curr.at(i,j) = R.at(i,j); and T_curr.at(i,3) =
      // state.p[i];
      Mat3x3 R_kf;
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          R_kf.m[i * 3 + j] = T_kf.at(i, j);
        }
      }
      state.q = mat33ToQuat(R_kf);
      state.p = {T_kf.at(0, 3), T_kf.at(1, 3), T_kf.at(2, 3)};
      state.v = {0, 0, 0}; // Reset velocity on relocalization

      // Simplified: just try to optimize from there
      // Note: In real ORB-SLAM3, this uses EPnP first.
      Optimizer::poseOptimization(matches, kps, cam, state, *map);

      if (matches.size() >= 10) {
        return true;
      }
    }
  }

  return false;
}

} // namespace orb_lite
