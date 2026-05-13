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

    double speed = norm(state.v);
    float dynamic_radius = 50.0f + (float)(speed * 30.0); // Add 30px per m/s
    if (dynamic_radius > 150.0f) dynamic_radius = 150.0f;

    const float radius = is_initialized ? dynamic_radius : 150.0f;
  const int grid_radius = (int)(radius / 10.0) + 1;

  for (size_t j = 0; j < map->points.size(); j++) {
    const auto &mp = map->points[j];
    if (mp.isBad) continue; // BUG FIX: Don't track against culled points!
    
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

    state.v.x *= 0.5;
    state.v.y *= 0.5;
    state.v.z *= 0.5;
    consecutive_lost_frames++;
    if (relocalize(descriptors, kps)) {
      tracking_state = OK;
      consecutive_lost_frames = 0;
      std::cout << "Tracking: RELOCALIZED!" << std::endl;
      
      // --- BUG FIX 3: Reset the frame timer and ABORT the rest of the loop
      // This prevents the velocity Complementary Filter from seeing a 60m/s spike!
      last_frame_time = timestamp; 
      
      // Update last_T_world_cam to the ACTUAL Relocalized Pose, not some old KeyFrame!
      Mat3x3 R_reloc = qToMat33(state.q);
      for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) last_T_world_cam.at(i, j) = R_reloc.at(i, j);
          last_T_world_cam.at(i, 3) = state.p[i];
      }
      
      return true; 
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
    // STATE 0: Capture Frame A
    if (last_kps.empty()) {
        last_kps = kps;
        last_descriptors = descriptors;
        
        Mat3x3 R = qToMat33(state.q);
        last_T_world_cam = Mat4x4::identity();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) last_T_world_cam.at(i, j) = R.at(i, j);
            last_T_world_cam.at(i, 3) = state.p[i];
        }
        std::cout << "Init: Locked Frame A." << std::endl;
        return false;
    }

    // STATE 1: Match against Frame A
    std::vector<std::pair<int, int>> matches;
    matchDescriptorsGrid(last_kps, last_descriptors, kps, descriptors, matches, 100.0f);

    Vec3 p_start = {last_T_world_cam.m[3], last_T_world_cam.m[7], last_T_world_cam.m[11]};
    double d_baseline = std::sqrt(std::pow(state.p.x - p_start.x, 2) +
                                  std::pow(state.p.y - p_start.y, 2) +
                                  std::pow(state.p.z - p_start.z, 2));

    // GATE 1: Did we look away completely? (Lost the scene)
    if (matches.size() < 15) {
        std::cout << "Init: Scene lost. Resetting Frame A." << std::endl;
        last_kps.clear(); // Forces a new Frame A on the next frame
        return false;
    }

    // GATE 2: Wait for baseline (DO NOT OVERWRITE Frame A here!)
    if (d_baseline < 0.05) {
        if (frame_count % 15 == 0) {
            std::cout << "Init: Waiting for baseline... " << d_baseline << "m" << std::endl;
        }
        return false; 
    }

    // STATE 2: We have baseline and matches. Attempt Triangulation!
    Mat4x4 T_curr = Mat4x4::identity();
    Mat3x3 R = qToMat33(state.q);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) T_curr.at(i, j) = R.at(i, j);
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
        
        if (pw.x == 0 && pw.y == 0 && pw.z == 0) continue;

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

    // GATE 3: Verify the map is structurally sound
    if (valid_count >= 20) {
        is_initialized = true;
        tracking_state = OK;
        std::cout << "Tracking: INITIALIZED! Map=" << map->points.size()
                  << " baseline=" << d_baseline << "m" << std::endl;
    } else {
        std::cout << "Init: Triangulation failed (bad geometry). Resetting Frame A." << std::endl;
        map->points.clear();
        last_kps.clear(); // Start over
    }
    
    return false; // Still return false on the initialization frame so track() doesn't proceed to optimization yet
} else {
    // TRACKING MODE
    tracked_map_indices.clear();
    searchMapPoints(descriptors, kps, tracked_map_indices);

    if ((int)tracked_map_indices.size() >= 12) {
      int inliers = Optimizer::poseOptimization(tracked_map_indices, kps, cam, state, *map);

      Vec3 p_before_opt = state.p;
      Quaternion q_before_opt = state.q;

      if (inliers >= 10) {
          std::cout << "Tracking: Optimized pos=[" << state.p.x << "," << state.p.y
                    << "," << state.p.z << "] inliers=" << inliers << std::endl;
          tracking_state = OK;
      } else {
          // Gradients diverged due to false matches. Roll back to pure IMU prediction.
          std::cout << "Tracking: Optimizer Diverged (" << inliers << " inliers). Rolling back to IMU." << std::endl;
          state.p = p_before_opt;
          state.q = q_before_opt;
          tracking_state = LOST;
      }
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
          
          // 1. Calculate the raw visual velocity truth
          Vec3 v_visual = {
              (state.p.x - p_before_track.x) / dt_opt,
              (state.p.y - p_before_track.y) / dt_opt,
              (state.p.z - p_before_track.z) / dt_opt
          };

          // 2. How much did the IMU velocity prediction deviate from reality?
          Vec3 v_err = {
              v_visual.x - state.v.x,
              v_visual.y - state.v.y,
              v_visual.z - state.v.z
          };

          // --- CONTINUOUS BIAS ESTIMATION ---
          // Rotate the World-Frame velocity error back into the Camera's Body-Frame
          Mat3x3 R_cw = mat33Transpose(qToMat33(state.q));
          Vec3 a_err_body = {
              R_cw.m[0]*v_err.x + R_cw.m[1]*v_err.y + R_cw.m[2]*v_err.z,
              R_cw.m[3]*v_err.x + R_cw.m[4]*v_err.y + R_cw.m[5]*v_err.z,
              R_cw.m[6]*v_err.x + R_cw.m[7]*v_err.y + R_cw.m[8]*v_err.z
          };

          // Slowly integrate this error back into the hardware bias!
          // This is the "Integral" term of a PID controller.
          const double bias_learning_rate = 0.005; // Tune this: higher = faster learning, but noisier
          state.ba.x -= a_err_body.x * bias_learning_rate;
          state.ba.y -= a_err_body.y * bias_learning_rate;
          state.ba.z -= a_err_body.z * bias_learning_rate;
          // ----------------------------------

          // 3. Keep the actual state velocity bounded
          const double alpha = 0.05; 
          state.v.x = state.v.x * (1.0 - alpha) + v_visual.x * alpha;
          state.v.y = state.v.y * (1.0 - alpha) + v_visual.y * alpha;
          state.v.z = state.v.z * (1.0 - alpha) + v_visual.z * alpha;

          double v_mag = std::sqrt(state.v.x * state.v.x + state.v.y * state.v.y + state.v.z * state.v.z);
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
        
        std::unordered_map<int, int> shared_point_counts;
        for (int mp_id : kf.mapPointIds) {
            if (mp_id < 0 || mp_id >= (int)map->points.size() || map->points[mp_id].isBad) continue;
            for (const auto& other_kf : map->keyframes) {
                for (int other_mp_id : other_kf.mapPointIds) {
                    if (other_mp_id == mp_id) shared_point_counts[other_kf.id]++;
                }
            }
        }
        
        for (const auto& pair : shared_point_counts) {
            if (pair.second >= 15) {
                kf.neighbors.push_back(pair.first);
                map->keyframes[pair.first].neighbors.push_back(kf.id); 
            }
        }
        
        // Ensure parent is always a neighbor to keep spanning tree connected
        if (std::find(kf.neighbors.begin(), kf.neighbors.end(), kf.parentId) == kf.neighbors.end()) {
            kf.neighbors.push_back(kf.parentId);
            map->keyframes[kf.parentId].neighbors.push_back(kf.id);
        }
      }

      if(new_pts > 10 || tracked_map_indices.size() >= 20){

      

        map->addKeyframe(kf);
        if (kf_db)
          kf_db->add(&map->keyframes.back());

        Optimizer::localBundleAdjustment(*map, kf.id, 5);

        // Loop Detection (every 5 KeyFrames)
        if (loop_closer && map->keyframes.size() % 5 == 0) {
          std::cout << "Tracking: Running Loop Detection..." << std::endl;
          KeyFrame *pLoopKF = nullptr;
          if (loop_closer->detectLoop(&map->keyframes.back(), pLoopKF)) {
            std::cout << "Tracking: Loop detected with KeyFrame " << pLoopKF->id
                      << "! Correcting..." << std::endl;
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
        else{
          std::cout << "Tracking: Skipping Loop Detection for KeyFrame " << kf.id
                    << std::endl;
        }

        std::cout << "Tracking: KeyFrame inserted. Matches="
                  << frame_matches.size() << " Triangulated=" << new_pts
                  << " Total Map=" << map->points.size() << std::endl;
        last_T_world_cam = T_curr;
        last_kps = kps;
        last_descriptors = descriptors;

      }
      else{
        std::cout << "Tracking: No KeyFrame inserted. Dist from last KF="
                  << dist_from_last_kf << "m Matches=" << tracked_map_indices.size()
                  << std::endl;
      }
    }
  }

  last_frame_time = timestamp;
  return (tracking_state == OK);
}

bool Tracking::relocalize(const std::vector<Descriptor> &descriptors,
                          const std::vector<KeyPoint> &kps) {
    if (!voc || !kf_db) return false;

    auto candidates = kf_db->detectRelocalizationCandidates(current_bow);
    if (candidates.empty()) return false;

    for (KeyFrame *pKF : candidates) {
        std::vector<std::pair<int, int>> bow_matches;
        SearchByBoW(pKF, descriptors, current_feat_vec, bow_matches);

        if (bow_matches.size() >= 15) {
            
            // --- BUG FIX 1: Translate 2D indices to 3D Global MapPoint IDs ---
            std::vector<std::pair<int, int>> opt_matches;
            for (const auto& match : bow_matches) {
                int idx_candidate = match.first;
                int idx_current = match.second;
                
                // Get the actual 3D point the candidate KeyFrame was looking at
                int global_mp_id = pKF->mapPointIds[idx_candidate];
                
                // Ensure the point is valid and exists in the candidate's map
                if (global_mp_id >= 0 && !pKF->pMap->points[global_mp_id].isBad) {
                    opt_matches.push_back({idx_current, global_mp_id});
                }
            }

            // Store the state in the current map before optimizing against target map
            NavState state_in_current_map = state;

            // Pre-emptively reset the state
            Mat4x4 T_kf = pKF->pose;
            Mat3x3 R_kf;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) R_kf.m[i * 3 + j] = T_kf.at(i, j);
            }
            state.q = mat33ToQuat(R_kf);
            state.p = {T_kf.at(0, 3), T_kf.at(1, 3), T_kf.at(2, 3)};
            state.v = {0, 0, 0}; 

            // --- BUG FIX 2: Optimize against the CANDIDATE's map, not the lost map ---
            Optimizer::poseOptimization(opt_matches, kps, cam, state, *pKF->pMap);

            if (opt_matches.size() >= 10) {
                if (pKF->pMap != this->map) {
                    std::cout << "Relocalize: Triggering Merge." << std::endl;
                    
                    // We need T_Wt_Wc.
                    // state has the pose in the Target Map (T_Wt_c).
                    // state_in_current_map has the pose in the Current Map (T_Wc_c).
                    // T_Wt_Wc = T_Wt_c * (T_Wc_c)^-1
                    
                    Mat3x3 R_Wt_c = qToMat33(state.q);
                    Vec3   t_Wt_c = state.p; // Camera center in target world
                    
                    Mat3x3 R_Wc_c = qToMat33(state_in_current_map.q);
                    Vec3   t_Wc_c = state_in_current_map.p; // Camera center in current world
                    
                    // The camera pose matrix T is [R^T, -R^T * C]
                    // Wait, our state.p is the camera center C. 
                    // Let's construct a 4x4 matrix representing the transform from local to world.
                    // T_Wt_Wc means converting a point in Wc to Wt.
                    // P_Wt = T_Wt_Wc * P_Wc
                    // We know C_Wt = T_Wt_Wc * C_Wc.
                    // And we know R_c_Wt = R_c_Wc * R_Wc_Wt.
                    // So R_Wt_Wc = R_Wt_c * (R_Wc_c)^T = R_Wt_c * mat33Transpose(R_Wc_c).
                    // Wait, usually R in state is R_wc (world to camera), wait!
                    // In Optimizer::poseOptimization math:
                    // Mat3x3 R_wc_init = qToMat33(state.q); -> R_wc means World to Camera? No, looking at Optimizer:
                    // P_c = R_cw * rel_p. where R_cw = R_wc^T. So R_wc is Camera to World!
                    
                    Mat3x3 R_Wt_Wc = mat33Mul(R_Wt_c, mat33Transpose(R_Wc_c));
                    
                    // C_Wt = R_Wt_Wc * C_Wc + t_Wt_Wc
                    // So t_Wt_Wc = C_Wt - R_Wt_Wc * C_Wc
                    Vec3 t_Wt_Wc = {
                        t_Wt_c.x - (R_Wt_Wc.m[0]*t_Wc_c.x + R_Wt_Wc.m[1]*t_Wc_c.y + R_Wt_Wc.m[2]*t_Wc_c.z),
                        t_Wt_c.y - (R_Wt_Wc.m[3]*t_Wc_c.x + R_Wt_Wc.m[4]*t_Wc_c.y + R_Wt_Wc.m[5]*t_Wc_c.z),
                        t_Wt_c.z - (R_Wt_Wc.m[6]*t_Wc_c.x + R_Wt_Wc.m[7]*t_Wc_c.y + R_Wt_Wc.m[8]*t_Wc_c.z)
                    };
                    
                    Mat4x4 T_Wt_Wc = Mat4x4::identity();
                    for(int i=0; i<3; i++) for(int j=0; j<3; j++) T_Wt_Wc.at(i,j) = R_Wt_Wc.at(i,j);
                    T_Wt_Wc.at(0,3) = t_Wt_Wc.x;
                    T_Wt_Wc.at(1,3) = t_Wt_Wc.y;
                    T_Wt_Wc.at(2,3) = t_Wt_Wc.z;
                    
                    atlas->mergeMaps(pKF->pMap, this->map, T_Wt_Wc);
                    this->map = atlas->getCurrentMap(); 
                }
                return true;
            }
        }
    }
    return false;
}

} // namespace orb_lite
