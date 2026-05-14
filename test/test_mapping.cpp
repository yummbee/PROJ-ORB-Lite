#include "DataLoader.hpp"
#include "Vision.hpp"
#include "IMU.hpp"
#include "Map.hpp"
#include "Atlas.hpp"
#include "Tracking.hpp"
#include "BoW.hpp"
#include "KeyFrameDatabase.hpp"
#include "LoopClosing.hpp"
#include "Optimizer.hpp"
#include <iostream>
#include <cmath>

#ifdef WITH_OPENCV
#include <opencv2/opencv.hpp>
#include <filesystem>
namespace fs = std::filesystem;
#endif

void drawMap(const orb_lite::Atlas& atlas, const orb_lite::NavState& state, int width, int height) {
#ifdef WITH_OPENCV
    cv::Mat mapVis = cv::Mat::zeros(height, width, CV_8UC3);
    
    // Virtual Camera for the view (could match tracking camera)
    orb_lite::Camera vcam = {width * 1.0, width * 1.0, width / 2.0, height / 2.0};
    
    orb_lite::Mat3x3 R_wc = orb_lite::qToMat33(state.q);
    orb_lite::Mat3x3 R_cw = orb_lite::mat33Transpose(R_wc);

    auto maps = const_cast<orb_lite::Atlas&>(atlas).getAllMaps();
    for (auto m : maps) {
        bool isCurrent = (m == const_cast<orb_lite::Atlas&>(atlas).getCurrentMap());
        
        for (const auto& p : m->points) {
            if (p.isBad) continue;
            
            orb_lite::Vec3 rel_p = {p.pos.x - state.p.x, p.pos.y - state.p.y, p.pos.z - state.p.z};
            orb_lite::Vec3 p_c = {
                R_cw.m[0] * rel_p.x + R_cw.m[1] * rel_p.y + R_cw.m[2] * rel_p.z,
                R_cw.m[3] * rel_p.x + R_cw.m[4] * rel_p.y + R_cw.m[5] * rel_p.z,
                R_cw.m[6] * rel_p.x + R_cw.m[7] * rel_p.y + R_cw.m[8] * rel_p.z};

            if (p_c.z > 0.1 && p_c.z < 50.0) {
                orb_lite::Vec2 uv = vcam.project(p_c);
                int ix = (int)uv.x;
                int iy = (int)uv.y;

                if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                    int b = (int)(255 * (1.0 - std::min(1.0, p_c.z / 30.0)));
                    cv::Scalar color = isCurrent ? cv::Scalar(0, b, 0) : cv::Scalar(b, b, b);
                    cv::circle(mapVis, cv::Point(ix, iy), 1, color, -1);
                }
            }
        }
    }

    cv::putText(mapVis, "VIRTUAL WORLD VIEW (Depth-Colored)", cv::Point(20, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);

    // --- ORIENTATION WIDGET ---
    int wSize = 40;
    cv::Point wCenter(width - 60, height - 60);
    orb_lite::Vec3 axes[3] = {{1,0,0}, {0,1,0}, {0,0,1}}; // X=Red, Y=Green, Z=Blue
    cv::Scalar aCols[3] = {cv::Scalar(0,0,255), cv::Scalar(0,255,0), cv::Scalar(255,0,0)};
    for (int i = 0; i < 3; i++) {
        orb_lite::Vec3 a_c = {
            R_cw.m[0]*axes[i].x + R_cw.m[1]*axes[i].y + R_cw.m[2]*axes[i].z,
            R_cw.m[3]*axes[i].x + R_cw.m[4]*axes[i].y + R_cw.m[5]*axes[i].z,
            R_cw.m[6]*axes[i].x + R_cw.m[7]*axes[i].y + R_cw.m[8]*axes[i].z};
        cv::Point aEnd(wCenter.x + (int)(a_c.x * wSize), wCenter.y + (int)(a_c.y * wSize));
        cv::line(mapVis, wCenter, aEnd, aCols[i], 2);
    }
    cv::circle(mapVis, wCenter, 3, cv::Scalar(200, 200, 200), -1);

    cv::imshow("ORB-Lite Atlas View", mapVis);
#endif
}

void runSyntheticOptimizerTest() {
    std::cout << "\n============================================\n";
    std::cout << "--- RUNNING SYNTHETIC OPTIMIZER TEST ---\n";
    
    // 1. Setup Perfect Virtual Camera (Standard VGA Intrinsics)
    orb_lite::Camera cam; 
    cam.fx = 500.0; cam.fy = 500.0; 
    cam.cx = 320.0; cam.cy = 240.0;

    orb_lite::Map map;
    std::vector<orb_lite::KeyPoint> kps;
    std::vector<std::pair<int, int>> matches;

    // 2. Create a perfect 1x1 meter square exactly 2 meters in front of the lens
    std::vector<orb_lite::Vec3> true_3d_points = {
        { 0.5,  0.5, 2.0}, // Bottom Right
        {-0.5,  0.5, 2.0}, // Bottom Left
        {-0.5, -0.5, 2.0}, // Top Left
        { 0.5, -0.5, 2.0}  // Top Right
    };

    for(int i = 0; i < 4; i++) {
        orb_lite::MapPoint mp; 
        mp.pos = true_3d_points[i];
        mp.isBad = false;
        map.points.push_back(mp);

        // Project the 3D points perfectly using the 0,0,0 origin pose
        orb_lite::KeyPoint kp;
        kp.x = cam.fx * (true_3d_points[i].x / true_3d_points[i].z) + cam.cx;
        kp.y = cam.fy * (true_3d_points[i].y / true_3d_points[i].z) + cam.cy;
        kp.octave = 0; // Highest resolution pyramid level
        kps.push_back(kp);

        // Lock a perfect 1-to-1 association
        matches.push_back({i, i}); 
    }

    // 3. Corrupt the Initial State (Simulate tracking drift)
    orb_lite::NavState state;
    
    // We intentionally tell the optimizer the camera is 20cm to the right (X=0.2)
    state.p = {0.2, 0.0, 0.0}; 
    state.v = {0,0,0};
    
    // We intentionally tell the optimizer the camera is yawed 5 degrees to the left
    double yaw_error = 5.0 * 3.1415926535 / 180.0;
    
    // Using your Z-axis rotation matrix (Wait, user says Y-axis in comments but code is Z-up)
    // Actually the user provided a Z-axis rotation in the snippet
    orb_lite::Mat3x3 R_yaw_err = {
        (float)std::cos(yaw_error), (float)-std::sin(yaw_error), 0,
        (float)std::sin(yaw_error), (float)std::cos(yaw_error), 0,
        0, 0, 1
    };
    state.q = orb_lite::mat33ToQuat(R_yaw_err);

    std::cout << "[START] Camera Pose   -> X: " << state.p.x << "m | Yaw: " << (yaw_error * 180.0 / 3.1415926535) << " deg\n";
    std::cout << "[GOAL]  Expected Pose -> X: 0.0m | Yaw: 0.0 deg\n\n";

    // 4. Run the Optimizer
    orb_lite::FeatureExtractor extractor; // Need extractor for sigma values
    std::cout << "Running pose optimization...\n";
    int inliers = orb_lite::Optimizer::poseOptimization(matches, kps, cam, state, map, extractor);

    // 5. Evaluate the Output
    orb_lite::Mat3x3 final_R = orb_lite::qToMat33(state.q);
    // Yaw for Z-up is atan2(m[1], m[0])
    double final_yaw = std::atan2(final_R.m[3], final_R.m[0]) * 180.0 / 3.1415926535;

    std::cout << "[RESULT] Camera Pose  -> X: " << state.p.x << "m | Yaw: " << final_yaw << " deg\n";
    std::cout << "[RESULT] Inliers kept -> " << inliers << " / 4\n";

    if (std::abs(state.p.x) < 0.01 && std::abs(final_yaw) < 0.5) {
        std::cout << ">>>> MATH STATUS: PERFECT. Jacobians and signs are 100% correct.\n";
    } else {
        std::cout << ">>>> MATH STATUS: FAILED. The optimizer is mathematically broken.\n";
    }
    std::cout << "============================================\n\n";
    
    // Pause execution so you can read the terminal
    exit(0); 
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: ./test_mapping <recording_dir> [voc_file]" << std::endl;
        std::cout << "   or: ./test_mapping latest [voc_file]" << std::endl;
        return 1;
    }

    std::string recordingDir = argv[1];
    if (recordingDir == "latest") {
        std::string dataPath = "../data";
        std::string latest = "";
        for (const auto & entry : fs::directory_iterator(dataPath)) {
            if (entry.is_directory()) {
                std::string name = entry.path().filename().string();
                if (name > latest) latest = name;
            }
        }
        if (!latest.empty()) {
            recordingDir = dataPath + "/" + latest + "/";
            std::cout << "-> Auto-selected latest: " << recordingDir << std::endl;
        } else {
            std::cout << "Error: No directories found in " << dataPath << std::endl;
            return 1;
        }
    }
    std::string vocFile = (argc > 2) ? argv[2] : "../Vocabulary/ORBvoc.txt";
    
    orb_lite::DataLoader loader(recordingDir);
    
    orb_lite::Vocabulary voc;
    std::cout << "Loading Vocabulary from " << vocFile << "..." << std::endl;
    
    std::string binFile = vocFile.substr(0, vocFile.find_last_of('.')) + ".bin";
    if (voc.loadFromBinaryFile(binFile)) {
        std::cout << "Success: Loaded binary vocabulary cache." << std::endl;
    } else if (voc.loadFromTextFile(vocFile)) {
        std::cout << "Success: Loaded text vocabulary. Saving binary cache for next time..." << std::endl;
        voc.saveToBinaryFile(binFile);
    } else {
        std::cout << "Error: Could not load vocabulary file!" << std::endl;
    }

    orb_lite::Atlas atlas;
    orb_lite::KeyFrameDatabase kf_db(voc);
    orb_lite::LoopClosing loop_closer(&atlas, &kf_db, &voc);

    orb_lite::Tracking tracker;
    tracker.atlas = &atlas;
    tracker.map = atlas.getCurrentMap();
    tracker.voc = &voc;
    tracker.kf_db = &kf_db;
    tracker.loop_closer = &loop_closer;
    tracker.cam = {700.0, 700.0, 320.0, 240.0}; 

    orb_lite::FrameData frame;
    std::vector<orb_lite::ImuSample> imu;

    int frameCount = 0;
    std::cout << "Starting Atlas SLAM Test: " << recordingDir << std::endl;

    while (loader.loadNext(frame, imu)) {
        orb_lite::Image img = {frame.imageData.data(), frame.width, frame.height, frame.width};
        
        tracker.track(img, imu, frame.timestamp);

#ifdef WITH_OPENCV
        cv::Mat vis;
        cv::Mat gray(frame.height, frame.width, CV_8UC1, frame.imageData.data());
        cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);

        for (const auto& kp : tracker.current_kps) {
            cv::circle(vis, cv::Point((int)kp.x, (int)kp.y), 1, cv::Scalar(0, 100, 0), -1);
        }

        cv::putText(vis, "Frame: " + std::to_string(frameCount), cv::Point(20, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        
        std::string statsStr = "Points: " + std::to_string(tracker.map->points.size()) + 
                               " Tracked: " + std::to_string(tracker.tracked_map_indices.size());
        cv::putText(vis, statsStr, cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);

        std::string stateStr = "INITIALIZING...";
        cv::Scalar stateCol(0, 255, 255);

        if (tracker.tracking_state == orb_lite::OK) {
            stateStr = "TRACKING OK";
            stateCol = cv::Scalar(0, 255, 0);
        } else if (tracker.tracking_state == orb_lite::LOST) {
            stateStr = "TRACKING LOST - RELOCALIZING...";
            stateCol = cv::Scalar(0, 0, 255);
        }

        cv::putText(vis, stateStr, cv::Point(20, 90), cv::FONT_HERSHEY_SIMPLEX, 0.5, stateCol, 1);

        // --- 3D ORIENTATION CUBE ---
        int cSize = 25;
        cv::Point cCenter(vis.cols - 40, 40);
        orb_lite::Vec3 cube[8] = {{-1,-1,-1}, {1,-1,-1}, {1,1,-1}, {-1,1,-1}, {-1,-1,1}, {1,-1,1}, {1,1,1}, {-1,1,1}};
        cv::Point p2d[8];
        orb_lite::Mat3x3 R_vis = orb_lite::qToMat33(tracker.state.q);
        for(int i=0; i<8; i++) {
            orb_lite::Vec3 p_r = {
                R_vis.m[0]*cube[i].x + R_vis.m[1]*cube[i].y + R_vis.m[2]*cube[i].z,
                R_vis.m[3]*cube[i].x + R_vis.m[4]*cube[i].y + R_vis.m[5]*cube[i].z,
                R_vis.m[6]*cube[i].x + R_vis.m[7]*cube[i].y + R_vis.m[8]*cube[i].z};
            p2d[i] = cv::Point(cCenter.x + (int)(p_r.x * cSize), cCenter.y + (int)(p_r.y * cSize));
        }
        int e[12][2] = {{0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7}};
        for(int i=0; i<12; i++) cv::line(vis, p2d[e[i][0]], p2d[e[i][1]], cv::Scalar(0, 255, 255), 1);
        cv::circle(vis, p2d[1], 2, cv::Scalar(0, 0, 255), -1); // Mark X+ axis corner

        cv::imshow("ORB-Lite Camera View", vis);
        drawMap(atlas, tracker.state, 600, 600);
        
        // 1. Create a color copy of the current grayscale frame using the existing vis image
        cv::Mat telemetry_img = vis.clone();

        orb_lite::Mat3x3 R_wc = orb_lite::qToMat33(tracker.state.q);
        orb_lite::Mat3x3 R_cw = orb_lite::mat33Transpose(R_wc);

        // 2. Draw every match the Optimizer just used
        for (const auto& match : tracker.tracked_map_indices) { // Or whatever array holds your final inliers
            int idx_2d = match.first;
            int idx_3d = match.second;

            // A. The Ground Truth Observation (Green Dot)
            cv::Point2f pt_2d(tracker.current_kps[idx_2d].x, tracker.current_kps[idx_2d].y);

            // B. Where the Map / IMU predicts it should be (Red Circle)
            if (idx_3d < 0 || idx_3d >= tracker.map->points.size()) continue;
            const auto& mp = tracker.map->points.at(idx_3d);
            orb_lite::Vec3 rel_p = {mp.pos.x - tracker.state.p.x, mp.pos.y - tracker.state.p.y, mp.pos.z - tracker.state.p.z};
            orb_lite::Vec3 P_c = {
                R_cw.m[0] * rel_p.x + R_cw.m[1] * rel_p.y + R_cw.m[2] * rel_p.z,
                R_cw.m[3] * rel_p.x + R_cw.m[4] * rel_p.y + R_cw.m[5] * rel_p.z,
                R_cw.m[6] * rel_p.x + R_cw.m[7] * rel_p.y + R_cw.m[8] * rel_p.z};
            
            if (P_c.z > 0) {
                orb_lite::Vec2 uv = tracker.cam.project(P_c);
                cv::Point2f pt_proj(uv.x, uv.y);

                // Draw them and connect with a line (The Residual)
                cv::circle(telemetry_img, pt_2d, 3, cv::Scalar(0, 255, 0), -1); // Green = Reality
                cv::circle(telemetry_img, pt_proj, 4, cv::Scalar(0, 0, 255), 1); // Red = Math
                cv::line(telemetry_img, pt_2d, pt_proj, cv::Scalar(255, 0, 0), 1); // Blue = Error

                // Add Depth Label
                cv::putText(telemetry_img, std::to_string(P_c.z).substr(0, 4) + "m", 
                            cv::Point((int)pt_proj.x + 5, (int)pt_proj.y - 5), 
                            cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar(0, 0, 255), 1);
            }
        }

        // 3. Render it to the screen!
        cv::imshow("SLAM Telemetry (Green=2D, Red=3D)", telemetry_img);

        static cv::VideoWriter writer;
        static bool writer_initialized = false;
        if (!writer_initialized) {
            writer.open("slam_telemetry.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30, telemetry_img.size());
            writer_initialized = true;
            std::cout << "-> Recording Telemetry to slam_telemetry.mp4" << std::endl;
        }
        writer.write(telemetry_img);

        int key = cv::waitKey(1);
        if (key == 27) break; 
#endif



        frameCount++;
    }

    

    std::cout << "SLAM Test Finished. Exporting Atlas data..." << std::endl;

    // 1. Retrieve all maps from the Atlas
    auto all_maps = atlas.getAllMaps();
    std::cout << "Atlas contains " << all_maps.size() << " separate maps." << std::endl;

    // 2. Save each map to a distinct CSV file
    int map_idx = 0;
    for (orb_lite::Map* m : all_maps) {
        // Only save maps that actually accumulated useful data to avoid clutter
        if (m->points.size() > 50) {
            std::string filename = "atlas_map_" + std::to_string(map_idx) + ".csv";
            m->saveCSV(filename.c_str());
            
            std::cout << " -> Saved " << filename << " (" 
                      << m->points.size() << " points, " 
                      << m->keyframes.size() << " KeyFrames)." << std::endl;
            map_idx++;
        } else {
            std::cout << " -> Ignored a fragmented map with only " << m->points.size() << " points." << std::endl;
        }
    }

    return 0;

}
