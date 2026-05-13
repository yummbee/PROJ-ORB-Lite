#include "DataLoader.hpp"
#include "Vision.hpp"
#include "Map.hpp"
#include "IMU.hpp"
#include <iostream>

#ifdef WITH_OPENCV
#include <opencv2/opencv.hpp>
#endif

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: ./test_bootstrapping <recording_dir> [num_frames_gap]" << std::endl;
        return 1;
    }

    std::string recordingDir = argv[1];
    int gap = (argc >= 3) ? std::stoi(argv[2]) : 5; // Default to 5 frames gap

    orb_lite::DataLoader loader(recordingDir);
    orb_lite::FrameData f1, f2;
    std::vector<orb_lite::ImuSample> imu_init, imu_step;
    std::vector<orb_lite::ImuSample> all_imu_for_alignment;

    // 1. Load Frame 1 and align gravity
    if (!loader.loadNext(f1, imu_init)) return 1;
    all_imu_for_alignment.insert(all_imu_for_alignment.end(), imu_init.begin(), imu_init.end());
    
    // If first frame has no IMU, peek ahead a bit for alignment
    if (all_imu_for_alignment.empty()) {
        orb_lite::FrameData f_aux; std::vector<orb_lite::ImuSample> i_aux;
        loader.loadNext(f_aux, i_aux);
        all_imu_for_alignment.insert(all_imu_for_alignment.end(), i_aux.begin(), i_aux.end());
    }

    orb_lite::Quaternion q_world_body = orb_lite::alignGravity(all_imu_for_alignment);
    orb_lite::NavState state;
    state.p = {0,0,0}; state.v = {0,0,0}; state.q = q_world_body;
    state.ba = {0,0,0}; state.bg = {0,0,0};

    // 2. Skip 'gap' frames and integrate IMU
    orb_lite::Preintegrated pre;
    double last_t = f1.timestamp;
    
    std::cout << "Bootstrapping: Frame 0 -> Frame " << gap << " (" << recordingDir << ")" << std::endl;

    for (int i = 0; i < gap; i++) {
        if (!loader.loadNext(f2, imu_step)) {
            std::cout << "Reached end of recording before gap." << std::endl;
            break;
        }
        for (const auto& s : imu_step) {
            double dt = s.timestamp - last_t;
            if (dt > 0 && dt < 0.1) {
                pre.update(s.acc, s.gyro, dt, state.ba, state.bg);
            }
            last_t = s.timestamp;
        }
    }

    // 3. Proper World-Frame Pose Prediction
    orb_lite::Vec3 g_world = {0, 0, -9.81};
    orb_lite::Mat3x3 R1 = orb_lite::qToMat33(state.q);
    
    orb_lite::Vec3 p2;
    p2.x = state.p.x + state.v.x * pre.dt + 0.5 * g_world.x * pre.dt * pre.dt + 
           (R1.m[0] * pre.dP.x + R1.m[1] * pre.dP.y + R1.m[2] * pre.dP.z);
    p2.y = state.p.y + state.v.y * pre.dt + 0.5 * g_world.y * pre.dt * pre.dt + 
           (R1.m[3] * pre.dP.x + R1.m[4] * pre.dP.y + R1.m[5] * pre.dP.z);
    p2.z = state.p.z + state.v.z * pre.dt + 0.5 * g_world.z * pre.dt * pre.dt + 
           (R1.m[6] * pre.dP.x + R1.m[7] * pre.dP.y + R1.m[8] * pre.dP.z);

    std::cout << "Resulting Movement after " << gap << " frames (" << pre.dt << "s):" << std::endl;
    std::cout << "  Delta P: " << p2.x << ", " << p2.y << ", " << p2.z << " meters" << std::endl;
    std::cout << "  Distance: " << std::sqrt(p2.x*p2.x + p2.y*p2.y + p2.z*p2.z) << " meters" << std::endl;

    // 4. Vision Match between f1 and the last loaded f2
    orb_lite::Image img1 = {f1.imageData.data(), f1.width, f1.height, f1.width};
    orb_lite::Image img2 = {f2.imageData.data(), f2.width, f2.height, f2.width};
    std::vector<orb_lite::KeyPoint> kps1, kps2;
    std::vector<orb_lite::Descriptor> desc1, desc2;
    orb_lite::detectFAST(img1, kps1, 20, true);
    orb_lite::computeORB(img1, kps1, desc1);
    orb_lite::detectFAST(img2, kps2, 20, true);
    orb_lite::computeORB(img2, kps2, desc2);
    std::vector<std::pair<int, int>> matches;
    orb_lite::matchDescriptors(desc1, desc2, matches);
    std::cout << "Vision Matches: " << matches.size() << std::endl;

#ifdef WITH_OPENCV
    cv::Mat vis;
    cv::Mat gray1(f1.height, f1.width, CV_8UC1, f1.imageData.data());
    cv::Mat gray2(f2.height, f2.width, CV_8UC1, f2.imageData.data());
    cv::hconcat(gray1, gray2, vis);
    cv::cvtColor(vis, vis, cv::COLOR_GRAY2BGR);
    for (const auto& m : matches) {
        cv::Point p1((int)kps1[m.first].x, (int)kps1[m.first].y);
        cv::Point p2((int)kps2[m.second].x + f1.width, (int)kps2[m.second].y);
        cv::line(vis, p1, p2, cv::Scalar(0, 255, 0), 1);
    }
    cv::imshow("Bootstrapping Matches (Gap: " + std::to_string(gap) + ")", vis);
    cv::waitKey(0);
#endif

    return 0;
}
