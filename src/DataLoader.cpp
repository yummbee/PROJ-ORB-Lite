#include "DataLoader.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace orb_lite {

DataLoader::DataLoader(const std::string& recordingDir) : recordingDir_(recordingDir) {
    loadImu();
    loadFramePaths();
}

void DataLoader::loadImu() {
    std::ifstream fa(recordingDir_ + "/Accelerometer.csv");
    std::ifstream fg(recordingDir_ + "/Gyroscope.csv");
    if (!fa.is_open() || !fg.is_open()) return;

    std::string line;
    std::getline(fa, line); // header
    std::getline(fg, line); // header

    struct RawImu { double t; Vec3 v; };
    std::vector<RawImu> accs, gyros;

    while (std::getline(fa, line)) {
        std::stringstream ss(line);
        std::string val;
        double t, x, y, z;
        std::getline(ss, val, ','); t = std::stod(val);
        std::getline(ss, val, ','); // skip seconds_elapsed
        std::getline(ss, val, ','); z = std::stod(val);
        std::getline(ss, val, ','); y = std::stod(val);
        std::getline(ss, val, ','); x = std::stod(val);
        accs.push_back({t * 1e-9, {x, y, z}});
    }

    while (std::getline(fg, line)) {
        std::stringstream ss(line);
        std::string val;
        double t, x, y, z;
        std::getline(ss, val, ','); t = std::stod(val);
        std::getline(ss, val, ','); // skip seconds_elapsed
        std::getline(ss, val, ','); z = std::stod(val);
        std::getline(ss, val, ','); y = std::stod(val);
        std::getline(ss, val, ','); x = std::stod(val);
        gyros.push_back({t * 1e-9, {x, y, z}});
    }

    // Aligned IMU: match gyro to nearest accel (simplified)
    size_t g_idx = 0;
    for (const auto& a : accs) {
        while (g_idx + 1 < gyros.size() && std::abs(gyros[g_idx + 1].t - a.t) < std::abs(gyros[g_idx].t - a.t)) {
            g_idx++;
        }
        // Transform IMU to Camera frame (X=Right, Y=Down, Z=Forward)
        // Sensor IMU: X=Right, Y=Forward, Z=Up
        // Mapping: CamX = ImuX, CamY = -ImuZ, CamZ = ImuY
        Vec3 acc_cam = {a.v.x, (float)-a.v.z, a.v.y};
        Vec3 gyro_cam = {gyros[g_idx].v.x, (float)-gyros[g_idx].v.z, gyros[g_idx].v.y};
        allImu_.push_back({acc_cam, gyro_cam, a.t});
    }
}

void DataLoader::loadFramePaths() {
    std::string cameraDir = recordingDir_ + "/Camera";
    for (const auto& entry : std::filesystem::directory_iterator(cameraDir)) {
        if (entry.path().extension() == ".jpg") {
            framePaths_.push_back(entry.path().string());
        }
    }
    std::sort(framePaths_.begin(), framePaths_.end());
    for (const auto& p : framePaths_) {
        std::string stem = std::filesystem::path(p).stem().string();
        frameTimestamps_.push_back(std::stod(stem) * 1e-3); // ms to s
    }
}

bool DataLoader::loadNext(FrameData& frame, std::vector<ImuSample>& imu) {
    if (nextFrameIdx_ >= framePaths_.size()) return false;

    frame.imagePath = framePaths_[nextFrameIdx_];
    frame.timestamp = frameTimestamps_[nextFrameIdx_];

    // Load image
    int channels;
    unsigned char* img = stbi_load(frame.imagePath.c_str(), &frame.width, &frame.height, &channels, 1);
    if (img) {
        frame.imageData.assign(img, img + (frame.width * frame.height));
        stbi_image_free(img);
    }

    // Sync IMU
    imu.clear();
    double prev_t = (nextFrameIdx_ == 0) ? -1.0 : frameTimestamps_[nextFrameIdx_ - 1];
    while (nextImuIdx_ < allImu_.size() && allImu_[nextImuIdx_].timestamp <= frame.timestamp) {
        if (allImu_[nextImuIdx_].timestamp > prev_t) {
            imu.push_back(allImu_[nextImuIdx_]);
        }
        nextImuIdx_++;
    }
    std::cout << "DataLoader: Frame " << nextFrameIdx_ << ", IMU samples: " << imu.size() << " timestamp: " << frame.timestamp << std::endl;

    nextFrameIdx_++;
    return true;
}

} // namespace orb_lite
