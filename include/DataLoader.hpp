#ifndef ORB_LITE_DATA_LOADER_HPP
#define ORB_LITE_DATA_LOADER_HPP

#include "Vision.hpp"
#include "IMU.hpp"
#include <string>
#include <vector>
#include <filesystem>

namespace orb_lite {

struct FrameData {
    double timestamp;
    std::string imagePath;
    std::vector<uint8_t> imageData; // Buffer for stb_image to fill
    int width, height;
};

class DataLoader {
public:
    DataLoader(const std::string& recordingDir);

    bool loadNext(FrameData& frame, std::vector<ImuSample>& imu);
    bool hasMore() const { return nextFrameIdx_ < framePaths_.size(); }

private:
    std::string recordingDir_;
    std::vector<std::string> framePaths_;
    std::vector<double> frameTimestamps_;
    std::vector<ImuSample> allImu_;
    size_t nextFrameIdx_ = 0;
    size_t nextImuIdx_ = 0;

    void loadImu();
    void loadFramePaths();
};

} // namespace orb_lite

#endif
