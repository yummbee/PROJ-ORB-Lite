#include "DataLoader.hpp"
#include "IMU.hpp"
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: ./test_imu <recording_dir>" << std::endl;
        return 1;
    }

    std::string recordingDir = argv[1];
    orb_lite::DataLoader loader(recordingDir);
    
    orb_lite::FrameData frame;
    std::vector<orb_lite::ImuSample> initialSamples;
    
    // Get first batch
    while (loader.loadNext(frame, initialSamples) && initialSamples.empty()) {
        // Skip frames until we get IMU data
    }

    if (initialSamples.empty()) {
        std::cout << "No IMU data found." << std::endl;
        return 1;
    }

    orb_lite::Preintegrated preint;
    orb_lite::Vec3 ba = {0,0,0}, bg = {0,0,0};
    
    double last_t = initialSamples.front().timestamp;
    double total_dt = 0;
    int sample_count = 0;

    std::cout << "Testing [Preintegrated] class on actual data: " << recordingDir << std::endl;

    bool first = true;
    while (first || loader.loadNext(frame, initialSamples)) {
        first = false;
        for (const auto& s : initialSamples) {
            double dt = s.timestamp - last_t;
            if (dt <= 0 || dt > 0.1) dt = 0.005;

            // Use the actual Preintegrated class logic
            preint.update(s.acc, s.gyro, dt, ba, bg);
            
            last_t = s.timestamp;
            total_dt += dt;
            sample_count++;
        }
        initialSamples.clear();

        if (sample_count > 0 && sample_count % 500 == 0) {
            std::cout << "  Samples: " << sample_count << " Total DT: " << total_dt << "s" << std::endl;
            std::cout << "    dP: " << preint.dP.x << ", " << preint.dP.y << ", " << preint.dP.z << std::endl;
            std::cout << "    dV: " << preint.dV.x << ", " << preint.dV.y << ", " << preint.dV.z << std::endl;
        }
    }

    std::cout << "Final Preintegrated Increments:" << std::endl;
    std::cout << "  dP: " << preint.dP.x << ", " << preint.dP.y << ", " << preint.dP.z << std::endl;
    std::cout << "  dV: " << preint.dV.x << ", " << preint.dV.y << ", " << preint.dV.z << std::endl;
    std::cout << "  dR: " << preint.dR.w << ", " << preint.dR.x << ", " << preint.dR.y << ", " << preint.dR.z << std::endl;

    return 0;
}
