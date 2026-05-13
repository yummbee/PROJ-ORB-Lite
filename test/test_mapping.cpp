#include "DataLoader.hpp"
#include "Vision.hpp"
#include "IMU.hpp"
#include "Map.hpp"
#include "Atlas.hpp"
#include "Tracking.hpp"
#include "BoW.hpp"
#include "KeyFrameDatabase.hpp"
#include "LoopClosing.hpp"
#include <iostream>

#ifdef WITH_OPENCV
#include <opencv2/opencv.hpp>
#endif

void drawMap(const orb_lite::Atlas& atlas, const orb_lite::NavState& state, int width, int height) {
#ifdef WITH_OPENCV
    cv::Mat mapVis = cv::Mat::zeros(height, width, CV_8UC3);
    
    static float scale = 200.0f; 
    float offsetX = width / 2.0f - (float)state.p.x * scale;
    float offsetY = height / 2.0f - (float)state.p.z * scale;

    auto maps = const_cast<orb_lite::Atlas&>(atlas).getAllMaps();
    int mapIdx = 0;
    for (auto m : maps) {
        cv::Scalar color = (m == const_cast<orb_lite::Atlas&>(atlas).getCurrentMap()) ? cv::Scalar(0, 255, 0) : cv::Scalar(100, 100, 100);
        int count = 0;
        for (const auto& p : m->points) {
            if (p.isBad) continue;
            count++;
            if (m->points.size() > 5000 && count % 5 != 0) continue; 
            
            int ix = (int)(p.pos.x * scale + offsetX);
            int iy = (int)(p.pos.z * scale + offsetY); 
            if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                mapVis.at<cv::Vec3b>(iy, ix) = cv::Vec3b(color[0], color[1], color[2]);
            }
        }
        mapIdx++;
    }

    int cx = (int)(state.p.x * scale + offsetX);
    int cz = (int)(state.p.z * scale + offsetY);
    cv::circle(mapVis, cv::Point(cx, cz), 4, cv::Scalar(0, 0, 255), -1);

    orb_lite::Mat3x3 R = orb_lite::qToMat33(state.p.x == 0 ? orb_lite::Quaternion{1,0,0,0} : state.q); // Guard
    int dx = (int)(R.m[2] * 15.0); 
    int dz = (int)(R.m[8] * 15.0);
    cv::line(mapVis, cv::Point(cx, cz), cv::Point(cx + dx, cz + dz), cv::Scalar(255, 255, 255), 1);

    cv::putText(mapVis, "Maps: " + std::to_string(maps.size()), cv::Point(10, 20), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

    cv::imshow("ORB-Lite Atlas View", mapVis);
#endif
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: ./test_mapping <recording_dir> [voc_file]" << std::endl;
        return 1;
    }

    std::string recordingDir = argv[1];
    std::string vocFile = (argc > 2) ? argv[2] : "Vocabulary/ORBvoc.txt";
    
    orb_lite::DataLoader loader(recordingDir);
    
    orb_lite::Vocabulary voc;
    std::cout << "Loading Vocabulary from " << vocFile << "..." << std::endl;
    if(!voc.loadFromTextFile(vocFile)) {
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

        cv::imshow("ORB-Lite Camera View", vis);
        drawMap(atlas, tracker.state, 600, 600);
        
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
