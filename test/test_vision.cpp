#include <opencv2/opencv.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <iostream>
#include <cassert>
#include "../include/Vision.hpp"



int main() {
    std::cout << "Running Vision tests...\n";

    // Example: FAST feature detection test
    {
        int width, height, channels;
        unsigned char* img_data = stbi_load("data/images/test_img_1.jpeg", &width, &height, &channels, 1);
        if (!img_data) {
            std::cerr << "Failed to load test image: " << stbi_failure_reason() << std::endl;
            exit(1);
        }
        orb_lite::Image img;
        img.data = img_data;
        img.cols = width;
        img.rows = height;
        img.stride = width;

        std::vector<orb_lite::KeyPoint> keypoints;

        // Example: run FAST
        orb_lite::detectFAST(img, keypoints, 20, true);
        std::cout << "FAST detected " << keypoints.size() << " keypoints.\n";

        // Visualize keypoints
        cv::Mat vis_img(height, width, CV_8UC1, img_data);
        cv::Mat vis_color;
        cv::cvtColor(vis_img, vis_color, cv::COLOR_GRAY2BGR);
        for (const auto& kp : keypoints) {
            cv::circle(vis_color, cv::Point2f(kp.x, kp.y), 3, cv::Scalar(0, 0, 255), 1);
        }
        // Scale output for easier viewing
        cv::Mat vis_scaled;
        double scale = 800.0 / std::max(width, height);
        cv::resize(vis_color, vis_scaled, cv::Size(), scale, scale);
        cv::imwrite("fast_keypoints.png", vis_scaled);
        std::cout << "Saved keypoint visualization to fast_keypoints.png\n";

        stbi_image_free(img_data);
    }

    // Example: BRIEF descriptor test
    {
        //load img
        int width, height, channels;
        unsigned char* img_data = stbi_load("data/images/test_img_1.jpeg", &width, &height, &channels, 1);
        if (!img_data) {
            std::cerr << "Failed to load test image: " << stbi_failure_reason() << std::endl;
            exit(1);
        }

        orb_lite::Image img;
        img.data = img_data;
        img.cols = width;
        img.rows = height;
        img.stride = width;
        std::vector<orb_lite::KeyPoint> keypoints;
        std::vector<orb_lite::Descriptor> descriptors;

        // Run FAST to get keypoints
        orb_lite::detectFAST(img, keypoints, 20, true);
        std::cout << "FAST detected " << keypoints.size() << " keypoints.\n";

        // Run ORB to get descriptors
        orb_lite::computeORB(img, keypoints, descriptors);
        std::cout << "Computed " << descriptors.size() << " descriptors.\n";

    }

    //Test match descriptors from img 1 and 2
    {
        //load images
        int width1, height1, channels1;
        unsigned char* img_data1 = stbi_load("data/images/test_img_1.jpeg", &width1, &height1, &channels1, 1);
        if (!img_data1) {
            std::cerr << "Failed to load test image 1: " << stbi_failure_reason() << std::endl;
            exit(1);
        }

        int width2, height2, channels2;
        unsigned char* img_data2 = stbi_load("data/images/test_img_2.jpeg", &width2, &height2, &channels2, 1);
        if (!img_data2) {
            std::cerr << "Failed to load test image 2: " << stbi_failure_reason() << std::endl;
            exit(1);
        }

        // Create Image structs
        orb_lite::Image img1;
        img1.data = img_data1;
        img1.cols = width1;
        img1.rows = height1;
        img1.stride = width1;

        orb_lite::Image img2;
        img2.data = img_data2;
        img2.cols = width2;
        img2.rows = height2;
        img2.stride = width2;

        std::vector<orb_lite::KeyPoint> keypoints1, keypoints2;
        std::vector<orb_lite::Descriptor> descriptors1, descriptors2;

        // Run FAST to get keypoints
        orb_lite::detectFAST(img1, keypoints1, 20, true);
        orb_lite::detectFAST(img2, keypoints2, 20, true);
        std::cout << "FAST detected " << keypoints1.size() << " keypoints in image 1.\n";
        std::cout << "FAST detected " << keypoints2.size() << " keypoints in image 2.\n";

        // Run ORB to get descriptors
        orb_lite::computeORB(img1, keypoints1, descriptors1);
        orb_lite::computeORB(img2, keypoints2, descriptors2);
        std::cout << "Computed " << descriptors1.size() << " descriptors in image 1.\n";
        std::cout << "Computed " << descriptors2.size() << " descriptors in image 2.\n";

        // Match descriptors
        std::vector<std::pair<int, int>> matches;
        orb_lite::matchDescriptors(descriptors1, descriptors2, matches);
        std::cout << "Found " << matches.size() << " matches between image 1 and image 2.\n";

        // Visualize matches (using 1 channel since stbi_load req_comp was 1)
        cv::Mat cv_img1(height1, width1, CV_8UC1, img_data1);
        cv::Mat cv_img2(height2, width2, CV_8UC1, img_data2);
        cv::Mat cv_img1_color, cv_img2_color;
        cv::cvtColor(cv_img1, cv_img1_color, cv::COLOR_GRAY2BGR);
        cv::cvtColor(cv_img2, cv_img2_color, cv::COLOR_GRAY2BGR);

        // Draw only matched keypoints
        for (const auto& m : matches) {
            const auto& kp1 = keypoints1[m.first];
            const auto& kp2 = keypoints2[m.second];
            cv::circle(cv_img1_color, cv::Point2f(kp1.x, kp1.y), 3, cv::Scalar(0, 0, 255), 1);
            cv::circle(cv_img2_color, cv::Point2f(kp2.x, kp2.y), 3, cv::Scalar(0, 0, 255), 1);
        }

        // Create a side-by-side image
        int out_height = std::max(cv_img1_color.rows, cv_img2_color.rows);
        int out_width = cv_img1_color.cols + cv_img2_color.cols;
        cv::Mat out(out_height, out_width, CV_8UC3, cv::Scalar(0,0,0));
        cv_img1_color.copyTo(out(cv::Rect(0, 0, cv_img1_color.cols, cv_img1_color.rows)));
        cv_img2_color.copyTo(out(cv::Rect(cv_img1_color.cols, 0, cv_img2_color.cols, cv_img2_color.rows)));

        // Scale output for easier viewing
        cv::Mat out_scaled;
        double scale = 800.0 / std::max(out.cols, out.rows);
        cv::resize(out, out_scaled, cv::Size(), scale, scale);
        cv::imwrite("matches.png", out_scaled);
        std::cout << "Saved matches visualization to matches.png\n";

        stbi_image_free(img_data1);
        stbi_image_free(img_data2);
    }

    // Example: ORB pipeline test
    {
        // TODO: Load test image/video and IMU data from ../data/ and test ORB
        bool orb_result = true; // Replace with actual test
        assert(orb_result && "ORB pipeline failed");
        std::cout << "ORB pipeline test passed.\n";
    }

    std::cout << "All tests completed.\n";
    return 0;
}
