cmake_minimum_required(VERSION 3.15)
project(ORB_Lite LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-O3 -march=native -ffast-math)

add_executable(test_ba test/test_ba.cpp)
target_link_libraries(test_ba orb_lite ${OpenCV_LIBS})

add_executable(test_jacobian test/test_jacobian.cpp)
target_link_libraries(test_jacobian orb_lite ${OpenCV_LIBS})

# Optional OpenCV for visualization tests only
find_package(OpenCV QUIET)

add_library(orb_lite STATIC
    src/Vision.cpp
    src/IMU.cpp
    src/Map.cpp
    src/Tracking.cpp
    src/Optimizer.cpp
    src/DataLoader.cpp
    src/BoW.cpp
    src/KeyFrameDatabase.cpp
    src/LoopClosing.cpp
    src/Atlas.cpp
)

target_include_directories(orb_lite PUBLIC include)

if(OpenCV_FOUND)
    target_compile_definitions(orb_lite PUBLIC WITH_OPENCV)
    target_link_libraries(orb_lite PUBLIC ${OpenCV_LIBS})
endif()

add_executable(orb_lite_test
    src/Main.cpp
)
target_link_libraries(orb_lite_test PRIVATE orb_lite)

add_executable(test_mapping
    test/test_mapping.cpp
)
target_link_libraries(test_mapping PRIVATE orb_lite)

add_executable(test_imu
    test/test_imu.cpp
)
target_link_libraries(test_imu PRIVATE orb_lite)

add_executable(test_bootstrapping
    test/test_bootstrapping.cpp
)
target_link_libraries(test_bootstrapping PRIVATE orb_lite)

if(OpenCV_FOUND)
    target_compile_definitions(test_mapping PRIVATE WITH_OPENCV)
    target_link_libraries(test_mapping PRIVATE ${OpenCV_LIBS})

    target_compile_definitions(test_bootstrapping PRIVATE WITH_OPENCV)
    target_link_libraries(test_bootstrapping PRIVATE ${OpenCV_LIBS})

    add_executable(test_vision
        test/test_vision.cpp
    )
    target_include_directories(test_vision PRIVATE include test)
    target_link_libraries(test_vision PRIVATE orb_lite ${OpenCV_LIBS})
endif()
