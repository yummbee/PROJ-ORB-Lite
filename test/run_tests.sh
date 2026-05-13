#!/bin/bash
set -e

# Go to project root
cd "$(dirname "$0")/.."

# Configure and build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# 2. Tell CMake to use 8 CPU cores to compile your .cpp files instantly
cmake --build build --parallel 8

TEST_TO_RUN=$1

run_test() {
    local t=$1
    if [[ -z "$TEST_TO_RUN" || "$TEST_TO_RUN" == "$t" ]]; then
        case $t in
            test_vision)
                echo "--- Running test_vision ---"
                ./build/test_vision
                ;;
            test_imu)
                LATEST_DATA=$(ls -td data/*/ 2>/dev/null | head -1)
                if [ -n "$LATEST_DATA" ]; then
                    echo "--- Running test_imu on $LATEST_DATA ---"
                    ./build/test_imu "$LATEST_DATA"
                else
                    echo "No data found for test_imu"
                fi
                ;;
            test_mapping)
                LATEST_DATA=$(ls -td data/*/ 2>/dev/null | head -1)
                if [ -n "$LATEST_DATA" ]; then
                    echo "--- Running test_mapping on $LATEST_DATA ---"
                    ./build/test_mapping "$LATEST_DATA"
                fi
                ;;
            test_bootstrapping)
                LATEST_DATA=$(ls -td data/*/ 2>/dev/null | head -1)
                if [ -n "$LATEST_DATA" ]; then
                    echo "--- Running test_bootstrapping on $LATEST_DATA (gap 10) ---"
                    ./build/test_bootstrapping "$LATEST_DATA" 50
                fi
                ;;
        esac
    fi
}

run_test test_vision
run_test test_imu
run_test test_mapping
run_test test_bootstrapping
