#ifndef COLORBANDMASK_H
#define COLORBANDMASK_H

#include "ProcessingConfig.h"

#include <algorithm>
#include <cstdint>
#include <opencv2/core.hpp>
#include <thread>
#include <vector>

// Shared helpers for the "several colors by proportion" pipeline.
// Both the streaming video generator (VideoCreation.cpp) and the standalone
// final-image generator (ImageCreation.cpp) build the exact same per-pixel
// band assignment; factoring it out here keeps the two in sync (DRY).

// Computes the sum of the B, G, R channels for every pixel, in parallel.
inline std::vector<int> computeRgbSums(const cv::Mat& baseImageMat) {
    const size_t pixelCount = static_cast<size_t>(baseImageMat.rows) * baseImageMat.cols;
    const unsigned int numThreads = std::thread::hardware_concurrency();
    const size_t chunkSize = pixelCount / numThreads;

    std::vector<int> rgbSums(pixelCount, 0);
    std::vector<std::thread> threads;
    threads.reserve(numThreads);

    for (unsigned int t = 0; t < numThreads; ++t) {
        threads.emplace_back([&, t]
        {
            const size_t start = t * chunkSize;
            const size_t end = t == numThreads - 1 ? pixelCount : (t + 1) * chunkSize;
            for (size_t i = start; i < end; ++i) {
                const int row = static_cast<int>(i / baseImageMat.cols);
                const int col = static_cast<int>(i % baseImageMat.cols);
                const cv::Vec3b& pixel = baseImageMat.at<cv::Vec3b>(row, col);
                rgbSums.at(i) = pixel[0] + pixel[1] + pixel[2];
            }
        });
    }
    for (auto& t : threads) { t.join(); }
    return rgbSums;
}

// Derives the per-step RGB-sum thresholds from the sorted RGB sums.
// thresholds[i] is the RGB sum at the cumulative-proportion boundary of step i.
inline std::vector<int> computeProportionThresholds(const std::vector<int>& sortedRGB) {
    const size_t pixelCount = sortedRGB.size();
    std::vector<int> thresholds(parameters::numProportionSteps);
    for (int i = 0; i < parameters::numProportionSteps; ++i) {
        const float cp = parameters::proportions.at(0) + static_cast<float>(i) * parameters::proportions.at(2);
        thresholds.at(i) = sortedRGB.at(
            std::min(static_cast<size_t>(static_cast<float>(pixelCount) * cp), pixelCount - 1));
    }
    return thresholds;
}

// Builds one boolean mask per proportion step: mask[i][pixelIdx] is true when
// the pixel's RGB sum falls in band i, i.e. (thresholds[i-1], thresholds[i]]
// (band 0 covers [0, thresholds[0]]).
inline std::vector<std::vector<bool>> computeColorBandMasks(const cv::Mat& baseImageMat) {
    const size_t pixelCount = static_cast<size_t>(baseImageMat.rows) * baseImageMat.cols;
    const unsigned int numThreads = std::thread::hardware_concurrency();
    const size_t chunkSize = pixelCount / numThreads;

    const std::vector<int> rgbSums = computeRgbSums(baseImageMat);

    std::vector<int> sortedRGB = rgbSums;
    std::ranges::sort(sortedRGB);
    const std::vector<int> thresholds = computeProportionThresholds(sortedRGB);

    std::vector pixelMask(parameters::numProportionSteps, std::vector<bool>(pixelCount, false));
    std::vector<std::thread> threads;
    threads.reserve(numThreads);

    for (int propIdx = 0; propIdx < parameters::numProportionSteps; ++propIdx) {
        const int lowerBound = propIdx == 0 ? -1 : thresholds.at(propIdx - 1);
        const int upperBound = thresholds.at(propIdx);

        for (unsigned int t = 0; t < numThreads; ++t) {
            threads.emplace_back([&, propIdx, t, lowerBound, upperBound]() {
                const size_t start = t * chunkSize;
                const size_t end = t == numThreads - 1 ? pixelCount : (t + 1) * chunkSize;
                for (size_t pixelIdx = start; pixelIdx < end; ++pixelIdx) {
                    const int s = rgbSums[pixelIdx];
                    pixelMask[propIdx][pixelIdx] = s > lowerBound && s <= upperBound;
                }
            });
        }
        for (auto& t : threads) { t.join(); }
        threads.clear();
    }
    return pixelMask;
}

// Sets every masked pixel of `target` to a flat grayscale color, in parallel.
inline void applyColorToMask(cv::Mat& target, const std::vector<bool>& mask, const uint8_t newColor) {
    const size_t pixelCount = mask.size();
    const unsigned int numThreads = std::thread::hardware_concurrency();
    const size_t chunkSize = pixelCount / numThreads;

    std::vector<std::thread> threads;
    threads.reserve(numThreads);

    for (unsigned int t = 0; t < numThreads; ++t) {
        threads.emplace_back([&, t]() {
            const size_t start = t * chunkSize;
            const size_t end = (t == numThreads - 1) ? pixelCount : (t + 1) * chunkSize;
            for (size_t pixelIdx = start; pixelIdx < end; ++pixelIdx) {
                if (mask[pixelIdx]) {
                    const size_t row = pixelIdx / target.cols;
                    const size_t col = pixelIdx % target.cols;
                    target.at<cv::Vec3b>(static_cast<int>(row), static_cast<int>(col)) =
                        cv::Vec3b(newColor, newColor, newColor);
                }
            }
        });
    }
    for (auto& t : threads) { t.join(); }
}

// Returns the "locked-in" final color for proportion band `bandIdx`, matching
// the alternating start/end pattern of the streaming animation in
// several_colors_transformations_streaming (VideoCreation.cpp): with
// reverseOrder toggling per band starting at false, even bands lock onto the
// last color nuance and odd bands lock onto the first.
inline uint8_t finalColorForBand(const int bandIdx) {
    return static_cast<uint8_t>(
        bandIdx % 2 == 0 ? parameters::colorNuances.at(1) : parameters::colorNuances.at(0));
}

#endif // COLORBANDMASK_H