#include "ColorConfig.h"
#include "EdgeDetector.h"
#include "VideoCreation.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <thread>

// Resets a thread-local working buffer with a fresh copy of the base image
// before each simplification phase.
static void resetWorkingBuffer(const Image& workingImg, const cv::Mat& baseImageMat) {
    std::memcpy(workingImg.data,
                baseImageMat.data,
                static_cast<size_t>(baseImageMat.cols)
                * baseImageMat.rows * baseImageMat.channels());
}

// Builds the per-frame callback shared by all phases: wraps the resulting
// Image buffer into a cv::Mat (no copy) and writes it to the video stream.
static auto makeFrameWriteCallback(const cv::Mat& baseImageMat, cv::VideoWriter& writer) {
    return [&baseImageMat, &writer](Image&& result) {
        const cv::Mat frame(
            baseImageMat.rows, baseImageMat.cols,
            baseImageMat.type(),
            const_cast<void*>(
                static_cast<const void*>(result.data)));
        writer.write(frame);
        return true;
    };
}

static void one_color_transformations_streaming(
    const std::string& baseName,
    const std::string& inputPath
) {
    if constexpr (!parameters::oneColor) { return; }
    auto imageOpt = loadImage(inputPath);
    if (!imageOpt) { return; }
    auto& baseImageMat = *imageOpt;

    const auto pipeline = OneColorPipeline::forStreaming();
    const size_t configFrames = pipeline.configCount();

    // ── Phase 0 config/frame count ────────────────────────────────────
    static constexpr std::array phase0Config = {1.F, 0.F, 0.F};
    static constexpr int phase0_tol_min  = parameters::toleranceOneColor.at(1);
    static constexpr int phase0_tol_max  = 120;
    static constexpr int phase0_tol_step = parameters::toleranceOneColor.at(2);
    constexpr size_t phase0Frames =
        static_cast<size_t>((phase0_tol_max - phase0_tol_min) / phase0_tol_step) + 1;

    const size_t phase1Frames = configFrames * pipeline.getTValues().size();
    constexpr size_t phase2Frames = parameters::numTolerance - 1;
    const size_t phase3Frames = pipeline.getLastValues().size();

    const size_t totalFrames = phase0Frames + phase1Frames + phase2Frames + phase3Frames;

    if (totalFrames - phase3Frames < static_cast<size_t>(parameters::fps)) {
        Logger::err(totalFrames - phase3Frames, " frames for ",  parameters::fps,
            " FPS (excluding phase3Frames) --> Video less than 1 second long, creation canceled");
        return;
    }

    Logger::log("Frames to process : ", totalFrames," --- configFrames ",
        configFrames, " --- TOLERANCE_RAM ", TOLERANCE_RAM, '\n');

    const std::string outputVideoPath =
        OutputPathBuilder::video_one_color(baseName, totalFrames, 0);

    cv::VideoWriter writer;
    writer.open(outputVideoPath,
                cv::VideoWriter::fourcc('m','p','4','v'),
                parameters::fps,
                cv::Size(baseImageMat.cols, baseImageMat.rows));
    if (!writer.isOpened()) {
        Logger::err("Error: Could not open video writer");
        return;
    }

    const int num_threads = computeNumThreads();

    std::vector<Image> thread_imgs;
    thread_imgs.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        thread_imgs.emplace_back(baseImageMat.cols, baseImageMat.rows,
                                 baseImageMat.channels());
    }

    // Single callback instance reused across all four phases
    const auto writeFrame = makeFrameWriteCallback(baseImageMat, writer);

    // ── Phase 0: single config {1,0,0}, decreasing tolerance ──────────────
    {
        const std::vector phase0ConfigVec(phase0Config.begin(), phase0Config.end());

        for (int tol = phase0_tol_max; tol >= phase0_tol_min; tol -= phase0_tol_step) {
            resetWorkingBuffer(thread_imgs.at(0), baseImageMat);

            thread_imgs.at(0).simplify_to_dominant_color_combinations(
                tol,
                &phase0ConfigVec,
                {},
                true,
                true,
                writeFrame
            );
        }
    }

    bool original_to_amended = true;
    // ── Phase 1: all configs at max tolerance ─────────────────────────────
    for (size_t phase = 0; phase < configFrames; ++phase) {
        resetWorkingBuffer(thread_imgs.at(0), baseImageMat);

        thread_imgs.at(0).simplify_to_dominant_color_combinations(
            parameters::toleranceOneColor.at(1),
            &pipeline.configs.at(phase),
            pipeline.getTValues(),
            false,
            original_to_amended,
            writeFrame
        );
        original_to_amended = !original_to_amended;
    }

    // ── Phase 2: last config, decreasing tolerance ────────────────────────
    constexpr int tol_min  = parameters::toleranceOneColor.at(0);
    constexpr int tol_max  = parameters::toleranceOneColor.at(1);
    constexpr int tol_step = parameters::toleranceOneColor.at(2);

    for (int tol = tol_max - tol_step; tol >= tol_min; tol -= tol_step) {
        const size_t configIdx = configFrames - 1;
        resetWorkingBuffer(thread_imgs.at(0), baseImageMat);

        thread_imgs.at(0).simplify_to_dominant_color_combinations(
            tol,
            &pipeline.configs.at(configIdx),
            {},
            true,
            true,
            writeFrame
        );
    }

    // ── Phase 3: transition from colored back to original (t: 1 → 0) ─────────
    const std::vector<float> phase3Config = { 1.F, 1.F, 1.F };

    resetWorkingBuffer(thread_imgs.at(0), baseImageMat);

    thread_imgs.at(0).simplify_to_dominant_color_combinations(
        parameters::toleranceOneColor.at(0),
        &phase3Config,
        pipeline.getLastValues(),
        false,
        false,
        writeFrame
    );

    writer.release();
    Logger::log(outputVideoPath, " --> video created");
}

static void reverse_transformations_by_proportion_streaming(
    const std::string& baseName,
    const std::string& inputPath
) {
    if constexpr (!parameters::totalReversal) { return; }

    const auto imageOpt = loadImage(inputPath);
    if (!imageOpt) { return; }
    const cv::Mat& baseImageMat = *imageOpt;

    const int width    = baseImageMat.cols;
    const int height   = baseImageMat.rows;
    const int channels = baseImageMat.channels();
    const int num_threads = computeNumThreads();

    for (size_t entryIdx = 0; entryIdx < reversal_suffixes.size(); ++entryIdx) {
        const std::string_view suffix = reversal_suffixes[entryIdx];
        const bool below = reversal_below_flag(entryIdx);

        constexpr int nFrames = parameters::numProportionSteps;

        if constexpr (nFrames < parameters::fps) {
            Logger::err("Video less than 1 second long, creation canceled for ", suffix);
            continue;
        }

        const std::string outputPath = OutputPathBuilder::video_reversal(baseName, suffix, nFrames);

        auto writerOpt = make_video_writer(outputPath, width, height, parameters::fps);
        if (!writerOpt) { continue; }
        cv::VideoWriter& writer = *writerOpt;

        // One working buffer per thread (reused between chunks)
        std::vector<Image> thread_imgs;
        thread_imgs.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t) {
            thread_imgs.emplace_back(width, height, channels);
        }

        for (int phase_base = 0; phase_base < nFrames; phase_base += TOLERANCE_RAM) {
            const int phase_end  = std::min(phase_base + TOLERANCE_RAM, nFrames);
            const int chunk_size = phase_end - phase_base;

            // A flat vector indexed by `local_phase` — one slot per frame of the chunk
            std::vector<Image> chunkResults(chunk_size, Image(0, 0, 0));

            #pragma omp parallel for schedule(static) num_threads(num_threads) \
                default(none) \
                shared(baseImageMat, thread_imgs, chunkResults, \
                       phase_base, phase_end, below, width, height, channels)
            for (int phase = phase_base; phase < phase_end; ++phase) {
                const float proportion =
                    parameters::proportions.at(0) +
                    static_cast<float>(phase) * parameters::proportions.at(2);

                if (proportion <= 0.0F) { continue; }

                const int tid = omp_get_thread_num();
                const int local_phase = phase - phase_base;

                Image& img = thread_imgs.at(tid);
                std::memcpy(img.data, baseImageMat.data,
                            static_cast<size_t>(width) * height * channels);

                img.reverse_by_proportion(proportion, below);

                // Copy to the flat slot — each local_phase is written by a single thread
                chunkResults.at(local_phase) = img;
            }

            // Sequential drain in frame order
            for (int phase = phase_base; phase < phase_end; ++phase) {
                const int local_phase = phase - phase_base;

                const float proportion =
                    parameters::proportions.at(0) +
                    static_cast<float>(phase) * parameters::proportions.at(2);

                if (proportion <= 0.0F) { continue; }

                const Image& result = chunkResults.at(local_phase);
                const cv::Mat frame(height, width,
                                    baseImageMat.type(),
                                    const_cast<void*>(
                                        static_cast<const void*>(result.data)));
                writer.write(frame);

                // Immediate release
                chunkResults.at(local_phase) = Image(0, 0, 0);
            }
        }

        writer.release();
        Logger::log("\n", outputPath, " created");
    }
}

static void several_colors_transformations_streaming(
    const std::string& baseName,
    const std::string& inputPath
) {
    if constexpr (!parameters::complete_transformation_colors_by_proportion) { return; }

    auto imageOpt = loadImage(inputPath);
    if (!imageOpt) { return; }
    cv::Mat& baseImageMat = *imageOpt;

    constexpr int nFrames =
    ((parameters::numProportionSteps + 1) / 2) * parameters::numColorNuances;

    if constexpr (nFrames < parameters::fps) {
        Logger::err("Video less than 1 second long, creation canceled");
        return;
    }

    Logger::log("Frames to process : ", nFrames);

    const std::string outputVideoPath = OutputPathBuilder::video_several_colors(baseName, nFrames);

    cv::VideoWriter video(outputVideoPath,
                          cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                          parameters::fps,
                          cv::Size(baseImageMat.cols, baseImageMat.rows));

    if (!video.isOpened()) {
        Logger::err("Error: Could not open video writer for ", outputVideoPath);
        return;
    }

    const size_t pixelCount = baseImageMat.rows * baseImageMat.cols;
    const unsigned int numThreads = std::thread::hardware_concurrency();
    const size_t chunkSize = pixelCount / numThreads;
    std::vector<std::thread> threads;

    // Compute per-pixel RGB sums in parallel
    std::vector<int> rgbSums(pixelCount, 0);

    for (unsigned int t = 0; t < numThreads; ++t) {
        threads.emplace_back([&, t]() {
            const size_t start = t * chunkSize;
            const size_t end = (t == numThreads - 1) ? pixelCount : (t + 1) * chunkSize;
            for (size_t i = start; i < end; ++i) {
                const int row = static_cast<int>(i / baseImageMat.cols);
                const int col = static_cast<int>(i % baseImageMat.cols);
                const cv::Vec3b& pixel = baseImageMat.at<cv::Vec3b>(row, col);
                rgbSums.at(i) = pixel[0] + pixel[1] + pixel[2];
            }
        });
    }
    for (auto& t : threads) { t.join(); }
    threads.clear();

    // Sort RGB sums to derive proportion thresholds
    std::vector<int> sortedRGB = rgbSums;
    std::ranges::sort(sortedRGB);

    std::vector<int> thresholds(parameters::numProportionSteps);
    for (int i = 0; i < parameters::numProportionSteps; ++i) {
        const float cp = parameters::proportions.at(0) + static_cast<float>(i) * parameters::proportions.at(2);
        thresholds.at(i) = sortedRGB.at(
            std::min(static_cast<size_t>(static_cast<float>(pixelCount) * cp), pixelCount - 1));
    }

    // Build per-step band masks: only pixels in the slice (thresholds[i-1], thresholds[i]]
    // Step 0 covers [0, thresholds[0]]; step i>0 covers (thresholds[i-1], thresholds[i]].
    std::vector pixelMask(parameters::numProportionSteps, std::vector<bool>(pixelCount, false));

    for (int propIdx = 0; propIdx < parameters::numProportionSteps; ++propIdx) {
        const int lowerBound = (propIdx == 0) ? -1 : thresholds.at(propIdx - 1);
        const int upperBound = thresholds.at(propIdx);

        for (unsigned int t = 0; t < numThreads; ++t) {
            threads.emplace_back([&, propIdx, t, lowerBound, upperBound]() {
                const size_t start = t * chunkSize;
                const size_t end = (t == numThreads - 1) ? pixelCount : (t + 1) * chunkSize;
                for (size_t pixelIdx = start; pixelIdx < end; ++pixelIdx) {
                    const int s = rgbSums[pixelIdx];
                    pixelMask[propIdx][pixelIdx] = s > lowerBound && s <= upperBound;
                }
            });
        }
        for (auto& t : threads) { t.join(); }
        threads.clear();
    }

    VideoWriterQueue queue(video);

    // Apply a color value to pixels selected by mask, in parallel
    auto applyColorTransform = [&](cv::Mat& target, const std::vector<bool>& mask, const uint8_t newColor) {
        for (unsigned int t = 0; t < numThreads; ++t) {
            threads.emplace_back([&, t, newColor]() {
                const size_t start = t * chunkSize;
                const size_t end = (t == numThreads - 1) ? pixelCount : (t + 1) * chunkSize;
                for (size_t pixelIdx = start; pixelIdx < end; ++pixelIdx) {
                    if (mask[pixelIdx]) {
                        const size_t row = pixelIdx / baseImageMat.cols;
                        const size_t col = pixelIdx % baseImageMat.cols;
                        target.at<cv::Vec3b>(static_cast<int>(row), static_cast<int>(col)) =
                            cv::Vec3b(newColor, newColor, newColor);
                    }
                }
            });
        }
        for (auto& t : threads) { t.join(); }
        threads.clear();
    };

    // accumulated holds the image state built up across all previous steps.
    // Each call to process() mutates only the band pixels of the current step,
    // leaving all previously colored pixels intact.
    cv::Mat accumulated = baseImageMat.clone();

    // Process one proportion step: animate the band pixels through colorNuances
    // up then down, starting from the current accumulated image state.
    auto process = [&](const int propIdx, const bool reverseOrder) {

        // Logger::log("auto process ", propIdx, "-", reverseOrder);

        const auto& mask = pixelMask.at(propIdx);

        const int start =
            reverseOrder
                ? parameters::colorNuances.at(1)
                : parameters::colorNuances.at(0);

        const int end =
            reverseOrder
                ? parameters::colorNuances.at(0)
                : parameters::colorNuances.at(1);

        const int step =
            reverseOrder
                ? -parameters::colorNuances.at(2)
                : parameters::colorNuances.at(2);

        for (int colorNuance = start;
             reverseOrder
                 ? colorNuance >= end
                 : colorNuance <= end;
             colorNuance += step) {

            applyColorTransform(
                accumulated,
                mask,
                static_cast<uint8_t>(colorNuance)
            );

            queue.enqueue(accumulated.clone());
             }

        // Locks in the final colour
        applyColorTransform(
            accumulated,
            mask,
            static_cast<uint8_t>(end)
        );
    };

    bool reverseOrder = false;
    for (int i = 0; i < parameters::numProportionSteps; ++i) {
        const float cp = parameters::proportions.at(0) + (static_cast<float>(i) * parameters::proportions.at(2));
        process(i, reverseOrder);
        reverseOrder = !reverseOrder;
        ++i;
    }

    queue.finish();
    video.release();

    if (const std::string outputImagePath =
        OutputPathBuilder::image_black_and_white(baseName, nFrames); !cv::imwrite(outputImagePath, accumulated)) {
        Logger::err("Error: Could not write last frame to ", outputImagePath);
    } else {
        Logger::log(outputImagePath, " created");
    }

    Logger::log("\n", outputVideoPath, " created");
}


static void edge_detector_video(
    const std::string& baseName,
    const std::string& inputVideoPath
) {
    cv::VideoCapture capture(inputVideoPath);

    if (!capture.isOpened()) {
        Logger::err("Error: cannot open ", inputVideoPath);
        return;
    }

    const int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    const int totalFrames = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));
    Logger::log("Number of frames in the original video : ", totalFrames);
    const double fps_edge_detector_video = capture.get(cv::CAP_PROP_FPS);

    constexpr int startFrame = std::max(0, parameters::frames.at(0));
    const int endFrame = (parameters::frames.at(1) == 0 || parameters::frames.at(1) > totalFrames)
                       ? totalFrames
                       : parameters::frames.at(1);
    const int framesToProcess = endFrame - startFrame;

    if (framesToProcess <= 0) {
        Logger::err("Error: invalid frame range [", startFrame, ", ", endFrame, "]");
        return;
    }
    Logger::log("Frames to process : ", framesToProcess);
    const std::string tempVideoPath = OutputPathBuilder::video_edge_detector_temp(baseName);

    cv::VideoWriter video(tempVideoPath,
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        fps_edge_detector_video,
        cv::Size(width, height));

    if (!video.isOpened()) {
        Logger::err("Error: cannot create ", tempVideoPath);
        return;
    }

    const auto imgSize = static_cast<const size_t>(width * height);
    capture.set(cv::CAP_PROP_POS_FRAMES, startFrame);

    // Initialize pipeline ONCE before loop
    EdgeDetectorPipeline pipeline(width, height);

    int frameIdx = 0;
    cv::Mat frameBGR(height, width, CV_8UC3);
    std::vector<uint8_t> grayData(imgSize);

    while (frameIdx < framesToProcess) {
        if (!capture.read(frameBGR)) { break;
}

        ++frameIdx;
        Logger::logProgress("Frame ", frameIdx, "/", framesToProcess);

        // BGR -> grayscale: average of B, G, R channels
        #pragma omp parallel for default(none) shared(imgSize, frameBGR, grayData)
            for (int i = 0; i < imgSize; ++i) {
                const uint8_t* pixel = frameBGR.data + (i * 3);
                grayData.at(i) = static_cast<uint8_t>((pixel[0] + pixel[1] + pixel[2]) / 3);
            }

        // Process frame through edge detection pipeline (zero allocation)
        const std::vector<uint8_t>& rgb = pipeline.process(grayData.data());

        // RGB -> BGR conversion for OpenCV compatibility
        #pragma omp parallel for default(none) shared(imgSize, frameBGR, rgb)
            for (int i = 0; i < imgSize; ++i) {
                frameBGR.data[static_cast<ptrdiff_t>(i * 3)]     = rgb.at((i * 3) + 2); // B
                frameBGR.data[i * 3 + 1] = rgb.at((i * 3) + 1); // G
                frameBGR.data[i * 3 + 2] = rgb.at(i * 3);     // R
            }

        video.write(frameBGR);
    }

    capture.release();
    video.release();

    Logger::log("\nMerging audio with FFmpeg...");

    const double startTime = static_cast<double>(startFrame) / fps_edge_detector_video;
    const double duration = static_cast<double>(framesToProcess) / fps_edge_detector_video;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << fps_edge_detector_video;

    std::string outputVideoPath = OutputPathBuilder::video_edge_detector(
        baseName, framesToProcess, totalFrames, fps_edge_detector_video);

    std::string ffmpegCmd =
        "ffmpeg -fflags +genpts "
        "-i \"" + tempVideoPath + "\" "
        "-ss " + std::to_string(startTime) + " "
        "-i \"" + inputVideoPath + "\" "
        "-t " + std::to_string(duration) + " "
        "-map 0:v:0 -map 1:a:0? "
        "-c:v copy -c:a aac "
        "-avoid_negative_ts make_zero "
        "-shortest -y \"" +
        outputVideoPath + "\" 2>&1";

    if (const int result = system(ffmpegCmd.c_str()); result == 0) {
        std::remove(tempVideoPath.c_str());
        Logger::log(outputVideoPath, " successfully created with audio");
    } else {
        Logger::err("Warning: FFmpeg failed. Video saved without audio: ", tempVideoPath);
    }
}

void processVideoTransforms(
    const std::string& baseName,
    const std::string& inputPath
) {
    // Checking extension MP4
    if (is_mp4_file(inputPath)) {
        Logger::log("MP4 file detected → edge_detector_video");
        edge_detector_video(baseName, inputPath);
    }
    else {
        Logger::log("Non-MP4 file detected → colored_transformations");
        several_colors_transformations_streaming(baseName, inputPath);
        one_color_transformations_streaming(baseName, inputPath);
        reverse_transformations_by_proportion_streaming(baseName, inputPath);
    }
}
