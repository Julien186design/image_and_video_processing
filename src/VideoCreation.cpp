#include "ColorBandMask.h"
#include "ColorConfig.h"
#include "EdgeDetector.h"
#include "ImageCreation.h"
#include "VideoCreation.h"
#include "CLD.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>

#include <unistd.h>
#include <sys/wait.h>
#include <vector>

[[nodiscard]]
static std::string format_fixed2(const double value) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << value;
    return oss.str();
}

// Runs ffmpeg via fork+execvp: no shell, so no injection vector.
// Returns the exit code of the child process, or -1 in case of fork/exec failure.
[[nodiscard]]
static int run_ffmpeg_merge(
    const std::string& tempVideoPath,
    const std::string& inputVideoPath,
    const std::string& outputVideoPath,
    const double startTime,
    const double duration
) {
    std::vector<std::string> args = {
        "ffmpeg",
        "-fflags", "+genpts",
        "-i", tempVideoPath,
        "-ss", format_fixed2(startTime),
        "-i", inputVideoPath,
        "-t", format_fixed2(duration),
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "aac",
        "-avoid_negative_ts", "make_zero",
        "-shortest",
        "-y",
        outputVideoPath
    };

    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    for (auto& a : args) argv.push_back(a.data()); // execvp exige char*, pas const char*
    argv.push_back(nullptr);

    const pid_t pid = fork();
    if (pid < 0) {
        Logger::err("Error: fork() failed");
        return -1;
    }

    if (pid == 0) {
        // Fils : fusionne stderr dans stdout (équivalent du "2>&1" du system() original)
        dup2(STDOUT_FILENO, STDERR_FILENO);
        execvp("ffmpeg", argv.data());
        // execvp only returns a result if it fails (ffmpeg not in the PATH, etc.)
        _exit(127);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        Logger::err("Error: waitpid() failed");
        return -1;
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

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
        OutputPathBuilder::video_one_color(baseName, totalFrames);

    cv::VideoWriter writer;
    writer.open(outputVideoPath,
                cv::VideoWriter::fourcc('m','p','4','v'),
                parameters::fps,
                cv::Size(baseImageMat.cols, baseImageMat.rows));
    if (!writer.isOpened()) {
        Logger::err("Error: Could not open video writer");
        return;
    }

    const size_t num_threads = computeNumThreads();

    std::vector<Image> thread_imgs;
    thread_imgs.reserve(num_threads);
    for (size_t t = 0; t < num_threads; ++t) {
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
    const size_t channels = baseImageMat.channels();
    const size_t num_threads = computeNumThreads();

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

                const size_t tid = omp_get_thread_num();
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
                const size_t local_phase = phase - phase_base;

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

    constexpr int nFrames = parameters::numProportionSteps * parameters::numColorNuances;

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

    // Per-pixel band membership derived from RGB-sum rank. Shared with the
    // standalone final-image generator in ImageCreation.cpp (see
    // ColorBandMask.h) so both produce the exact same band assignment.
    const std::vector<std::vector<bool>> pixelMask = computeColorBandMasks(baseImageMat);

    VideoWriterQueue queue(video);

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

            applyColorToMask(
                accumulated,
                mask,
                static_cast<uint8_t>(colorNuance)
            );

            queue.enqueue(accumulated.clone());
             }

        // Locks in the final colour
        applyColorToMask(
            accumulated,
            mask,
            static_cast<uint8_t>(end)
        );
    };

    bool reverseOrder = false;
    for (int i = 0; i < parameters::numProportionSteps; ++i) {
        process(i, reverseOrder);
        reverseOrder = !reverseOrder;
    }

    queue.finish();
    video.release();

    // Final-frame image creation is handled independently by
    // several_colors_final_image (ImageCreation.cpp).
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
    const int endFrame = parameters::frames.at(1) == 0 || parameters::frames.at(1) > totalFrames
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
            for (size_t i = 0; i < imgSize; ++i) {
                const uint8_t* pixel = frameBGR.data + i * 3;
                grayData.at(i) = static_cast<uint8_t>((pixel[0] + pixel[1] + pixel[2]) / 3);
            }

        // Process frame through edge detection pipeline (zero allocation)
        const std::vector<uint8_t>& rgb = pipeline.process(grayData.data());

        // RGB -> BGR conversion for OpenCV compatibility
        #pragma omp parallel for default(none) shared(imgSize, frameBGR, rgb)
            for (size_t i = 0; i < imgSize; ++i) {
                frameBGR.data[static_cast<ptrdiff_t>(i * 3)]     = rgb.at(i * 3 + 2); // B
                frameBGR.data[static_cast<ptrdiff_t>(i * 3 + 1)] = rgb.at(i * 3 + 1); // G
                frameBGR.data[static_cast<ptrdiff_t>(i * 3 + 2)] = rgb.at(i * 3);     // R
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

    if (const int result = run_ffmpeg_merge(tempVideoPath, inputVideoPath, outputVideoPath, startTime, duration); result == 0) {
        std::remove(tempVideoPath.c_str());
        Logger::log(outputVideoPath, " successfully created with audio");
    } else {
        Logger::err("Warning: FFmpeg failed. Video saved without audio: ", tempVideoPath);
    }
}

static void coherent_line_drawing_streaming(
    const std::string& baseName,
    const std::string& inputPath
) {
    if constexpr (!parameters::coherentLineDrawing) { return; }

    const auto [tau_min, tau_max, tau_step] = parameters::cld_tau_range;
    const int nFrames = static_cast<int>(std::round((tau_max - tau_min) / tau_step)) + 1;

    if (nFrames < parameters::fps) {
        Logger::err("Video less than 1 second long, creation canceled for CLD");
        return;
    }

    CLD cld;
    cld.sigma_c = parameters::cld_sigma_c;
    cld.sigma_m = parameters::cld_sigma_m;
    cld.rho     = parameters::cld_rho;

    cld.readSrc(inputPath);
    for (int i = 0; i < parameters::cld_ETF_iter; ++i) {
        cld.etf.refine_ETF(parameters::cld_ETF_kernel);
    }
    cld.genCLD();  // remplit cld.FDoG une bonne fois ; le cld.result produit ici est jeté

    const std::string outputVideoPath = OutputPathBuilder::video_cld(baseName, nFrames);
    auto writerOpt = make_video_writer(outputVideoPath, cld.originalImg.cols, cld.originalImg.rows, parameters::fps);
    if (!writerOpt) { return; }
    cv::VideoWriter& writer = *writerOpt;

    VideoWriterQueue queue(writer);
    cv::Mat frame(cld.FDoG.size(), CV_8UC1);   // pré-allouée : binaryThresholding ne l'alloue pas elle-même

    for (int i = 0; i < nFrames; ++i) {
        const double tau = tau_min + static_cast<double>(i) * tau_step;
        cld.binaryThresholding(cld.FDoG, frame, tau);
        cv::Mat frameBGR;
        cv::cvtColor(frame, frameBGR, cv::COLOR_GRAY2BGR);
        queue.enqueue(std::move(frameBGR));
    }

    queue.finish();
    writer.release();
    Logger::log(outputVideoPath, " created");
}


void processVideoTransforms(
    const std::string& baseName,
    const std::string& inputPath
) {

    if (is_mp4_file(inputPath)) {
        Logger::log("MP4 file detected → edge_detector_video");
        edge_detector_video(baseName, inputPath);
    }
    else {
        Logger::log("Non-MP4 file detected → colored_transformations");
        several_colors_transformations_streaming(baseName, inputPath);
        several_colors_final_image(baseName, inputPath);
        one_color_transformations_streaming(baseName, inputPath);
        reverse_transformations_by_proportion_streaming(baseName, inputPath);
        coherent_line_drawing_streaming(baseName, inputPath);
    }
}