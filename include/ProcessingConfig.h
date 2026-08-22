#ifndef IMAGE_PROCESSING_PROCESSINGCONFIG_H
#define IMAGE_PROCESSING_PROCESSINGCONFIG_H

#define THREAD_OFFSET 1
#define TOLERANCE_RAM 48

#include <string>
#include <utility>
#include <vector>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <omp.h>
#include <array>
#include <iostream>
#include <filesystem>

struct parameters {
    static constexpr std::array<float, 3> proportions = {0.F, 1.F, .05F};
    static constexpr std::array<int, 3> colorNuances = {80, 250, 5}; // {first colorNuance, last colorNuance, step}
    static constexpr std::array<int, 2> frames = {0, 0};
    static constexpr int fps = 10;
    static constexpr int fraction = 2;
    static constexpr std::array<int, 2> rectangles = {40, 63};
    static constexpr std::array<int, 3> toleranceOneColor = {0, 40, 1};
    static constexpr std::array<float, 3> weightOfRGB = {0.F, 1.F, .01F};
    static constexpr std::array<float, 3> passesRGB = {1.F, 1.F, 1.F}; // 0 = original image, 1 = colored image
    static constexpr bool complete_transformation_colors_by_proportion = false;
    static constexpr bool oneColor = false;
    static constexpr bool totalReversal = false;
    static constexpr bool partial = false;
    static constexpr bool partialInDiagonal = false;
    static constexpr int numProportionSteps =
        static_cast<int>((std::get<1>(proportions) - std::get<0>(proportions)) / std::get<2>(proportions)) + 1;

    static constexpr int numColorNuances =
        (std::get<1>(colorNuances) - std::get<0>(colorNuances)) / std::get<2>(colorNuances) + 1;

    static constexpr int numTolerance =
        (std::get<1>(toleranceOneColor) - std::get<0>(toleranceOneColor)) / std::get<2>(toleranceOneColor) + 1;

    static constexpr int iterationsRGB =
        static_cast<int>((std::get<1>(weightOfRGB) - std::get<0>(weightOfRGB)) / std::get<2>(weightOfRGB)) + 1;

    static constexpr int transition_to_the_original = 4 * fps;

};

inline const std::string folder_output = "Output/";

struct ThresholdParams {
    bool below;
    bool dark;
};

constexpr std::array<std::string_view, 2> reversal_suffixes = {
    "Reversal-BT",
    "Reversal-WT",
};

// ─── Transformation entry: groups a suffix with its output directory ──────────

struct TransformationEntry {
    std::string suffix;
    std::string output_dir;

    // Returns the "partial/square" variant of this entry
    [[nodiscard]] auto partial() const -> TransformationEntry {
        std::string dir = output_dir;
        if (!dir.empty() && dir.back() == '/') {
            dir.pop_back();
}
        return { .suffix=suffix + " Partial", .output_dir=dir + " Square/" };
    }
};

// ─── Threshold parameter sets ─────────────────────────────────────────────────

constexpr std::array<ThresholdParams, 4> transformation_params = {{
    {.below = true,  .dark = true },   // BelowDark
    {.below = true,  .dark = false},   // BelowLight
    {.below = false, .dark = true },   // AboveDark
    {.below = false, .dark = false}    // AboveLight
}};



extern const std::string folder_colors_nuances;
extern const std::string folder_videos;
extern const std::string folder_edgedetector;
extern const std::string folder_onecolor;


extern const std::vector<TransformationEntry> total_step_by_step_entries;
extern const std::vector<TransformationEntry> reversal_step_by_step_entries;
extern const std::vector<TransformationEntry> total_black_and_white_entries;

void image_and_video_processing(const std::string& inputFile);

inline void sendNotification(const std::string& title, const std::string &message)
{
    const std::string command =
        "notify-send '" + title + "' '" + message + "'";

    (void)std::system(command.c_str());
}

inline std::vector<int> generateDiagonalRectangles() {
    if constexpr (parameters::fraction <= 0) { return {};
}
    constexpr int taille = 1U << parameters::fraction;  // 2^fraction via bit-shift
    constexpr int pas = taille + 1;

    std::vector<int> vector;
    vector.reserve(taille);

    vector.push_back(-1);
    for (int i = 0; i < taille; ++i) {
        vector.push_back(i * pas);
    }
    return vector;
}

inline auto decode_rectangles() -> std::pair<bool, std::vector<int>> {
    if constexpr (parameters::rectangles.empty()) { return {false, {}};
}

    if constexpr (parameters::rectangles.at(0) == -1) {
        // Diagonal case: rectangles are explicitly listed after the flag
        return {true, std::vector<int>(parameters::rectangles.begin() + 1, parameters::rectangles.end())};
    }

    // Non-diagonal case: input is {start, end}, expand to range
    if constexpr (parameters::rectangles.size() != 2) { return {false, {}};
}
    constexpr int start = std::min(parameters::rectangles.at(0), parameters::rectangles.at(1));
    constexpr int end   = std::max(parameters::rectangles.at(0), parameters::rectangles.at(1));

    std::vector<int> result;
    result.reserve(end - start + 1);
    for (int i = start; i <= end; ++i) {
        result.push_back(i);
}
    return {false, result};
}



inline bool is_mp4_file(const std::string& path) {
    const auto ext = std::filesystem::path(path).extension().string();
    std::string lower = ext;
    std::ranges::transform(lower, lower.begin(),
        [](const unsigned char c) { return std::tolower(c); });
    return lower == ".mp4";
}

inline int computeNumThreads() {
    return std::max(1, omp_get_max_threads() - THREAD_OFFSET);
}

// ─── Partial variant generation ───────────────────────────────────────────────

inline std::vector<TransformationEntry> generatePartialEntries(
    const std::vector<TransformationEntry>& entries)
{
    std::vector<TransformationEntry> result;
    result.reserve(entries.size());
    std::ranges::transform(entries, std::back_inserter(result),
        [](const TransformationEntry& e) { return e.partial(); });
    return result;
}

class OutputPathBuilder {
public:
    // Formats a float with up to 2 decimal places, no trailing zeros
    static std::string formatProportion(const float value) {
        char buf[16];
        int len = std::snprintf(buf, sizeof(buf), "%.4f", value);
        while (len > 0 && buf[len - 1] == '0') { --len;
}
        if (len > 0 && buf[len - 1] == '.') { --len;
}
        return {buf, static_cast<size_t>(len)};
    }

    // Builds a standard path prefix
    static auto buildStandard(const std::string& outputDir,
                                     const std::string& baseName,
                                     const std::string& suffix,
                                     const float proportion) -> std::string {
        return std::format(
            "{}{} - {} {} % ", outputDir, baseName, suffix, formatProportion(100.F * proportion));
    }

    // Simple image paths
    static std::string image_edge_detector(const std::string& baseName) {
        return std::format("{}{} - GT.png", folder_edgedetector, baseName);
    }

    static auto image_one_color(const std::string& baseName,
                                    const std::string& colors, int tolerance, size_t idx) -> std::string {
        return std::format("{}{} - Tolerance {} - {} - {}-2.png",
                           folder_onecolor,
                           baseName,
                           tolerance,
                           colors,
                           idx);
    }

    // Complete / reverse / partial paths
    static auto image_complete(const std::string& outputDir,
                                const std::string& baseName,
                                const std::string& suffix,
                                const float proportion,
                                int colorNuance) -> std::string {
        return buildStandard(outputDir, baseName, suffix, proportion) + std::format("- CN {}.png", colorNuance);
    }

    static auto image_reverse(const std::string& baseName,
                               const float proportion) -> std::string {
        return buildStandard(std::string(folder_output) + "Reversal/", baseName, "Reversal", proportion) + ".png";
    }

    static std::string image_partial(const std::string& outputDir,
                               const std::string& baseName,
                               const std::string& suffix,
                               const float proportion,
                               int colorNuance,
                               const bool diagonal,
                               const std::vector<int>& rectangles) {
        const std::string rectInfo = diagonal
            ? std::format("- {} Squares", rectangles.size() - 1)
            : std::format("- Rectangles {} - {}", rectangles.front(), rectangles.back());
        return buildStandard(outputDir, baseName, suffix, proportion) + rectInfo + std::format(
                   " - CN {}.png", colorNuance);
    }

    static std::string image_black_and_white(const std::string& baseName, int nFrames) {
        return std::format("{}{} - {} images - {{{}-{}-{}}} - {{{}-{}-{}}} last frame.png",
                           folder_colors_nuances, baseName, nFrames,
                           formatProportion(parameters::proportions.at(0)),
                           formatProportion(parameters::proportions.at(1)),
                           formatProportion(parameters::proportions.at(2)),
                           parameters::colorNuances.at(0),
                           parameters::colorNuances.at(1),
                           parameters::colorNuances.at(2));
    }

    // Video paths
    static std::string video_several_colors(const std::string& baseName, int nFrames) {
        return std::format("{}{} - {} images - {} fps - {{{}-{}-{}}} - {{{}-{}-{}}}.mp4",
                           folder_colors_nuances, baseName, nFrames, parameters::fps,
                           formatProportion(parameters::proportions.at(0)),
                           formatProportion(parameters::proportions.at(1)),
                           formatProportion(parameters::proportions.at(2)),
                           parameters::colorNuances.at(0),
                           parameters::colorNuances.at(1),
                           parameters::colorNuances.at(2));
    }

    static std::string video_edge_detector(const std::string& baseName,
                                           int framesToProcess,
                                           const int totalFrames,
                                           const double fps) {
        std::string range = framesToProcess == totalFrames ? " - " :
                            std::format("{{{}-{}}} ", parameters::frames.at(0), parameters::frames.at(1));
        return std::format("{}{} - {} frames {}{} fps.mp4",
                           folder_edgedetector, baseName, framesToProcess, range,
                           formatProportion(static_cast<float>(fps)));
    }

    static std::string video_edge_detector_temp(const std::string& baseName) {
        return std::format("{}{}_temp.mp4", folder_edgedetector, baseName);
    }

    static std::string video_one_color(const std::string& baseName, size_t nFrames) {
        return std::format("{}{} - {} images - {} fps - {} - {} - {}.mp4",
                                       folder_onecolor, baseName, nFrames,
                                       parameters::fps,
                                       formatToleranceColors(parameters::toleranceOneColor),
                                       writingWeightedColors(parameters::weightOfRGB),
                                       writingWeightedColors(parameters::passesRGB));
    }

    static std::string video_reversal(
        const std::string& baseName,
        const std::string_view suffix,
        const int nFrames
    ) {
        return std::format("{}{} - {} - {} images - {} fps.mp4",
                           std::string(folder_output) + "Reversal/", baseName, suffix,
                           nFrames, parameters::fps);
    }

    static std::string formatToleranceColors(const std::span<const int> toleranceOneColor) {
        std::string string = "{";
        for (size_t i = 0; i < 3; ++i) {
            if (i > 0) { string += '-';
}
            string += std::to_string(toleranceOneColor[i]);

        }
        string += '}';
        return string;
    }
    // Weighted RGB as "{r-g-b}"
    static std::string writingWeightedColors(const std::span<const float> weightOfRGB) {
        if (weightOfRGB.size() < 3) {
            throw std::invalid_argument("weightOfRGB must contain at least 3 elements");
        }

        if (weightOfRGB[0] == weightOfRGB[1] && weightOfRGB[1] == weightOfRGB[2]) {
            return formatProportion(weightOfRGB[0]);
        }

        std::string string = "{";
        for (size_t i = 0; i < 3; ++i) {
            if (i > 0) { string += '-';
}
            string += formatProportion(weightOfRGB[i]);
        }
        string += '}';
        return string;
    }
};

class ProgressNotifier {
public:
    ProgressNotifier(std::string title, const std::size_t totalWork)
        : title_(std::move(title)),
          total_(totalWork),
          start_(std::chrono::steady_clock::now())
    {}

    ~ProgressNotifier() {
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start_).count();

        const std::string message =
            "100% completed\n"
            "Total time: " + std::to_string(elapsed) + " s";
    }

    void update(const std::size_t current) {
        const double ratio =
            static_cast<double>(current) / static_cast<double>(total_);
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start_).count();

        for (std::size_t i = 0; i < kMilestones.size(); ++i) {
            if (ratio >= kMilestones[i] / 100.0)
                notifyOnce(milestonesSent_[i], kMilestones[i], elapsed);
        }
    }

    void notifyStep(const std::string& stepName,
                    const std::chrono::steady_clock::time_point stepStart) const {
        const auto duration = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - stepStart).count();

        const std::string message =
            stepName + " completed\n"
            "Duration: " + std::to_string(duration) + " s";

        sendNotification(title_, message);
    }

private:
    void notifyOnce(std::atomic<bool>& flag, const int percent, const long elapsed) const {
        if (bool expected = false; flag.compare_exchange_strong(expected, true)) {
            const std::string message =
                std::to_string(percent) + "% completed\n"
                "Elapsed: " + std::to_string(elapsed) + " s";
            sendNotification(title_, message);
        }
    }

    static constexpr std::array<int, 4> kMilestones = {20, 40, 60, 80};

    std::string title_;
    std::size_t total_;
    std::array<std::atomic<bool>, kMilestones.size()> milestonesSent_{};
    std::chrono::steady_clock::time_point start_;
};

class Logger {
public:
    // Variadic template: handles any number/type of arguments
    template<typename... Args>
    static void log(Args&&... args) {
        (std::cout << ... << std::forward<Args>(args)) << std::endl;
    }

    // Overload with '\r' + flush for progress lines (like frame counter)
    template<typename... Args>
    static void logProgress(Args&&... args) {
        (std::cout << ... << std::forward<Args>(args)) << "\r";
        std::cout.flush();
    }

    template<typename... Args>
    static void err(Args&&... args) {
        (std::cerr << ... << std::forward<Args>(args)) << std::endl;
    }
};

#endif //IMAGE_PROCESSING_PROCESSINGCONFIG_H