#include "ProcessingConfig.h"
#include "ImageCreation.h"
#include "VideoProcessing.h"

#include <chrono>



// Definitions of variables declared as `extern` in the header
const std::string folder_colors_nuances          = std::string(folder_output) + "Colors nuances/";
const std::string folder_videos       = std::string(folder_output) + "Videos/";
const std::string folder_edgedetector = std::string(folder_output) + "Edge Detector/";
const std::string folder_onecolor     = std::string(folder_output) + "One Color/";
const std::string folder_cld       = std::string(folder_output) + "CLD/";

const std::vector<TransformationEntry> total_step_by_step_entries = {
    { .suffix = "BTB", .output_dir = std::string(folder_output) + "BTB/" },
    { .suffix = "BTW", .output_dir = std::string(folder_output) + "BTW/" },
    { .suffix = "WTB", .output_dir = std::string(folder_output) + "WTB/" },
    { .suffix = "WTW", .output_dir = std::string(folder_output) + "WTW/" },
};


const std::vector<TransformationEntry> total_black_and_white_entries = {
    { .suffix = "Black and white - Original", .output_dir = std::string(folder_output) + "Original black and white/" },
    { .suffix = "Black and white - Reversed", .output_dir = std::string(folder_output) + "Reversed black and white/" },
};

void image_and_video_processing(const std::string& inputFile)
{
    const auto [baseName, inputPath] = extractImageInfo(inputFile);

    const ProgressNotifier notifier("Process Transform", 2);

    auto stepStart = std::chrono::steady_clock::now();

    if (processImageTransforms(baseName, inputPath)) {
        notifier.notifyStep("processImageTransforms", stepStart);
    }

    stepStart = std::chrono::steady_clock::now();
    processVideoTransforms(baseName, inputPath);
    notifier.notifyStep("processVideoTransforms", stepStart);
}
