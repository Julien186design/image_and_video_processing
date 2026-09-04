#include "ColorBandMask.h"
#include "EdgeDetector.h"
#include "ImageCreation.h"
#include "ColorConfig.h"
#include "CLD.h"


#include <opencv2/imgcodecs.hpp>
#include <utility>
#include <opencv2/imgproc.hpp>

#include "VideoCreation.h"


static void oneColorTransformations(
    const Image& baseImage,
    const std::string& baseName
) {
    if constexpr (!parameters::oneColor) { return; }
    const auto pipeline = OneColorPipeline::forStatic();

    constexpr int tol_min = std::get<0>(parameters::toleranceOneColor);
    constexpr int tol_max = std::get<1>(parameters::toleranceOneColor);
    constexpr std::array tol_values = { tol_min, tol_max };

    const size_t total_iterations = pipeline.configCount() * tol_values.size();

#pragma omp parallel default(none) \
    shared(baseImage, pipeline, baseName, total_iterations, tol_values)
    {
        std::string path;
        path.reserve(256);
        std::error_code ec;

#pragma omp for schedule(dynamic)
        for (size_t iter = 0; iter < 14; ++iter) {

            const size_t config_idx = iter / tol_values.size();
            const int tole          = tol_values.at(iter % tol_values.size());

            const auto params = pipeline.buildParams(config_idx);
            const auto weightedColors = params.weightedColors; // or std::get<0>(params) depending on buildParams() return type

            // Check whether all output images for this (config, tolerance) already exist.
            // If so, skip the computation entirely.
            static constexpr size_t expected = 5;
            bool all_exist = true;
            for (size_t idx = 0; idx < expected; ++idx) {
                path = OutputPathBuilder::image_one_color(baseName, weightedColors, tole, idx);
                if (!std::filesystem::exists(path, ec)) {
                    all_exist = false;
                    break;
                }
            }
            if (all_exist) { continue; }

            size_t idx = 0;
            baseImage.simplify_to_dominant_color_combinations(
                tole, &pipeline.configs.at(config_idx), {}, false, true,
                [&](Image&& result) {
                    path = OutputPathBuilder::image_one_color(
                        baseName, weightedColors, tole, idx);
                    if (!std::filesystem::exists(path, ec)) {
                        result.write(path.c_str());
                    }
                    return ++idx < expected;
                }
            );
        }
    }
}

static void complete_transformations_by_proportion(
    const Image& baseImage,
    const std::string& baseName
) {
    if constexpr (!parameters::complete_transformation_colors_by_proportion) { return; }

    const std::vector proportions = { 0.25f, 0.50f, 0.75f };

    auto apply = [](Image& img, const float proportion, const size_t transformIdx, const int colNua) -> bool {
        if (proportion <= 0.0f) { return false; }
        const auto [below, dark] = transformation_params.at(transformIdx);
        img.proportion_complete(proportion, colNua, dark, below);
        return true;
    };

    auto buildPath = [](const std::string& dir, const std::string& base,
                        const std::string& suffix, const float proportion, const int colNua) {
        return OutputPathBuilder::image_complete(dir, base, suffix, proportion, colNua);
    };

    constexpr int c0   = parameters::colorNuances.at(0);
    constexpr int c1   = parameters::colorNuances.at(1);
    constexpr int step = c0 != c1 ? c1 - c0 : 1;
    constexpr std::array twoColors = { c0, c1, step };

    run_transformations_by_proportion(
        baseImage, baseName, total_step_by_step_entries,
        apply, buildPath,
        &twoColors,
        &proportions
    );
}

static void partial_transformations_by_proportion(
    const Image& baseImage,
    const std::string& baseName,
    const std::vector<int> &rectangles
) {
    const std::vector<TransformationEntry> partialEntries =
        generatePartialEntries(total_step_by_step_entries);

    // Diagonal mode is indicated by the -1 sentinel at index 0
    const bool diagonal = !rectangles.empty() && rectangles.at(0) == -1;;

    // Capture rectangles, fraction, diagonal by value — safe across threads
    auto apply = [rectangles](Image& img, const float proportion, const size_t transformIdx, const int
        colNua) -> bool {
        if (proportion <= 0.0F) { return false;
}
        const auto [below, dark] = transformation_params.at(transformIdx);
        img.proportion_region_fraction(proportion, colNua, parameters::fraction, rectangles, dark, below);
        return true;
    };

    auto buildPath = [diagonal, rectangles](const std::string& dir, const std::string& base,
                                             const std::string& suffix, const float proportion, const int colNua) {
        return OutputPathBuilder::image_partial(dir, base, suffix, proportion, colNua, diagonal, rectangles);
    };

    run_transformations_by_proportion(
        baseImage, baseName, partialEntries,
        apply, buildPath,
        &parameters::colorNuances
    );
}

static void reverse_transformations_by_proportion(
    const Image& baseImage,
    const std::string& baseName
) {
    if constexpr (!parameters::totalReversal) { return; }

    ImageBuffer modified(baseImage.w, baseImage.h, baseImage.channels);
    modified.resetFrom(baseImage);
    modified.get().reverse_by_proportion(1.0F, true);
    modified.saveAs(
        OutputPathBuilder::image_reverse(baseName, 1.0F).c_str()
    );
}

static void edge_detector_image(
	const Image& baseImage,
	const std::string& baseName
) {
	Image img = baseImage;
	img.grayscale_avg();
    const size_t img_size = static_cast<size_t>(img.w) * img.h;

    std::vector<uint8_t> grayData(img_size);

    const std::span<const uint8_t> dataSpan(img.data, img_size * img.channels);

    for (size_t k = 0, src = 0; k < img_size; ++k, src += static_cast<size_t>(img.channels)) {
        grayData[k] = dataSpan[src];
    }

	EdgeDetectorPipeline pipeline(img.w, img.h);
	const std::vector<uint8_t>& rgb = pipeline.process(grayData.data());

	const Image gradient(img.w, img.h, 3);
	std::memcpy(gradient.data, rgb.data(), rgb.size());

	const std::string outputPath = OutputPathBuilder::image_edge_detector(baseName);
	gradient.write(outputPath.c_str());
}

// Independently regenerates the "several colors by proportion" final frame
// that several_colors_transformations_streaming (VideoCreation.cpp) used to
// write as the last frame of its video. Reuses the shared band-mask
// computation from ColorBandMask.h so the result is pixel-identical to that
// video's last frame, without generating any video.
void several_colors_final_image(
    const std::string& baseName,
    const std::string& inputPath
) {
    if constexpr (!parameters::complete_transformation_colors_by_proportion) {
        return;
    }

    const auto imageOpt = loadImage(inputPath);
    if (!imageOpt) {
        return;
    }

    const cv::Mat& baseImageMat = *imageOpt;

    constexpr int nFrames =
        parameters::numProportionSteps * parameters::numColorNuances;

    const std::vector<std::vector<bool>> pixelMask =
        computeColorBandMasks(baseImageMat);

    cv::Mat image = baseImageMat.clone();

    for (int bandIdx = 0;
         bandIdx < parameters::numProportionSteps;
         ++bandIdx) {
        applyColorToMask(
            image,
            pixelMask.at(bandIdx),
            finalColorForBand(bandIdx)
        );
         }

    const std::string outputImagePath =
        OutputPathBuilder::image_black_and_white(baseName, nFrames);

    if (!cv::imwrite(outputImagePath, image)) {
        Logger::err(
            "Error: Could not write final image to ",
            outputImagePath
        );
        return;
    }

    Logger::log(outputImagePath, " created");
}

static void coherent_line_drawing_image(
    const std::string& inputPath,
    const std::string& baseName
) {
    if constexpr (!parameters::coherentLineDrawing) { return; }

    const std::string outputPath = OutputPathBuilder::image_cld(baseName);
    if (std::filesystem::exists(outputPath)) { return; }

    CLD cld;
    cld.sigma_c = parameters::cld_sigma_c;
    cld.sigma_m = parameters::cld_sigma_m;
    cld.rho     = parameters::cld_rho;
    cld.tau     = parameters::cld_tau_final;

    cld.readSrc(inputPath);   // fait son propre imread grayscale + appelle etf.initial_ETF en interne

    for (int i = 0; i < parameters::cld_ETF_iter; ++i) {
        cld.etf.refine_ETF(parameters::cld_ETF_kernel);
    }

    cld.genCLD();
    for (int i = 0; i < parameters::cld_CLD_iter; ++i) {
        cld.combineImage();
        cld.genCLD();
    }

    cv::cvtColor(cld.result, cld.result, cv::COLOR_GRAY2RGB);
    if (!cv::imwrite(outputPath, cld.result)) {
        Logger::err("Error: could not write CLD output to ", outputPath);
    }
}

bool processImageTransforms(
    const std::string& baseName,
    const std::string& inputPath
) {

    if (is_mp4_file(inputPath)) {
        Logger::log("MP4 file detected, no transformation applied.");
        return false;
    }

    Logger::log(inputPath);

    const Image image(inputPath.c_str(), 0);

    edge_detector_image(image, baseName);
    coherent_line_drawing_image(inputPath, baseName);
    oneColorTransformations(image, baseName);
    complete_transformations_by_proportion(image, baseName);
    reverse_transformations_by_proportion(image, baseName);
    several_colors_final_image(baseName, inputPath);

    for (const bool useDiagonal : {false, true}) {
        if (useDiagonal ? !parameters::partialInDiagonal : !parameters::partial) {
            continue;
}

        const std::vector<int> rectangles = useDiagonal
            ? generateDiagonalRectangles()
            : decode_rectangles().second;

        partial_transformations_by_proportion(image, baseName, rectangles);
    }
    return true;

}