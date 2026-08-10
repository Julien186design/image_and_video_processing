#ifndef IMAGE_PROCESSING_IMAGECREATION_H
#define IMAGE_PROCESSING_IMAGECREATION_H

#include "ImageBuffer.h"

#include <string>

template<typename ApplyFunc, typename PathFunc>
void run_transformations_by_proportion(
    const Image& baseImage,
    const std::string& baseName,
    const std::vector<TransformationEntry>& entries,
    ApplyFunc apply,
    PathFunc buildPath,
    const std::array<int, 3>* colorNuances,
    // Optional override: if non-null, iterate over these proportions
    // instead of the range derived from parameters::proportions.
    const std::vector<float>* proportions_override = nullptr
) {
    const int num_threads = computeNumThreads();

    // Build the proportion sequence once, before the parallel region.
    std::vector<float> proportion_values;
    if (proportions_override) {
        proportion_values = *proportions_override;
    } else {
        proportion_values.reserve(parameters::numProportionSteps);
        for (size_t p = 0; p < static_cast<size_t>(parameters::numProportionSteps); ++p) {
            proportion_values.push_back(
                parameters::proportions[0] + static_cast<float>(p) * parameters::proportions[2]
            );
        }
    }

    auto process_one = [&](const float proportion, const size_t i, const int colNua) {
        const auto& [suffix, output_dir] = entries[i];
        ImageBuffer modified(baseImage.w, baseImage.h, baseImage.channels);
        modified.resetFrom(baseImage);
        if (!apply(modified.get(), proportion, i, colNua)) { return; }
        modified.saveAs(buildPath(output_dir, baseName, suffix, proportion, colNua).c_str());
    };

#pragma omp parallel for schedule(dynamic) num_threads(num_threads) \
    default(none) \
    shared(baseImage, baseName, entries, colorNuances, apply, buildPath, \
           process_one, proportion_values)
    for (float proportion : proportion_values) {
        for (size_t i = 0; i < entries.size(); ++i) {
            if (colorNuances) {
                for (int col_nua = (*colorNuances)[0];
                     col_nua <= (*colorNuances)[1];
                     col_nua += (*colorNuances)[2]) {
                    process_one(proportion, i, col_nua);
                }
            } else {
                process_one(proportion, i, 0);
            }
        }
    }
}

bool processImageTransforms(
    const std::string& baseName,
    const std::string& inputPath
);

#endif //IMAGE_PROCESSING_IMAGECREATION_H