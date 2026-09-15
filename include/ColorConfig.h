#ifndef IMAGE_PROCESSING_COLORCONFIG_H
#define IMAGE_PROCESSING_COLORCONFIG_H

#include <mutex>
#include <vector>

#include "Image.h"
#include "ProcessingConfig.h"


inline auto generateColorConfigs(
    const bool binaryOnly) -> std::vector<std::vector<float>>
{
    constexpr int size = parameters::iterationsRGB;

    std::vector<std::vector<float>> configs;
    configs.reserve(size * size * size);

    auto toWeight = [](const int idx) -> float {
        return std::get<0>(parameters::weightOfRGB) +
               static_cast<float>(idx) * std::get<2>(parameters::weightOfRGB);
    };

    if (binaryOnly) {
        const std::array<std::array<float,3>, 7> corners = {{
            {0, 0, 1},
            {0, 1, 0},
            {1, 0, 0},
            {0, 1, 1},
            {1, 1, 0},
            {1, 0, 1},
            {1, 1, 1},
        }};
        for (const auto& [k, j, i] : corners) { 
            configs.push_back({ k, j, i });
}
        return configs;
    }

    for (int i = 0; i < parameters::iterationsRGB; ++i) {
        configs.push_back({ 1, 0, toWeight(i) });
    }

    for (int i = 1; i < parameters::iterationsRGB; ++i) {
        configs.push_back({ 1, toWeight(i), 1 });
    }

    return configs;
}

struct OneColorPipeline {
    const std::vector<std::vector<float>> configs;
    const std::vector<float> tValues;
    const std::vector<float> lastValues;

    static OneColorPipeline forStatic()    { return OneColorPipeline(true); }
    static OneColorPipeline forStreaming() { return OneColorPipeline(false);  }

    struct ConfigParams {
        std::string weightedColors;
    };

    [[nodiscard]] static size_t outputCount() {
        return parameters::numProportionSteps + 1;
    }

    [[nodiscard]] ConfigParams buildParams(const size_t configIdx) const {
        return { OutputPathBuilder::writingWeightedColors(configs.at(configIdx)) };
    }

    [[nodiscard]] size_t configCount() const { return configs.size(); }
    [[nodiscard]] size_t passCount()   const { return tValues.size(); }
    [[nodiscard]] const std::vector<float>& getTValues() const { return tValues; }
    [[nodiscard]] const std::vector<float>& getLastValues() const { return lastValues; }

    // passes = parameters::numProportionSteps (empty span triggers that path).
    [[nodiscard]] auto applyStatic(
        const Image& img, const int tolerance, const size_t configIdx
    ) const -> std::vector<Image> {
        std::vector<Image> out;
        out.reserve(outputCount());
        img.simplify_to_dominant_color_combinations(
            tolerance, &configs.at(configIdx), {}, false, true,
            [&out](Image&& result) {
                out.push_back(std::move(result));
                return true; // always continue
            }
        );
        return out;
    }

    [[nodiscard]] std::vector<Image> applyStreaming(
        const Image& img, const int tolerance, const size_t configIdx
    ) const {
        std::vector<Image> out;
        out.reserve(tValues.size());
        img.simplify_to_dominant_color_combinations(
            tolerance, &configs.at(configIdx), tValues, true, true,
            [&out](Image&& result) {
                out.push_back(std::move(result));
                return true;
            }
        );
        return out;
    }

private:
    mutable std::once_flag tValuesPrinted;

    explicit OneColorPipeline(const bool binaryOnly)
        : configs(generateColorConfigs(binaryOnly))
        , tValues(buildTValues())
        , lastValues(fromColorsToOriginal())
    {}

    static std::vector<float> buildTValues() {
        std::vector<float> values;
        const auto [from, to, step] = parameters::passesRGB;
        const int steps  = static_cast<int>((to - from) / step);

        for (int i = 0; i <= steps; ++i) {
            const float tVal = from + static_cast<float>(i) * step;
            values.push_back(std::clamp(tVal, std::min(from, to), std::max(from, to)));
        }
        return values;
    }

    static std::vector<float> fromColorsToOriginal() {
        std::vector<float> values;

        for (int i = 0; i < parameters::transition_to_the_original; ++i) {
            values.push_back(static_cast<float>(i) / static_cast<float>(parameters::transition_to_the_original - 1));
        }
        return values;
    }
};

#endif //IMAGE_PROCESSING_COLORCONFIG_H