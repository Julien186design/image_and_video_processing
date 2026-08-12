#ifndef IMAGE_PROCESSING_VIDEOCREATION_H
#define IMAGE_PROCESSING_VIDEOCREATION_H

#include <string>

void reverse_transformations_by_proportion_streaming(
    const std::string& baseName,
    const std::string& inputPath
);

void processVideoTransforms(
    const std::string& baseName ,
    const std::string& inputPath
);

#endif //IMAGE_PROCESSING_VIDEOCREATION_H