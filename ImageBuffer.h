#ifndef IMAGE_PROCESSING_IMAGEBUFFER_H
#define IMAGE_PROCESSING_IMAGEBUFFER_H

#include "Image.h"
#include <cassert>
#include <cstring>

class ImageBuffer {
    Image buffer;
public:
    ImageBuffer(const int w, const int h, const int channels) : buffer(w, h, channels) {}

    void resetFrom(const Image& source) {
        assert(buffer.size == source.size);
        std::memcpy(buffer.data, source.data, buffer.size);
    }

    Image& get() { return buffer; }

    void saveAs(const char* path) { buffer.write(path); }
};

#endif //IMAGE_PROCESSING_IMAGEBUFFER_H