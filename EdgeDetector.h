#ifndef IMAGE_PROCESSING_EDGEDETECTOR_H
#define IMAGE_PROCESSING_EDGEDETECTOR_H

#include "Image.h"

#include <cstring>
#include <vector>

class EdgeDetectorPipeline {
    const int width;
    const int height;
    const size_t imgSize;
    const double threshold;

    // 3x3 Gaussian kernel (normalized by 1/16)
    static constexpr double inv16 = 1.0 / 16.0;
    double gaussKernel[9] = {
        inv16, 2 * inv16, inv16,
        2 * inv16, 4 * inv16, 2 * inv16,
        inv16, 2 * inv16, inv16
    };

    // Internal working buffers (allocated once)
    std::vector<double> blurData;
    std::vector<double> tx, ty;
    std::vector<double> gx, gy;
    std::vector<double> g;
    std::vector<uint8_t> outputRGB;

    // Persistent grayscale image buffer (avoids per-frame allocation)
    Image tempImg;

public:

    // Constructor: allocate all buffers once
    EdgeDetectorPipeline(const int width, const int height, const double thresh = 0.09)
        : width(width),
          height(height),
          imgSize(static_cast<size_t>(width) * height),
          threshold(thresh),
          blurData(imgSize),
          tx(imgSize),
          ty(imgSize),
          gx(imgSize),
          gy(imgSize),
          g(imgSize),
          outputRGB(imgSize * 3),
          tempImg(width, height, 1)
    {}

    const std::vector<uint8_t>& process(const uint8_t* grayData)
    {
        // Copy grayscale input into persistent image buffer
        std::memcpy(tempImg.data, grayData, imgSize);

        // Apply Gaussian blur in-place
        tempImg.convolve_linear(0, 3, 3, gaussKernel, 1, 1);

        // -------------------------
        // Stage 1: Normalize to double
        // -------------------------
        #pragma omp parallel for schedule(static) default(none) \
    		shared(blurData)
        for (size_t k = 0; k < imgSize; ++k) {
            blurData.at(k) = tempImg.data[k] / 255.0;
}

        // -------------------------
        // Stage 2: Horizontal derivative pass
        // -------------------------
		#pragma omp parallel for collapse(2) schedule(static) default(none) \
			shared(blurData, tx, ty)
        for (int r = 0; r < height; ++r)
        {
            for (int c = 1; c < width - 1; ++c)
            {
                const size_t idx = static_cast<size_t>(r) * width + c;

                tx.at(idx) = blurData.at(idx + 1) - blurData.at(idx - 1);

                ty.at(idx) = 47.0 * blurData.at(idx + 1)
                        + 162.0 * blurData.at(idx)
                        + 47.0 * blurData.at(idx - 1);
            }
        }

        // -------------------------
        // Stage 3: Vertical derivative pass
        // -------------------------
		#pragma omp parallel for collapse(2) schedule(static) default(none) \
			shared(tx, ty, gx, gy)
        for (int c = 1; c < width - 1; ++c)
        {
            for (int r = 1; r < height - 1; ++r)
            {
                const size_t idx = static_cast<size_t>(r) * width + c;

                gx.at(idx) = 47.0  * tx.at(static_cast<size_t>(r + 1) * width + c)
                           + 162.0 * tx.at(static_cast<size_t>(r)     * width + c)
                           + 47.0  * tx.at(static_cast<size_t>(r - 1) * width + c);

                gy.at(idx) = ty.at(static_cast<size_t>(r + 1) * width + c)
                           - ty.at(static_cast<size_t>(r - 1) * width + c);
            }
        }

        // -------------------------
        // Stage 4: Gradient magnitude and orientation
        // -------------------------
		#pragma omp parallel for schedule(static) default(none) \
			shared(g, gx, gy)
        for (size_t k = 0; k < imgSize; ++k)
        {
            g[k] = std::sqrt(gx[k] * gx[k] + gy[k] * gy[k]);
        }

        // -------------------------
        // Stage 5: Parallel min/max reduction (portable version)
        // -------------------------
        double mx = -std::numeric_limits<double>::infinity();
        double mn =  std::numeric_limits<double>::infinity();

		#pragma omp parallel default(none) shared(g, mx, mn)
        {
            double local_max = -std::numeric_limits<double>::infinity();
            double local_min =  std::numeric_limits<double>::infinity();

            #pragma omp for nowait
            for (size_t k = 0; k < imgSize; ++k)
            {
                if (g[k] > local_max) local_max = g[k];
                if (g[k] < local_min) local_min = g[k];
            }

            #pragma omp critical
            {
                if (local_max > mx) mx = local_max;
                if (local_min < mn) mn = local_min;
            }
        }

        // -------------------------
        // Stage 6: HSL → RGB mapping
        // -------------------------
		#pragma omp parallel for schedule(static) default(none) \
			shared(g, outputRGB) firstprivate(mx, mn)
        for (size_t k = 0; k < imgSize; ++k)
        {
        	const double h = std::atan2(gy[k], gx[k]) * 180.0 / M_PI + 180.0;
            double v = (mx == mn) ? 0.0 : (g[k] - mn) / (mx - mn);
            v = (v > threshold) ? v : 0.0;

            const double s = v;
            const double l = v;

            const double c = (1.0 - std::fabs(2.0 * l - 1.0)) * s;
            const double x = c * (1.0 - std::fabs(std::fmod(h / 60.0, 2.0) - 1.0));
            const double m = l - c / 2.0;

            double rt = 0.0, gt = 0.0, bt = 0.0;

            if      (h < 60.0)  { rt = c; gt = x; }
            else if (h < 120.0) { rt = x; gt = c; }
            else if (h < 180.0) { gt = c; bt = x; }
            else if (h < 240.0) { gt = x; bt = c; }
            else if (h < 300.0) { bt = c; rt = x; }
            else                { bt = x; rt = c; }

            outputRGB[k * 3    ] = static_cast<uint8_t>(255.0 * (rt + m));
            outputRGB[k * 3 + 1] = static_cast<uint8_t>(255.0 * (gt + m));
            outputRGB[k * 3 + 2] = static_cast<uint8_t>(255.0 * (bt + m));
        }

        return outputRGB;
    }
};

#endif //IMAGE_PROCESSING_EDGEDETECTOR_H