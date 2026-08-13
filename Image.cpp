#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define BYTE_BOUND(value) value < 0 ? 0 : (value > 255 ? 255 : value)

/*
 *CLion on Linux Mint
 *Run/Edit Configurations
 *Working directory : $PROJECT8DIR$ | /[...]/CLion Projects/Image Processing

*/
#include "Image.h"

#include <stb_image.h>
#include <stb_image_write.h>
#include <cstdio>
#include <algorithm>
#include <cmath>     // for std::round
#include <ranges>
#include <vector>
#include <optional>
#include <array>

Image::Image(const char* filename, const int channel_force) : channels(0) {
	if (read(filename, channel_force)) {
		Logger::log("Read ", filename);
		size = w * h * channels;
	} else {
		Logger::err("Failed to read ", filename);
	}
}

Image::Image(const int w, const int h, const int channels)
	: w(w), h(h), channels(channels) {
	size = w * h * channels;
	data = new uint8_t[size];
	memset(data, 0, size);
}

Image::Image(const Image& img) : Image(img.w, img.h, img.channels) {
	memcpy(data, img.data, size);
}

Image::~Image() {
	// stbi_image_free(data);
	delete[] data;

}

bool Image::read(const char* filename, const int channel_force) {
	data = stbi_load(filename, &w, &h, &channels, channel_force);
	channels = channel_force == 0 ? channels : channel_force;
	return data != nullptr; // used to be NULL
}

bool Image::write(const char* filename) const {
	const ImageType type = get_file_type(filename);
	int success = 0;
	switch (type) {
	case PNG:
		success = stbi_write_png(filename, w, h, channels, data, w * channels);
		break;
	case BMP:
		success = stbi_write_bmp(filename, w, h, channels, data);
		break;
	case JPG:
		success = stbi_write_jpg(filename, w, h, channels, data, 100);
		break;
	case TGA:
		success = stbi_write_tga(filename, w, h, channels, data);
		break;
	}
	if (success != 0) {
		// Green: "Wrote", Cyan: filename, then image dimensions (width, height, channels, size)
		Logger::log("\e[32mWrote \e[36m", filename, "\e[0m, ", w, ", ", h, ", ", channels, ", ", size);
		return true;
	}
	// Bold red: "Failed to write", Cyan: filename, then image dimensions
	Logger::err("\e[31;1mFailed to write \e[36m", filename, "\e[0m, ", w, ", ", h, ", ", channels, ", ", size);
	return false;
}

ImageType Image::get_file_type(const char* filename) {
	if(const char* ext = strrchr(filename, '.'); ext != nullptr) {
		if(strcmp(ext, ".png") == 0) {
			return PNG;
		}
		if(strcmp(ext, ".jpg") == 0) {
			return JPG;
		}
		if(strcmp(ext, ".bmp") == 0) {
			return BMP;
		}
		if(strcmp(ext, ".tga") == 0) {
			return TGA;
		}
	}
	return PNG;
}


void Image::simplify_pixel(
	uint8_t& red, uint8_t& green, uint8_t& blue,
	const SimplifyParams& param
) {
	const bool r_eq_g = approx_equal(red, green, param.tolerance);
	const bool r_eq_b = approx_equal(red, blue, param.tolerance);
	const bool g_eq_b = approx_equal(green, blue, param.tolerance);

	// Case 1: all channels equal
	if (r_eq_g && r_eq_b) {
		red   = param.third.r;
		green = param.third.g;
		blue  = param.third.b;
		return;
	}

	// Case 2: two channels equal
	if (r_eq_g) {
		red   = param.half.r;
		green = param.half.g;
		blue  = param.zero.b;
		return;
	}
	if (r_eq_b) {
		red   = param.half.r;
		green = param.zero.g;
		blue  = param.half.b;
		return;
	}
	if (g_eq_b) {
		red   = param.zero.r;
		green = param.half.g;
		blue  = param.half.b;
		return;
	}

	// Case 3: all distinct
	const auto [max, mid, min] = get_channel_indices(red, green, blue);

	const uint8_t full[3] = {param.full.r, param.full.g, param.full.b};
	const uint8_t half[3] = {param.half.r, param.half.g, param.half.b};
	const uint8_t zero[3] = {param.zero.r, param.zero.g, param.zero.b};

	uint8_t new_vals[3];
	new_vals[max] = full[max];
	new_vals[mid] = half[mid];
	new_vals[min] = zero[min];

	red   = new_vals[0];
	green = new_vals[1];
	blue  = new_vals[2];
}


void Image::simplify_to_dominant_color_combinations(
    const int tolerance,
    const std::vector<float>* weightOfRGB,
    const std::span<const float> tValues,
    const bool one_shot,
    const bool original_to_amended,
    const std::function<bool(Image&&)>& onResult
) const {
    const size_t passes = one_shot ? 1 : tValues.empty()
        ? 3 : tValues.size();

    const size_t pixelCount = size >= static_cast<size_t>(channels)
        ? size - static_cast<size_t>(channels - 1)
        : 0;

	for (size_t i = 0; i < passes; ++i) {
		const size_t passe = original_to_amended ? i : (passes - 1 - i);
        // One copy at a time: replaces the bulk allocation.
        Image result(*this);

        const float t = tValues.empty()
            ? ((passes > 1)
                ? static_cast<float>(passe) / static_cast<float>(passes - 1)
                : 1.0F)
            : tValues[passe];

        // Parallelise over pixels for this single pass.
        #pragma omp parallel for schedule(static) \
            default(none) shared(result, weightOfRGB) \
            firstprivate(t, tolerance, pixelCount)
        for (size_t j = 0; j < pixelCount; j += static_cast<size_t>(channels)) {
            uint8_t& red   = result.data[j];
            uint8_t& green = result.data[j + 1];
            uint8_t& blue  = result.data[j + 2];

            const uint8_t orig_r = red;
            const uint8_t orig_g = green;
            const uint8_t orig_b = blue;

            auto blend = [&](const float orig, const float weight, const float intensity) {
                const float source       = orig * weight;
                const float target       = weight * intensity;
                const float interpolated = source + t * (target - source);
                return static_cast<uint8_t>(std::clamp(interpolated, 0.0F, 255.0F));
            };

            const uint8_t r_third = blend(orig_r, (*weightOfRGB)[0], SimpleColors::ONE_THIRD);
            const uint8_t g_third = blend(orig_g, (*weightOfRGB)[1], SimpleColors::ONE_THIRD);
            const uint8_t b_third = blend(orig_b, (*weightOfRGB)[2], SimpleColors::ONE_THIRD);

            const uint8_t r_half  = blend(orig_r, (*weightOfRGB)[0], SimpleColors::HALF);
            const uint8_t g_half  = blend(orig_g, (*weightOfRGB)[1], SimpleColors::HALF);
            const uint8_t b_half  = blend(orig_b, (*weightOfRGB)[2], SimpleColors::HALF);

            const uint8_t r_full  = blend(orig_r, (*weightOfRGB)[0], SimpleColors::FULL);
            const uint8_t g_full  = blend(orig_g, (*weightOfRGB)[1], SimpleColors::FULL);
            const uint8_t b_full  = blend(orig_b, (*weightOfRGB)[2], SimpleColors::FULL);

            const uint8_t r_zero  = blend(orig_r, (*weightOfRGB)[0], 0);
            const uint8_t g_zero  = blend(orig_g, (*weightOfRGB)[1], 0);
            const uint8_t b_zero  = blend(orig_b, (*weightOfRGB)[2], 0);

            SimplifyParams const param{
                .third = {.r = r_third, .g = g_third, .b = b_third},
                .half  = {.r = r_half,  .g = g_half,  .b = b_half},
                .full  = {.r = r_full,  .g = g_full,  .b = b_full},
                .zero  = {.r = r_zero,  .g = g_zero,  .b = b_zero},
                .tolerance = tolerance
            };

            simplify_pixel(red, green, blue, param);
        }

        // Hand off the completed pass; stop early if the caller signals abort.
        if (!onResult(std::move(result))) { break; }
    }
}


std::optional<int> Image::sorting_pixels_by_brightness(const float proportion, const bool below) const
{
	// Reject invalid proportions
	if (proportion <= 0.0F || proportion > 1.0F)
		return std::nullopt;

	// Number of pixels (assuming RGB interleaved)
	const size_t pixelCount = size / static_cast<size_t>(channels);

	// Histogram for possible RGB sums [0, 765] (765 = 3 * 255)
	std::array<size_t, 766> histogram{};
	histogram.fill(0);

	// Build histogram
	for (size_t i = 0; i < size; i += static_cast<size_t>(channels))
	{
		const int sum = data[i] + data[i + 1] + data[i + 2];
		++histogram[sum];
	}

	// Target rank in sorted order
	const size_t targetRank =
		static_cast<float>(pixelCount) * proportion;

	size_t cumulative = 0;

	if (below)
	{

		// Traverse from darkest to brightest
		for (int value = 0; value <= 765; ++value)
		{
			cumulative += histogram[value];
			if (cumulative > targetRank) {
				return value;
}
		}
		return 765;
	}
	// Traverse from brightest to darkest
	for (int value = 765; value >= 0; --value)
	{
		cumulative += histogram[value];
		if (cumulative > targetRank)
			return value;
	}
	return 0;
}

auto Image::proportion_complete(
	const float proportion,
	const int colorNuance,
	const bool useDarkNuance,
	const bool below
) -> Image& {
	const auto threshold = sorting_pixels_by_brightness(proportion, below);
	if (!threshold) return *this;

	const int newColor = useDarkNuance ? colorNuance : 255 - colorNuance;
	for (size_t i = 0; i < size; i += static_cast<size_t>(channels)) {
		if (const int sum = data[i] + data[i + 1] + data[i + 2]; (below && sum <= *threshold) ||
			(!below && sum >= *threshold))
			data[i] = data[i + 1] = data[i + 2] = newColor;
	}
	return *this;
}

Image& Image::reverse_by_proportion(const float proportion, const bool below) {
	const auto threshold = sorting_pixels_by_brightness(proportion, below);
	if (!threshold) return *this;

	const int thresh = *threshold;
	const auto cmp = below
		? [](const int sum, const int t){ return sum <= t; }
		: [](const int sum, const int t){ return sum >= t; };

	for (size_t i = 0; i < size; i += channels) {
		if (cmp(data[i] + data[i+1] + data[i+2], thresh)) {
			data[i]   = 255 - data[i];
			data[i+1] = 255 - data[i+1];
			data[i+2] = 255 - data[i+2];
		}
	}
	return *this;
}

Image& Image::black_and_white(const float proportion, const bool below) {
	// Compute the threshold based on the given proportion of pixels
	const auto threshold = sorting_pixels_by_brightness(proportion, below);
	if (!threshold) return *this;

	const int thresh = *threshold;

	// Define the comparison function based on the 'below' parameter
	const auto cmp = below
		? [](const int sum, const int t) { return sum <= t; }
	: [](const int sum, const int t) { return sum >= t; };

	// Iterate over each pixel in the image
	for (size_t i = 0; i < size; i += channels) {
		const int rgb_sum = data[i] + data[i + 1] + data[i + 2];

		// Apply the comparison to determine if the pixel should be black or white
		const uint8_t value = cmp(rgb_sum, thresh) ? 0 : 255;

		// Set the RGB values to either 0 (black) or 255 (white)
		data[i] = data[i + 1] = data[i + 2] = value;
	}

	return *this;
}

//fraction by rectangles

template<typename TransformFunc>
// Applies a proportional transformation to specified regions of the image based on RGB intensity.
// The transformation is applied to pixels either below or above a calculated threshold.
// Parameters:
//   - proportion: Fraction of pixels to consider for threshold calculation (0.0 to 1.0).
//   - fraction: Determines the number of rectangles per row/column (2^fraction).
//   - rectanglesToModify: Indices of rectangles to process.
//   - below: If true, transform pixels below the threshold; otherwise, transform pixels above.
//   - transformation: Function returning the value to apply to transformed pixels.
Image& Image::apply_proportion_transformation_region_fraction(
    const float proportion,
    const int fraction,
    const std::vector<int>& rectanglesToModify,
    const bool below,
    TransformFunc transformation
) {
    // Early exit if inputs are invalid
    if (rectanglesToModify.empty() || proportion <= 0.0f || proportion > 1.0f) {
        return *this;
    }

    // Precompute constants to avoid repeated calculations
    const int numRectanglesPerRow = 1 << fraction;
    const int totalRectangles = numRectanglesPerRow * numRectanglesPerRow;
    const int rectWidth = w / numRectanglesPerRow;
    const int rectHeight = h / numRectanglesPerRow;
    const uint8_t transformValue = transformation();

    // Precompute the threshold index once
    const size_t regionPixelCount = rectWidth * rectHeight;
    const size_t thresholdIndex = static_cast<size_t>((regionPixelCount - 1) * proportion);

    // Process each rectangle
    for (const int rectIndex : rectanglesToModify) {
        if (rectIndex < 0 || rectIndex >= totalRectangles) {
            continue; // Skip invalid indices
        }

        // Calculate rectangle boundaries
        const int rectRow = rectIndex / numRectanglesPerRow;
        const int rectCol = rectIndex % numRectanglesPerRow;
        const int startX = rectCol * rectWidth;
        const int startY = rectRow * rectHeight;
        const int endX = std::min(startX + rectWidth, w);
        const int endY = std::min(startY + rectHeight, h);

        // Extract RGB values for the current region
        std::vector<int> rgbValues;
        rgbValues.reserve(regionPixelCount);

        for (int y = startY; y < endY; ++y) {
            const size_t rowOffset = y * w * channels;
            for (int x = startX; x < endX; ++x) {
                const size_t pixelOffset = rowOffset + x * channels;
                rgbValues.push_back(data[pixelOffset] + data[pixelOffset + 1] + data[pixelOffset + 2]);
            }
        }

        // Sort the RGB values to find the threshold
    	std::vector<int> sortedValues = rgbValues;

    	const size_t nthIdx = below
			? thresholdIndex
			: (sortedValues.size() - 1 - thresholdIndex);

    	std::ranges::nth_element(sortedValues, sortedValues.begin() + nthIdx
	    );

    	const int threshold = sortedValues[nthIdx];

        // Apply the transformation to the region
        for (int y = startY; y < endY; ++y) {
            const size_t rowOffset = y * w * channels;
            for (int x = startX; x < endX; ++x) {
                const size_t pixelOffset = rowOffset + x * channels;

                if (const int rgb = data[pixelOffset] + data[pixelOffset + 1] + data[pixelOffset + 2];
	                (below && rgb <= threshold) || (!below && rgb >= threshold)) {
                    data[pixelOffset] = data[pixelOffset + 1] = data[pixelOffset + 2] = transformValue;
                }
            }
        }
    }
    return *this;
}


Image& Image::proportion_region_fraction(
    const float proportion,
    const int colorNuance,
    const int fraction,
    const std::vector<int>& rectanglesToModify,
    const bool useDarkNuance,
    const bool below) {

    const int newColor = useDarkNuance ? colorNuance : 255 - colorNuance;
    return apply_proportion_transformation_region_fraction(
        proportion,
        fraction,
        rectanglesToModify,
        below,
        [newColor]() -> uint8_t { return newColor; }
    );
}


Image& Image::operator=(const Image& img) {
	if (this == &img) {
		return *this;  // Protection auto-affectation
	}

	// Freeing up the old buffer
	delete[] data;

	// Copying the new dimensions
	w = img.w;
	h = img.h;
	channels = img.channels;
	size = img.size;

	// Allocation and copying of the new buffer
	data = new uint8_t[size];
	memcpy(data, img.data, size);

	return *this;
}


Image& Image::std_convolve_clamp_to_0(const int channel, const int ker_w, const int ker_h, const double *ker, int cr, int c) {

    // Checking the image dimensions
    if (w <= 0 || h <= 0 || data == nullptr) {
        fprintf(stderr, "Error: Invalid image (incorrect dimensions or data).\n");
        return *this;
    }

    // Dynamic allocation of the temporary buffer
    auto *new_data = new uint8_t[w * h];
    if (new_data == nullptr) {
        fprintf(stderr, "Error: Memory allocation failed for new_data.\n");
        return *this;
    }

    // Initialising the temporary buffer
    memset(new_data, 0, w * h);

    // Calcul de la convolution
    const int half_ker_w = ker_w / 2;
    const int half_ker_h = ker_h / 2;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            double sum = 0.0;

            // Application du noyau de convolution
            for (int ky = -half_ker_h; ky <= half_ker_h; ++ky) {
                for (int kx = -half_ker_w; kx <= half_ker_w; ++kx) {
                    const int pixel_x = x + kx;
                    const int pixel_y = y + ky;

                    // Edge management : clamp to 0
                    if (pixel_x < 0 || pixel_x >= w || pixel_y < 0 || pixel_y >= h) {
                        continue;
                    }

                    // Calculating the pixel’s index
                    const int idx = pixel_y * w + pixel_x;

                    // Calculating the pixel’s contribution to the result
                    const int ker_idx = (ky + half_ker_h) * ker_w + (kx + half_ker_w);
                    sum += data[channels * idx + channel] * ker[ker_idx];
                }
            }

            // Storing the result in `new_data`
            const int idx = y * w + x;
            new_data[idx] = static_cast<uint8_t>(sum);
        }
    }

    // Copy the result to this > data
    for (uint64_t i = 0; i < static_cast<uint64_t>(w) * h; ++i) {
        data[channels * i + channel] = new_data[i];
    }

    // Libération de la mémoire
    delete[] new_data;

    return *this;
}



Image& Image::std_convolve_clamp_to_border(const uint8_t channel, const uint32_t ker_w,
	const uint32_t ker_h, const double ker[], const uint32_t cr, const uint32_t cc) {
	std::vector<uint8_t> new_data(static_cast<size_t>(w) * h);
	const uint64_t center = cr*ker_w + cc;
	for(uint64_t k=channel; k<size; k+=channels) {
		double c = 0;
		for (long i = -static_cast<long>(cr); i < static_cast<long>(ker_h - cr); ++i) {
			long row = (static_cast<long>(k / channels) / w) - i;
			if(row < 0) {
				row = 0;
			} else if(row > h-1) {
				row = h-1;
			}
			for (long j = -static_cast<long>(cc); j < static_cast<long>(ker_w - cc); ++j) {
				long col = static_cast<long>(k / channels) % w - j;
				if(col < 0) {
					col = 0;
				} else if(col > w-1) {
					col = w-1;
				}
				const uint64_t ker_idx = center + i * static_cast<long>(ker_w) + j;
				c += ker[ker_idx] * data[(row * w + col) * channels + channel];
			}
		}
		new_data[k / channels] = static_cast<uint8_t>(BYTE_BOUND(round(c)));
	}
	for(uint64_t k=channel; k<size; k+=channels) {
		data[k] = new_data[k/channels];
	}
	return *this;
}


Image& Image::std_convolve_cyclic(
	const uint8_t channel, const uint32_t ker_w, const uint32_t ker_h, double ker[], const uint32_t cr, const uint32_t cc) {
	std::vector<uint8_t> new_data(static_cast<size_t>(w) * h);
	const uint64_t center = cr*ker_w + cc;
	for(uint64_t k=channel; k<size; k+=channels) {
		double c = 0;
		for(long i = -static_cast<long>(cr); i<static_cast<long>(ker_h)-cr; ++i) {
			long row = static_cast<long>(k)/channels/w-i;
			if(row < 0) {
				row = row%h + h;
			} else if(row > h-1) {
				row %= h;
			}
			for(long j = -static_cast<long>(cc); j<static_cast<long>(ker_w)-cc; ++j) {
				long col = (static_cast<long>(k)/channels)%w-j;
				if(col < 0) {
					col = col%w + w;
				} else if(col > w-1) {
					col %= w;
				}
				c += ker[center+i*static_cast<long>(ker_w)+j]*data[(row*w+col)*channels+channel];
			}
		}
		new_data[k/channels] = static_cast<uint8_t>(BYTE_BOUND(round(c)));

	}
	for(uint64_t k=channel; k<size; k+=channels) {
		data[k] = new_data[k/channels];
	}
	return *this;
}




uint32_t Image::rev(const uint32_t n, const uint32_t a) {
	const auto max_bits = static_cast<uint8_t>(ceil(log2(n)));
	uint32_t reversed_a = 0;
	for(uint8_t i=0; i<max_bits; ++i) {
		if(a & (1<<i)) {
			reversed_a |= (1<<(max_bits-1-i));
		}
	}
	return reversed_a;
}

void Image::bit_rev(const uint32_t n, std::complex<double> a[], std::complex<double>* A) {
	for(uint32_t i=0; i<n; ++i) {
		A[rev(n,i)] = a[i];
	}
}

void Image::fft(const uint32_t n, std::complex<double> x[], std::complex<double>* X) {
	// x in standard order
	if (x != X) {
		memcpy(X, x, n * sizeof(std::complex<double>));
	}

	// Gentleman-Sande butterfly
	uint32_t sub_probs = 1;
	uint32_t sub_prob_size = n;

	while (sub_prob_size > 1) {
		const uint32_t half = sub_prob_size >> 1;
		const std::complex w_step(cos(-2 * M_PI / sub_prob_size), sin(-2 * M_PI / sub_prob_size));

		for (uint32_t i = 0; i < sub_probs; ++i) {
			const uint32_t j_begin = i * sub_prob_size;
			const uint32_t j_end = j_begin + half;
			std::complex<double> w(1, 0);

			for (uint32_t j = j_begin; j < j_end; ++j) {
				std::complex<double> tmp1 = X[j];
				std::complex<double> tmp2 = X[j + half];
				X[j] = tmp1 + tmp2;
				X[j + half] = (tmp1 - tmp2) * w;
				w *= w_step;
			}
		}

		sub_probs <<= 1;
		sub_prob_size = half;
	}
	// X in bit reversed order
}


void Image::ifft(const uint32_t n, std::complex<double> X[], std::complex<double>* x) {
	// X in bit reversed order
	if (X != x) {
		memcpy(x, X, n * sizeof(std::complex<double>));
	}

	// Cooley-Tukey butterfly
	uint32_t sub_probs = n >> 1;
	uint32_t half = 1;

	while (half < n) {
		const uint32_t sub_prob_size = half << 1;
		const std::complex w_step(cos(2 * M_PI / sub_prob_size), sin(2 * M_PI / sub_prob_size));

		for (uint32_t i = 0; i < sub_probs; ++i) {
			const uint32_t j_begin = i * sub_prob_size;
			const uint32_t j_end = j_begin + half;
			std::complex<double> w(1, 0);

			for (uint32_t j = j_begin; j < j_end; ++j) {
				std::complex<double> tmp1 = x[j];
				std::complex<double> tmp2 = w * x[j + half];
				x[j] = tmp1 + tmp2;
				x[j + half] = tmp1 - tmp2;
				w *= w_step;
			}
		}

		sub_probs >>= 1;
		half = sub_prob_size;
	}

	for (uint32_t i = 0; i < n; ++i) {
		x[i] /= n;
	}
	// x in standard order
}


void Image::dft_2D(const uint32_t m, const uint32_t n, std::complex<double> x[], std::complex<double>* X) {
	//x in row-major & standard order
	auto* intermediate = new std::complex<double>[m*n];
	//rows
	for(uint32_t i=0; i<m; ++i) {
		fft(n, x+i*n, intermediate+i*n);
	}
	//cols
	for(uint32_t j=0; j<n; ++j) {
		for(uint32_t i=0; i<m; ++i) {
			X[j*m+i] = intermediate[i*n+j]; //row-major --> col-major
		}
		fft(m, X+j*m, X+j*m);
	}
	delete[] intermediate;
	//X in column-major & bit-reversed (in rows then columns)
}

void Image::idft_2D(const uint32_t m, const uint32_t n, std::complex<double> X[], std::complex<double>* x) {
	//X in column-major & bit-reversed (in rows then columns)
	auto* intermediate = new std::complex<double>[m*n];
	//cols
	for(uint32_t j=0; j<n; ++j) {
		ifft(m, X+j*m, intermediate+j*m);
	}
	//rows
	for(uint32_t i=0; i<m; ++i) {
		for(uint32_t j=0; j<n; ++j) {
			x[i*n+j] = intermediate[j*m+i]; //row-major <-- col-major
		}
		ifft(n, x+i*n, x+i*n);
	}
	delete[] intermediate;
	//x in row-major & standard order
}

void Image::pad_kernel(const uint32_t ker_w, const uint32_t ker_h, double ker[], const uint32_t cr, const uint32_t cc, const uint32_t pw, const uint32_t ph, std::complex<double>* pad_ker) {
	//padded so center of kernel is at top left
	for(long i=-static_cast<long>(cr); i<static_cast<long>(ker_h)-cr; ++i) {
		const uint32_t r = i<0 ? i+ph : i;
		for(long j=-static_cast<long>(cc); j<static_cast<long>(ker_w)-cc; ++j) {
			const uint32_t c = j<0 ? j+pw : j;
			pad_ker[r*pw+c] = std::complex<double>(ker[(i+cr)*ker_w+(j+cc)], 0);
		}
	}
}
void Image::pointwise_product(const uint64_t l, std::complex<double> a[], std::complex<double> b[], std::complex<double>* p) {
	for(uint64_t k=0; k<l; ++k) {
		p[k] = a[k]*b[k];
	}
}

Image& Image::fd_convolve_clamp_to_0(
	const uint8_t channel, const uint32_t ker_w, const uint32_t ker_h, double ker[], const uint32_t cr, const uint32_t cc) {
	//calculate padding
	/* 1.0
	uint32_t pw = 1<<((uint8_t)ceil(log2(w+ker_w-1)));
	uint32_t ph = 1<<((uint8_t)ceil(log2(h+ker_h-1)));
	uint64_t psize = pw*ph;
	*/
	const uint32_t pw = 1 << static_cast<uint8_t>(ceil(log2(static_cast<double>(w + ker_w - 1))));
	const uint32_t ph = 1 << static_cast<uint8_t>(ceil(log2(static_cast<double>(h + ker_h - 1))));
	const uint64_t psize = static_cast<uint64_t>(pw) * static_cast<uint64_t>(ph);

	//pad image
	auto* pad_img = new std::complex<double>[psize];
	for(uint32_t i=0; i<h; ++i) {
		for(uint32_t j=0; j<w; ++j) {
			pad_img[i*pw+j] = std::complex<double>(data[(i*w+j)*channels+channel],0);
		}
	}

	//pad kernel
	auto* pad_ker = new std::complex<double>[psize];
	pad_kernel(ker_w, ker_h, ker, cr, cc, pw, ph, pad_ker);

	//convolution
	dft_2D(ph, pw, pad_img, pad_img);
	dft_2D(ph, pw, pad_ker, pad_ker);
	pointwise_product(psize, pad_img, pad_ker, pad_img);
	idft_2D(ph, pw, pad_img, pad_img);

	//update pixel data
	for(uint32_t i=0; i<h; ++i) {
		for(uint32_t j=0; j<w; ++j) {
			data[(i*w+j)*channels+channel] = BYTE_BOUND((uint8_t)round(pad_img[i*pw+j].real()));
		}
	}

	return *this;
}
Image& Image::fd_convolve_clamp_to_border(
	const uint8_t channel, const uint32_t ker_w, const uint32_t ker_h, double ker[], const uint32_t cr, const uint32_t cc) {
	//calculate padding
	const uint32_t pw = 1 << static_cast<uint8_t>(ceil(log2(w + ker_w - 1)));
	const uint32_t ph = 1 << static_cast<uint8_t>(ceil(log2(h+ker_h-1)));
	const uint64_t psize = pw*ph;

	//pad image
	auto* pad_img = new std::complex<double>[psize];
	for(uint32_t i=0; i<ph; ++i) {
		const uint32_t r = (i<h) ? i : ((i<h+cr ? h-1 : 0));
		for(uint32_t j=0; j<pw; ++j) {
			const uint32_t c = (j<w) ? j : ((j<w+cc ? w-1 : 0));
			pad_img[i*pw+j] = std::complex<double>(data[(r*w+c)*channels+channel],0);
		}
	}

	//pad kernel
	auto* pad_ker = new std::complex<double>[psize];
	pad_kernel(ker_w, ker_h, ker, cr, cc, pw, ph, pad_ker);

	//convolution
	dft_2D(ph, pw, pad_img, pad_img);
	dft_2D(ph, pw, pad_ker, pad_ker);
	pointwise_product(psize, pad_img, pad_ker, pad_img);
	idft_2D(ph, pw, pad_img, pad_img);

	//update pixel data
	for(uint32_t i=0; i<h; ++i) {
		for(uint32_t j=0; j<w; ++j) {
			data[(i*w+j)*channels+channel] = BYTE_BOUND((uint8_t)round(pad_img[i*pw+j].real()));
		}
	}

	return *this;
}

Image& Image::fd_convolve_cyclic(const uint8_t channel, const uint32_t ker_w,
	const uint32_t ker_h, double ker[], const uint32_t cr, const uint32_t cc) {
	//calculate padding
	const uint32_t pw = 1 << static_cast<uint8_t>(ceil(log2(w + ker_w - 1)));
	const uint32_t ph = 1 << static_cast<uint8_t>(ceil(log2(h + ker_h - 1)));

	const uint64_t psize = pw*ph;

	//pad image
	auto* pad_img = new std::complex<double>[psize];
	for(uint32_t i=0; i<ph; ++i) {
		const uint32_t r = (i<h) ? i : ((i<h+cr ? i%h : h-ph+i));
		for(uint32_t j=0; j<pw; ++j) {
			const uint32_t c = (j<w) ? j : ((j<w+cc ? j%w : w-pw+j));
			pad_img[i*pw+j] = std::complex<double>(data[(r*w+c)*channels+channel],0);
		}
	}

	//pad kernel
	auto* pad_ker = new std::complex<double>[psize];
	pad_kernel(ker_w, ker_h, ker, cr, cc, pw, ph, pad_ker);

	//convolution
	dft_2D(ph, pw, pad_img, pad_img);
	dft_2D(ph, pw, pad_ker, pad_ker);
	pointwise_product(psize, pad_img, pad_ker, pad_img);
	idft_2D(ph, pw, pad_img, pad_img);

	//update pixel data
	for(uint32_t i=0; i<h; ++i) {
		for(uint32_t j=0; j<w; ++j) {
			data[(i*w+j)*channels+channel] = BYTE_BOUND((uint8_t)round(pad_img[i*pw+j].real()));
		}
	}

	return *this;
}


Image& Image::convolve_linear(const uint8_t channel, const uint32_t ker_w, const uint32_t ker_h,
	double ker[], const uint32_t cr, const uint32_t cc) {
	if (static_cast<uint64_t>(ker_w) * static_cast<uint64_t>(ker_h) > 224) {
		return fd_convolve_clamp_to_0(channel, ker_w, ker_h, ker, cr, cc);
	}
	return std_convolve_clamp_to_0(
		channel, static_cast<int>(ker_w), static_cast<int>(ker_h), ker, static_cast<int>(cr), static_cast<int>(cc));
}

Image& Image::convolve_clamp_to_border(const uint8_t channel, const uint32_t ker_w,
	const uint32_t ker_h, double ker[], const uint32_t cr, const uint32_t cc) {
	if(ker_w*ker_h > 224) {
		return fd_convolve_clamp_to_border(channel, ker_w, ker_h, ker, cr, cc);
	}
	return std_convolve_clamp_to_border(channel, ker_w, ker_h, ker, cr, cc);
}

Image applyDenoise(const Image& input, const float strength) {
	if (strength <= 0.0f) {
		return input; // copie
	}

	Image result(input);

	// simple 3×3 Gaussian kernel
	double kernel[9] = {
		1, 2, 1,
		2, 4, 2,
		1, 2, 1
	};

	// normalisation
	for (double& v : kernel) {
		v /= 16.0;
	}

	for (int c = 0; c < result.channels; ++c) {
		result.convolve_clamp_to_border(c, 3, 3, kernel, 1, 1);
	}

	return result;
}

Image& Image::remove_haze_black_level() {

	std::array<uint8_t, 3> min_val = {255,255,255};
	std::array<uint8_t, 3> max_val = {0,0,0};

	// 1. Find min/max per channel
	for (size_t i = 0; i < size; i += channels) {
		for (int c = 0; c < 3; ++c) {
			min_val[c] = std::min(min_val[c], data[i + c]);
			max_val[c] = std::max(max_val[c], data[i + c]);
		}
	}

	// Avoid division by zero
	for (int c = 0; c < 3; ++c) {
		if (max_val[c] == min_val[c]) {
			max_val[c] = min_val[c] + 1;
		}
	}

	// 2. Normalize
#pragma omp parallel for
	for (size_t i = 0; i < size; i += channels) {
		for (int c = 0; c < 3; ++c) {
			float val = data[i + c];
			val = (val - min_val[c]) * 255.0f / (max_val[c] - min_val[c]);
			data[i + c] = static_cast<uint8_t>(std::clamp(val, 0.0f, 255.0f));
		}
	}

	return *this;
}

Image& Image::local_contrast(const float strength) {

	Image blurred(*this);

	// même kernel que ton denoise
	double kernel[9] = {
		1,2,1,
		2,4,2,
		1,2,1
	};

	for (double& v : kernel) v /= 16.0;

	for (int c = 0; c < channels; ++c) {
		blurred.convolve_clamp_to_border(c, 3, 3, kernel, 1, 1);
	}

#pragma omp parallel for
	for (size_t i = 0; i < size; ++i) {
		float val = data[i] + strength * (data[i] - blurred.data[i]);
		data[i] = static_cast<uint8_t>(std::clamp(val, 0.0f, 255.0f));
	}

	return *this;
}

Image& Image::convolve_cyclic(const uint8_t channel, const  uint32_t ker_w, const uint32_t ker_h, double ker[], const uint32_t cr, const uint32_t cc) {
	if(ker_w*ker_h > 224) {
		return fd_convolve_cyclic(channel, ker_w, ker_h, ker, cr, cc);
	}
	return std_convolve_cyclic(channel, ker_w, ker_h, ker, cr, cc);
}


Image& Image::diffmap(const Image& img) {
	const int compare_width = std::min(w, img.w);
	const int compare_height = std::min(h, img.h);
	const int compare_channels = std::min(channels, img.channels);

	for (uint32_t i = 0; i < compare_height; ++i) {
		for (uint32_t j = 0; j < compare_width; ++j) {
			for (int k = 0; k < compare_channels; ++k) { // Changed uint8_t to int
				data[(i * w + j) * channels + k] = static_cast<uint8_t>(BYTE_BOUND(abs(data[(i * w + j) * channels + k] - img.data[(i * img.w + j) * img.channels + k])));
			}
		}
	}
	return *this;
}



Image& Image::diffmap_scale(Image& img, uint8_t scl) {
	const int compare_width = std::min(w, img.w);
	const int compare_height = std::min(h, img.h);
	const int compare_channels = std::min(channels, img.channels);

	uint8_t largest = 0;
	for(uint32_t i=0; i<compare_height; ++i) {
		for(uint32_t j=0; j<compare_width; ++j) {
			for(uint8_t k=0; k<compare_channels; ++k) {
				data[(i*w+j)*channels+k] = BYTE_BOUND(abs(data[(i*w+j)*channels+k] -
					img.data[(i*img.w+j)*img.channels+k]));
				largest = std::max(largest, data[(i*w+j)*channels+k]);
			}
		}
	}
	scl = static_cast<uint8_t>(255 / fmax(1, fmax(scl, largest)));
	for(int i=0; i<size; ++i) {
		data[i] *= scl;
	}
	return *this;
}


Image& Image::grayscale_avg() {
	if(channels < 3) {
		Logger::err("Image ", this, " has less than 3 channels, it is assumed to already be grayscale.");
	} else {
		for(size_t i = 0; i < size; i+=static_cast<size_t>(channels)) {
			//(r+g+b)/3
			const int gray = (data[i] + data[i+1] + data[i+2])/3;
			memset(data+i, gray, 3);
		}
	}
	return *this;
}


Image& Image::grayscale_lum() {
	if(channels < 3) {
		Logger::err("Image ", this, " has less than 3 channels, it is assumed to already be grayscale.");
	} else {
		for(size_t i = 0; i < size; i+=static_cast<size_t>(channels)) {
			const double gray_d = 0.2126*data[i] + 0.7152*data[i+1] + 0.0722*data[i+2];
			const int gray = static_cast<int>(std::round(gray_d));
			memset(data+i, gray, 3);
		}
	}
	return *this;
}

Image& Image::color_mask(const float r, const float g, const float b) {
	if (channels < 3) {
		Logger::err("\033[31m[ERROR] Color mask requires at least 3 channels, but this image has ", channels, " channels\033[0m");
	} else {
		for (size_t i = 0; i < size; i += static_cast<size_t>(channels)) {
			data[i]   = static_cast<uint8_t>(std::clamp(std::round(static_cast<float>(data[i])   * r), 0.0f, 255.0f));
			data[i+1] = static_cast<uint8_t>(std::clamp(std::round(static_cast<float>(data[i+1]) * g), 0.0f, 255.0f));
			data[i+2] = static_cast<uint8_t>(std::clamp(std::round(static_cast<float>(data[i+2]) * b), 0.0f, 255.0f));
		}
	}
	return *this;
}


Image& Image::encode_message(const char* message) {
	const uint32_t len = strlen(message) * 8;
	if(len + STEG_HEADER_SIZE > size) {
		Logger::err("\033[31m[ERROR] This message is too large (", len + STEG_HEADER_SIZE, " bits / ", size, " bits)\033[0m");
		return *this;
	}

	for(size_t i = 0; i < STEG_HEADER_SIZE; ++i) {
		data[i] &= 0xFE;
		data[i] |= (len >> (STEG_HEADER_SIZE - 1 - i)) & 1UL;
	}

	for(uint32_t i = 0; i < len; ++i) {
		data[i+STEG_HEADER_SIZE] &= 0xFE;
		data[i+STEG_HEADER_SIZE] |= (message[i/8] >> ((len-1-i)%8)) & 1;
	}

	return *this;
}

Image& Image::decode_message(char* buffer, size_t* messageLength) {
	constexpr uint32_t len = 0;
	for(size_t i = 0; i < STEG_HEADER_SIZE; ++i) {
		data[i] &= 0xFE;
		data[i] |= (len >> (STEG_HEADER_SIZE - 1 - i)) & 1UL;
	}

	*messageLength = len / 8;

	for(uint32_t i = 0; i < len; ++i) {
		buffer[i/8] = static_cast<char>((buffer[i/8] << 1) | (data[i+STEG_HEADER_SIZE] & 1));
	}


	return *this;
}




Image& Image::flipX() {
	uint8_t tmp[4];
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w / 2; ++x) {
			uint8_t* px1 = &data[(x + y * w) * channels];
			uint8_t* px2 = &data[((w - 1 - x) + y * w) * channels];

			// Utilisation de std::copy_n pour copier `channels` éléments
			std::copy_n(px1, channels, tmp);
			std::copy_n(px2, channels, px1);
			std::copy_n(tmp, channels, px2);
		}
	}
	return *this;
}


Image& Image::flipY() {
	uint8_t tmp[4];
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w / 2; ++x) {
			uint8_t* px1 = &data[(x + y * w) * channels];
			uint8_t* px2 = &data[((w - 1 - x) + y * w) * channels];

			// Utilisation de std::copy_n pour copier `channels` éléments
			std::copy_n(tmp, channels, px1);
			std::copy_n(px1, channels, px2);
			std::copy_n(px2, channels, tmp);
		}
	}
	return *this;
}


Image& Image::overlay(const Image& source, const int x, const int y) {
	for (int sy = 0; sy < source.h; ++sy) {
		if (sy + y < 0) continue;
		if (sy + y >= h) break;

		for (int sx = 0; sx < source.w; ++sx) {
			if (sx + x < 0) continue;
			if (sx + x >= w) break;

			uint8_t* srcPx = &source.data[(sx + sy * source.w) * source.channels];
			uint8_t* dstPx = &data[(sx + x + (sy + y) * w) * channels];

			const float srcAlpha = (source.channels < 4) ? 1.0f : (static_cast<float>(srcPx[3]) / 255.0f);


			if (const float dstAlpha = (channels < 4) ? 1.0f : (static_cast<float>(dstPx[3]) / 255.0f); srcAlpha > 0.99f && dstAlpha > 0.99f) {
				if (source.channels >= channels) {
					std::copy_n(srcPx, channels, dstPx);
				} else {
					std::fill_n(dstPx, channels, srcPx[0]);
				}
			} else {
				if (const float outAlpha = srcAlpha + dstAlpha * (1.0f - srcAlpha); outAlpha < 0.01f) {
					std::fill_n(dstPx, channels, 0);
				} else {
					for (int chnl = 0; chnl < channels; ++chnl) {
						dstPx[chnl] = static_cast<uint8_t>(
							BYTE_BOUND((srcPx[chnl] / 255.0f * srcAlpha +
									   dstPx[chnl] / 255.0f * dstAlpha * (1.0f - srcAlpha)) /
									  outAlpha * 255.0f)
						);
					}
					if (channels > 3) {
						dstPx[3] = static_cast<uint8_t>(BYTE_BOUND(outAlpha * 255.0f));
					}
				}
			}
		}
	}
	return *this;
}


Image& Image::crop(const uint16_t cx, const uint16_t cy, const uint16_t cw, const uint16_t ch) {
	size = cw * ch * channels;
	auto* croppedImage = new uint8_t[size];
	memset(croppedImage, 0, size);

	for(uint16_t y = 0; y < ch; ++y) {
		if(y + cy >= h) {
			break;
		}
		for(uint16_t x = 0; x < cw; ++x) {
			if(x + cx >= w) {
				break;
			}
			memcpy(&croppedImage[(x + y * cw) * channels], &data[(x + cx + (y + cy) * w) * channels], channels);
		}
	}

	w = cw;
	h = ch;


	delete[] data;
	data = croppedImage;
	croppedImage = nullptr;

	return *this;
}


Image& Image::resizeNN(const uint16_t nw, const uint16_t nh) {
	size = nw * nh * channels;
	auto* newImage = new uint8_t[size];

	const float scaleX = static_cast<float>(nw) / static_cast<float>(w);
	const float scaleY = static_cast<float>(nh) / static_cast<float>(h);

	for(uint16_t y = 0; y < nh; ++y) {
		const auto sy = static_cast<uint16_t>(std::round(static_cast<float>(y) / scaleY));
		for(uint16_t x = 0; x < nw; ++x) {
			const auto sx = static_cast<uint16_t>(std::round(static_cast<float>(x) / scaleX));


			memcpy(&newImage[(x + y * nw) * channels], &data[(sx + sy * w) * channels], channels);

		}
	}

	w = nw;
	h = nh;
	delete[] data;
	data = newImage;
	newImage = nullptr;

	return *this;
}
