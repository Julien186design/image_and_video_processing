#ifndef IMAGE_H
#define IMAGE_H

#include "ProcessingConfig.h"
#include "schrift.h"

#include <cstdio>
#include <complex>
#include <vector>
#include <cmath>
#include <functional>
#include <optional>
#include <span>

//legacy feature of C
#undef __STRICT_ANSI__
#define USE_MATH_DEFINES
#ifndef M_PI
	#define M_PI (3.14159265358979323846)
#endif


#define STEG_HEADER_SIZE sizeof(uint32_t) * 8

inline std::ostream& operator<<(std::ostream& os, const std::vector<float>& vec)
{
	os << "[";
	for (size_t i = 0; i < vec.size(); ++i) {
		os << vec[i];
		if (i != vec.size() - 1) {
			os << ", ";
		}
	}
	os << "]";
	return os;
}

struct RGB {
	uint8_t r, g, b;
};

struct SimplifyParams {
	RGB third;
	RGB half;
	RGB full;
	RGB zero;
	int tolerance;
};

enum ImageType {
	PNG, JPG, BMP, TGA
};

namespace SimpleColors {
	constexpr uint8_t ONE_THIRD = 85;
	constexpr uint8_t HALF = 128;
	constexpr uint8_t FULL = 255;
}


struct Font;

struct EdgeDetectorResult {
    std::vector<uint8_t> outputRGB;
    double minGradient, maxGradient;
};

struct ImageInfo {
	std::string baseName;
	std::string inputPath;
};

struct Font {
	SFT sft = {nullptr, 12, 12, 0, 0, SFT_DOWNWARD_Y|SFT_RENDER_IMAGE};
	Font(const char* fontfile, const uint16_t size) {
		if((sft.font = sft_loadfile(fontfile)) == nullptr) {
			Logger::err("\033[31m[ERROR] Failed to load ", fontfile, "\033[0m");
			return;
		}
		setSize(size);
	}
	~Font() {
		sft_freefont(sft.font);
	}
	void setSize(const uint16_t size) {
		sft.xScale = size;
		sft.yScale = size;
	}
};

struct Image {
	uint8_t* data = nullptr;
	size_t size = 0;
	int w;
	int h;
	int channels;

	// Structure for storing channel indices
	struct ChannelIndices {
		int max;
		int mid;
		int min;
	};

	// Function helper for determining the order of color channels
	static ChannelIndices get_channel_indices(const uint8_t r, const  uint8_t g, const uint8_t b) {
		if (r > g && g > b) return {0, 1, 2};
		if (r > b && b > g) return {0, 2, 1};
		if (g > r && r > b) return {1, 0, 2};
		if (g > b && b > r) return {1, 2, 0};
		if (b > r && r > g) return {2, 0, 1};
		return {2, 1, 0}; // Cas par défaut
	}

	explicit Image(const char* filename, int channel_force = 0);
	Image(int w, int h, int channels = 3);
	Image(const Image& img);
	Image& operator=(const Image& img);
	~Image();



	bool read(const char* filename, int channel_force = 0);
	bool write(const char* filename) const;

	static ImageType get_file_type(const char* filename);

	Image& std_convolve_clamp_to_0(int channel, int ker_w, int ker_h, const double *ker, int cr, int c);
	Image& std_convolve_clamp_to_border(uint8_t channel, uint32_t ker_w, uint32_t ker_h, const double ker[], uint32_t cr, uint32_t cc);

	Image& std_convolve_cyclic(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc);

	static bool approx_equal(const uint8_t a, const  uint8_t b, const uint8_t tol) {
		const int diff = static_cast<int>(a) - static_cast<int>(b);
		return diff >= 0 ? diff <= tol : -diff <= tol;
	}

	static uint8_t avg_u8_round(const uint8_t a, const uint8_t b) {
		return static_cast<uint8_t>(
			(static_cast<unsigned int>(a) +
			 static_cast<unsigned int>(b) + 1u) / 2u
		);
	}

	// SimplifyWeights must be declared before any method that uses it
	struct SimplifyWeights {
		std::vector<uint8_t> r_third, g_third, b_third;
		std::vector<uint8_t> r_half,  g_half,  b_half;
		std::vector<uint8_t> r_full,  g_full,  b_full;
	};



	static uint32_t rev(uint32_t n, uint32_t a);
	static void bit_rev(uint32_t n, std::complex<double> a[], std::complex<double>* A);

	static void fft(uint32_t n, std::complex<double> x[], std::complex<double>* X);
	static void ifft(uint32_t n, std::complex<double> X[], std::complex<double>* x);
	static void dft_2D(uint32_t m, uint32_t n, std::complex<double> x[], std::complex<double>* X);
	static void idft_2D(uint32_t m, uint32_t n, std::complex<double> X[], std::complex<double>* x);

	static void pad_kernel(uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc, uint32_t pw, uint32_t ph, std::complex<double>* pad_ker);
	static inline void pointwise_product(uint64_t l, std::complex<double> a[], std::complex<double> b[], std::complex<double>* p);

	Image& fd_convolve_clamp_to_0(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc);
	Image& fd_convolve_clamp_to_border(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc);
	Image& fd_convolve_cyclic(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc);


	Image& convolve_linear(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc);
	Image& convolve_clamp_to_border(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc);
	Image& remove_haze_black_level();
	Image& local_contrast(float strength);
	Image& convolve_cyclic(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc);


	Image& diffmap(const Image& img);
	Image& diffmap_scale(Image& img, uint8_t scl = 0);

	Image& grayscale_avg();

	[[nodiscard]] std::optional<int> sorting_pixels_by_brightness(float proportion, bool below) const;
	Image& proportion_complete(float proportion, int colorNuance, bool useDarkNuance, bool below);
	Image& reverse_by_proportion(float proportion, bool below);

	Image& black_and_white(float proportion, bool below);

	static void simplify_pixel(
		uint8_t& red, uint8_t& green, uint8_t& blue,
		const SimplifyParams& param);


	void simplify_to_dominant_color_combinations(
		int tolerance,
		const std::vector<float>* weightOfRGB,
		std::span<const float> tValues,
		bool one_shot,
		bool original_to_amended,
		const std::function<bool(Image&&)>& onResult
	) const;


	template<typename TransformFunc>
	Image& apply_proportion_transformation_region_fraction(
		float proportion,
		int fraction,
		const std::vector<int>& rectanglesToModify,
		bool below,
		TransformFunc transformation
	);

	Image& proportion_region_fraction(float proportion, int colorNuance, int fraction,
	const std::vector<int>& rectanglesToModify, bool useDarkNuance, bool below);

	Image& grayscale_lum();

	Image& color_mask(float r, float g, float b);

	Image& encode_message(const char* message);
	Image& decode_message(char* buffer, size_t* messageLength);

	Image& flipX();
	Image& flipY();

	Image& overlay(const Image& source, int x, int y);

	Image& overlayText(const char* txt, const Font& font, int x, int y, uint8_t r = 255, uint8_t g = 255, uint8_t b = 255, uint8_t a = 255);

	Image& crop(uint16_t cx, uint16_t cy, uint16_t cw, uint16_t ch);

	Image& resizeNN(uint16_t nw, uint16_t nh);


};

Image applyDenoise(const Image& input, float strength);

inline ImageInfo extractImageInfo(const std::string& inputFile) {
	const size_t dotPos = inputFile.find_last_of('.');
	const size_t slashPos = inputFile.find_last_of('/');

	std::string baseName;
	if (slashPos == std::string::npos) {
		baseName = inputFile.substr(0, dotPos);
	} else {
		baseName = inputFile.substr(slashPos + 1, dotPos - (slashPos + 1));
	}

	const std::string inputPath = "Input/" + inputFile;
	return {baseName, inputPath};
}


EdgeDetectorResult process_edge_detection_core(
	const uint8_t* grayData,
	int width,
	int height,
	double threshold = 0.09
);

#endif // IMAGE_H