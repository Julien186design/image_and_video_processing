# Image & Video Processing (Fork & Integration)

A C++20 application for advanced image and video processing, built upon the foundational work of **Code-Break0** and **SSARCandy**. 

---

## 📜 Acknowledgments & Attribution

This project is a derivative work that integrates and expands upon concepts, algorithms, and implementations from two open-source projects. 

* **[Code-Break0/Image-Processing](https://github.com/Code-Break0/Image-Processing)**  
  * **Original License:** GNU General Public License v3.0 (GPL-3.0)  
  * **Role:** Provided the original architecture and base concepts for processing image data.
* **[SSARCandy/Coherent-Line-Drawing](https://github.com/SSARCandy/Coherent-Line-Drawing)**  
  * **Original License:** MIT License  
  * **Role:** Provided the implementation of the *Coherent Line Drawing* (CLD) and *Flow-based Difference of Gaussians* (FDoG) algorithms based on the paper by Kang et al.

### Statement of Intent & Context

The modifications, refactoring, and additions introduced in this repository are non-intrusive adaptations designed to extend functionality for specific use cases (such as multithreading performance, video processing pipelines, and localized system integration). 

> **Note:** The changes made here do not invalidate, question, or replace the original works. The operating contexts, goals, and execution environments of this project differ from the upstream repositories. For pure, unmodified, or reference implementations, please consult the respective upstream projects linked above.

---

## ⚖️ Licensing

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)** to maintain compatibility with copyleft requirements from upstream components. GPL-3.0 governs the repository and any distributed build as a whole.

* The full GPL-3.0 text is in [`LICENSE`](LICENSE).
* Third-party MIT-licensed components from **SSARCandy/Coherent-Line-Drawing** are preserved under their original terms in [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md), which also lists the exact files they cover (`include/CLD.h`, `include/ETF.h`, `include/const.h`, `include/gui.h`, `include/postProcessing.h`, `src/CLD.cpp`, `src/ETF.cpp`, `src/cmd.cpp`, `src/gui.cpp`, `src/main.cpp`, `src/postProcessing.cpp`).
* All original copyright notices of **Code-Break0** and **SSARCandy** are preserved.
* Files carrying substantive modifications from their upstream originals are marked as such in a header comment, per GPL-3.0 §5(a).

See [`LICENSE`](LICENSE) for the GPL-3.0 text and [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) for third-party notices.

---

## 💻 Environment & System Compatibility

This software has been developed, compiled, and benchmarked under a specific environment:

* **OS:** Linux Mint
* **CPU Architecture:** Intel x86_64

### Compatibility Notice

> ⚠️ **Platform & Hardware Disclaimer:**  
> Development and testing were strictly conducted on a **Linux Mint system powered by an Intel processor**. Due to hardware-specific optimizations, compiler flags, and environment-dependent multithreading setups, **full functionality on other operating systems (e.g., macOS, Windows) or different CPU architectures (e.g., ARM/Apple Silicon) cannot be guaranteed**. Users compiling on alternative platforms may need to adjust CMake configs or platform-specific dependencies.

---

## ⚙️ Build & Dependencies

### Prerequisites

* **C++ Compiler:** C++20 compliant (`g++` or `clang++`)
* **Build System:** CMake (≥ 3.16)
* **Libraries:**
  * OpenCV 4.x
  * OpenMP (for parallel processing)
 
---

## 📁 Directory Structure

The program expects two primary folders at the root directory: `Input/` and `Output/`.

### Input
Place your raw image files inside categorized sub directories within `Input/`:

Input/
└── Category/
├── sample_image_1.jpg
└── sample_image_2.jpg

Output/
├── BTB/
├── BTB Square/
├── BTW/
├── BTW Square/
├── Colors nuances/
├── Edge Detector/
├── One Color/
├── Reversal/
├── Videos/
├── WTB/
├── WTB Square/
├── WTW/
└── WTW Square/
