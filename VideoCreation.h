#ifndef VIDEOPROCESSING_H
#define VIDEOPROCESSING_H

#include <condition_variable>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <queue>
#include <string>
#include <thread>

// VideoWriterQueue.h
class VideoWriterQueue {
public:
    explicit VideoWriterQueue(cv::VideoWriter& writer,
                               const std::size_t maxSize = 30)
        : writer_(writer), maxSize_(maxSize), done_(false)
    {
        writerThread_ = std::thread([this] { run(); });
    }

    ~VideoWriterQueue() { finish(); }

    // Enqueues a frame; blocks if queue is full
    void enqueue(cv::Mat&& frame) {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [&] { return queue_.size() < maxSize_; });
        queue_.push(std::move(frame));
        lock.unlock();
        cv_.notify_one();
    }

    // Signals end of production and joins the writer thread
    void finish() {
        if (done_.exchange(true)) return; // idempotent
        cv_.notify_one();
        if (writerThread_.joinable()) writerThread_.join();
    }

private:
    void run() {
        while (true) {
            std::unique_lock lock(mutex_);
            cv_.wait(lock, [&] {
                return !queue_.empty() || done_;
            });
            if (queue_.empty() && done_) { break;
}

            cv::Mat frame = std::move(queue_.front());
            queue_.pop();
            lock.unlock();
            cv_.notify_one();

            writer_.write(frame);
        }
    }

    cv::VideoWriter&           writer_;
    std::size_t                maxSize_;
    std::atomic<bool>          done_;
    std::queue<cv::Mat>        queue_;
    std::mutex                 mutex_;
    std::condition_variable    cv_;
    std::thread                writerThread_;
};

[[nodiscard]]
inline std::optional<cv::Mat> loadImage(const std::string& path) {
    cv::Mat m = cv::imread(path, cv::IMREAD_COLOR);
    if (m.empty()) {
        Logger::err("Error: could not load image ", path);
        return std::nullopt;
    }
    return m;
}

// Returns the 'below' flag for reversal entry index i.
// Index 0 → Reversal-BT (below = true : dark pixels inverted first)
// Index 1 → Reversal-WT (below = false : bright pixels inverted first)
[[nodiscard]]
inline bool reversal_below_flag(const size_t entryIndex) {
    return entryIndex == 0;
}

// Opens a VideoWriter and returns it, or std::nullopt on failure.
// Centralises the open+check pattern used by every streaming function.
[[nodiscard]]
inline std::optional<cv::VideoWriter> make_video_writer(
    const std::string& path,
    const int width,
    const int height,
    const int fps
) {
    cv::VideoWriter writer;
    writer.open(path,
                cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                fps,
                cv::Size(width, height));
    if (!writer.isOpened()) {
        Logger::err("Error: could not open video writer for ", path);
        return std::nullopt;
    }
    return writer;
}

#endif // VIDEOPROCESSING_H