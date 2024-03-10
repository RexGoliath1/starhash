#include <iostream>
#include <chrono>

const int HOUR_TO_MS = (1000 * 60 * 60);
const int MIN_TO_MS = (1000 * 60);
const int SEC_TO_MS = 1000;

class ProgressBar {
public:
    ProgressBar(int total, double alpha = 0.1, int width = 50)
        : total_(total), alpha_(alpha), width_(width), start_time_(std::chrono::steady_clock::now()),
          avg_time_per_iteration_(0.0), ewma_time_per_iteration_(0.0) {}

    void show_progress(int ii) {
        if (ii == 0) {
            start_time_ = std::chrono::steady_clock::now();
        } else {
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time_).count();
            double time_per_iteration = static_cast<double>(elapsed_time) / ii;

            if (avg_time_per_iteration_ == 0.0) {
                avg_time_per_iteration_ = time_per_iteration;
                ewma_time_per_iteration_ = time_per_iteration;
            } else {
                ewma_time_per_iteration_ = alpha_ * time_per_iteration + (1 - alpha_) * ewma_time_per_iteration_;
            }
        }

        std::cout << "\r";

        int progress = (ii * width_) / total_;

        auto remain_time_microseconds = static_cast<long long>(ewma_time_per_iteration_ * (total_ - ii));

        int hours = static_cast<int>(remain_time_microseconds / (HOUR_TO_MS));
        int minutes = static_cast<int>((remain_time_microseconds / (MIN_TO_MS)) % 60);
        int seconds = static_cast<int>((remain_time_microseconds / SEC_TO_MS) % 60);

        std::cout << "[";
        for (int i = 0; i < width_; ++i) {
            if (i < progress)
                std::cout << "=";
            else
                std::cout << " ";
        }
        std::cout << "] ";
        std::cout << "Time remain: " << hours << "H, " << minutes << "m, " << seconds << "s (" << ii << "/" << total_ << ")";

        if(ii == total_) {
            auto elapsed_time = std::chrono::steady_clock::now() - start_time_;
            auto elapsed_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count();
            hours = static_cast<int>(elapsed_time_ms / (HOUR_TO_MS));
            minutes = static_cast<int>((elapsed_time_ms / (MIN_TO_MS)) % 60);
            seconds = static_cast<int>((elapsed_time_ms / SEC_TO_MS) % 60);

            std::cout << "Total time: " << hours << "H, " << minutes << "m, " << seconds << "s (" << ii << "/" << total_ << ")";

        }

        std::fflush(stdout);
    }

private:
    int total_;
    double alpha_;
    int width_;
    std::chrono::steady_clock::time_point start_time_;
    double avg_time_per_iteration_;
    double ewma_time_per_iteration_;
};
