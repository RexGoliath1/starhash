#ifndef UTILITIES_SH
#define UTILITIES_SH

#include <filesystem>

// Check for the operating system
#if defined(__linux__)
#include <unistd.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

namespace fs = std::filesystem;

fs::path get_executable_path();
void show_progress_bar(int ii, int total, std::chrono::time_point<std::chrono::steady_clock> start_time);

#endif // UTILITIES_SH
