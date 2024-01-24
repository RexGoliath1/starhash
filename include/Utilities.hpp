#include <iostream>
#include <filesystem>
#include <limits.h>

// Check for the operating system
#if defined(__linux__)
#include <unistd.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

namespace fs = std::filesystem;

fs::path get_executable_path();
