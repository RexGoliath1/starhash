#include "Utilities.hpp"
#include <limits>
#include <limits.h>
#include <iostream>

fs::path get_executable_path() {
#if defined(__linux__)
  char result[PATH_MAX];
  ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
  std::cout << "result: " << result << std::endl;
  std::cout << "parent result: " << fs::path(result).parent_path() << std::endl;
  if (count != -1) {
    return fs::path(result).parent_path();
    // return fs::canonical(fs::path(result)).parent_path();
  }
#elif defined(__APPLE__)
  char path[PATH_MAX];
  uint32_t size = sizeof(path);
  if (_NSGetExecutablePath(path, &size) == 0) {
    return fs::canonical(fs::path(path)).parent_path();
  }
#endif
  // If we can't find the executable path, return the current path
  return fs::current_path();
}
