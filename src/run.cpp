#include "StarCatalogGenerator.hpp"
#include "StarSolver.hpp"

#include <filesystem>
#include <iostream>

// #define SKIP_CATALOG_CREATE

namespace fs = std::filesystem;

bool create_out_directory(fs::path dir_name) {
  std::error_code err;
  if (!fs::create_directory(dir_name, err)) {
    if (fs::exists(dir_name)) {
      return true;
    } else {
      std::printf("Failed to create [%s], err:%s\n", dir_name.c_str(),
                  err.message().c_str());
      return false;
    }
  } else {
    return true;
  }
}

int main(int argc, char **argv) {
  fs::path base_path, data_path, hipparcos_file, output_path, p_cat_out_file;
  // fs::path test_image = fs::current_path() / ".." / "data" /
  // "star_tracker_image.jpeg";
  fs::path test_image =
      fs::current_path() / ".." / "data" / "large_star_image.JPG";

  int max_contours = 100;
  int max_points_per_contour = 1000;

  // Until we have full config controls, use defaults
  if (argc == 1) {
    std::cout << "Creating new catalog with defaults" << std::endl;
    base_path = fs::current_path() / "..";

    output_path = base_path / "results";
    p_cat_out_file = output_path / "output.h5";
  }

  create_out_directory(output_path);

#ifndef SKIP_CATALOG_CREATE
  StarCatalogGenerator catalog;
  catalog.run();
#endif

  StarSolver solver(max_contours, max_points_per_contour, output_path);
  solver.load_image(test_image);
  // solver.get_centroids();
  solver.get_gauss_centroids();

  return 0;
}
