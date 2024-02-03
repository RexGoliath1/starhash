#include "H5Cpp.h"
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <assert.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <inttypes.h>
#include <iostream>
#include <list>
#include <math.h>
#include <memory>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>
#include <type_traits>

#include "eigen_mods.hpp"
#include "Utilities.hpp"

// Some macro defines to debug various functions before valgrid setup
// #define DEBUG_HIP
// #define DEBUG_PM
// #define DEBUG_HASH
// #define DEBUG_GET_NEARBY_STARS
// #define DEBUG_GET_NEARBY_STAR_PATTERNS
#define DEBUG_PATTERN_CATALOG

namespace fs = std::filesystem;

const unsigned int hip_rows = 117955;
const unsigned int hip_cols = 10;

// Hash function for Eigen Matricies
// Ignore warnings about std::unary_function and std::binary_function.
// TODO: convert unary_function deprecated.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
template <typename T> struct matrix_hash : std::unary_function<T, size_t> {
  std::size_t operator()(T const &matrix) const {
    // Note that it is oblivious to the storage order of Eigen matrix (column-
    // or row-major). It will give you the same hash value for two different
    // matrices if they are the transpose of each other in different storage
    // order.
    size_t seed = 0;
    for (size_t i = 0; i < (size_t)matrix.size(); ++i) {
      auto elem = *(matrix.data() + i);
      seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 + (seed << 6) +
              (seed >> 2);
    }

    return seed;
  }
};
#pragma GCC diagnostic pop

// Coarse sky map (pre computed hash table). This is only used internally to
// define the database.
using CoarseSkyMap = std::unordered_map<Eigen::Vector3i, std::vector<int>,
                                        matrix_hash<Eigen::Vector3i>>;

// Define eigen csv formatter
const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision,
                                       Eigen::DontAlignCols, ", ", "\n");

class StarCatalogGenerator {
public:
  StarCatalogGenerator(const std::string &in_file, const std::string &out_file);
  explicit StarCatalogGenerator();
  ~StarCatalogGenerator();

  bool load_pattern_catalog();
  // Pipeline method for everything
  void run_pipeline();

private:
  fs::path app_path = get_executable_path();
  fs::path default_hipparcos_path = app_path / "../data/hipparcos.tsv";
  fs::path default_catalog_path = app_path / "../results/output.h5";

  fs::path input_catalog_file;
  fs::path output_catalog_file;
  Eigen::MatrixXd input_catalog_data;
  Eigen::MatrixXd proper_motion_data;

  // filter_star_separation outputs
  Eigen::MatrixXd
      star_table; // Post star separation check table with PM corrected ICRS
                  // angles for "verificeation stars".
  std::vector<int> pattern_stars; // Vector of "Pattern stars" satisfying
                                  // separation thresholds

  CoarseSkyMap coarse_sky_map;
  Eigen::ArrayXd edges;

  bool regenerate_catalog = true;

  // TODO: Load dynamically through some kind of configuration
  const unsigned int pattern_size = 4;
  const unsigned int num_pattern_angles =
      (pattern_size * (pattern_size - 1)) / 2;
  ;

  int total_catalog_stars = 0;
  int number_hippo_stars_bright = 0;

  const double deg2rad = M_PI / 180.0;
  const double arcsec2deg = (1.0 / 3600.0);
  const double mas2arcsec = (1.0 / 1000.0);
  const double mas2rad = mas2arcsec * arcsec2deg * deg2rad;
  const double au2km = 149597870.691;

  float hip_byear = 1991.25; // Hipparcos Besellian Epoch
  unsigned int hip_columns = 10;
  unsigned int hip_header_rows = 55;
    // TODO: Replace with astropy script input. For now it's manual.
  float current_byear = 2024.0921411361237;// Current Besellian Epoch

  Eigen::MatrixXd bcrf_frame;

  // Default thresholding parameters (Default tetra amounts are in readme)
  float brightness_thresh = 11; // Minimum brightness of db
  //float brightness_thresh = 0.0; // Minimum brightness of db. Checking entire catalog prop
  // float brightness_thresh = 6.5; // Minimum brightness of db
  double min_separation_angle =
      0.3; // Minimum angle between 2 stars (ifov degrees or equivilent for
           // dealing with double / close stars)
  double min_separation = std::cos(
      min_separation_angle * deg2rad); // Minimum norm distance between 2 stars
  unsigned int pattern_stars_per_fov = 10;
  unsigned int catalog_stars_per_fov = 20;
  double max_fov_angle = 42;
  double max_fov_dist = std::cos(max_fov_angle * deg2rad);
  double max_half_fov_dist = std::cos(max_fov_angle * deg2rad / 2.0);
  unsigned int temp_star_bins = 4;
  int pattern_bins = 25;

  // Global counter for pattern_list
  int pattern_list_size = 0;
  int pattern_list_growth = 20000;
  int index_pattern_debug_freq = 10000;
  int separation_debug_freq = 10000;

  enum {
    RA_J2000 = 0,
    DE_J2000,
    HID,
    RA_ICRS,
    DE_ICRS,
    PLX,
    PMRA,
    PMDE,
    HPMAG,
    COLOUR,
    SIZE_ELEMS
  };

  // TODO: Inspect if python is doing things with this.. Currently > INT_MAX so
  // modulo is of -1640531535
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
  const int magic_number = 2654435761;
#pragma GCC diagnostic pop

  void create_new_catalog();

  bool pattern_catalog_file_exists();

  // Initial catalog transforms
  bool read_hipparcos();
  void convert_hipparcos();
  void sort_star_magnitudes();
  void init_besselian_year(); // TODO: Time is hard
  void init_bcrf();
  void correct_proper_motion();

  // Star angular thresholding (wrt other starts and optical axis)
  void filter_star_separation();
  bool is_star_pattern_in_fov(Eigen::MatrixXi &pattern_list,
                              std::vector<int> nearby_star_pattern);
  void get_nearby_stars(Eigen::Vector3d star_vector,
                        std::vector<int> &nearby_stars);
  void get_star_edge_pattern(Eigen::VectorXi pattern);
  void get_nearby_star_patterns(Eigen::MatrixXi &pattern_list,
                                std::vector<int> nearby_stars, int star_id);

  // Intermediate hash table functions
  void init_output_catalog();

  // Final star edge pattern hash table (from paper / code)
  int key_to_index(Eigen::VectorXi hash_code, const unsigned int pattern_bins,
                   const unsigned int catalog_length);
  void generate_output_catalog();

  template <typename Derived>
  void output_hdf5(std::string filepath, std::string dataset, const Eigen::MatrixBase<Derived>& matrix, bool truncate = false) {

      // Decide if overwritting or appending
      H5::H5File h5_file;
      if (truncate) {
        h5_file = H5::H5File(filepath.c_str(), H5F_ACC_TRUNC);
      } else {
        h5_file = H5::H5File(filepath.c_str(), H5F_ACC_RDWR);
      }

      hsize_t dim[2];
      dim[0] = matrix.rows();
      dim[1] = matrix.cols();
      
      // Determine the data type
      H5::DataType datatype;
      if (std::is_same<typename Derived::Scalar, int>::value) {
          datatype = H5::PredType::NATIVE_INT;
      } else if (std::is_same<typename Derived::Scalar, double>::value) {
          datatype = H5::PredType::NATIVE_DOUBLE;
      } else {
          // Handle other types or throw an error
      }
      
      // Create the data array
      auto arr = new typename Derived::Scalar*[dim[0]];
      for (hsize_t i = 0; i < dim[0]; i++) {
          arr[i] = new typename Derived::Scalar[dim[1]];
      }
      
      // Copy the data into the array
      for (hsize_t j = 0; j < dim[1]; j++) {
          for (hsize_t i = 0; i < dim[0]; i++) {
              arr[i][j] = matrix(i, j);
          }
      }

      if (h5_file.getId() < 0) {
        // The file is not open or not valid.
        throw std::runtime_error("Unable to open HDF5 file.");
      }
      
      H5::DataSpace ds(2, dim);
      H5::DataSet pc_dataset = h5_file.createDataSet(dataset.c_str(), datatype, ds);

      if (!pc_dataset.getId()) {
        // The dataset was not created successfully.
        throw std::runtime_error("Unable to create HDF5 dataset.");
      }

      pc_dataset.write(&arr[0][0], datatype);

      // Free the allocated memory
      for (hsize_t i = 0; i < dim[0]; i++) {
          delete[] arr[i];
      }
      delete[] arr;

      // for (size_t i = pc_cols; i > 0; ) {
      //     delete[] data_arr[--i];
      // }
      // delete[] data_arr;
  }

  // Generic Eigen 2 csv writer
  template <typename Derived>
  void write_to_csv(std::string name,
                    const Eigen::MatrixBase<Derived> &matrix) {
    std::ofstream file(name.c_str());
    file << matrix.format(CSVFormat);
    file.close();
  }
};
