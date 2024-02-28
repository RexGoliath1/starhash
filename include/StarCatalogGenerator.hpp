#ifndef CATALOG_GENERATOR_SH
#define CATALOG_GENERATOR_SH

#include "H5Cpp.h"
#include "yaml-cpp/yaml.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <sys/stat.h>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "Utilities.hpp"
#include "eigen_mods.hpp"

namespace fs = std::filesystem;

// Defaults
inline fs::path app_path = get_executable_path();
inline fs::path base_path = app_path.parent_path();
inline fs::path default_config_path = base_path / "data/config.yaml";

// Constants
const double deg2rad    = M_PI / 180.0;
const double arcsec2deg = (1.0 / 3600.0);
const double mas2arcsec = (1.0 / 1000.0);
const double mas2rad    = mas2arcsec * arcsec2deg * deg2rad;
const double au2km      = 149597870.691;

// Input Catalog Columns
enum {
  RA_J2000 = 0,
  DE_J2000,
  HIP,
  RA_ICRS,
  DE_ICRS,
  PLX,
  PMRA,
  PMDE,
  HPMAG,
  COLOUR,
  SIZE_ELEMS
};

// Some macro defines to debug various functions before valgrid setup
// #define DEBUG_HIP
#define DEBUG_INPUT_CATALOG
#define DEBUG_PM
// #define DEBUG_HASH
// #define DEBUG_GET_NEARBY_STARS
// #define DEBUG_GET_NEARBY_STAR_PATTERNS
// #define DEBUG_PATTERN_CATALOG


const unsigned int hip_rows = 117955;
const unsigned int hip_cols = 10;

// Hash function for Eigen Matricies
// Ignore warnings about std::unary_function and std::binary_function.
// TODO: convert unary_function deprecated.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

template <typename T, typename Result> struct unary_function {
  using argument_type = T;
  using result_type = Result;
};

template <typename T> struct matrix_hash : unary_function<T, std::size_t> {
  std::size_t operator()(T const &matrix) const {
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
  StarCatalogGenerator();
  ~StarCatalogGenerator();

  bool load_pattern_catalog();
  void run();

private:
  fs::path input_catalog_file;
  fs::path output_catalog_file;
  Eigen::MatrixXd input_catalog_data;
  Eigen::MatrixXd proper_motion_data;

  // @brief Post star separation check table with PM corrected ICRS
  Eigen::MatrixXd star_table; 
  // @brief Vector of "Pattern stars" satisfying separation thresholds
  std::vector<int> pattern_stars; 
  CoarseSkyMap coarse_sky_map;
  Eigen::ArrayXd edges;
  bool regenerate;
  unsigned int pattern_size;
  unsigned int num_pattern_angles;
  int total_catalog_stars = 0;

  bool ut_pm;
  unsigned int hip_columns = 10;
  unsigned int hip_header_rows = 53;

  float catalog_jyear;
  float target_jyear;
  std::string year_str;

  // @brief Observer position relative to Barycentric Celestial Reference System (get_earth_ssb.py)
  Eigen::RowVector3d bcrf_position;
  Eigen::MatrixXd bcrf_frame;

  // @brief Default thresholding parameters
  float magnitude_thresh;
  // @brief Minimum angle between 2 stars (ifov degrees or equivilent for double stars
  double min_separation_angle;
  // @brief Minimum norm distance between 2 stars
  double min_separation;
  unsigned int pattern_stars_per_fov;
  unsigned int catalog_stars_per_fov;
  double max_fov_angle;
  double max_fov_dist;
  double max_half_fov_dist;
  unsigned int intermediate_star_bins;
  // @brief TODO: check why int
  int pattern_bins;

  // Global counter for pattern_list
  int pattern_list_size = 0;
  int pattern_list_growth;
  int index_pattern_debug_freq;
  int separation_debug_freq;

  // TODO: Inspect if python is doing things with this.. Currently > INT_MAX so
  // modulo is of -1640531535
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
  const int magic_number = 2654435761;
#pragma GCC diagnostic pop

  void read_yaml(fs::path config_path = default_config_path);

  // Initial catalog transforms
  bool read_input_catalog();
  void convert_hipparcos();
  void sort_star_magnitudes();
  void init_bcrf();
  void correct_proper_motion();

  void filter_star_separation();
  bool is_star_pattern_in_fov(Eigen::MatrixXi &pattern_list, std::vector<int> nearby_star_pattern);
  void get_nearby_stars(Eigen::Vector3d star_vector, std::vector<int> &nearby_stars);
  void get_star_edge_pattern(Eigen::VectorXi pattern);
  void get_nearby_star_patterns(Eigen::MatrixXi &pattern_list, std::vector<int> nearby_stars, int star_id);

  // Intermediate hash table functions
  void init_output_catalog();

  // Final star edge pattern hash table (from paper / code)
  int key_to_index(Eigen::VectorXi hash_code, const unsigned int pattern_bins, const unsigned int catalog_length);
  void generate_output_catalog();

  template <typename Derived>
  void output_hdf5(std::string filepath, std::string dataset, const Eigen::MatrixBase<Derived> &matrix, bool truncate = false) {

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
      throw std::runtime_error("Unknown datatype");
    }

    // Create the data array
    auto arr = new typename Derived::Scalar *[dim[0]];
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
    H5::DataSet pc_dataset =
        h5_file.createDataSet(dataset.c_str(), datatype, ds);

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

#endif // CATALOG_GENERATOR_SH
