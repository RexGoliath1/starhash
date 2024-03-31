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

// TODO: A few references to hipparcos to generalize to Gaia / UCAC4

namespace fs = std::filesystem;

// Defaults
inline fs::path app_path = get_executable_path();
inline fs::path base_path = app_path.parent_path();
inline fs::path default_config_path = base_path / "data/config.yaml";

// Constants
const double deg2rad = M_PI / 180.0;
const double arcsec2deg = (1.0 / 3600.0);
const double mas2arcsec = (1.0 / 1000.0);
const double mas2rad = mas2arcsec * arcsec2deg * deg2rad;
const double au2km = 149597870.691;

typedef enum {
  HIP,
  RArad,
  DErad,
  PLX,
  PMRA,
  PMDE,
  HPMAG,
  B_V,
  E_PLX,
  E_PMRA,
  E_PMDE,
  SHP,
  e_B_V,
  V_I,
  RAdeg,
  DEdeg,
  RA_J2000,
  DE_J2000,
  RA_ICRS,
  DE_ICRS,
  CATALOG_COLUMNS
} column;

// Debugging functions used for validation (TODO: Move to config)
// #define DEBUG_HIP
#define DEBUG_INPUT_CATALOG
#define DEBUG_PM
#define DEBUG_HASH
// #define DEBUG_GET_NEARBY_STARS
// #define DEBUG_GET_NEARBY_STAR_PATTERNS
#define DEBUG_STAR_TABLE
#define DEBUG_PATTERN_LIST
#define DEBUG_PATTERN_CATALOG

const unsigned int catalog_rows = 117955;

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
const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision,
                                       Eigen::DontAlignCols, ", ", "\n");

class StarCatalogGenerator {
public:
  StarCatalogGenerator();
  ~StarCatalogGenerator();

  bool load_pattern_catalog();
  void run();

private:
  fs::path input_catalog_file;
  fs::path output_directory;
  fs::path output_catalog_file;
  Eigen::MatrixXd input_catalog_data;
  Eigen::MatrixXd proper_motion_data;

  // @brief Post star separation check table with PM corrected ICRS
  Eigen::MatrixXd star_table;
  Eigen::MatrixXd pat_star_table;
  // @brief Vector of "Pattern stars" satisfying separation thresholds
  std::vector<int> pattern_stars;
  CoarseSkyMap coarse_sky_map;
  Eigen::ArrayXd edges;
  Eigen::ArrayXd pat_angles;
  Eigen::ArrayXd edge_angles;
  int code_size;
  Eigen::Array<uint64_t, Eigen::Dynamic, 1> pat_bin_cast;
  Eigen::Array<uint64_t, Eigen::Dynamic, 1> key_range;
  Eigen::Array<uint64_t, Eigen::Dynamic, 1> indicies;

  bool regenerate;
  unsigned int pattern_size;
  unsigned int num_pattern_angles;
  int total_catalog_stars = 0;

  bool ut_pm;
  bool ut_no_motion;
  float catalog_jyear;
  float target_jyear;
  std::string year_str;

  // @brief Observer position relative to Barycentric Celestial Reference System
  // (get_earth_ssb.py)
  Eigen::RowVector3d bcrf_position;
  Eigen::MatrixXd bcrf_frame;

  // @brief Default min magnitude threshold parameter
  double min_magnitude_thresh;
  // @brief Default max magnitude threshold parameter
  double max_magnitude_thresh;
  // @brief Default parallax thresholding parameters
  double plx_thresh;
  // @brief Minimum angle between 2 stars (Distance)
  double min_separation;
  // @brief Max Pattern stars per fov
  unsigned int pattern_stars_per_fov;
  // @brief Max Catalog stars per fov
  unsigned int catalog_stars_per_fov;
  // @brief Max Expected FOV (Radians)
  double max_fov;
  // @brief Max Expected FOV (Distance)
  double max_fov_dist;
  // @brief Max Expected Half FOV (Distance)
  double max_hfov_dist;
  // @brief Intermediate hash number of bins
  unsigned int intermediate_star_bins;
  // @brief 
  uint64_t pattern_bins;
  // @brief 
  uint64_t catalog_size_multiple;

  // @brief Global counter for pattern_list
  int pattern_list_size = 0;
  // @brief pattern_list dynamic growth
  int pattern_list_growth;
  // @brief pattern printout frequency
  int index_pattern_debug_freq;
  // @brief star sep printout frequency
  int separation_debug_freq;

  // @brief path to debug hash
  fs::path debug_hash_file;

  // TODO: Inspect if python is doing things with this.. Currently > INT_MAX so
  // modulo is of -1640531535
  // @brief magic hash number TODO: Check GMP or Boost::Multiprecision
  const uint64_t magic_number = 2654435761;

  // Sort column used for HIP and HpMag
  static column g_sort_column;

  void read_yaml(fs::path config_path = default_config_path);

  // Initial catalog transforms
  bool read_input_catalog();
  void convert_hipparcos();
  static int sort_compare(const void *a, const void *b);
  void sort_star_columns(column col);
  void init_bcrf();
  void correct_proper_motion();

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
  uint64_t key_to_index(const Eigen::Array<uint64_t, Eigen::Dynamic, 1> hash_code,
                        const uint64_t catalog_length);
  void generate_output_catalog();

  template <typename T> struct hdf5_type;

  template <typename Derived>
  void write_eigen_to_hdf5(const std::string &filename,
                              const std::string &datasetName,
                              const Eigen::MatrixBase<Derived> &matrix) {
    const auto &derivedMatrix = matrix.derived();
    Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> rowMajorMatrix = derivedMatrix;

    // Open or create the file
    H5::H5File file(filename, H5F_ACC_RDWR | H5F_ACC_CREAT);
    hsize_t dimensions[2] = {static_cast<hsize_t>(rowMajorMatrix.rows()), static_cast<hsize_t>(rowMajorMatrix.cols())};

    H5::DataSpace dataspace(2, dimensions);
    auto datatype = hdf5_type<typename Derived::Scalar>::get();

    if (H5Lexists(file.getId(), datasetName.c_str(), H5P_DEFAULT)) {
        // Delete the existing dataset
        file.unlink(datasetName.c_str());
    }

    H5::DataSet dataset = file.createDataSet(datasetName, datatype, dataspace);
    dataset.write(rowMajorMatrix.data(), datatype);
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

template <> struct StarCatalogGenerator::hdf5_type<float> {
  static H5::DataType get() { return H5::PredType::NATIVE_FLOAT; }
};

template <> struct StarCatalogGenerator::hdf5_type<double> {
  static H5::DataType get() { return H5::PredType::NATIVE_DOUBLE; }
};

template <> struct StarCatalogGenerator::hdf5_type<int> {
  static H5::DataType get() { return H5::PredType::NATIVE_INT; }
};

#endif // CATALOG_GENERATOR_SH
