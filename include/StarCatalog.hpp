#include <string>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <sys/stat.h>
#include <fstream>
#include <inttypes.h>
#include <memory>
#include <array>
#include <math.h>
#include <Eigen/Dense>
#include <chrono>
#include <ctime>
#include <assert.h>
#include <math.h>
#include <algorithm>
#include <functional>
#include <experimental/filesystem>
#include <list>
#include <unordered_map>
#include <vector>
#include <exception>
#include "H5Cpp.h"

namespace fs = std::experimental::filesystem;

static const fs::path default_hipparcos_path = "/../data/hipparcos.tsv"; // Default relative path
static const fs::path default_catalog_path("/../results/output.h5"); // Default relative path

const unsigned int hip_rows = 117955;

// Hash function for Eigen Matricies 
template<typename T> struct matrix_hash : std::unary_function<T, size_t> {
    std::size_t operator()(T const& matrix) const {
        // Note that it is oblivious to the storage order of Eigen matrix (column- or
        // row-major). It will give you the same hash value for two different matrices if they
        // are the transpose of each other in different storage order.
        size_t seed = 0;
        for (size_t i = 0; i < (size_t)matrix.size(); ++i) {
        auto elem = *(matrix.data() + i);
        seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    return seed;
    }
};

// Coarse sky map (pre computed hash table). This is only used internally to define the database.
using CoarseSkyMap = std::unordered_map<Eigen::Vector3i, std::vector<int>, matrix_hash<Eigen::Vector3i>>;

// Define dynamic eigen binary arrays
typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;

// Define eigen csv formatter
const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

class StarCatalog
{
public:
    StarCatalog(const std::string &in_file, const std::string &out_file);
    explicit StarCatalog();
    ~StarCatalog();

    bool load_pattern_catalog();
    // Pipeline method for everything
    void run_pipeline();


private:
    fs::path input_catalog_file;
    fs::path output_catalog_file;
    Eigen::MatrixXd input_catalog_data;
    Eigen::MatrixXd proper_motion_data;

    // filter_star_separation outputs
    Eigen::MatrixXd star_table; // Post star separation check table with PM corrected ICRS angles for "verificeation stars".
    std::vector<int> pattern_stars; // Vector of "Pattern stars" satisfying separation thresholds

    CoarseSkyMap coarse_sky_map;
    Eigen::ArrayXd edges;

    // TODO: Load dynamically through some kind of configuration
    const unsigned int pattern_size = 4;
    const unsigned int num_pattern_angles = (pattern_size * (pattern_size - 1)) / 2;;

    int total_catalog_stars = 0;
    int number_hippo_stars_bright = 0;

    const double deg2rad = M_PI / 180.0;
    const double arcsec2deg =  (1.0 / 3600.0);
    const double mas2arcsec = (1.0 / 1000.0);
    const double mas2rad = mas2arcsec * arcsec2deg * deg2rad;
    const double au2km = 149597870.691;

    float hip_byear = 1991.25; // Hipparcos Besellian Epoch
    unsigned int hip_columns = 10;
    unsigned int hip_header_rows = 55;
    float current_byear;

    Eigen::MatrixXd bcrf_frame;

    // Default thresholding parameters (Default tetra amounts are in readme)
    float brightness_thresh = 11.4; // Minimum brightness of db
    double min_separation_angle = 0.3; // Minimum angle between 2 stars (ifov degrees or equivilent for dealing with double / close stars)
    double min_separation = std::cos(min_separation_angle * deg2rad); // Minimum norm distance between 2 stars
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

    // TODO: Inspect if python is doing things with this.. Currently > INT_MAX so modulo is of -1640531535
    const int magic_number = 2654435761;

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
    bool is_star_pattern_in_fov(Eigen::MatrixXi &pattern_list, std::vector<int> nearby_star_pattern);
    void get_nearby_stars(Eigen::Vector3d star_vector, std::vector<int> &nearby_stars);
    void get_star_edge_pattern(Eigen::VectorXi pattern);
    void get_nearby_star_patterns(Eigen::MatrixXi &pattern_list, std::vector<int> nearby_stars, int star_id);

    // Intermediate hash table functions
    void init_output_catalog();

    // Final star edge pattern hash table (from paper / code)
    int key_to_index(Eigen::VectorXi hash_code, const unsigned int pattern_bins, const unsigned int catalog_length);
    void generate_output_catalog();


    // Generic Eigen 2 csv writer
    template <typename Derived> void write_to_csv(std::string name, const Eigen::MatrixBase<Derived>& matrix)
    {
        std::ofstream file(name.c_str());
        file << matrix.format(CSVFormat);
        file.close();
    }

};