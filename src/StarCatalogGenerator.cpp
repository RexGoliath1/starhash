#include "StarCatalogGenerator.hpp"
#include "ProgressBar.hpp"
#include <algorithm>
#include <assert.h>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <limits>

StarCatalogGenerator::StarCatalogGenerator() {
  // First let's get the catalog length
  read_yaml();

  // Config dependant variables
  num_pattern_angles = (pattern_size * (pattern_size - 1)) / 2;
  max_angle = max_fov; // TODO: When changing to ISA normalize by 180 instead if still relevant. Or better don't use true angle, just dot product


  if (use_star_centroid) {
    pat_edges.resize(pattern_size);
    pat_edges.setZero();
    pat_angles.resize(pattern_size);
    pat_angles.setZero();
    edge_size = pat_edges.size();
    angle_size = pat_angles.size();
    max_edge = std::sin(max_fov / 2);
  } else {
    max_edge = 2 * std::sin(max_fov / 2);
    pat_edges.resize(num_pattern_angles);
    pat_edges.setZero();
    pat_angles.resize(num_pattern_angles);
    pat_angles.setZero();
    // Tetra Techniques
    if (max_measured_norm) {
      edge_size = pat_edges.size() - 1;
      angle_size = pat_angles.size() - 1;
    } else {
      edge_size = pat_edges.size();
      angle_size = pat_angles.size();
    }
  }

  if (use_angles) {
    pat_edge_angles.resize(edge_size + angle_size);
    pat_edge_angles.setZero();
  } else {
    pat_edge_angles.resize(edge_size);
    pat_edge_angles.setZero();
  }

  code_size = pat_edge_angles.size();
  pat_bin_cast.resize(code_size); 
  key_range.resize(code_size);
  indicies.resize(code_size);
  key_range = Eigen::VectorXi::LinSpaced(code_size, 0, code_size - 1).cast<uint64_t>();
  pat_bin_cast = pattern_bins * Eigen::VectorXi::Ones(code_size).cast<uint64_t>();

  double log_max = std::log(UINT64_MAX / 2.0);
  bool will_overflow = code_size * std::log(pattern_bins) > log_max;
  assert(!will_overflow);

  init_bcrf();
}

StarCatalogGenerator::~StarCatalogGenerator() {}

void StarCatalogGenerator::read_yaml(fs::path config_path) {
  YAML::Node config = YAML::LoadFile(config_path.c_str());

  regenerate = config["catalog"]["regenerate"].as<bool>();
  target_jyear = config["catalog"]["target_jyear"].as<double>();
  catalog_jyear = config["catalog"]["catalog_jyear"].as<double>();

  std::vector<double> target_bcrf_position;
  target_bcrf_position = config["catalog"]["target_bcrf_position"].as<std::vector<double>>();
  bcrf_position = Eigen::Map<Eigen::RowVectorXd, Eigen::Unaligned>(target_bcrf_position.data(), target_bcrf_position.size());

  // TODO: Extend to multiple input catalogs
  input_catalog_file = config["catalog"]["input_catalog_file"].as<std::string>();
  input_catalog_file = base_path / input_catalog_file;
  output_directory = config["catalog"]["output_directory"].as<std::string>();
  output_directory = base_path / output_directory;
  output_catalog_file = config["catalog"]["output_catalog_file"].as<std::string>();
  output_catalog_file = output_directory / output_catalog_file;
  pattern_size = config["catalog"]["pattern_size"].as<unsigned int>();
  pattern_stars_per_fov = config["catalog"]["pattern_stars_per_fov"].as<unsigned int>();
  catalog_stars_per_fov = config["catalog"]["catalog_stars_per_fov"].as<unsigned int>();
  min_magnitude_thresh = config["catalog"]["min_magnitude_thresh"].as<double>();
  max_magnitude_thresh = config["catalog"]["max_magnitude_thresh"].as<double>();
  max_fov = deg2rad * config["catalog"]["max_fov_angle"].as<double>();
  max_fov_dist = std::cos(max_fov);
  max_hfov_dist = std::cos(max_fov / 2.0);
  intermediate_star_bins = config["catalog"]["intermediate_star_bins"].as<unsigned int>();
  pattern_bins = config["catalog"]["pattern_bins"].as<int>();
  double min_separation_angle = config["catalog"]["min_separation_angle"].as<double>();
  min_separation = std::cos(min_separation_angle * deg2rad); 
  catalog_size_multiple = config["catalog"]["catalog_size_multiple"].as<unsigned int>();
  use_angles = config["catalog"]["use_angles"].as<bool>();
  max_measured_norm = config["catalog"]["max_measured_norm"].as<bool>();
  quadprobe_max = config["catalog"]["quadprobe_max"].as<uint64_t>();
  use_star_centroid = config["catalog"]["use_star_centroid"].as<bool>();

  primes = config["catalog"]["primes"].as<std::vector<uint64_t>>();

  pattern_list_growth = config["debug"]["pattern_list_growth"].as<int>();
  index_pattern_debug_freq = config["debug"]["index_pattern_debug_freq"].as<int>();
  separation_debug_freq = config["debug"]["separation_debug_freq"].as<int>();
  debug_hash_file = config["debug"]["debug_hash_file"].as<std::string>();
  debug_hash_file = output_directory / debug_hash_file;

  ut_pm = config["unit_test"]["proper_motion"].as<bool>();
  ut_no_motion = config["unit_test"]["no_motion"].as<bool>();

  // Check if the directory exists and create it if not
  if (!fs::exists(output_directory)) {
    fs::create_directories(output_directory);
  }

  // Checks for valid inputs
  std::cout << "input: " << input_catalog_file << std::endl;
  assert(fs::exists(input_catalog_file));
  assert(pattern_size >= 4);

  if (ut_no_motion) {
    year_str = "ut_no_motion";
  } else {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2) << target_jyear;
    year_str = stream.str();
  }
}

bool StarCatalogGenerator::load_pattern_catalog() {
  // TODO: Load from HDF5 and put into star_table and pattern_catalog
  // TODO: Need to save off parameter?
  // TODO: Should the catalog be loading the initial camera parameters?
  // Probably not, but maybe have separate set of catalog parameters as some
  // kind of assert /check Have some other routine that creates a new catalog at
  // runtime if parameters don't match expectations
  std::cout << "Loading existing pattern_catalog " << output_catalog_file
            << std::endl;

  return true;
}

bool StarCatalogGenerator::read_input_catalog() {
  std::string lstr, num;
  std::vector<unsigned int> idx;
  unsigned int rcnt = 0, ccnt = 0;
  unsigned int min_mag_dropped = 0, max_mag_dropped = 0, plx_dropped = 0, col_dropped = 0;

  Eigen::MatrixXd temp_catalog;
  temp_catalog.resize(catalog_rows, CATALOG_COLUMNS);
  temp_catalog.setZero();

  // Check catalog exists
  if (!fs::exists(input_catalog_file)) {
    std::cout << "DNE: " << input_catalog_file << std::endl;
    return false;
  }

  // Skip header info
  std::ifstream data(input_catalog_file.c_str());
  data.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  while (std::getline(data, lstr)) {
    std::stringstream iss(lstr);
    ccnt = 0;

    while (std::getline(iss, num, ',')) {
#ifdef DEBUG_HIP
      std::cout << std::stod(num) << " ";
#endif
      if (ccnt < CATALOG_COLUMNS) {
        temp_catalog(rcnt, ccnt) = std::stod(num);
        ccnt++;
      } else {
        return false;
      }
    }

    // If magnitude is above/below threshold, zero out row and continue
    bool min_mag_check = (double(temp_catalog(rcnt, HPMAG)) <= min_magnitude_thresh);
    bool max_mag_check = (double(temp_catalog(rcnt, HPMAG)) >= max_magnitude_thresh);
    // TODO: Confirm if zero is okay
    bool plx_check = (double(temp_catalog(rcnt, PLX)) > plx_thresh);
    bool col_check = ccnt > 0;

    min_mag_dropped += (unsigned int)min_mag_check;
    max_mag_dropped += (unsigned int)max_mag_check;
    plx_dropped += (unsigned int)plx_check;
    col_dropped += (unsigned int)col_check;

    if (col_check && min_mag_check && max_mag_check && plx_check) {
      idx.push_back(rcnt);
      rcnt++;
    } else if (col_check) {
      temp_catalog.row(rcnt).setZero();
    } else {
      break;
    }

#ifdef DEBUG_HIP
    std::cout << std::endl;
    if (ccnt != CATALOG_COLUMNS)
      std::printf("Expected %u cols, got %u\n", CATALOG_COLUMNS, ccnt);
#endif
  }

  total_catalog_stars += rcnt;

  // Perform resizing of large matricies here
  input_catalog_data = temp_catalog(idx, Eigen::indexing::all);
  proper_motion_data.resize(total_catalog_stars, 3);
  bcrf_frame.resize(total_catalog_stars, 3);

  std::printf("Catalog contains %u bright stars out of %u\n", rcnt, catalog_rows);
  std::printf("Number above min magnitude threshold (Mag >= %f) %u \n", min_magnitude_thresh, min_mag_dropped);
  std::printf("Number below max magnitude threshold (Mag <= %f) %u \n", max_magnitude_thresh, max_mag_dropped);
  std::printf("Number above parallax threshold (Plx > %f) %u \n", plx_thresh, plx_dropped);
  std::printf("Number above column check (cols > 0) %u \n", col_dropped);

  return true;
}

// Filter out dim stars, change some units to radians
void StarCatalogGenerator::convert_hipparcos() {
  input_catalog_data.col(RA_J2000) *= deg2rad;
  input_catalog_data.col(DE_J2000) *= deg2rad;
  input_catalog_data.col(RA_ICRS) *= deg2rad;
  input_catalog_data.col(DE_ICRS) *= deg2rad;
  input_catalog_data.col(PMRA) *= mas2rad;
  input_catalog_data.col(PMDE) *= mas2rad;
  input_catalog_data.col(PLX) *= mas2rad;
}

// Static members must be declared inside of the class cpp
column StarCatalogGenerator::g_sort_column;

int StarCatalogGenerator::sort_compare(const void* a, const void* b) {
    const Eigen::VectorXd* vec_a = static_cast<const Eigen::VectorXd*>(a);
    const Eigen::VectorXd* vec_b = static_cast<const Eigen::VectorXd*>(b);
    if ((*vec_a)(g_sort_column) < (*vec_b)(g_sort_column)) return -1;
    if ((*vec_a)(g_sort_column) > (*vec_b)(g_sort_column)) return 1;
    return 0;
}

void StarCatalogGenerator::sort_star_columns(column col) {
  std::vector<Eigen::VectorXd> vec(input_catalog_data.rows());
  g_sort_column = col;

  for (int64_t i = 0; i < input_catalog_data.rows(); ++i) {
      vec[i] = input_catalog_data.row(i);
  }

  // // Use qsort to sort the vector
  // qsort(vec.data(), vec.size(), sizeof(Eigen::VectorXd), sort_compare);

  // // Assuming input_catalog_data can be accessed via non-const references
  // for (int64_t i = 0; i < input_catalog_data.rows(); i++) {
  //   input_catalog_data.row(i) = vec[i];
  // }

  // for (int64_t i = 0; i < input_catalog_data.rows(); i++)
  //   vec.push_back(input_catalog_data.row(i));

  std::stable_sort(vec.begin(), vec.end(),
            [col](Eigen::VectorXd const &t1, Eigen::VectorXd const &t2) {
              return t1(col) < t2(col);
            });

  // for (int64_t i = 0; i < input_catalog_data.rows(); i++)
  //   input_catalog_data.row(i) = vec[i];
  // Assuming input_catalog_data can be accessed via non-const references
  for (int64_t i = 0; i < input_catalog_data.rows(); i++) {
    input_catalog_data.row(i) = vec[i];
  }

#ifdef DEBUG_INPUT_CATALOG
  fs::path debug_input =
      output_catalog_file.parent_path() / "input_catalog.csv";
  write_to_csv(debug_input, input_catalog_data);
#endif
}

void StarCatalogGenerator::init_bcrf() {
  // Initialize BCRF (Barycentric Celestial Reference System) aka observer position relative to sun
  if (ut_pm) {
    bcrf_position << -0.18428431, 0.88477935, 0.383819; // Earth J2000
  }
  // TODO: Don't be dumb with memory and just use matrix vector multiply
  bcrf_frame = bcrf_position.replicate<catalog_rows, 1>();
}

double angleBetweenVectors(const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
  double dot = a.dot(b);
  double denom = a.norm() * b.norm();
  double value = std::max(-1.0, std::min(1.0, dot / denom));
  return 180.0 / M_PI * std::acos(value); // angle in radians
}

void StarCatalogGenerator::correct_proper_motion() {
  // TODO: Make sure this is appropriate for other catalogs (UCAC4, Tycho, Gaia,
  // etc)
  // TODO: Determine numpy vs eigen 7th decimal place differences in this math
  assert(input_catalog_data.rows() > input_catalog_data.cols());

  // TODO: Move this unit test and use gtest
  // Unit Test to check PM Calculation is within bounds for J2000

  int ra_row = RA_J2000;
  int de_row = DE_J2000;

  if (ut_pm) {
    catalog_jyear = 1991.25;
    target_jyear = 2000.0;
    ra_row = RA_ICRS;
    de_row = DE_ICRS;
  }

  Eigen::MatrixXd plx(input_catalog_data.rows(), 3);
  Eigen::MatrixXd los(input_catalog_data.rows(), 3);
  Eigen::MatrixXd p_hat(input_catalog_data.rows(), 3);
  Eigen::MatrixXd q_hat(input_catalog_data.rows(), 3);

  los.col(0) = input_catalog_data.col(de_row).array().cos() *
               input_catalog_data.col(ra_row).array().cos();
  los.col(1) = input_catalog_data.col(de_row).array().cos() *
               input_catalog_data.col(ra_row).array().sin();
  los.col(2) = input_catalog_data.col(de_row).array().sin();

  p_hat.col(0) = -1 * input_catalog_data.col(ra_row).array().sin();
  p_hat.col(1) = input_catalog_data.col(ra_row).array().cos();
  p_hat.col(2).setZero();

  q_hat.col(0) = -1 * input_catalog_data.col(de_row).array().sin() *
                 input_catalog_data.col(de_row).array().cos();
  q_hat.col(1) = -1 * input_catalog_data.col(de_row).array().sin() *
                 input_catalog_data.col(de_row).array().sin();
  q_hat.col(2) = input_catalog_data.col(de_row).array().cos();

  proper_motion_data =
      (target_jyear - catalog_jyear) *
      (p_hat.array().colwise() * input_catalog_data.col(PMRA).array() +
       q_hat.array().colwise() * input_catalog_data.col(PMDE).array());

  plx = (bcrf_position.transpose() * input_catalog_data.col(PLX).transpose()).transpose();

  if (ut_no_motion) {
    std::cout << "Applying no Proper Motion or Parallax" << std::endl;
    proper_motion_data = los;
  } else {
    proper_motion_data = los + proper_motion_data - plx;
  }

  proper_motion_data.rowwise().normalize();

  if (ut_pm) {
    // Check against PM J2000 calculations provided by VizieR
    Eigen::MatrixXd hproper_motion_data(input_catalog_data.rows(), 3);
    hproper_motion_data.col(0) =
        input_catalog_data.col(DE_J2000).array().cos() *
        input_catalog_data.col(RA_J2000).array().cos();
    hproper_motion_data.col(1) =
        input_catalog_data.col(RA_J2000).array().sin() *
        input_catalog_data.col(DE_J2000).array().cos();
    hproper_motion_data.col(2) = input_catalog_data.col(DE_J2000).array().sin();
    hproper_motion_data.rowwise().normalize();

    std::vector<double> ut_angles(proper_motion_data.rows());
    for (int ii = 0; ii < proper_motion_data.rows(); ++ii) {
      ut_angles[ii] = angleBetweenVectors(proper_motion_data.row(ii),
                                       hproper_motion_data.row(ii));
    }

    // Calculate angles of each row
    int countnan = 0;
    for (int ii = 0; ii < hproper_motion_data.rows(); ++ii) {
      if (std::isnan(ut_angles[ii])) {
        std::cout << "NaN Angle: \t PM ID" << ii << "\t Angle: \t" << ut_angles[ii]
                  << "\tv_prop = " << proper_motion_data.row(ii)
                  << "\tv_truth = " << hproper_motion_data.row(ii) << std::endl;
        ut_angles[ii] = 0.0;
        countnan++;
      }
    }

    std::cout << "NaN Rows: " << countnan << std::endl;

    // Calculate min, mean, and max
    double min_norm = *std::min_element(ut_angles.begin(), ut_angles.end());
    double max_norm = *std::max_element(ut_angles.begin(), ut_angles.end());
    double mean_norm =
        std::accumulate(ut_angles.begin(), ut_angles.end(), 0.0) / ut_angles.size();

    // Calculate median
    std::sort(ut_angles.begin(), ut_angles.end());
    double median_norm = ut_angles[ut_angles.size() / 2];

    // Calculate standard deviation (sigma)
    double sq_sum =
        std::inner_product(ut_angles.begin(), ut_angles.end(), ut_angles.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / ut_angles.size() - mean_norm * mean_norm);

    // Output results
    std::cout << "Proper Motion Angle Errors (Degrees): " << std::endl;
    std::cout << "Min:\t" << min_norm << "\tMax:\t" << max_norm << "\tMean:\t"
              << mean_norm << "\tMedian:\t" << median_norm << "\tSigma:\t"
              << stdev << std::endl;

    exit(1);
  }

#ifdef DEBUG_PM
  fs::path debug_pm = output_catalog_file.parent_path() /
                      ("proper_motion_data_" + year_str + ".csv");
  write_to_csv(debug_pm, proper_motion_data);
#endif
}

void StarCatalogGenerator::filter_star_separation() {
  /* Separate stars into "pattern stars" vs "verification stars"
      "Pattern Stars": Star idicies that are minimum separated and have less
     than pattern_stars_per_fov stars in any single fov. "Verification Stars":
     Star idicies that are minimum separated and have less than
     catalog_stars_per_fov stars in any single fov. This is more stars than
     actually used for attitude solution. Why? .. Reason: This table is used to
     double check you're not using a potentially "bad" edge solution by
     confusing stars with each other. Why? You want to limit the number of
     patterns to search through, so we make sure to reduce stars to min
     separation. The min sep angle is large enough to reduce most effects of
     double stars (except for unresolvable ~ ifov/2 ones. TODO?). But, when we
     go to process an image, we may have nearby stars that "distort" the solved
     solution. Why? By using a for loop that iterates through brightest stars
     only one time, we have accepted stars near other ones that have not been
     tested yet, but have cut out the next set. A solution to this could be to
     keep iterating on this set until we remove all "near" stars with some
     minimum radius distance. (TODO?) Another solution could be to build a
     Vorinoi space that only use stars lower that this min sep threshold. That
     make hipparcos very unusable. To be clear, this absolutely is a tradeoff of
     the algorithm. The original paper does not do this, but instead !!only!!
     accepts single star relative area solutions (no stars near patterns stars.
     default thresh of 0.005) 0.005 is a normalized value take from the edge
     computed (normed to max edge as usual). This could be bad, in that stars
     that are near other stars will never contribute. This needs analysis of
     camera geometry and if solution is bad enough to merit more thought. Steve
     G Comment: This seems like an okay trade off if multiple patterns can be
     used for the final solution. Hard kill if otherwise though. All other kinds
     of solutions (Triangle Pyramid, ISA values) have this problem. I think the
     right mitigation is to use several clusters, but that might need compute
     limits. Something like random forest or XGB might be good for finding an
     optimal solution space here.
  */

  // Is star "pattern star" (updated in loop)
  ArrayXb ang_pattern_idx = ArrayXb::Constant(proper_motion_data.rows(), false);

  // Is star "verification star" (updated in loop)
  ArrayXb ang_verify_idx = ArrayXb::Constant(proper_motion_data.rows(), false);

  // Verification vector used to index final star table before creation of
  // catalog
  std::vector<int> verification_stars;

  // Brightest star (after sort) is always in pattern / verification
  ang_verify_idx(0) = true;
  ang_pattern_idx(0) = true;

  pattern_stars.push_back(0);
  verification_stars.push_back(0);

  double num_stars_in_fov = -1;

  ProgressBar pb(proper_motion_data.rows());
  for (int ii = 1; ii < proper_motion_data.rows(); ii++) {
    pb.show_progress(ii);

    // Determine angle between current star and all other stars
    Eigen::VectorXd current_star = proper_motion_data.row(ii);
    Eigen::VectorXd current_star_angles =
        proper_motion_data.topRows(ii) * current_star;

    // Find idicies that pass angle test
    ArrayXb separated_star_indicies =
        (current_star_angles.array() < min_separation);

    // Explanation: Apply min_sep check to only marked pattern stars
    ArrayXb separated_pattern_indicies =
        (separated_star_indicies && ang_pattern_idx.topRows(ii)) ||
        !ang_pattern_idx.topRows(ii);

    // Explanation: Apply min_sep check to only marked verification stars
    ArrayXb separated_verify_indicies =
        (separated_star_indicies && ang_verify_idx.topRows(ii)) ||
        !ang_verify_idx.topRows(ii);

    // Pattern test: Limit "close stars" by number of stars used for pattern
    // matching per FOV

    // Check that all pattern stars are close
    if (separated_pattern_indicies.all()) {

      // separated_pattern_indicies is reused to count number of close stars
      // Multiplied by zero if non-pattern star to separate out non-patterns.
      Eigen::ArrayXd current_pattern_star_angles =
          current_star_angles.array() *
          ang_pattern_idx.topRows(ii).cast<double>().array();

      separated_pattern_indicies =
          current_pattern_star_angles.array() > max_hfov_dist;

      num_stars_in_fov = separated_pattern_indicies.cast<int>().sum();

      if (num_stars_in_fov < pattern_stars_per_fov) {
        ang_verify_idx(ii) = true;
        ang_pattern_idx(ii) = true;

        // Explanation: We are later indexing with this into an already indexed
        // matrix of verification stars. This is how tetra does it, but this
        // could likely be improved for readability
        pattern_stars.push_back(verification_stars.size());

        verification_stars.push_back(ii);
        continue;
      }
    }

    // Verification test: Limit number of "close stars" used for further
    // verification per FOV
    if (separated_verify_indicies.all()) {

      Eigen::ArrayXd current_verify_star_angles =
          current_star_angles.array() *
          ang_verify_idx.topRows(ii).cast<double>().array();

      separated_verify_indicies =
          current_verify_star_angles.array() > max_hfov_dist;

      num_stars_in_fov = separated_verify_indicies.cast<int>().sum();

      if (num_stars_in_fov < catalog_stars_per_fov) {
        ang_verify_idx(ii) = true;
        verification_stars.push_back(ii);
      }
    }
  }

  star_table = proper_motion_data(verification_stars, Eigen::indexing::all);
  pat_star_table = star_table(pattern_stars, Eigen::indexing::all);
  std::cout << std::endl;
  std::cout << "Found " << star_table.rows() << " verification stars for catalog." << std::endl;
  std::cout << "Found " << pattern_stars.size() << " pattern stars for catalog." << std::endl;
#ifdef DEBUG_STAR_TABLE
  fs::path debug_table;
  debug_table = output_directory / ("star_table_" + year_str + ".csv");
  write_to_csv(debug_table, star_table);

  debug_table = output_directory / ("pattern_star_table_" + year_str + ".csv");
  write_to_csv(debug_table, pat_star_table);

  Eigen::MatrixXd verif_input_catalog_data = input_catalog_data(verification_stars, Eigen::indexing::all);
  debug_table = output_directory / ("verif_input_catalog_" + year_str + ".csv");
  write_to_csv(debug_table, verif_input_catalog_data);

  Eigen::MatrixXd pattern_input_catalog_data = input_catalog_data(verification_stars, Eigen::indexing::all);
  debug_table = output_directory / ("pattern_input_catalog_" + year_str + ".csv");
  write_to_csv(debug_table, pattern_input_catalog_data);
#endif 
}

void StarCatalogGenerator::get_nearby_stars(Eigen::Vector3d star_vector,
                                            std::vector<int> &nearby_stars) {
  // Vector to fill in with hash codes for indexing
  Eigen::Vector3i low_codes, high_codes;
  // Eigen::MatrixXi codes = Eigen::MatrixXi(pat_star_table.cols(), pat_star_table.rows());

  Eigen::Vector3i zeros = Eigen::Vector3i::Zero();
  Eigen::Vector3i bin_limit = int(2 * intermediate_star_bins) * Eigen::Vector3i::Ones();
  std::vector<int> star_ids;
  std::vector<int> all_star_ids;

  // TODO: Why do we use radians here for radius error ...
  low_codes = (intermediate_star_bins * (star_vector.array() + 1.0 - max_fov)).cast<int>();
  low_codes = low_codes.array().max(zeros.array());
  high_codes = (1 + (intermediate_star_bins * (star_vector.array() + 1.0 + max_fov))).cast<int>();
  high_codes = high_codes.array().min(bin_limit.array());

#ifdef DEBUG_GET_NEARBY_STARS
  std::cout << "Low Codes: " << low_codes.transpose() << std::endl;
  std::cout << "High Codes: " << high_codes.transpose() << std::endl;
#endif

  // For all nearby star hash codes (+/- FOV) get list of nearby stars for new
  // hash map
  // TODO: Codes should never be negative. May want to cast all associated
  // hashes to unsigned
  for (int ii = low_codes(0); ii < high_codes(0); ii++) {
    for (int jj = low_codes(1); jj < high_codes(1); jj++) {
      for (int kk = low_codes(2); kk < high_codes(2); kk++) {
        Eigen::Vector3i code;
        code[0] = ii;
        code[1] = jj;
        code[2] = kk;

        // TODO: temp_coarse_sky_map[hash_code].remove(pattern[0])
        // Do we need to remove stars from the unordered map?
        //
        star_ids = coarse_sky_map[code];

#ifdef DEBUG_GET_NEARBY_STARS
        std::cout << "coarse_sky_map[" << code.transpose() << "]: [";
#endif

        for (const int star_id : star_ids) {

#ifdef DEBUG_GET_NEARBY_STARS
          std::printf("%d, ", star_id);
#endif

          all_star_ids.push_back(star_id);
          double dp = star_vector.dot(star_table.row(star_id));
          if ((dp > max_fov_dist)) {
          //if ((dp > max_fov_dist) && (dp < min_separation)) {
            nearby_stars.push_back(star_id);
          }
        }

#ifdef DEBUG_GET_NEARBY_STARS
        std::cout << "]" << std::endl;
#endif
      }
    }
  }
}

bool StarCatalogGenerator::is_star_pattern_in_fov(
    Eigen::MatrixXi &pattern_list, std::vector<int> nearby_star_pattern) {
  // Make sure passed in Star ID combination matches pattern_size
  if (nearby_star_pattern.size() != pattern_size) {
    std::cout << "nearby_star_pattern.size() : " << nearby_star_pattern.size() << std::endl;
    return false;
  }

  // Checking all pair angles
  std::vector<int> star_pair, selector(pattern_size);
  std::fill(selector.begin(), selector.begin() + 2, 1);

  bool all_stars_in_fov = true;
  double dot_p;
  unsigned int cnt = 0;

  // TODO: Make Fixed Vector3d. Requires proper_motion_data to be Matrix3d
  Eigen::VectorXd star_vector_1, star_vector_2;

  do {
    // Check if number checked exceeds number of actual permutations
    if (cnt > num_pattern_angles)
      throw std::runtime_error("Star FOV check exceeded expected permutations\n");

    for (unsigned int ii = 0; ii < pattern_size; ii++) {
      if (selector[ii]) {
        star_pair.push_back(nearby_star_pattern[ii]);
      }
    }

    // Filter out stars outsid>e FOV and compute edges
    star_vector_1 = star_table.row(star_pair[0]);
    star_vector_2 = star_table.row(star_pair[1]);
    dot_p = star_vector_1.dot(star_vector_2);

    if (dot_p < max_fov_dist) {
      all_stars_in_fov = false;
      star_pair.clear();
      break;
    }

    star_pair.clear();
    cnt++;
  } while (std::prev_permutation(selector.begin(), selector.end()));

  return all_stars_in_fov;
}

void StarCatalogGenerator::get_star_centroid_pattern(Eigen::VectorXi pattern) {
  assert(pattern.size() == pattern_size);
  Eigen::MatrixXd centroid_vectors(pattern_size, 3);
  Eigen::Vector3d centroid_vector;
  centroid_vector.setZero();
  std::vector<std::pair<Eigen::VectorXd, double>> vec(pattern_size);

  // First find radial edges from centroid
  for (unsigned int ii = 0; ii < pattern_size; ii++) {
    centroid_vectors.row(ii) = star_table.row(pattern[ii]);
  }

  centroid_vector = centroid_vectors.colwise().mean();
  Eigen::MatrixXd replicated_centroid_vector = centroid_vector.transpose().replicate(pattern_size, 1);
  centroid_vectors -= replicated_centroid_vector;

  pat_edges = centroid_vectors.rowwise().norm();

  for (unsigned int ii = 0; ii < pattern_size; ii++ ) {
    assert(pat_edges(ii) > 0);
    if (pat_edges(ii) > 0) { // Avoid division by zero
        centroid_vectors.row(ii) /= pat_edges(ii);
    } else {
      // TODO: Skip or require more interesting patterns?
      std::cout << "Centroid Vector is zero.." << std::endl;
    }
    vec[ii] = std::make_pair(centroid_vectors.row(ii), pat_edges[ii]);
  }

  // Sort the vector of pairs by the norm
  std::stable_sort(vec.begin(), vec.end(),
                   [](const std::pair<Eigen::VectorXd, double>& t1, const std::pair<Eigen::VectorXd, double>& t2) {
                       return t1.second < t2.second;
                   });

  // Assign the sorted rows back to centroid_vectors
  for (unsigned int ii = 0; ii < pattern_size; ii++) {
      centroid_vectors.row(ii) = vec[ii].first;
      pat_edges[ii] = vec[ii].second / max_edge;
  }

  // Now's the tricky part... Map dot and cross product to circle quadrants. No trig functions.
  // pat_angles represents the counter-clockwise rotation from largest edge to smallest
  for (unsigned int ii = 0; ii < pattern_size; ii++) {
    auto next_index = (ii + 1) % pattern_size;
    Eigen::Vector3d v1 = centroid_vectors.row(ii);
    Eigen::Vector3d v2 = centroid_vectors.row(next_index);

    auto angle = v1.dot(v2);
    auto direction = v1.cross(v2).norm();
    assert((-1 <= angle) || (angle <= 1));
    assert((-1 <= direction) || (direction <= 1));

    // Quadrant 1
    if (angle >= 0 && direction >= 0)
      pat_angles[ii] = 0.25 * (1 - angle);
    // Quadrant 2
    else if (angle <= 0 && direction >= 0)
      pat_angles[ii] = 0.25 - 0.25 * angle;
    // Quadrant 3
    else if (angle <= 0 && direction <= 0)
      pat_angles[ii] = 0.5 + 0.25 * (1 + angle);
    // Quadrant 4
    else if (angle >= 0 && direction <= 0)
      pat_angles[ii] = 0.75 + 0.25 * angle;
    // ???
    else
      throw std::runtime_error("Unknown centroid angle quadrant");

    assert((-1 <= pat_angles[ii]) || (pat_angles[ii] <= 1));
  }

  if (use_angles) {
    pat_edge_angles.head(edge_size) = pat_edges.head(edge_size);
    pat_edge_angles.tail(angle_size) = pat_angles.head(angle_size);
  } else {
    pat_edge_angles = pat_edges.head(edge_size);
  }

  // std::cout << pat_edge_angles.transpose() << std::endl;

}

void StarCatalogGenerator::get_star_edge_pattern(Eigen::VectorXi pattern) {
  assert(pattern.size() == pattern_size);
  std::vector<int> star_pair, selector(pattern_size);

  // Only checking all pair angles. This selector only picks 2 vectors at a time, through all permutations of some pattern size.
  std::fill(selector.begin(), selector.begin() + 2, 1);

  double dot_p;
  unsigned int cnt = 0;

  // TODO: Make Fixed Vector3d. Requires proper_motion_data to be Matrix3d
  Eigen::VectorXd star_vector_1, star_vector_2;

  do {
    // Check if number checked exceeds number of actual permutations
    if (cnt > num_pattern_angles)
      throw std::out_of_range(
          "Star FOV check exceeded expected permutations\n");

    for (unsigned int ii = 0; ii < pattern_size; ii++) {
      if (selector[ii]) {
        star_pair.push_back(pattern[ii]);
      }
    }

    // Filter out stars outside FOV and compute edges
    star_vector_1 = star_table.row(star_pair[0]);
    star_vector_2 = star_table.row(star_pair[1]);
    dot_p = star_vector_1.dot(star_vector_2);

    star_vector_1 -= star_vector_2;
    pat_edges[cnt] = star_vector_1.norm();
    pat_angles[cnt] = std::acos(dot_p);

    // If star pattern contains angles outside FOV, somthing went wrong in prior
    // pattern_list creation
    assert(dot_p > max_fov_dist);


    star_pair.clear();
    cnt++;
  } while (std::prev_permutation(selector.begin(), selector.end()));

  // If edge count != num_pattern_angles, expected combination not correct
  assert(cnt == num_pattern_angles);

  std::stable_sort(pat_edges.begin(), pat_edges.end());
  std::stable_sort(pat_angles.begin(), pat_angles.end());

  if (max_measured_norm) {
    pat_edges /= pat_edges.maxCoeff();
    pat_angles /= pat_angles.maxCoeff();
  } else {
    pat_edges /= max_edge;
    pat_angles /= max_angle;
  }

  // Remove last element of pat_edges.or angles (the one used to normalize)
  if (use_angles) {
    pat_edge_angles.head(edge_size) = pat_edges.head(edge_size);
    pat_edge_angles.tail(angle_size) = pat_angles.head(angle_size);
    // std::cout << "\n" << std::endl;
    // std::cout << "pat_edges. " << pat_edges.transpose() << std::endl;
    // std::cout << "angles: " << pat_angles.transpose() << std::endl;
    // std::cout << "pat_edge_angles: " << pat_edge_angles.transpose() << std::endl;
  } else {
    pat_edge_angles = pat_edges.head(edge_size);
  }
}

uint64_t modularExponentiation(uint64_t base, uint64_t exponent, uint64_t modulus) {
    uint64_t result = 1;
    base %= modulus;

    for (uint64_t i = 0; i < exponent; ++i) {
        result = (result * base) % modulus;
    }

    return result;
}

// uint64_t modularMultiplication(uint64_t a, uint64_t b, uint64_t modulus) {
//     uint64_t result = 0;
//     a %= modulus;
// 
//     for (uint64_t i = 0; i < b; ++i) {
//         result = (result + a) % modulus;
//     }
// 
//     return result;
// }

uint64_t modularMultiplication(uint64_t a, uint64_t b, uint64_t modulus) {
    uint64_t result = 0;
    a %= modulus;

    int iterations = 64 - __builtin_clzll(b); // Count the number of bits in 'b'

    for (int i = 0; i < iterations; ++i) {
        if (b & 1) { // If the least significant bit of 'b' is 1
            result = (result + a) % modulus;
        }
        a = (2 * a) % modulus;
        b >>= 1; // Right shift 'b' to divide by 2
    }

    return result;
}

uint64_t StarCatalogGenerator::key_to_index(const Eigen::Array<uint64_t, Eigen::Dynamic, 1> hash_code,
                      const uint64_t catalog_length) {
    
    uint64_t sum = 0;
    for (int i = 0; i < code_size; ++i) {
        uint64_t expResult = modularExponentiation(pattern_bins, i, catalog_length);
        uint64_t term = modularMultiplication(hash_code[i], expResult, catalog_length);
        sum = (sum + term) % catalog_length;
    }

    // Final multiplication by magic number and modulo operation
    return modularMultiplication(sum, magic_number, catalog_length);
}

// TODO: May have to go through and convert all integers into uint64_t
uint64_t StarCatalogGenerator::key_to_index_edges(const Eigen::Array<uint64_t, Eigen::Dynamic, 1> hash_code,
                                       const uint64_t catalog_length) {
  indicies = hash_code.array() * Eigen::pow(pat_bin_cast.array(), key_range.array());

  indicies = magic_number * indicies;
  for(int ii = 0; ii < code_size; ii++) {
    indicies[ii] = indicies[ii] % static_cast<uint64_t>(catalog_length);
  }

  uint64_t sum = indicies.sum();

  // TODO: Carefully check python types to see if this matches TETRA Logic
  return ((sum * magic_number) % static_cast<uint64_t>(catalog_length));
}

void StarCatalogGenerator::get_nearby_star_patterns(
    Eigen::MatrixXi &pattern_list, std::vector<int> nearby_stars, int star_id) {

  int n = nearby_stars.size();
  std::vector<int> nearby_star_pattern;
  std::vector<int> selector(n);

  // pattern_size - 1 : Find combinations of stars with current star
  std::fill(selector.begin(), selector.begin() + pattern_size - 1, 1);

  do {
    nearby_star_pattern.push_back(star_id);
    for (int ii = 0; ii < n; ii++) {
      if (selector[ii]) {
        nearby_star_pattern.push_back(nearby_stars[ii]);
      }
    }

    // TODO: Start being more picky. Check max angles
    if (is_star_pattern_in_fov(pattern_list, nearby_star_pattern)) {
#ifdef DEBUG_GET_NEARBY_STAR_PATTERNS
      std::cout << "nearby_star_pattern = ";
      for (auto pat: nearby_star_pattern) {
        std::cout << pat << ", ";
      }
      std::cout << std::endl;
#endif

      int *pat_ptr = &nearby_star_pattern[0];
      Eigen::Map<Eigen::VectorXi> star_pattern_vec(pat_ptr,
                                                   nearby_star_pattern.size());

      // Add pattern to pattern_list
      pattern_list_size++;
      if (pattern_list_size > pattern_list.rows()) {
        // pattern_list.resize(pattern_list.rows() + pattern_list_growth,
        // Eigen::NoChange);
        pattern_list.conservativeResize(
            pattern_list.rows() + pattern_list_growth, Eigen::NoChange);
      }
      pattern_list.row(pattern_list_size - 1) = star_pattern_vec;
    }

    nearby_star_pattern.clear();
  } while (std::prev_permutation(selector.begin(), selector.end()));
}

void StarCatalogGenerator::init_output_catalog() {
  Eigen::MatrixXi codes = Eigen::MatrixXi(pat_star_table.cols(), pat_star_table.rows());
  codes = (static_cast<double>(intermediate_star_bins) * (pat_star_table.array() + 1)).cast<int>();

// Debug IO
#ifdef DEBUG_HASH
  const Eigen::IOFormat fmt(Eigen::FullPrecision, Eigen::DontAlignCols, ", ");
  fs::remove(debug_hash_file);
  std::ofstream ofs(debug_hash_file);
  ofs.close();
  YAML::Node node;
  std::ofstream fout(debug_hash_file.c_str());

  std::stringstream ss;
#endif

  for (int ii = 0; ii < codes.rows(); ii++) {
#ifdef DEBUG_HASH
    // std::cout << "Star Table Row: " << star_table.row(ii).format(fmt) << std::endl;
    // std::cout << "Codes Row: " << codes.row(ii).format(fmt) << std::endl;

    for (int jj = 0; jj < codes.cols(); jj++) {
      ss << codes(ii, jj);
      if (jj < codes.cols() - 1) {
        ss << ", ";
      }
    }

    // std::cout << ss.str() << std::endl;

  node[ss.str()].push_back(std::to_string(pattern_stars[ii]));
  ss.str("");
#endif

    coarse_sky_map[static_cast<Eigen::Vector3i>(codes.row(ii))].push_back(pattern_stars[ii]);
  }

#ifdef DEBUG_HASH
  fout << node;
  fout.close();

  Eigen::Vector3i test(3, 1);
  test = ((double)intermediate_star_bins * (star_table.row(0).array() + 1)).cast<int>();
  // std::cout << "List: ";
  // const std::vector<int> llist = coarse_sky_map[test];
  // if (llist.empty())
  //   std::cout << "Hash map is empty for tested code/key" << std::endl;
  // else {
  //   for (const auto &item : llist)
  //     std::cout << item << ' ';
  //   std::cout << '\n';
  // }
#endif
}

void pop_vector(std::vector<int>& vec, int value) {
    auto it = std::find(vec.begin(), vec.end(), value);
    assert(it != vec.end());
    vec.erase(it);
}

void StarCatalogGenerator::generate_output_catalog() {
  std::vector<int> nearby_stars;
  std::vector<int> nearby_star_combos;
  Eigen::Vector3d star_vector;
  Eigen::Matrix3d star_vector_combos;
  int star_id;
  uint64_t hash_index;

  // TODO: uint16 matrix class necessary? (Could pull this up one level)
  Eigen::MatrixXi pattern_list(1, pattern_size);
  Eigen::VectorXi pattern(pattern_size);
  uint64_t quadprobe_count;

#if defined(DEBUG_GET_NEARBY_STARS) || defined(DEBUG_GET_NEARBY_STAR_PATTERNS)
  time_t tstart;
#endif

  std::cout << "Generating all patterns " << std::endl;
  ProgressBar pb1(pattern_stars.size());

  for (long unsigned int ii = 0; ii < pattern_stars.size(); ii++) {
    pb1.show_progress(ii);

    nearby_stars.clear();
    star_id = pattern_stars[ii];
    star_vector = star_table.row(star_id);

    Eigen::Vector3i code = ((intermediate_star_bins) * (star_vector.array() + 1.0)).cast<int>();
    // TODO: Matching Tetra here, but doesn't this completely remove the star from further patterns?
    pop_vector(coarse_sky_map[code], star_id);

#ifdef DEBUG_GET_NEARBY_STARS
    tstart = time(0);
#endif
    // For each star kept for pattern matching and verificaiton, find all nearby
    // stars in FOV
    get_nearby_stars(star_vector, nearby_stars);

    if (nearby_stars.size() < (pattern_size - 1))
      continue;

#ifdef DEBUG_GET_NEARBY_STARS
    std::cout << "get_nearby_stars took " << difftime(time(0), tstart)
              << " Seconds." << std::endl;
#endif

#ifdef DEBUG_GET_NEARBY_STAR_PATTERNS
    tstart = time(0);
#endif
    // For all stars nearby, find each star pattern combination (pattern_size)
    // If pattern contains star angles within FOV limits, add to pattern_list
    get_nearby_star_patterns(pattern_list, nearby_stars, star_id);

#ifdef DEBUG_GET_NEARBY_STAR_PATTERNS
    std::cout << "get_nearby_star_patterns took " << difftime(time(0), tstart)
              << " Seconds." << std::endl;
#endif
  }

  std::cout << "Done. Resizing pattern list and allocating catalog" << std::endl;

  pattern_list.conservativeResize(pattern_list_size, Eigen::NoChange);

#ifdef DEBUG_GET_NEARBY_STAR_PATTERNS
  std::printf("Found %d patterns \n", pattern_list_size);
#endif

#ifdef DEBUG_PATTERN_LIST
  fs::path debug_pl = output_catalog_file.parent_path() /
                      ("pattern_list_" + year_str + ".csv");
  write_to_csv(debug_pl, pattern_list);
#endif

  // TODO: Move this into higher level Class / Structure. This is our catalog
  auto it = std::lower_bound(primes.begin(), primes.end(), catalog_size_multiple * static_cast<uint64_t>(pattern_list.rows()));
  if (it == primes.end()) {
    // Handle the case where all primes are smaller than catalog_size
    throw std::runtime_error("No prime number found in the vector that is greater than the catalog size.");
  } else {
    catalog_length = *it;
  }

  // WARNING: This is not how Tetra does this. They init to zeros.. But that is
  // (possibly) a legitimate star in the pattern Starhash inits to -1 (TODO:
  // Macro of -1 ) to avoid star ID conflicts Other TODO: This usees a base
  // matrix of double which I think uses more memory than needed (but is nice).
  // Other TODO: May want to copy unordered map logic for output
  // pattern_catalog. Could reduce lookup timing. (currently sparse quad probing
  // of large matrix)
  Eigen::MatrixXi pattern_catalog = Eigen::MatrixXi(catalog_length, pattern_size).setZero();
  // Eigen::MatrixXi pattern_catalog = -1 * Eigen::MatrixXi::Ones(catalog_length, pattern_size);

  std::cout << std::endl;
  std::cout << "Inputting Hash into catalog" << std::endl;

  // For all patterns in pattern_list, find hash and insert into pattern_catalog
  ProgressBar pb2(pattern_list.rows());
  for (long unsigned int ii = 0; ii < (unsigned int)pattern_list.rows(); ii++) {
    pb2.show_progress(ii);

    // For each pattern, get pat_edges.
    pattern = pattern_list.row(ii);

    if (use_star_centroid) {
      get_star_centroid_pattern(pattern);
    } else {
      get_star_edge_pattern(pattern);
    }

    // Eigen::VectorXi hash_code = (pat_edges.Eigen::seqN(0, pat_edges.size() - 1)) * (double)pattern_bins).cast<int>();
    Eigen::Array<uint64_t, Eigen::Dynamic, 1> hash_code(code_size);
    hash_code = (pat_edge_angles * (double)pattern_bins).cast<uint64_t>();
    // std::cout << "hash_code " << hash_code.transpose() << std::endl;

    // Must use first implementation to avoid uint64 overflow due to high exponentials with bigger array
    hash_index = key_to_index(hash_code, catalog_length);
    // if (use_angles) {
    //   hash_index = key_to_index(hash_code, catalog_length);
    // } else {
    //   hash_index = key_to_index_edges(hash_code, catalog_length);
    // }

#ifdef DEBUG_PATTERN_CATALOG
    // std::printf("pattern[%llu].hash_code = [", hash_index);
    // for (auto code: hash_code) {
    //   std::cout << code << ", ";
    // }
    // std::cout << "]";

    // std::cout << " (pat_edges. [";
    // for (auto edge: pat_edges. {
    //   std::cout << edge << ", ";
    // }
    // std::cout << "])" << std::endl;
    // exit(-1);
#endif

    // Use quadratic probing to find an open space in the pattern catalog to insert
    // TODO: bound quadratic probe
    quadprobe_count = 0;

    while (true) {
      uint64_t index = (hash_index + (uint64_t)std::pow(quadprobe_count, 2)) % (uint64_t)pattern_catalog.rows();
      if (pattern_catalog.row(index).sum() == 0) {
        pattern_catalog.row(index) = pattern;
        break;
      }
      quadprobe_count++;
      if (quadprobe_count > quadprobe_max) {
        std::cout << "index: " << index << std::endl;
        std::cout << quadprobe_count << " > " << quadprobe_max << std::endl;
        throw std::runtime_error(&"Quadprobe indexing exceeded " [ quadprobe_max]);
      }
    }
  }

  write_eigen_to_hdf5(output_catalog_file, "input_catalog_data", input_catalog_data);
  write_eigen_to_hdf5(output_catalog_file, "star_table", star_table);
  write_eigen_to_hdf5(output_catalog_file, "pattern_catalog", pattern_catalog);
  // output_hdf5(output_catalog_file, "proper_motion_data", proper_motion_data);

#ifdef DEBUG_PATTERN_CATALOG
  fs::path debug_catalog = output_catalog_file.parent_path() /
                           ("pattern_catalog_" + year_str + ".csv");
  write_to_csv(debug_catalog, pattern_catalog);
#endif
}

void StarCatalogGenerator::run() {
  // Load pre-existing catalog if exists, otherwise create new database (hash
  // table)
  if (!fs::exists(output_catalog_file) || regenerate) {

    std::cout << "Reading Hipparcos Catalog" << std::endl;
    assert(read_input_catalog());

    std::cout << "Sorting Stars by HIP" << std::endl;
    sort_star_columns(column::HIP);

    std::cout << "Sorting Stars by Magnitude" << std::endl;
    sort_star_columns(column::HPMAG);

    std::cout << "Convert Hipparcos" << std::endl;
    convert_hipparcos();

    std::cout << "Correcting Proper Motion" << std::endl;
    correct_proper_motion();

    std::cout << "Filtering Star Separation" << std::endl;
    filter_star_separation();

    std::cout << "Initializing output catalog" << std::endl;
    init_output_catalog();

    std::cout << "Generating output catalog" << std::endl;
    generate_output_catalog();
    
  } else {
    std::cout << "Loading existing output catalog" << std::endl;
    load_pattern_catalog();
  }
}
