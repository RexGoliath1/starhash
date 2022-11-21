#include "starhash.hpp"

/* SBG List of Major TODO's */
// 1. Vector -> Array 
//      - Limit vector growth.
//      - Need to determine max star neighborhoods for given FOV / Star Catalog Parameters
//      - This is only neccessary for FSW, not this database generation
// 2. Possible double to float conversion (in FSW, fine here. MatrixXd -> MatrixXf).
// 3. Make invariant to catalog (GAIA + BCBS import)
// 4. For all vectors that can't become arrays (do these exist?), make sure to explicitly clear before losing scope (break, continue, etc).
// 5. Run Code coverage / static analysis to check above works.

namespace fs = std::experimental::filesystem;

/* Helpful debug flags */
// #define DEBUG_HIP 1
// #define DEBUG_PM
// #define DEBUG_CSV_OUTPUTS
// #define DEBUG_HASH
// #define DEBUG_CATALOG_GENERATE_NEIGHBORS

const double deg2rad = M_PI / 180.0;
const double arcsec2deg =  (1.0 / 3600.0);
const double mas2arcsec = (1.0 / 1000.0);
const double mas2rad = mas2arcsec * arcsec2deg * deg2rad;
const double au2km = 149597870.691;

// Hipparcos Info
const fs::path default_hip_path("/../data/hipparcos.tsv"); // Default relative path
const fs::path default_catalog_path("/../results/output.h5"); // Default relative path
const float hip_byear = 1991.25; // Hipparcos Besellian Epoch
const unsigned int hip_columns = 10;
const unsigned int hip_rows = 117955;
const unsigned int hip_header_rows = 55;
// TODO: Inspect if python is doing things with this.. Currently > INT_MAX so modulo is of -1640531535
const int magic_number = 2654435761;

// Tetra thresholding
//const double default_b_thresh = 6.5; // Minimum brightness of db
//const double min_separation_angle = 0.5; // Minimum angle between 2 stars (ifov degrees)
//const double min_separation = std::cos(min_separation_angle * deg2rad); // Minimum norm distance between 2 stars
//const unsigned int pattern_stars_per_fov = 10;
//const unsigned int catalog_stars_per_fov = 20;
//const float max_fov_angle = 20.0; 
//const float max_half_fov_dist = std::cos(max_fov_angle * deg2rad / 2.0);

// Database thresholding
const double default_b_thresh = 11.4; // Minimum brightness of db
const double min_separation_angle = 0.3; // Minimum angle between 2 stars (ifov degrees or equivilent for dealing with double / close stars)
const double min_separation = std::cos(min_separation_angle * deg2rad); // Minimum norm distance between 2 stars
const unsigned int pattern_stars_per_fov = 10;
const unsigned int catalog_stars_per_fov = 20;
const double max_fov_angle = 42;
const double max_fov_dist = std::cos(max_fov_angle * deg2rad);
const double max_half_fov_dist = std::cos(max_fov_angle * deg2rad / 2.0);
const unsigned int temp_star_bins = 4;
const unsigned int pattern_size = 4;
const unsigned int num_pattern_angles = (pattern_size * (pattern_size - 1)) / 2;
const int pattern_bins = 25;

// Global counter for pattern_list
int pattern_list_size = 0;
int pattern_list_growth = 20000;
int index_pattern_debug_freq = 10000;


// Database user settings

// Camera settings

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


// Hash function for Eigen Matricies 
template<typename T>
struct matrix_hash : std::unary_function<T, size_t> {
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

typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;

const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

template <typename Derived>
void write_to_csv(std::string name, const Eigen::MatrixBase<Derived>& matrix)
{
    std::ofstream file(name.c_str());
    file << matrix.format(CSVFormat);
    file.close();
}

inline bool file_exists(const std::string& name) 
{
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0);
}

bool create_dir(fs::path dir_name)
{
    std::error_code err;
    if(!fs::create_directory(dir_name, err))
    {
        if(fs::exists(dir_name))
        {
            return true;
        }
        else 
        {
            std::printf("Failed to create [%s], err:%s\n", dir_name.c_str(), err.message().c_str());
            return false;
        }
    }
    else 
    {
        return true;
    }
}

int read_hipparcos(fs::path h_file, Eigen::MatrixXd &hippo_data, const double b_thresh)
{
    std::ifstream data(h_file.c_str());
    unsigned int rcnt = 0, ccnt = 0;
    std::string lstr;

    // Skip header info
   for (unsigned int ii = 0; ii < hip_header_rows; ii++)
      data.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    std::string num;

    while(std::getline(data, lstr))
    {
        std::stringstream iss(lstr);
        ccnt = 0;

        while( std::getline(iss, num, '\t') ) 
        {
#ifdef DEBUG_HIP
            std::cout << std::stod(num) << " "; 
#endif
            if (ccnt < hip_columns)
            {
                hippo_data(rcnt, ccnt) = std::stod(num);
                ccnt++;
            }
            else 
            {
                return -rcnt;
            }
        }

        // If magnitude is below threshold, zero out row and continue
        if ((ccnt > 0) && (float(hippo_data(rcnt, HPMAG)) > float(b_thresh))) 
        {
            rcnt++;
        }
        else
        {
            hippo_data.row(rcnt).setZero();
        }

#ifdef DEBUG_HIP
        std::cout << std::endl;

        if(ccnt != hip_columns)
            std::printf("Expected %x cols, got %x\n", hip_columns, ccnt);
#endif
    }

#ifdef DEBUG_HIP
    printf("Hippo contains %d bright stars out of %ld\n", rcnt, hip_rows);
    if(rcnt != hip_rows)
        std::printf("Expected %x rows, got %x\n", hip_rows, rcnt);

#endif
     return rcnt;   
}

// Filter out dim stars, change some units to radians
void convert_hipparcos(Eigen::MatrixXd &hippo_data)
{
    hippo_data.col(RA_ICRS) *= deg2rad;
    hippo_data.col(DE_ICRS) *= deg2rad;
    hippo_data.col(PMRA) *= mas2rad;
    hippo_data.col(PMDE) *= mas2rad;
    hippo_data.col(PLX) *= mas2rad;
}

void sort_star_magnitude(Eigen::MatrixXd &hippo_data)
{
    std::vector<Eigen::VectorXd> vec;
    for (int64_t i = 0; i < hippo_data.rows(); ++i)
        vec.push_back(hippo_data.row(i));

    std::sort(vec.begin(), vec.end(), [](Eigen::VectorXd const& t1, Eigen::VectorXd const& t2){ return t1(HPMAG) > t2(HPMAG); } );

    for (int64_t i = 0; i < hippo_data.rows(); ++i)
        hippo_data.row(i) = vec[i];
}

void filter_star_separation(Eigen::MatrixXd pmc, ArrayXb &ang_pattern_idx, ArrayXb &ang_verify_idx, std::vector<int>&ang_pattern_vec, std::vector<int>&ang_verify_vec)
{
    // TODO: Remove stars near one another
    Eigen::ArrayXd ang_pattern_stars = Eigen::ArrayXd(pmc.rows());
    Eigen::ArrayXd temp_ang = Eigen::ArrayXd(pmc.rows());
    ArrayXb temp_ang_pattern_idx = ArrayXb::Constant(pmc.rows(),false);
    ang_pattern_idx(0) = true;
    ang_verify_idx(0) = true;
    ang_pattern_vec.push_back(0);
    ang_verify_vec.push_back(0);
    double num_stars_in_fov = -1;
    

    for (int ii = 1; ii < pmc.rows(); ii++)
    {
        // Pattern test: Number of pattern stars (hash/ISA) per FOV
        ang_pattern_stars = pmc(ii, Eigen::all) * pmc.transpose();
        temp_ang_pattern_idx = (ang_pattern_stars < min_separation); 
        temp_ang_pattern_idx = temp_ang_pattern_idx || !ang_pattern_idx;
        if (temp_ang_pattern_idx.all())
        {
            temp_ang = ang_pattern_stars.cwiseProduct(ang_pattern_idx.cast <double> ());
            temp_ang_pattern_idx = temp_ang.array() > max_half_fov_dist;
            num_stars_in_fov = temp_ang_pattern_idx.cast <int>().sum();
            if (num_stars_in_fov < pattern_stars_per_fov)
            {
                ang_pattern_idx(ii) = true;
                ang_verify_idx(ii) = true;
                ang_pattern_vec.push_back(ii);
                ang_verify_vec.push_back(ii);
            }
        }
        
        // Verification test: Number of catalog stars (hash/ISA) per FOV
        temp_ang_pattern_idx = (ang_pattern_stars < min_separation);
        temp_ang_pattern_idx = temp_ang_pattern_idx || !ang_verify_idx;
        if (temp_ang_pattern_idx.all())
        {
            temp_ang = ang_pattern_stars.cwiseProduct(ang_verify_idx.cast <double> ());
            temp_ang_pattern_idx = temp_ang.array() > max_half_fov_dist;
            num_stars_in_fov = temp_ang_pattern_idx.cast<int>().sum();
            if(num_stars_in_fov < catalog_stars_per_fov)
            {
                ang_verify_idx(ii) = true;
                ang_verify_vec.push_back(ii);
            }
        }

    }

}


void init_hash_table(Eigen::MatrixXd star_table, std::unordered_map<Eigen::Vector3i,std::vector<int>, matrix_hash<Eigen::Vector3i>> &course_sky_map) 
{
    Eigen::MatrixXi codes = Eigen::MatrixXi(star_table.cols(), star_table.rows());
    codes = ((double)temp_star_bins * (star_table.array() + 1)).cast<int>();

// Debug IO
#ifdef DEBUG_HASH
const Eigen::IOFormat fmt(Eigen::StreamPrecision, Eigen::DontAlignCols,  ", ");
#endif

    for(int ii = 0; ii < codes.rows(); ii++)
    {
#ifdef DEBUG_HASH
        std::cout << "Star Table Row: " << star_table.row(ii).format(fmt) << std::endl; 
        std::cout << "Codes Row: " << codes.row(ii).format(fmt) << std::endl;
#endif
        course_sky_map[codes.row(ii)].push_back(ii);
    }

#ifdef DEBUG_HASH
            Eigen::Vector3i test(3,1);
            test = ((double)temp_star_bins * (star_table.row(0).array() + 1)).cast<int>();
            std::cout << "List: ";
            const std::vector<int> llist = course_sky_map[test];
            if (llist.empty())
                std::cout << "Hash map is empty for tested code/key" << std::endl;
            else
            {
                for ( const auto &item : llist ) std::cout << item << ' ';
                std::cout << '\n';
            }
#endif
}


void get_nearby_stars(std::unordered_map<Eigen::Vector3i,std::vector<int>, matrix_hash<Eigen::Vector3i>> course_sky_map, Eigen::MatrixXd verify_star_map, Eigen::Vector3d star_vector, std::vector<int> &nearby_stars) 
{
    // TODO: Figure out why map is empty ... (Also skip self-id from entry lookup)
    // Vector to fill in with hash codes for indexing
    Eigen::Vector3i low_codes, high_codes;
    Eigen::Vector3i codes; 
    Eigen::Vector3i zeros = Eigen::Vector3i::Zero();
    Eigen::Vector3i bin_limit = int(2 * temp_star_bins) * Eigen::Vector3i::Ones();
    std::vector<int> star_ids;

    low_codes = (temp_star_bins * (star_vector.array() + 1.0 - max_fov_dist)).cast <int> ();
    low_codes.array().max(zeros.array());
    high_codes = (temp_star_bins * (star_vector.array() + 1.0 + max_fov_dist)).cast <int> ();
    high_codes.array().min(bin_limit.array());


    // For all nearby star hash codes (+/- FOV) get list of nearby stars for new hash map
    // TODO: Codes should never be negative. May want to cast all associated hashes to unsigned
    for(int ii = low_codes(0); ii <= high_codes(0); ii++)
    {
        for(int jj = low_codes(1); jj <= high_codes(1); jj++)
        {
            for(int kk = low_codes(2); kk <= high_codes(2); kk++)
            {
                codes[0] = ii;
                codes[1] = jj;
                codes[2] = kk;
                star_ids = course_sky_map[codes];
                
                for (const int & star_id : star_ids)
                {
                    double dp = star_vector.dot(verify_star_map.row(star_id));
                    if((dp > max_fov_dist) && (dp < min_separation))
                    {
                        nearby_stars.push_back(star_id);
                    }
                }
            }
        }
    }


}

bool is_star_pattern_in_fov(Eigen::MatrixXi &pattern_list, std::vector<int> nearby_star_pattern, Eigen::MatrixXd star_table) 
{
    // Make sure passed in Star ID combination matches pattern_size
    assert(nearby_star_pattern.size() == pattern_size);
    
    // Checking all pair angles
    std::vector<int> star_pair, selector(pattern_size);
    std::fill(selector.begin(), selector.begin() + 2, 1);

    bool all_stars_in_fov = true;
    double dot_p;
    unsigned int cnt = 0;

    // TODO: Make Fixed Vector3d. Requires star_table to be Matrix3d 
    Eigen::VectorXd star_vector_1, star_vector_2;

    do {
        // Check if number checked exceeds number of actual permutations
        if(cnt > num_pattern_angles)
            throw std::runtime_error("Star FOV check exceeded expected permutations\n");

        for(unsigned int ii = 0; ii < pattern_size; ii++)
        {
            if(selector[ii])
            {
                star_pair.push_back(nearby_star_pattern[ii]);
            }
        }

        // Filter out stars outsid>e FOV and compute edges        
        star_vector_1 = star_table.row(star_pair[0]);
        star_vector_2 = star_table.row(star_pair[1]);
        dot_p = star_vector_1.dot(star_vector_2);

        if (dot_p < max_fov_dist)
        {
            all_stars_in_fov = false;
            star_pair.clear();
            break;
        }

        star_pair.clear();
        cnt++;
    }
    while(std::prev_permutation(selector.begin(), selector.end()));

    return all_stars_in_fov;


}

void get_star_edge_pattern(Eigen::VectorXi nearby_star_combo, Eigen::MatrixXd star_table, Eigen::Array<double, num_pattern_angles, 1> &edges)
{
    assert(nearby_star_combo.size() == pattern_size);
    std::vector<int> star_pair, selector(pattern_size);
    
    // Only checking all pair angles
    std::fill(selector.begin(), selector.begin() + 2, 1);

    double dot_p;
    unsigned int cnt = 0;

    // TODO: Make Fixed Vector3d. Requires star_table to be Matrix3d 
    Eigen::VectorXd star_vector_1, star_vector_2;

    do {
        // Check if number checked exceeds number of actual permutations
        if(cnt > num_pattern_angles)
            throw std::out_of_range("Star FOV check exceeded expected permutations\n");

        for(unsigned int ii = 0; ii < pattern_size; ii++)
        {
            if(selector[ii])
            {
                star_pair.push_back(nearby_star_combo[ii]);
            }
        }

        // Filter out stars outside FOV and compute edges        
        star_vector_1 = star_table.row(star_pair[0]);
        star_vector_2 = star_table.row(star_pair[1]);
        dot_p = star_vector_1.dot(star_vector_2);

        
        star_vector_1 -= star_vector_2;
        edges[cnt] = star_vector_1.norm();

        // If star pattern contains angles outside FOV, somthing went wrong in prior pattern_list creation
        assert(dot_p > max_fov_dist);

        star_pair.clear();
        cnt++;
    }
    while(std::prev_permutation(selector.begin(), selector.end()));

    // If edge count != num_pattern_angles, expected combination not correct
    assert(cnt == num_pattern_angles);

    std::sort(edges.begin(), edges.end());
    edges /= edges.maxCoeff();
}

int key_to_index(Eigen::VectorXi hash_code, const unsigned int pattern_bins, const unsigned int catalog_length)
{
	const unsigned int rng_size = hash_code.size();
	Eigen::VectorXi key_range = Eigen::VectorXi::LinSpaced(rng_size, 0, rng_size - 1);
    Eigen::VectorXi pat_bin_cast= pattern_bins * Eigen::VectorXi::Ones(rng_size);
	Eigen::VectorXi index = hash_code.array() * Eigen::pow(pat_bin_cast.array(), key_range.array());

    // TODO: Carefully check python types to see if this matches TETRA Logic
	return (int(index.sum() * magic_number) % catalog_length);
}

void get_nearby_star_patterns(Eigen::MatrixXi &pattern_list, std::vector<int> nearby_stars, Eigen::MatrixXd star_table, int star_id)
{
    int n = nearby_stars.size();
    std::vector<int> nearby_star_pattern;
    std::vector<int> selector(n);

    // pattern_size - 1 : Find combinations of stars with current star
    std::fill(selector.begin(), selector.begin() + pattern_size - 1, 1);


    do {
        nearby_star_pattern.push_back(star_id);
        for (int ii = 0; ii < n; ii++)
        {
            if(selector[ii])
            {
                nearby_star_pattern.push_back(nearby_stars[ii]);
            }
        }

        if(is_star_pattern_in_fov(pattern_list, nearby_star_pattern, star_table))
        {
            int* pat_ptr = &nearby_star_pattern[0];
            Eigen::Map<Eigen::VectorXi> star_pattern_vec(pat_ptr, nearby_star_pattern.size()); 

            // Add pattern to pattern_list
            pattern_list_size++;
            if(pattern_list_size > pattern_list.rows())
            {
                // pattern_list.resize(pattern_list.rows() + pattern_list_growth, Eigen::NoChange);
                pattern_list.conservativeResize(pattern_list.rows() + pattern_list_growth, Eigen::NoChange);
            }
            pattern_list.row(pattern_list_size - 1) =  star_pattern_vec;
        }

        nearby_star_pattern.clear();
    }
    while(std::prev_permutation(selector.begin(), selector.end()));
}


void generate_pattern_catalog(std::unordered_map<Eigen::Vector3i,std::vector<int>, matrix_hash<Eigen::Vector3i>> &course_sky_map, Eigen::MatrixXd star_table, std::vector<int> ang_verify_vec) 
{
    std::vector<int> nearby_stars;
    std::vector<int> nearby_star_combos;
    Eigen::Vector3d star_vector; 
    Eigen::Matrix3d star_vector_combos;
    int star_id;

    // TODO: uint16 matrix class necessary? (Could pull this up one level)
    Eigen::MatrixXi pattern_list(1, pattern_size);
    Eigen::VectorXi pattern(pattern_size);
    int quadprobe_count;

    Eigen::Array<double, num_pattern_angles, 1> edges; 

#ifdef DEBUG_CATALOG_GENERATE_NEIGHBORS
    time_t tstart, tend;
#endif

    for(long unsigned int ii = 0; ii < ang_verify_vec.size(); ii++)
    {
#ifdef DEBUG_CATALOG_GENERATE_NEIGHBORS
        tstart = time(0);
#endif
        nearby_stars.clear(); // lol, quite important.
        star_id = ang_verify_vec[ii];
        star_vector = star_table.row(ang_verify_vec[ii]);
// #ifdef DEBUG_CATALOG_GENERATE_NEIGHBORS
        std::cout << "Looking for patterns near star id " << star_id << std::endl;
// #endif

        // For each star kept for pattern matching and verificaiton, find all nearby stars in FOV
        get_nearby_stars(course_sky_map, star_table, star_vector, nearby_stars);
#ifdef DEBUG_CATALOG_GENERATE_NEIGHBORS
        std::cout << "Number of Neighbors = " << nearby_stars.size() << std::endl;
#endif

        // For all stars nearby, find each star pattern combination (pattern_size)
        // If pattern contains star angles within FOV limits, add to pattern_list 
        get_nearby_star_patterns(pattern_list, nearby_stars, star_table, star_id);

#ifdef DEBUG_CATALOG_GENERATE_NEIGHBORS
        tend = time(0);
        std::cout << "Took " << difftime(tend, tstart) << " Seconds." << std::endl;
#endif
    }

    pattern_list.conservativeResize(pattern_list_size, Eigen::NoChange);

    // TODO: Move this into higher level Class / Structure. This is our catalog
    int catalog_length = 2 * pattern_list.rows();
    // WARNING: This is not how Tetra does this. They init to zeros.. But that is (possibly) a legitimate star in the pattern
    // Starhash inits to -1 (TODO: Macro of -1 ) to avoid star ID conflicts
    Eigen::MatrixXi pattern_catalog = -1 * Eigen::MatrixXi::Ones(catalog_length, pattern_size);

    // For all patterns in pattern_list, find hash and insert into pattern_catalog
    for(long unsigned int ii = 0; ii < (unsigned int)pattern_list.rows(); ii++)
    {
        if ((ii % index_pattern_debug_freq) == 0)
            std::cout << "Indexing pattern " << ii << " of " << pattern_list.rows() << std::endl;

        quadprobe_count = 0;

        // For each pattern, get edges
        pattern = pattern_list.row(ii);
        get_star_edge_pattern(pattern, star_table, edges);
        Eigen::VectorXi hash_code = (edges(Eigen::seqN(0, edges.size() - 1)) * (double)pattern_bins).cast<int>();
        int hash_index = key_to_index(hash_code, pattern_bins, catalog_length);

        // Use quadratic probing to find an open space in the pattern catalog to insert
        // TODO: Check if quad probe bounding is required to avoid infinite loops
        while(true)
        {
            int index = (hash_index + (int)std::pow(quadprobe_count, 2)) % (int)pattern_catalog.rows();
            if (pattern_catalog(index, 0) == -1)
            {
                // This doesn't work. Need to change from vector to array and be careful.
                pattern_catalog.row(index) = pattern;
                break;
            }
            quadprobe_count++;
        }
    }

    // Save everthing off to HDF5
    std::cout << "Saving off catalog." << std::endl;
    fs::path output = fs::current_path() / default_catalog_path;
    H5::H5File hf_file(output.string(), H5F_ACC_TRUNC);
    hsize_t dim[2];
    dim[0] = pattern_catalog.rows();
    dim[1] = pattern_catalog.cols();
    int NX = (int)dim[1];
    int NY = (int)dim[0];
    int **data_arr = new int*[NX];
    for (size_t i = 0; (int)i < NX; i++) {
        data_arr[i] = new int[NY];
    }
   
    for (int jj = 0; jj < NX; jj++)
    {
        for (int ii = 0; ii < NY; ii++)
        {
            data_arr[jj][ii] = pattern_catalog(ii, jj);
        }
    }
    H5::DataSpace ds(2, dim);
    H5::DataSet dataset = hf_file.createDataSet("catalog", H5::PredType::NATIVE_INT, ds);
    dataset.write(data_arr, H5::PredType::NATIVE_INT);

    for (size_t i = NX; i > 0; ) {
        delete[] data_arr[--i];
    }
    delete[] data_arr;
    
    fs::path debug_catalog = output / "pattern_catalog.csv";
    // write_to_csv(pattern_catalog, debug_catalog);
}

const float get_besselian_year()
{
    // MAJOR TODO: Replace with chrono / ERFA equivilent byear calculation. (See astropy.time.Time())
    const float cur_byear = 2022.6583374268196;// "Current" Besellian Epoch (08/2022)
    return cur_byear;   
}

// Coorect for stars motion per year from catalog date
void proper_motion_correction(const Eigen::MatrixXd hippo_data, const Eigen::MatrixXd rBCRF, Eigen::MatrixXd &pmc)
{
    // TODO: Determine numpy vs eigen 7th decimal place differences in this math
    assert(hippo_data.rows() > hippo_data.cols());
    Eigen::MatrixXd proper_motion(hippo_data.rows(), 3);
    Eigen::MatrixXd plx(hippo_data.rows(), 3);
    Eigen::MatrixXd los(hippo_data.rows(), 3);
    Eigen::MatrixXd p_hat(hippo_data.rows(), 3);
    Eigen::MatrixXd q_hat(hippo_data.rows(), 3);
    

#ifdef DEBUG_PM
    // const float cur_byear = 2000.0012775136654;
    const float cur_byear = 2022.7028553603136; // Sept 13, 2022 21:30 CST
#else
    const float cur_byear = get_besselian_year();
#endif

    los.col(0) = hippo_data.col(RA_ICRS).array().cos() * hippo_data.col(DE_ICRS).array().cos();
    los.col(1) = hippo_data.col(RA_ICRS).array().sin() * hippo_data.col(DE_ICRS).array().cos();
    los.col(2) = hippo_data.col(DE_ICRS).array().sin();

    p_hat.col(0) = -1 * hippo_data.col(RA_ICRS).array().sin();
    p_hat.col(1) = hippo_data.col(RA_ICRS).array().cos();
    p_hat.col(2).setZero(); 

    q_hat.col(0) = -1 * hippo_data.col(DE_ICRS).array().sin() * hippo_data.col(DE_ICRS).array().cos();
    q_hat.col(1) = -1 * hippo_data.col(DE_ICRS).array().sin() * hippo_data.col(DE_ICRS).array().sin();
    q_hat.col(2) = -1 * hippo_data.col(DE_ICRS).array().cos();

    proper_motion = (cur_byear  - hip_byear) * (p_hat.array().colwise() * hippo_data.col(PMRA).array() + q_hat.array().colwise() * hippo_data.col(PMDE).array());
    plx = rBCRF.array().colwise() * hippo_data.col(PLX).array();

    pmc = los + proper_motion - plx;
    pmc.rowwise().normalize();

    #ifdef DEBUG_PM

    // Check against PM J2000 calculations provided by VizieR
    Eigen::MatrixXd hpmc(hippo_data.rows(), 3);
    hpmc.col(0) = hippo_data.col(RA_J2000).array().cos() * hippo_data.col(RA_J2000).array().cos();
    hpmc.col(1) = hippo_data.col(RA_J2000).array().sin() * hippo_data.col(DE_J2000).array().cos();
    hpmc.col(2) = hippo_data.col(DE_J2000).array().sin();
    hpmc.rowwise().normalize();

    hpmc = hpmc - pmc;
    std::cout << "Difference between VizieR proper motion is " << hpmc.sum() << std::endl;

    #endif

}

int main(int argc, char **argv)
{
    fs::path base = fs::current_path() / "..";
    fs::path output = base / "results";
    fs::path h_file = base / "data" / "hipparcos.tsv";
    create_dir(output);

    if (!fs::exists(h_file))
        std::cout << "Does not exist: " << h_file.string() << std::endl;
    else
        std:: cout << "Hippo Path: " << h_file.string() << std::endl;

    std::string db_name = "hippo";
    Eigen::MatrixXd all_data(hip_rows, hip_columns);
    Eigen::MatrixXd pmc(hip_rows, 3);
    std::unordered_map<Eigen::Vector3i, std::vector<int>, matrix_hash<Eigen::Vector3i>> course_sky_map;
    
    // Barycentric Celestial Reference System (observer position relative to sun)
    // TODO: Replace to include parallax, currently ignoring
    // Eigen::RowVectorXd rBCRF(1, 3) 
    // rBCRF << au2km, 0,0, 0.0;
    Eigen::RowVectorXd rBCRF(3);
    rBCRF << 0, 0, 0;
    Eigen::MatrixXd rBCRF_Mat(hip_rows, 3);
    rBCRF_Mat = rBCRF.replicate<hip_rows, 1>();

    if(!(file_exists(db_name)))
    {
        int num_stars = read_hipparcos(h_file, all_data, default_b_thresh); 
        if ( num_stars > 0)
        {
            Eigen::VectorXi idx(num_stars);
            idx = Eigen::VectorXi::LinSpaced(num_stars, 0, num_stars - 1);
            Eigen::MatrixXd bright_data(num_stars, hip_columns);
            std::vector<int> ang_pattern_vec;
            std::vector<int> ang_verify_vec; 
            ArrayXb ang_pattern_idx = ArrayXb::Constant(num_stars, false);
            ArrayXb ang_verify_idx = ArrayXb::Constant(num_stars, false);
            
            bright_data = all_data(idx, Eigen::all);
            convert_hipparcos(bright_data);
            sort_star_magnitude(bright_data);
            proper_motion_correction(bright_data, rBCRF_Mat(idx, Eigen::all), pmc); 
            filter_star_separation(pmc, ang_pattern_idx, ang_verify_idx, ang_pattern_vec, ang_verify_vec);
            std::printf("Retained %d out of %d stars for pattern matching\n", (int)ang_pattern_idx.cast<int>().sum(), (int)ang_pattern_idx.size());
            std::printf("Retained %d out of %d stars for pattern verification\n", (int)ang_verify_idx.cast<int>().sum(), (int)ang_verify_idx.size());
            std::cout << std::endl;
            Eigen::MatrixXd valid_pmc_table = pmc(ang_verify_vec, Eigen::all); 
            Eigen::MatrixXd pattern_pmc_table = pmc(ang_pattern_vec, Eigen::all); 
            init_hash_table(valid_pmc_table, course_sky_map); 
#ifdef DEBUG_HASH
            Eigen::Vector3i codes;
            codes << 4, 0, 5;
            std::vector<int> testing;
            testing = course_sky_map[codes];
            for (auto v : testing)
                std::cout << v << "\n";

            std::cout << std::endl;
#endif


            std::cout << "Generating all possible patterns" << std::endl;
            generate_pattern_catalog(course_sky_map, pmc, ang_pattern_vec);


#ifdef DEBUG_CSV_OUTPUTS
            fs::path bright_stars = output / "bright.csv";
            write_to_csv(bright_stars, bright_data);
#endif

        }
        else
        {
            std::cout << "Failed to read hipparcos" << std::endl;
        }
    }

#ifdef DEBUG_CSV_OUTPUTS
    fs::path pmc_path = output / "pmc.csv";
    write_to_csv(pmc_path, pmc);
#endif

    std::cout << "End Starhash" << std::endl;
    return 0;
}