#include "StarCatalog.hpp"

//const int StarCatalog::pattern_size = 4;
//const int StarCatalog::num_pattern_angles = (pattern_size * (pattern_size - 1)) / 2;

StarCatalog::StarCatalog(const std::string &in_file, const std::string &out_file) {
    input_catalog_file = in_file;
    output_catalog_file = out_file;

    edges.resize(num_pattern_angles);
    edges.setZero();
}

StarCatalog::StarCatalog(): StarCatalog(default_hipparcos_path, default_catalog_path) {

}

StarCatalog::~StarCatalog() {

}

bool StarCatalog::pattern_catalog_file_exists()
{
    struct stat buffer;
    return (stat (output_catalog_file.c_str(), &buffer) == 0);
}

bool StarCatalog::load_pattern_catalog()
{
    // TODO: Load from HDF5 and put into star_table and pattern_catalog
    // TODO: Need to save off parameter? 
    // TODO: Should the catalog be loading the initial camera parameters?
        // Probably not, but maybe have separate set of catalog parameters as some kind of assert /check
        // Have some other routine that creates a new catalog at runtime if parameters don't match expectations
    std::cout << "Loading existing pattern_catalog " << output_catalog_file << std::endl;

    return true;
}

bool StarCatalog::read_hipparcos()
{
    std::ifstream data(input_catalog_file.c_str());
    unsigned int rcnt = 0, ccnt = 0;
    std::string lstr;

    // Skip header info
   for (unsigned int ii = 0; ii < hip_header_rows; ii++)
      data.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    std::string num;

    // Hipparcos input is tsv/csv with star per line. This is basically just pandas.read_csv
    // First pass brightness check before storing
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
                input_catalog_data(rcnt, ccnt) = std::stod(num);
                ccnt++;
            }
            else 
            {
                return false;
            }
        }

        // If magnitude is below threshold, zero out row and continue
        if ((ccnt > 0) && (float(input_catalog_data(rcnt, HPMAG)) > brightness_thresh)) 
        {
            rcnt++;
        }
        else
        {
            input_catalog_data.row(rcnt).setZero();
        }

#ifdef DEBUG_HIP
        std::cout << std::endl;

        if(ccnt != hip_columns)
            std::printf("Expected %x cols, got %x\n", hip_columns, ccnt);
#endif
    }

    total_catalog_stars += rcnt;

#ifdef DEBUG_HIP
    printf("Hippo contains %d bright stars out of %ld\n", rcnt, hip_rows);
    if(rcnt != hip_rows)
        std::printf("Expected %x rows, got %x\n", hip_rows, rcnt);

#endif

    return true;

}

// Filter out dim stars, change some units to radians
void StarCatalog::convert_hipparcos()
{
    input_catalog_data.col(RA_ICRS) *= deg2rad;
    input_catalog_data.col(DE_ICRS) *= deg2rad;
    input_catalog_data.col(PMRA) *= mas2rad;
    input_catalog_data.col(PMDE) *= mas2rad;
    input_catalog_data.col(PLX) *= mas2rad;
}

void StarCatalog::sort_star_magnitudes()
{
    std::vector<Eigen::VectorXd> vec;
    for (int64_t i = 0; i < input_catalog_data.rows(); ++i)
        vec.push_back(input_catalog_data.row(i));

    std::sort(vec.begin(), vec.end(), [](Eigen::VectorXd const& t1, Eigen::VectorXd const& t2){ return t1(HPMAG) > t2(HPMAG); } );

    for (int64_t i = 0; i < input_catalog_data.rows(); ++i)
        input_catalog_data.row(i) = vec[i];
}

void StarCatalog::init_besselian_year()
{
    // MAJOR TODO: Replace with chrono / ERFA equivilent byear calculation. (See astropy.time.Time())
    current_byear = 2022.6583374268196;// "Current" Besellian Epoch (08/2022)
}

void StarCatalog::init_bcrf()
{
    // Initialize BCRF (Barycentric Celestial Reference System) aka observer position relative to sun
    // Hard TODO: Replace to include parallax, currently ignoring

    // Eigen::RowVectorXd rBCRF(1, 3) 
    // rBCRF << au2km, 0,0, 0.0;

    Eigen::RowVectorXd row_bcrf_obs_pos(3);
    row_bcrf_obs_pos << 0, 0, 0;

    // Simple TODO: Don't be dumb with memory and just use matrix vector multiply
    bcrf_frame = row_bcrf_obs_pos.replicate<hip_rows, 1>();
}

void StarCatalog::correct_proper_motion()
{
    // TODO: Make sure this is appropriate for other catalogs (UCAC4, Tycho, Gaia, etc)
    // TODO: Determine numpy vs eigen 7th decimal place differences in this math
    assert(input_catalog_data.rows() > input_catalog_data.cols());
    Eigen::MatrixXd proper_motion(input_catalog_data.rows(), 3);
    Eigen::MatrixXd plx(input_catalog_data.rows(), 3);
    Eigen::MatrixXd los(input_catalog_data.rows(), 3);
    Eigen::MatrixXd p_hat(input_catalog_data.rows(), 3);
    Eigen::MatrixXd q_hat(input_catalog_data.rows(), 3);

    los.col(0) = input_catalog_data.col(RA_ICRS).array().cos() * input_catalog_data.col(DE_ICRS).array().cos();
    los.col(1) = input_catalog_data.col(RA_ICRS).array().sin() * input_catalog_data.col(DE_ICRS).array().cos();
    los.col(2) = input_catalog_data.col(DE_ICRS).array().sin();

    p_hat.col(0) = -1 * input_catalog_data.col(RA_ICRS).array().sin();
    p_hat.col(1) = input_catalog_data.col(RA_ICRS).array().cos();
    p_hat.col(2).setZero(); 

    q_hat.col(0) = -1 * input_catalog_data.col(DE_ICRS).array().sin() * input_catalog_data.col(DE_ICRS).array().cos();
    q_hat.col(1) = -1 * input_catalog_data.col(DE_ICRS).array().sin() * input_catalog_data.col(DE_ICRS).array().sin();
    q_hat.col(2) = -1 * input_catalog_data.col(DE_ICRS).array().cos();

    proper_motion = (current_byear  - hip_byear) * (p_hat.array().colwise() * input_catalog_data.col(PMRA).array() + q_hat.array().colwise() * input_catalog_data.col(PMDE).array());
    plx = bcrf_frame.array().colwise() * input_catalog_data.col(PLX).array();

    proper_motion_data = los + proper_motion - plx;
    proper_motion_data.rowwise().normalize();

    #ifdef DEBUG_PM
    // Check against PM J2000 calculations provided by VizieR
    Eigen::MatrixXd hproper_motion_data(input_catalog_data.rows(), 3);
    hproper_motion_data.col(0) = input_catalog_data.col(RA_J2000).array().cos() * input_catalog_data.col(RA_J2000).array().cos();
    hproper_motion_data.col(1) = input_catalog_data.col(RA_J2000).array().sin() * input_catalog_data.col(DE_J2000).array().cos();
    hproper_motion_data.col(2) = input_catalog_data.col(DE_J2000).array().sin();
    hproper_motion_data.rowwise().normalize();

    hproper_motion_data = hproper_motion_data - proper_motion_data;
    std::cout << "Difference between VizieR proper motion is " << hproper_motion_data.sum() << std::endl;
    #endif
}

void StarCatalog::filter_star_separation()
{
    /* Separate stars into "pattern stars" vs "verification stars" 
        "Pattern Stars": Star idicies that are minimum separated and have less than pattern_stars_per_fov stars in any single fov.
        "Verification Stars": Star idicies that are minimum separated and have less than catalog_stars_per_fov stars in any single fov. This is more stars than actually used for attitude solution. Why? ..
        Reason: This table is used to double check you're not using a potentially "bad" edge solution by confusing stars with each other. Why?
            You want to limit the number of patterns to search through, so we make sure to reduce stars to min separation.
            The min sep angle is large enough to reduce most effects of double stars (except for unresolvable ~ ifov/2 ones. TODO?).
            But, when we go to process an image, we may have nearby stars that "distort" the solved solution. Why?
            By using a for loop that iterates through brightest stars only one time, we have accepted stars near other ones that have not been tested yet, but have cut out the next set. 
            A solution to this could be to keep iterating on this set until we remove all "near" stars with some minimum radius distance. (TODO?)
            Another solution could be to build a Vorinoi space that only use stars lower that this min sep threshold. 
            That make hipparcos very unusable. To be clear, this absolutely is a tradeoff of the algorithm.
            The original paper does not do this, but instead !!only!! accepts single star relative area solutions (no stars near patterns stars. default thresh of 0.005)
                0.005 is a normalized value take from the edge computed (normed to max edge as usual).
            This could be bad, in that stars that are near other stars will never contribute. 
            This needs analysis of camera geometry and if solution is bad enough to merit more thought.
            Steve G Comment: This seems like an okay trade off if multiple patterns can be used for the final solution. Hard kill if otherwise though.
                All other kinds of solutions (Triangle Pyramid, ISA values) have this problem. I think the right mitigation is to use several clusters, but that might need compute limits.
                Something like random forest or XGB might be good for finding an optimal solution space here.
    */
    int num_bright_stars = proper_motion_data.rows();
    // Angles (ICRS) between current star and all others
    Eigen::ArrayXd current_star_angles = Eigen::ArrayXd(num_bright_stars);
    // Angles (ICRS) between current star and all pattern stars
    Eigen::ArrayXd current_pattern_star_angles = Eigen::ArrayXd(num_bright_stars);
    // Angles (ICRS) between current star and all verification stars
    Eigen::ArrayXd current_verify_star_angles = Eigen::ArrayXd(num_bright_stars);
    // Minimum separation check for all stars
    ArrayXb separated_star_indicies = ArrayXb::Constant(num_bright_stars,false);
    // Minimum separation check for all pattern stars
    ArrayXb separated_pattern_indicies = ArrayXb::Constant(num_bright_stars,false);
    // Minimum separation check for all verification stars
    ArrayXb separated_verify_indicies = ArrayXb::Constant(num_bright_stars,false);

    // Is star "pattern star" (updated in loop)
    ArrayXb ang_pattern_idx = ArrayXb::Constant(proper_motion_data.rows(), false);
    // Is star "verification star" (updated in loop)
    ArrayXb ang_verify_idx = ArrayXb::Constant(proper_motion_data.rows(), false);
    // Verification vector used to index final star table before creation of catalog
    std::vector<int> verification_stars;
    
    // Brightest star (after sort) is always in pattern / verification
    ang_verify_idx(0) = true;
    ang_pattern_idx(0) = true;
    pattern_stars.push_back(0);

    double num_stars_in_fov = -1;

    for (int ii = 1; ii < proper_motion_data.rows(); ii++)
    {
        // Determine angle between current star and all other stars
        current_star_angles = proper_motion_data(ii, Eigen::all) * proper_motion_data.transpose();

        // Find idicies that pass angle test
        separated_star_indicies = (current_star_angles < min_separation); 

        // Explanation: Slight difference from tetra. In Tetra they look at pattern stars only. 
        // In starhash we look at all stars every loop, so we need to flag all "non-pattern stars" as true so that
        // the .all() check is really only checking the ang_pattern_idx indicies. Super confusing.
        // Same logic applies for verification.
        separated_pattern_indicies = separated_star_indicies || !ang_pattern_idx;

        // Pattern test: Limit "close stars" by number of stars used for pattern matching per FOV

        // Check that all pattern stars are close
        if (separated_pattern_indicies.all())
        {
            // separted_pattern_indicies is reused to count number of close stars
            current_pattern_star_angles = current_star_angles.cwiseProduct(ang_pattern_idx.cast <double> ());
            separated_pattern_indicies = current_pattern_star_angles.array() > max_half_fov_dist;
            num_stars_in_fov = separated_pattern_indicies.cast <int>().sum();
            if (num_stars_in_fov < pattern_stars_per_fov)
            {
                ang_verify_idx(ii) = true;
                ang_pattern_idx(ii) = true;
                pattern_stars.push_back(ii);
            }
        }
        
        separated_verify_indicies = separated_star_indicies || !ang_verify_idx;

        // Verification test: Limit number of "close stars" used for further verification per FOV
        if (separated_verify_indicies.all())
        {
            current_verify_star_angles = current_star_angles.cwiseProduct(ang_verify_idx.cast <double> ());
            separated_verify_indicies = current_verify_star_angles.array() > max_half_fov_dist;
            num_stars_in_fov = separated_verify_indicies.cast<int>().sum();
            if(num_stars_in_fov < catalog_stars_per_fov)
            {
                ang_verify_idx(ii) = true;
                verification_stars.push_back(ii);
            }
        }

    }

    star_table = proper_motion_data(verification_stars, Eigen::all);
}

void StarCatalog::get_nearby_stars(Eigen::Vector3d star_vector, std::vector<int> &nearby_stars) 
{
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
                star_ids = coarse_sky_map[codes];
                
                for (const int & star_id : star_ids)
                {
                    double dp = star_vector.dot(star_table.row(star_id));
                    if((dp > max_fov_dist) && (dp < min_separation))
                    {
                        nearby_stars.push_back(star_id);
                    }
                }
            }
        }
    }
}

bool StarCatalog::is_star_pattern_in_fov(Eigen::MatrixXi &pattern_list, std::vector<int> nearby_star_pattern) 
{
    // Make sure passed in Star ID combination matches pattern_size
    assert(nearby_star_pattern.size() == pattern_size);
    
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
        star_vector_1 = proper_motion_data.row(star_pair[0]);
        star_vector_2 = proper_motion_data.row(star_pair[1]);
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

void StarCatalog::get_star_edge_pattern(Eigen::VectorXi pattern) 
{
    assert(pattern.size() == pattern_size);
    std::vector<int> star_pair, selector(pattern_size);
    
    // Only checking all pair angles
    std::fill(selector.begin(), selector.begin() + 2, 1);

    double dot_p;
    unsigned int cnt = 0;

    // TODO: Make Fixed Vector3d. Requires proper_motion_data to be Matrix3d 
    Eigen::VectorXd star_vector_1, star_vector_2;

    do {
        // Check if number checked exceeds number of actual permutations
        if(cnt > num_pattern_angles)
            throw std::out_of_range("Star FOV check exceeded expected permutations\n");

        for(unsigned int ii = 0; ii < pattern_size; ii++)
        {
            if(selector[ii])
            {
                star_pair.push_back(pattern[ii]);
            }
        }

        // Filter out stars outside FOV and compute edges        
        star_vector_1 = proper_motion_data.row(star_pair[0]);
        star_vector_2 = proper_motion_data.row(star_pair[1]);
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

int StarCatalog::key_to_index(Eigen::VectorXi hash_code, const unsigned int pattern_bins, const unsigned int catalog_length)
{
	const unsigned int rng_size = hash_code.size();
	Eigen::VectorXi key_range = Eigen::VectorXi::LinSpaced(rng_size, 0, rng_size - 1);
    Eigen::VectorXi pat_bin_cast= pattern_bins * Eigen::VectorXi::Ones(rng_size);
	Eigen::VectorXi index = hash_code.array() * Eigen::pow(pat_bin_cast.array(), key_range.array());

    // TODO: Carefully check python types to see if this matches TETRA Logic
	return (int(index.sum() * magic_number) % catalog_length);
}

void StarCatalog::get_nearby_star_patterns(Eigen::MatrixXi &pattern_list, std::vector<int> nearby_stars, int star_id)
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

        if(is_star_pattern_in_fov(pattern_list, nearby_star_pattern))
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

void StarCatalog::init_output_catalog()
{
    Eigen::MatrixXi codes = Eigen::MatrixXi(proper_motion_data.cols(), proper_motion_data.rows());
    codes = ((double)temp_star_bins * (proper_motion_data.array() + 1)).cast<int>();

// Debug IO
#ifdef DEBUG_HASH
const Eigen::IOFormat fmt(Eigen::StreamPrecision, Eigen::DontAlignCols,  ", ");
#endif

    for(int ii = 0; ii < codes.rows(); ii++)
    {
#ifdef DEBUG_HASH
        std::cout << "Star Table Row: " << proper_motion_data.row(ii).format(fmt) << std::endl; 
        std::cout << "Codes Row: " << codes.row(ii).format(fmt) << std::endl;
#endif
        coarse_sky_map[codes.row(ii)].push_back(ii);
    }

#ifdef DEBUG_HASH
            Eigen::Vector3i test(3,1);
            test = ((double)temp_star_bins * (proper_motion_data.row(0).array() + 1)).cast<int>();
            std::cout << "List: ";
            const std::vector<int> llist = coarse_sky_map[test];
            if (llist.empty())
                std::cout << "Hash map is empty for tested code/key" << std::endl;
            else
            {
                for ( const auto &item : llist ) std::cout << item << ' ';
                std::cout << '\n';
            }
#endif
}

void StarCatalog::generate_output_catalog()
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

#ifdef DEBUG_CATALOG_GENERATE_NEIGHBORS
    time_t tstart, tend;
#endif

    for(long unsigned int ii = 0; ii < pattern_stars.size(); ii++)
    {
#ifdef DEBUG_CATALOG_GENERATE_NEIGHBORS
        tstart = time(0);
#endif
        nearby_stars.clear(); // lol, quite important.
        star_id = pattern_stars[ii];
        star_vector = star_table.row(star_id);
// #ifdef DEBUG_CATALOG_GENERATE_NEIGHBORS
        std::cout << "Looking for patterns near star id " << star_id << std::endl;
// #endif

        // For each star kept for pattern matching and verificaiton, find all nearby stars in FOV
        get_nearby_stars(star_vector, nearby_stars);
#ifdef DEBUG_CATALOG_GENERATE_NEIGHBORS
        std::cout << "Number of Neighbors = " << nearby_stars.size() << std::endl;
#endif

        // For all stars nearby, find each star pattern combination (pattern_size)
        // If pattern contains star angles within FOV limits, add to pattern_list 
        get_nearby_star_patterns(pattern_list, nearby_stars, star_id);

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
    // Other TODO: This usees a base matrix of double which I think uses more memory than needed (but is nice). 
    // Other TODO: May want to copy unordered map logic. Could reduce lookup timing.
    Eigen::MatrixXi pattern_catalog = -1 * Eigen::MatrixXi::Ones(catalog_length, pattern_size);

    // For all patterns in pattern_list, find hash and insert into pattern_catalog
    for(long unsigned int ii = 0; ii < (unsigned int)pattern_list.rows(); ii++)
    {
        if ((ii % index_pattern_debug_freq) == 0)
            std::cout << "Indexing pattern " << ii << " of " << pattern_list.rows() << std::endl;

        quadprobe_count = 0;

        // For each pattern, get edges
        pattern = pattern_list.row(ii);
        get_star_edge_pattern(pattern); 
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
    // Need both pattern catalog and star table for real time operation

    fs::path output = fs::current_path() / default_catalog_path;
    H5::H5File hf_file(output.string(), H5F_ACC_TRUNC);

    std::cout << "Saving off catalog." << std::endl;
    hsize_t dim[2];
    dim[0] = pattern_catalog.rows();
    dim[1] = pattern_catalog.cols();
    int pc_cols = (int)dim[1];
    int pc_rows = (int)dim[0];
    int **data_arr = new int*[pc_cols];
    for (size_t i = 0; (int)i < pc_cols; i++) {
        data_arr[i] = new int[pc_rows];
    }
   
    for (int jj = 0; jj < pc_cols; jj++)
    {
        for (int ii = 0; ii < pc_rows; ii++)
        {
            data_arr[jj][ii] = pattern_catalog(ii, jj);
        }
    }
    H5::DataSpace pc_ds(2, dim);

    H5::DataSet pc_dataset = hf_file.createDataSet("pattern_catalog", H5::PredType::NATIVE_INT, pc_ds);
    pc_dataset.write(data_arr, H5::PredType::NATIVE_INT);

    std::cout << "Saving off star table." << std::endl;
    hsize_t st_dim[2];
    st_dim[0] = star_table.rows();
    st_dim[1] = star_table.cols();
    int st_cols = (int)st_dim[1];
    int st_rows = (int)st_dim[0];
    double **st_data_arr = new double*[st_cols];
    for (size_t i = 0; (int)i < st_cols; i++) {
        st_data_arr[i] = new double[st_rows];
    }
   
    for (int jj = 0; jj < st_cols; jj++)
    {
        for (int ii = 0; ii < st_rows; ii++)
        {
            st_data_arr[jj][ii] = star_table(ii, jj);
        }
    }

    H5::DataSpace st_ds(2, dim);

    H5::DataSet st_dataset = hf_file.createDataSet("star_table", H5::PredType::NATIVE_DOUBLE, st_ds);
    st_dataset.write(st_data_arr, H5::PredType::NATIVE_DOUBLE);

    for (size_t i = pc_cols; i > 0; ) {
        delete[] data_arr[--i];
    }
    delete[] data_arr;

    for (size_t i = st_cols; i > 0; ) {
        delete[] st_data_arr[--i];
    }
    delete[] st_data_arr;

#ifdef DEBUG_PATTERN_CATALOG 
    fs::path debug_catalog = output / "pattern_catalog.csv";
    write_to_csv(pattern_catalog, debug_catalog);
#endif

}

// Dumb pipeline to keep things organized
void StarCatalog::run_pipeline()
{
    // Load pre-existing catalog if exists, otherwise create new database (hash table)
    if (!pattern_catalog_file_exists())
    {
        std::cout << "Reading Hipparcos Catalog" << std::endl;
        if (read_hipparcos())
        {
            std::cout << "Sorting Stars" << std::endl;
            sort_star_magnitudes();
            std::cout << "Correcting Proper Motion" << std::endl;
            correct_proper_motion();
            std::cout << "Filtering Star Separation" << std::endl;
            filter_star_separation();
            std::cout << "Initializing output catalog" << std::endl;
            init_output_catalog();
            std::cout << "Generating output catalog" << std::endl;
            generate_output_catalog();
        }
    }
    else
    {
        std::cout << "Loading existing pattern catalog and star table" << std::endl;
        load_pattern_catalog();
    }

}