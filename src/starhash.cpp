#include "starhash.hpp"

namespace fs = std::experimental::filesystem;

/* Helpful debug flags */
//#define DEBUG_HIP 1
#define DEBUG_PM
#define DEBUG_CSV_OUTPUTS
#define DEBUG_HASH

const double deg2rad = M_PI / 180.0;
const double arcsec2deg =  (1.0 / 3600.0);
const double mas2arcsec = (1.0 / 1000.0);
const double mas2rad = mas2arcsec * arcsec2deg * deg2rad;
const double au2km = 149597870.691;

// Hipparcos Info
const fs::path default_hip_path("/../data/hipparcos.tsv"); // Default relative path
const float hip_byear = 1991.25; // Hipparcos Besellian Epoch
const unsigned int hip_columns = 10;
const unsigned int hip_rows = 117955;
const unsigned int hip_header_rows = 55;

// Tetra thresholding
//const double default_b_thresh = 6.5; // Minimum brightness of db
//const double min_separation_angle = 0.5; // Minimum angle between 2 stars (ifov degrees)
//const double min_separation = std::cos(min_separation_angle * deg2rad); // Minimum norm distance between 2 stars
//const unsigned int pattern_stars_per_fov = 10;
//const unsigned int catalog_stars_per_fov = 20;
//const float max_fov_angle = 20.0; 
//const float max_half_fov_dist = std::cos(max_fov_angle * deg2rad / 2.0);

// Database thresholding
const double default_b_thresh = 6.0; // Minimum brightness of db
const double min_separation_angle = 0.3; // Minimum angle between 2 stars (ifov degrees)
const double min_separation = std::cos(min_separation_angle * deg2rad); // Minimum norm distance between 2 stars
const unsigned int pattern_stars_per_fov = 30;
const unsigned int catalog_stars_per_fov = 60;
const double max_fov_angle = 65.8;
const double max_fov_dist = std::cos(max_fov_angle * deg2rad);
const double max_half_fov_dist = std::cos(max_fov_angle * deg2rad / 2.0);
const unsigned int temp_star_bins = 4;
const unsigned int pattern_size = 4;


// Database user settings

// Camera settings

enum {
    RA_J2000,
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
        if ((ccnt > 0) && (float(hippo_data(rcnt, HPMAG)) < float(b_thresh))) 
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


void init_hash_table(Eigen::MatrixXd star_table, std::unordered_map<Eigen::MatrixXi,std::list<int>, matrix_hash<Eigen::MatrixXi>> &course_sky_map) 
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
            Eigen::MatrixXi test(3,1);
            test = ((double)temp_star_bins * (star_table.row(0).array() + 1)).cast<int>();
            std::cout << "List: ";
            const std::list<int> llist = course_sky_map[test];
            if (llist.empty())
                std::cout << "Hash map is empty for tested code/key" << std::endl;
            else
            {
                for ( const auto &item : llist ) std::cout << item << ' ';
                std::cout << '\n';
            }
#endif


}

void get_nearby_stars(std::unordered_map<Eigen::MatrixXi,std::list<int>, matrix_hash<Eigen::MatrixXi>> &course_sky_map, Eigen::MatrixXd verify_star_map, Eigen::Vector3d star_vector, std::list<int> &nearby_stars) 
{
    // Vector to fill in with hash codes for indexing
    Eigen::Vector3i codes, low_codes, high_codes;
    Eigen::Vector3i zeros = Eigen::Vector3i::Zero();
    Eigen::Vector3i bin_limit = int(2 * temp_star_bins) * Eigen::Vector3i::Ones();
    std::list<int> star_ids;

    low_codes = (temp_star_bins * (star_vector.array() + 1.0 - max_fov_dist)).cast <int> ();
    low_codes.array().max(zeros.array());
    high_codes = (temp_star_bins * (star_vector.array() + 1.0 + max_fov_dist)).cast <int> ();
    high_codes.array().min(bin_limit.array());


    // For all nearby star hash codes (+/- FOV) get list of nearby stars for new hash map
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
                    if(dp > max_fov_dist)
                    {
                        nearby_stars.push_back(star_id);
                    }
                }
            }
        }
    }


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
    std::unordered_map<Eigen::MatrixXi, std::list<int>, matrix_hash<Eigen::MatrixXi>> course_sky_map;
    
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
            std::cout << "Generating all possible patterns" << std::endl;



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