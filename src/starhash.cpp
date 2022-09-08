#include "opencv2/opencv.hpp"
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

//#define DEBUG_HIP 1
#define DEBUG_PM
const double deg2rad = M_PI / 180.0;
const double arcsec2deg =  (1.0 / 3600.0);
const double mas2arcsec = (1.0 / 1000.0);
const double mas2rad = mas2arcsec * arcsec2deg * deg2rad;
const double au2km = 149597870.691;

// Hipparcos Info
const float hip_byear = 1991.25; // Hipparcos Besellian Epoch
const unsigned int hip_columns = 10;
const unsigned int hip_rows = 117955;
const unsigned int hip_header_rows = 55;
const double default_b_thresh = 6.0; // Minimum brightness of db
const double default_a_thresh = 0.3; // Minimum angle between 2 stars

// Database user settings


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
    COLOUR
};

inline bool file_exists(const std::string& name) {
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0);
}

inline long factorial(const int n)
{
    long f = 1;
    for (int i=1; i<=n; ++i)
        f *= i;
    return f;
}

int read_hipparcos(std::string h_file, Eigen::MatrixXd &hippo_data, const double b_thresh)
{
    std::ifstream data(h_file.c_str());
    unsigned int rcnt = 0, ccnt = 0;
    std::string::size_type sz;

    if(!data.is_open())
        return -1;

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
            hippo_data(rcnt, ccnt) = std::stod(num);
            ccnt++;
        }

        // If magnitude is below threshold, zero out row and continue
        if (float(hippo_data(rcnt, HPMAG)) > float(b_thresh))
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

void filter_star_separation(Eigen::MatrixXd &hippo_data, double min_separation)
{
    // TODO: Remove stars near one another
    // TODO: Remove 
    
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
    assert(hippo_data.rows() > hippo_data.cols());
    Eigen::MatrixXd proper_motion(hippo_data.rows(), 3);
    Eigen::MatrixXd plx(hippo_data.rows(), 3);
    Eigen::MatrixXd los(hippo_data.rows(), 3);
    Eigen::MatrixXd p_hat(hippo_data.rows(), 3);
    Eigen::MatrixXd q_hat(hippo_data.rows(), 3);
    

#ifdef DEBUG_PM
    const float cur_byear = 2000.0012775136654;
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

int main()
{
    std::string h_file = "Hipparcos.tsv";
    std::string db_name = "hippo";
    Eigen::MatrixXd all_data(hip_rows, hip_columns);
    Eigen::MatrixXd pmc(hip_rows, 3);
    Eigen::MatrixXd star_pairs(factorial(hip_rows), 2);
    long test = factorial(hip_rows);
    
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
            bright_data = all_data(idx, Eigen::placeholders::all);
            convert_hipparcos(bright_data);
            sort_star_magnitude(bright_data);
            proper_motion_correction(bright_data, rBCRF_Mat(idx, Eigen::placeholders::all), pmc); 
        }
        else
        {
            std::cout << "Failed to read hipparcos" << std::endl;
        }
    }

    std::cout << "End Starhash" << std::endl;
    return 0;
}