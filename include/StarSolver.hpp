#ifndef STAR_SOLVER_H
#define STAR_SOLVER_H

#include <math.h>
#include <opencv2/opencv.hpp>
#include "H5Cpp.h"
#include <vector>
#include <experimental/filesystem>

// Some macro defines to debug various functions before valgrid setup
#define DEBUG_BACKGROUND_SUB
#define DEBUG_SIGMA_FILTER
#define DEBUG_MORPH_OPEN

namespace fs = std::experimental::filesystem;

#define MAX_CONTOURS 100
#define MAX_POINTS_PER_CONTOUR 1000

static const fs::path default_output_path = fs::current_path() / ".." / "results";

typedef enum bkgd_sub_mode {
    LOCAL_MEDIAN = 0,
    LOCAL_MEAN,
    GLOBAL_MEDIAN,
    GLOBAL_MEAN
} bkgd_sub_mode_t;

typedef enum sigma_mode {
    LOCAL_MEDIAN_ABS = 0,
    LOCAL_ROOT_SQUARE,
    GLOBAL_MEDIAN_ABS,
    GLOBAL_ROOT_SQUARE
} sigma_mode_t;

class StarSolver {
    public:
        StarSolver(int maxContours, int maxPointsPerContour, fs::path output_path);
        ~StarSolver(){};
        void set_frame(cv::Mat img);
        void solve_from_image();
        void get_centroids();
        void load_catalog();
        void load_image(fs::path img_path);
        void compute_vectors(float fov);
        void get_roi();
        void sub_darkframe();
        double get_median(cv::Mat input, int n);
        void get_gauss_centroids();

        void findContours();
        void computeMoments();
        

        // Image parameters
        int width;
        int height;
        cv::Mat cur_img; // TODO: Convert to float / double for precision
        cv::Mat prev_img;
        cv::Mat dark_frame; // Dark frame to subtract from image
        cv::Mat thresh_img; // Binary mask containing thresholded image
        cv::Mat std_img;
        cv::Mat filter_buffer;
        cv::Mat sigma_buffer;
        cv::Mat kernel;

        // Gaussian centroiding parameters
        bool denoise = true;


        // Centroiding parameters
        int max_contours;
        int max_points_per_contour;
        int num_contours;
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Moments> moments;

        // FOV solver threshold parameters
        float fov_estimate; 
        float fov_max_error;

        // Star matching parameters
        unsigned int num_pattern_stars;
        float match_radius; 
        double match_threshold;

        // Centroiding filtering parameters 
        bkgd_sub_mode background_sub_mode = LOCAL_MEDIAN;
        sigma_mode_t sigma_sub_mode = LOCAL_MEDIAN_ABS;
        unsigned int filter_size = 31;
        float img_threshold = -1;
        float img_std;
        float sigma = 0.05;
        unsigned int centroid_window_size;
        bool binary_close = true;
        bool binary_open = !binary_close;
        float med_sigma_coef = 3;
        int morph_elem = 2;
        int morph_size = 31;
			

        // Centroiding spot parameters
        int max_spot_area;
        int min_spot_area;
        int max_spot_sum;
        int min_spot_sum;
        float max_axis_ratio;
        unsigned int max_num_spots;

        // Optional return parameters
        bool return_moments;
        bool return_images;

        // ROI Cropping Parameters
        bool apply_roi;
        unsigned int roi_x_min;
        unsigned int roi_x_max;
        unsigned int roi_y_min;
        unsigned int roi_y_max;

        //File Handling Paramters
        fs::path output_path;
        std::string config_file;
        std::string catalog_file;
        std::string debug_folder;
        int debug_level;
};

#endif // STAR_SOLVER_H