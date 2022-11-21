#include <math.h>
#include <opencv2/opencv.hpp>
#include "H5Cpp.h"

typedef enum bkgd_sub_mode {
    LOCAL_MEDIAN = 0,
    LOCAL_MEAN,
    GLOBAL_MEDIAN,
    GLOBAL_MEAN
} bkrd_sub_mode_t;

enum sigma_mode {
    LOCAL_MEDIAN_ABS = 0,
    LOCAL_ROOT_SQUARE,
    GLOBAL_MEDIAN_ABS,
    GLOBAL_ROOT_SQUARE
} sigma_mode_t;

class Solver {
    public:
        Solver();
        ~Solver();
        void set_frame(cv::Mat img);
        void solve_from_img();

        // Image parameters
        unsigned int width;
        unsigned int height;
        cv::Mat cur_img; // TODO: Convert to float / double for precision
        cv::Mat prev_img;
        cv::Mat filter_buffer;
        cv::Mat sigma_buffer;

        // FOV solver threshold parameters
        float fov_estimate; 
        float fov_max_error;

        // Star matching parameters
        unsigned int num_pattern_stars;
        float match_radius; 
        double match_threshold;

        // Centroiding filtering parameters 
        bkrd_sub_mode_t b_mode;
        sigma_mode_t s_mode;
        unsigned int filter_size;
        float img_threshold;
        float sigma;
        unsigned int centroid_window_size;
        bool binary_open;

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
        std::string config_file;
        std::string catalog_file;
        std::string debug_folder;
        int debug_level;

}