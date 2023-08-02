#include "StarSolver.hpp"
#include <algorithm>
#include <random>
#include <iterator>
#include <iostream>
#include <random>
#include <unordered_set>
#include "iterators.hpp"
#include "eigen_mods.hpp"

// #define DEBUG_FLATTEN_IMAGE


// SBG Ongoing TODOs
// 1. ROI Mode

StarSolver::StarSolver(int max_contours, int max_points_per_contour, fs::path output_path) :
        max_contours(max_contours), max_points_per_contour(max_points_per_contour), output_path(output_path) {}

float StarSolver::eigen_median(Eigen::VectorXf vec) {
    const auto median_it = vec.begin() + vec.size() / 2;
    std::nth_element(vec.begin(), median_it , vec.end());
    auto median = *median_it;
    return median;
}

float StarSolver::get_stddev(Eigen::VectorXf vec) {
    return std::sqrt((vec.array() - vec.mean()).square().sum()/(vec.size()));
}

float StarSolver::get_stddev_cutoff(Eigen::VectorXf vec, float sigma_cutoff) {
    /* Standard Deviation calculation with sigma cutoff */
    auto inliers = vec.array() <= sigma_cutoff;
    auto r = (vec.array() - vec.mean()) * inliers.cast<float>();
    auto var = r.cwiseProduct(r).sum() / inliers.cast<float>().sum();
    return std::sqrt(var);
}


void StarSolver::get_gauss_centroids()
{
    auto rng = std::mt19937{std::random_device{}()};

    if (denoise) {
        cv::GaussianBlur(cur_img, cur_img, cv::Size(3, 3), 0);
    }
    
    // Convert gray to fp32 and subtract med blur
    // TODO: I think this is still broken for the median or the image is actually flat.
    cv::Mat cur_img_fp32, mb_img_fp32;
    cur_img.convertTo(cur_img_fp32, CV_32FC1);
    cv::medianBlur(cur_img_fp32, mb_img_fp32, 5);
    flat_image = cur_img_fp32 - mb_img_fp32;

    /* randomly sample part of image, that are at least num_edge_pixels away from the edge */
    int num_pixels = (width - num_edge_pixels) * (height - num_edge_pixels);
    assert(num_pixels > num_flat_pixels);
    std::sample(boxed_iterator{num_edge_pixels}, boxed_iterator{width}, std::back_inserter(flat_col_samples), num_flat_pixels, rng);
    std::sample(boxed_iterator{num_edge_pixels}, boxed_iterator{height}, std::back_inserter(flat_row_samples), num_flat_pixels, rng);
    std::shuffle(flat_row_samples.begin(), flat_row_samples.end(), rng);
    std::shuffle(flat_col_samples.begin(), flat_col_samples.end(), rng);
    Eigen::VectorXf diffs(num_flat_pixels);

    for (int ii = 0; ii < num_flat_pixels; ii++) {
        auto row = flat_row_samples[ii];
        auto col = flat_col_samples[ii];
        float diff = flat_image.at<uchar>(row + 5, col + 5) - flat_image.at<uchar>(row, col);
        diffs[ii] = diff;

#ifdef DEBUG_FLATTEN_IMAGE
        std::cout << "Sampled col: " << row << ", row: " << col;
        std::cout << "  Pixel + 5: " << static_cast<int>(flat_image.at<uchar>(row + 5, col + 5));
        std::cout << "  Pixel: " << static_cast<int>(flat_image.at<uchar>(row, col));
        std::cout << "  Diff =  " << diffs[ii] << std::endl;
#endif

    }

    /* Identify outliers to remove for random sample by thresholding the residual median standard deviation */
    float median = eigen_median(diffs);
    Eigen::VectorXf median_diffs(num_flat_pixels);
    Eigen::VectorXf median_sigma(num_flat_pixels);

    median_diffs = diffs.array() - median;
    median_diffs = Eigen::abs(median_diffs.array());

    float median_diff = eigen_median(median_diffs);
    float mean_diff = median_diffs.mean();

    // Don't divide by zero (TODO: Check if mean is zero)
    if (std::abs(median_diff) > std::numeric_limits<float>::epsilon()) {
        median_sigma = 1.4826 * median_diffs / median_diff;
    }
    else {
        median_sigma = median_diffs / mean_diff;
    }

    // Compute std deviation excluding outliers
    flat_stddev = get_stddev_cutoff(median_sigma, sigma_cutoff);

#ifdef DEBUG_FLATTEN_IMAGE
    auto inliers = median_sigma.array() < sigma_cutoff;
    float stddev_norm = get_stddev(median_sigma);
    for (int ii = 0; ii < num_flat_pixels; ii++) {
        std::cout << "Index: " << ii << 
            "   Inlier: " << inliers[ii] << 
            "  Diff: " << diffs[ii] << 
            "  Median: " << median << 
            "  Mean: " << mean_diff << 
            "  Median Diffs: " << median_diffs[ii] << 
            "  Median Diff: " << median_diff << 
            "  Median Sigma: " << median_sigma[ii] << std::endl;
    }

    std::cout << "Number Inliers: " << inliers.cast<float>().sum() << std::endl;
    std::cout << "Flattened Image Stddev: " << flat_stddev << std::endl;
    std::cout << "Flattened Image Stddev with outliers: " << stddev_norm << std::endl;
#endif


}

void StarSolver::sub_darkframe()
{
    cur_img = cur_img - dark_frame;
}

double StarSolver::get_median(cv::Mat input, int n)
{
    // COMPUTE HISTOGRAM OF SINGLE CHANNEL MATRIX
    float range[] = { 0, (float)n};
    const float* hist_range = { range };
    bool uniform = true; 
    bool accumulate = false;
    cv::Mat hist;
    cv::calcHist(&input, 1, 0, cv::Mat(), hist, 1, &n, &hist_range, uniform, accumulate);

    // COMPUTE CUMULATIVE DISTRIBUTION FUNCTION (CDF)
    cv::Mat cdf;
    hist.copyTo(cdf);
    for (int i = 1; i <= n - 1; i++){
        cdf.at<float>(i) += cdf.at<float>(i - 1);
    }
    cdf /= input.total();

    // COMPUTE MEDIAN
    double medianVal;
    for (int i = 0; i <= n - 1; i++)
    {
        if(cdf.at<float>(i) >= 0.5) 
        {
            medianVal = i;
            break; 
        }
    }
    return medianVal / n; 
}

void StarSolver::load_catalog()
{
    H5::H5File hf_file(catalog_file, H5F_ACC_RDONLY);
    H5::DataSet ds_cat = hf_file.openDataSet("/catalog");
    H5::DataSpace dspace = ds_cat.getSpace();
    // TODO: Access catalog hdf5 data
}

void StarSolver::compute_vectors(float fov)
{
    float cx = width / 2.0f;
    float cy = height / 2.0f;

    float scale_factor = std::tan(fov / 2 ) / cx;
    // TODO: Implement
    std::cout << cx << cy << scale_factor << std::endl;
    // Loop over centroids and compute norm vectors
}

void StarSolver::load_image(fs::path img_path)
{
    cur_img = cv::imread(img_path.string(), cv::IMREAD_GRAYSCALE);
    if(cur_img.empty())
    {
        std::cout << "Error loading image" << std::endl;
        exit(1);
    }
    width = cur_img.cols;
    height = cur_img.rows;
    std::cout << "Loaded image: " << img_path << std::endl;
}

void StarSolver::set_frame(cv::Mat img)
{
    // TODO: Float frame
    if(img.size() != cur_img.size())
        cv::resize(img, cur_img, cur_img.size(), cv::INTER_LINEAR);
    else
        cur_img = img;
}

void StarSolver::get_centroids()
{
    switch(background_sub_mode)
    {
        case LOCAL_MEDIAN:
        {
            assert(filter_size > 0);
            assert(filter_size % 2 == 1);
            cv::medianBlur(cur_img, filter_buffer, filter_size);
            break;
        }
        case LOCAL_MEAN:
        {
            assert(filter_size > 0);
            assert(filter_size % 2 == 1);
            cv::Mat kernel = cv::Mat::ones(filter_size, filter_size, CV_32F) / (filter_size * filter_size);
            cv::filter2D(cur_img, filter_buffer, -1, kernel);
            break;
        }
        case GLOBAL_MEDIAN:
        {
            double img_median = get_median(cur_img, cur_img.cols * cur_img.rows);
            filter_buffer = img_median * cv::Mat::ones(cur_img.rows, cur_img.cols, cur_img.type());
            break;
        }
        case GLOBAL_MEAN:
        {
            filter_buffer = cv::mean(cur_img) * cv::Mat::ones(cur_img.rows, cur_img.cols, cur_img.type());
            break;
        }
        default:
        {
            throw std::runtime_error("Background Mode incorrect type");
        }
    }

#ifdef DEBUG_BACKGROUND_SUB
    fs::path bkgrd_debug_path = output_path / "background_sub.png";
    cv::imwrite(bkgrd_debug_path.string(), filter_buffer);
#endif

    cur_img = cur_img - filter_buffer;

    if(img_threshold < 0)
    {
        assert(sigma > 0);

        switch(sigma_sub_mode)
        {
            default:
            {
                assert(filter_size > 0);
                assert(filter_size % 2 == 1);
            }
            case LOCAL_MEDIAN_ABS:
            {
                cv::medianBlur(cur_img, filter_buffer, filter_size);
                filter_buffer *= med_sigma_coef;
                break;
            }
            case LOCAL_ROOT_SQUARE:
            {
                cv::Mat kernel = cv::Mat::ones(filter_size, filter_size, CV_32F) / (filter_size * filter_size);

                cur_img.convertTo(filter_buffer, CV_32F);
                filter_buffer = cv::abs(filter_buffer);
                cv::pow(filter_buffer, 2, filter_buffer);
                cv::filter2D(filter_buffer, filter_buffer, -1, kernel);
                cv::pow(filter_buffer, 2, filter_buffer);
                break;
            }
            case GLOBAL_MEDIAN_ABS:
            {
                cur_img.convertTo(filter_buffer, CV_32F);
                filter_buffer = cv::abs(filter_buffer);
                double img_median = get_median(filter_buffer, (int)(filter_buffer.cols * filter_buffer.rows));
                filter_buffer = med_sigma_coef * img_median * cv::Mat::ones(filter_buffer.rows, filter_buffer.cols, cur_img.type());
                break;
            }
            case GLOBAL_ROOT_SQUARE:
            {
                cur_img.convertTo(filter_buffer, CV_32F);
                cv::pow(filter_buffer, 2, filter_buffer);
                cv::Scalar img_sq_mean = cv::mean(filter_buffer);
                cv::pow(img_sq_mean, 0.5, img_sq_mean);
                filter_buffer = img_sq_mean * cv::Mat::ones(filter_buffer.rows, filter_buffer.cols, cur_img.type());
                break;
            }
        }
        sigma_buffer = filter_buffer * sigma;
    }

    thresh_img = cur_img < sigma_buffer;

#ifdef DEBUG_SIGMA_FILTER
    double minVal; 
    double maxVal; 
    cv::Point minLoc; 
    cv::Point maxLoc;
    
    cv::minMaxLoc( sigma_buffer, &minVal, &maxVal, &minLoc, &maxLoc );
    std::cout << "Sigma Filter Max: " << maxVal << " at " << maxLoc << std::endl;
    std::cout << "Sigma Filter Min: " << minVal << " at " << minLoc << std::endl;

    fs::path sigma_debug_path = output_path / "sigma_filter.png";
    fs::path thresh_debug_path = output_path / "threshold_image.png";
    cv::imwrite(sigma_debug_path.string(), sigma_buffer);
    cv::imwrite(thresh_debug_path.string(), thresh_img);
#endif

    cv::Mat element;
    if (binary_close)
    {
        //element = cv::getStructuringElement(morph_elem, cv::Size(2 * morph_size + 1, 2 * morph_size + 1), cv::Point(morph_size, morph_size));
        element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(morph_size, morph_size));
        cv::morphologyEx(thresh_img, thresh_img, cv::MORPH_CLOSE, element);
        //element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * morph_size + 1, 2 * morph_size + 1), cv::Point(morph_size, morph_size));
        //cv::morphologyEx(thresh_img, thresh_img, cv::MORPH_OPEN, element);
    }

    if (binary_open)
    {
        element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(morph_size, morph_size));
        cv::morphologyEx(thresh_img, thresh_img, cv::MORPH_OPEN, element);
        cv::morphologyEx(thresh_img, thresh_img, cv::MORPH_CLOSE, element);
    }

#ifdef DEBUG_MORPH_OPEN
    fs::path open_debug_path = output_path / "morph_open_filter.png";
    cv::imwrite(open_debug_path.string(), thresh_img);
#endif

    filter_buffer.copyTo(std_img);

    // find moments of the image
    findContours();
    computeMoments();
}

void StarSolver::findContours() {
    cv::findContours(cur_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
}

void StarSolver::computeMoments() {
    float area;
    //float m20, m2_xx, m2_yy, major, minor;

    cv::Moments moment;
    // compute moments of the contours and save off ones that meet criteria
    for (long unsigned int ii = 0; ii < contours.size(); ii++) {
        area = cv::contourArea(contours[ii]);

        if (area < min_spot_area || area > max_spot_area)
            continue;

        cv::Moments moment = cv::moments(contours[ii], false);

        if (moment.m00 < min_spot_sum || moment.m00 > max_spot_sum)
            continue;

        //m2_xx = max(0, np.sum((x - m1_x)**2 * a) / m0)
        //m2_yy = max(0, np.sum((y - m1_y)**2 * a) / m0)
        //m2_xy = np.sum((x - m1_x) * (y - m1_y) * a) / m0
        //major = np.sqrt(2 * (m2_xx + m2_yy + np.sqrt((m2_xx - m2_yy)**2 + 4 * m2_xy**2)))
        //minor = np.sqrt(2 * max(0, m2_xx + m2_yy - np.sqrt((m2_xx - m2_yy)**2 + 4 * m2_xy**2)))        

        moments.push_back(moment);
    }
}
