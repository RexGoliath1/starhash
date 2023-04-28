#include "StarSolver.hpp"

// SBG Ongoing TODOs
// 1. ROI Mode

StarSolver::StarSolver(int maxContours, int maxPointsPerContour) :
        maxContours(maxContours), maxPointsPerContour(maxPointsPerContour) {}


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
    // H5T_class_t type_class = ds_cat.getTypeClass();
    hsize_t dims[2];
    // hsize_t rank = dspace.getSimpleExtentDims(dims, NULL); 

    hsize_t dimsm[1];
    dimsm[0] = dims[0];
    H5::DataSpace memspace(1, dimsm);

    std::vector<float> data;
    data.resize(dims[0]);
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
    switch(b_mode)
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

    cur_img = cur_img - filter_buffer;

    if(img_threshold < 0)
    {
        assert(sigma > 0);

        switch(s_mode)
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

    thresh_img = cur_img > sigma_buffer;

    if (binary_open)
    {
        cv::Mat element = cv::getStructuringElement(morph_elem, cv::Size(2 * morph_size + 1, 2 * morph_size + 1), cv::Point(morph_size, morph_size));
        cv::morphologyEx(thresh_img, thresh_img, cv::MORPH_OPEN, element);
    }

    filter_buffer.copyTo(std_img);


    // find moments of the image
    findContours();
    computeMoments();
}

void StarSolver::findContours() {
    cv::findContours(cur_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
}

void StarSolver::computeMoments() {
    for (int ii = 0; ii < contours.size(); ii++) {
        moments.push_back(cv::moments(contours[ii], false));
    }
}