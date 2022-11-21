#include "solver.hpp"

Solver::Solver()
{
    ;
}

void Solver::get_roi(cv::Mat img)
{
    // Avoiding cropping. TODO: Try pyramid techniques
    cur_img = img(cv::Range(roi_x_min,roi_y_min), cv::Range(roi_x_max,roi_y_max));
}

void Solver::check_filter_size()
{
}

void Solver::get_centroids_from_image()
{
    ;
}

void Solver::load_generated_catalog(TODO star_centroids, float fov)
{
    H5::File hf_file(catalog_file, HF5_ACC_RDONLY);
    H5::Dataset ds_cat = hf_file.openDataSet("/catalog");
    H5::DataSpace dspace = ds_cat.getSpace();
    H5T_class_t type_class = ds_cat.getTypeClass();
    hsize_t dims[2];
    hsize_t rank = dspace.getSimpleExtentDims(dims, NULL); 

    hsize_t dimsm[1];
    dimsm[0] = dims[0];
    H5::DataSpace memspace(1, dimsm);

    vector<float> data;
    data.resize(dims[0]);

}

void Solver::compute_vectors()
{
    float cx = width / 2.0f;
    float cy = height / 2.0f;

    float scale_factor = std::tan()
}

void Solver::set_frame(cv::Mat img)
{
    // TODO: Float frame
    if(img.size() != cur_img.size()))
        cur_img = cv::resize(img, cur_img, cur_img.size(1), cur_img.size(0));
    else
        cur_img = img;
}

Solver::solve_from_image()
{

    switch(b_mode)
    {
        case: LOCAL_MEDIAN
            assert(filter_size > 0);
            assert(filter_size % 2 == 1);
            cv::medianBlur(cur_img, filter_buffer, filter_size);
            break;
        case: LOCAL_MEAN
            assert(filter_size > 0);
            assert(filter_size % 2 == 1);
            cv::blur(cur_img, filter_buffer, Size(filter_size, filter_size));
            break;
        case: GLOBAL_MEDIAN
            cv::medianBlur(cur_img, filter_buffer, cur_img.size());
            break;
        case: GLOBAL_MEAN
            filter_buffer = cv::mean(cur_img) * cv::Mat::ones(cur_img.rows(), cur_img.cols(), cur_img.type());
            break;
        default:
            throw std::runtime_error("Background Mode incorrect type");
    }

    cur_img = cur_img - filter_buffer;


    if(img_threshold < 0)
    {
        assert(sigma > 0);

        switch(s_mode)
        {
            case: LOCAL_MEDIAN_ABS
                assert(filter_size > 0);
                assert(filter_size % 2 == 1);
            case: LOCAL_ROOT_SQUARE
                //
            case: GLOBAL_MEDIAN_ABS
                //
            case: GLOBAL_ROOT_SQUARE
                //
            default:
                throw std::runtime_error("Threshold Mode incorrect type");
        }
    }
}