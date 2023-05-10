#include "StarCatalogGenerator.hpp"
#include "StarSolver.hpp"

#include <iostream>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

bool create_out_directory(fs::path dir_name)
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

int main(int argc, char **argv)
{
    fs::path base_path, data_path, hipparcos_file, output_path, p_cat_out_file;
    //fs::path test_image = fs::current_path() / ".." / "data" / "star_tracker_image.jpeg";
    fs::path test_image = fs::current_path() / ".." / "data" / "large_star_image.JPG";

    int max_contours = 100;
    int max_points_per_contour = 1000;

    // Until we have full config controls, use defaults
    if (argc == 1) {
        std::cout << "Creating new catalog with defaults" << std::endl; 
        base_path = fs::current_path() / "..";

        data_path = base_path / "data";
        hipparcos_file = data_path / "hipparcos.tsv";

        output_path = base_path / "results";
        p_cat_out_file = output_path / "output.h5";
    }

    create_out_directory(output_path);

    if (fs::exists(hipparcos_file))
    {
        std:: cout << "Hipparcos Catalog Path: " << hipparcos_file.string() << std::endl;
    }
    else
    {
        std::cout << "Hipparcos Catalog does not exist: " << hipparcos_file.string() << std::endl;
        return 1;
    }

    StarCatalogGenerator catalog(hipparcos_file, p_cat_out_file);
    catalog.run_pipeline();

    StarSolver solver(max_contours, max_points_per_contour, output_path);
    solver.load_image(test_image);
    solver.get_centroids();

    return 0;
}
