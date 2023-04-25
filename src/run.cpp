#include "StarCatalog.hpp"

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
    fs::path output;
    fs::path h_file;
    fs::path out_file;

    if (argc == 1) {
        std::cout << "Creating new catalog with defaults" << std::endl; 
        fs::path base = fs::current_path() / "..";
        output = base / "results";
        h_file = base / "data" / "hipparcos.tsv";
        out_file = base / "results" / "output.h5";
    }
    create_out_directory(output);

    if (fs::exists(h_file))
    {
        std:: cout << "Hipparcos Catalog Path: " << h_file.string() << std::endl;
    }
    else
    {
        std::cout << "Hipparcos Catalog does not exist: " << h_file.string() << std::endl;
        return 1;
    }

    StarCatalog catalog(h_file, out_file);
    catalog.run_pipeline();

    return 0;
}