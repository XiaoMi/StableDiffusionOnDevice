#include "utils/utils_file.h"

bool file_exists(const std::string &file_path) {
    struct stat buffer;
    return (stat(file_path.c_str(), &buffer) == 0);
}