#include "utils/utils_vec.h"

template<typename T>
void vec_print(const std::string &name, const std::vector<T> &vec) {
    LOGI("==============  %s", name.c_str());
    std::ostringstream oss;
    for (const T &i: vec) {
        oss << i << "  ";
    }
    LOGI("%s", oss.str().c_str());
}

template void vec_print<int>(const std::string &name, const std::vector<int> &vec);

template void vec_print<float>(const std::string &name, const std::vector<float> &vec);

template void vec_print<std::string>(const std::string &name, const std::vector<std::string> &vec);

template void
vec_print<unsigned long>(const std::string &name, const std::vector<unsigned long> &vec);

template void vec_print<long>(const std::string &name, const std::vector<long> &vec);

template void vec_print<uchar>(const std::string &name, const std::vector<uchar> &vec);