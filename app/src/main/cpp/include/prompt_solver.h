#include "utils/utils.h"
#include "utils/utils_tokenizer.h"
#include "models/TextEncoderModel.h"

using namespace std;

class PromptSolver {
public:
    int load(const std::string &path);

    cv::Mat get_conditioning(const string &prompt_ch, const string &prompt_en,
                             const string &default_prompt_ch,
                             const string &default_prompt_en, int language_mode);

private:

    cv::Mat get_conditioning_v2_en(const string &prompt, const string &default_prompt);

    TextEncoderModel textEncoder_en;
    CLIPTokenizer tokenizer_en;
};