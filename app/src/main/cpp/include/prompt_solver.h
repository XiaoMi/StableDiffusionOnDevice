#include "utils/utils.h"
#include "utils/utils_tokenizer.h"
#include "models/TextEncoderModel.h"

using namespace std;

class PromptSolver {
public:
    int load(const std::string &path);
    int get_conditioning(const string &prompt_en,const string &default_prompt_en, cv::Mat &res_cond);
    int unload();

private:
    int get_conditioning_v2_en(const string &prompt, const string &default_prompt,cv::Mat &res_cond);
    TextEncoderModel textEncoder_en;
    CLIPTokenizer tokenizer_en;
};