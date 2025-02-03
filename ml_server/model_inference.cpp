#include <iostream>
#include <optional>
#include <string>
#include <vector>

enum MLMode {
    NO_MODEL = 0,
    SUCKER = 1,
    LIMB = 2,
    FULL = 3
};

class InferenceParams {
public:
    MLMode ml_mode;
    int message_index;
    float input_value;
    InferenceParams(MLMode ml_mode, int message_index, float input_value) {
        this->ml_mode = ml_mode;
        this->message_index = message_index;
        this->input_value = input_value;
    }
};

std::vector<std::string> split(const std::string& s, const std::string& delimiter) {
    std::vector<std::string> tokens;
    size_t pos = 0;
    std::string token;
    std::string s_copy = s;
    while ((pos = s_copy.find(delimiter)) != std::string::npos) {
        token = s_copy.substr(0, pos);
        tokens.push_back(token);
        s_copy.erase(0, pos + delimiter.length());
    }
    tokens.push_back(s_copy);
    return tokens;
}

InferenceParams parse_input(std::string input) {
    // Parse the input string into 
    std::string delimiter = " ";
    std::vector<std::string> params = split(input, delimiter);
    MLMode ml_mode = MLMode::NO_MODEL;
    InferenceParams res(ml_mode, -1, -1.0);

    if (params[0] == "Sucker") {
        res.ml_mode = MLMode::SUCKER;
    } else if (params[0] == "Limb") {
        res.ml_mode = MLMode::LIMB;
    } else if (params[0] == "Full") {
        res.ml_mode = MLMode::FULL;
    } else {
        return res;
    }
    res.message_index = std::stoi(params[1]);
    res.input_value = std::stof(params[2]);
    return res;
}

std::optional<std::string> infer(int msg_ix, char* buffer) {
    // Run inference
    std::string success_message = "Success";

    std::cout << "Parsing \"" << buffer << "\"" << std::endl;
    InferenceParams params = parse_input(buffer);

    std::string result = success_message + " " + std::to_string(msg_ix) + " " + std::to_string(params.ml_mode) + " " + std::to_string(params.message_index) + " " + std::to_string(params.input_value);

    return std::make_optional(result);
}