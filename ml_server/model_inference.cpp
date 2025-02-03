#include <iostream>
#include <optional>
#include <string>

std::optional<std::string> infer(char* buffer) {
    std::string success_message = "Success";

    std::cout << "Parsing \"" << buffer << "\"" << std::endl;

    return std::make_optional(success_message);
}