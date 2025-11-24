#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>

int main() {
    std::ifstream file("main.txt");
    if (!file.is_open()) {
        std::cerr << "Failed to open main.txt" << std::endl;
        return 1;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string text = buffer.str();

    // Escape double quotes in the text to safely pass in the command line
    std::string escaped_text;
    for (char c : text) {
        if (c == '"') {
            escaped_text += "\\\"";
        } else {
            escaped_text += c;
        }
    }

    // Call eSpeak with the text
    std::string command = "espeak \"" + escaped_text + "\"";

    int result = std::system(command.c_str());
    if (result != 0) {
        std::cerr << "Failed to execute espeak command" << std::endl;
        return 1;
    }

    return 0;
}

