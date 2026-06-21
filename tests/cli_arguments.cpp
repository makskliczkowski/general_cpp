#include "UserInterface/ui.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main() {
    try {
        using genutils::cli::Arguments;
        using genutils::cli::ParseError;

        Arguments arguments({"--count=4", "--scale", "-1.25", "--verbose",
                             "--name", "first", "--name", "second", "input.dat",
                             "--", "-literal"});
        require(arguments.require<int>("count") == 4, "integral option");
        require(arguments.require<double>("--scale") == -1.25, "negative real option");
        require(arguments.flag("verbose"), "flag option");
        require(arguments.require<std::string>("name") == "second", "last repeated value");
        require(arguments.values_as<std::string>("name") ==
                    std::vector<std::string>({"first", "second"}),
                "all repeated values");
        require(arguments.positional() == std::vector<std::string>({"input.dat", "-literal"}),
                "positional and terminator parsing");
        require(arguments.get_or<int>("missing", 7) == 7, "typed fallback");

        bool rejected = false;
        try {
            static_cast<void>(Arguments({"--count", "bad"}).require<int>("count"));
        } catch (const ParseError&) {
            rejected = true;
        }
        require(rejected, "invalid typed value rejected");

        const auto path = std::filesystem::temp_directory_path() /
                          "genutils_cli_arguments_test.conf";
        {
            std::ofstream output(path);
            output << "# comment\n--title \"two words\"\n--enabled=yes\n-value -3\n";
        }
        const auto file_arguments = Arguments::from_file(path);
        std::filesystem::remove(path);
        require(file_arguments.require<std::string>("title") == "two words",
                "quoted file value");
        require(file_arguments.flag("enabled"), "explicit boolean flag");
        require(file_arguments.require<int>("value") == -3, "negative file value");

        std::cout << "GenUtils CLI tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "GenUtils CLI test failure: " << error.what() << '\n';
        return 1;
    }
}
