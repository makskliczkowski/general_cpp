#include "../src/UserInterface/ui.h"

#include <cctype>
#include <fstream>

namespace genutils::cli {

Arguments::Arguments(std::vector<std::string> tokens) {
    parse_tokens(std::move(tokens));
}

Arguments Arguments::from_argv(int argc, const char* const* argv, std::size_t first) {
    if (argc < 0 || (argc > 0 && argv == nullptr)) {
        throw ParseError("invalid argc/argv pair");
    }
    const auto count = static_cast<std::size_t>(argc);
    if (first > count) throw ParseError("argument start index is out of range");
    std::vector<std::string> tokens;
    tokens.reserve(count - first);
    for (std::size_t index = first; index < count; ++index) {
        if (argv[index] == nullptr) throw ParseError("null command-line argument");
        tokens.emplace_back(argv[index]);
    }
    return Arguments(std::move(tokens));
}

Arguments Arguments::from_file(const std::filesystem::path& path) {
    return Arguments(tokenize_file(path));
}

bool Arguments::contains(std::string_view option) const {
    return options_.contains(canonical_name(option));
}

bool Arguments::flag(std::string_view option) const {
    const auto found = options_.find(canonical_name(option));
    if (found == options_.end()) return false;
    const auto& entry = found->second.back();
    return !entry.value || parse<bool>(*entry.value, option);
}

std::optional<std::string_view> Arguments::value(std::string_view option) const {
    const auto name = canonical_name(option);
    const auto found = options_.find(name);
    if (found == options_.end()) return std::nullopt;
    const auto& entry = found->second.back();
    if (!entry.value) throw ParseError("option requires a value: " + name);
    return *entry.value;
}

std::vector<std::string_view> Arguments::values(std::string_view option) const {
    const auto name = canonical_name(option);
    const auto found = options_.find(name);
    if (found == options_.end()) return {};
    std::vector<std::string_view> result;
    result.reserve(found->second.size());
    for (const auto& entry : found->second) {
        if (!entry.value) throw ParseError("option requires a value: " + name);
        result.emplace_back(*entry.value);
    }
    return result;
}

std::string Arguments::canonical_name(std::string_view option) {
    while (!option.empty() && option.front() == '-') option.remove_prefix(1);
    if (option.empty()) throw ParseError("empty option name");
    return std::string(option);
}

bool Arguments::is_option_token(std::string_view token) {
    if (token.size() < 2 || token.front() != '-') return false;
    const unsigned char second = static_cast<unsigned char>(token[1]);
    return token[1] == '-' || (!std::isdigit(second) && token[1] != '.');
}

void Arguments::parse_tokens(std::vector<std::string> tokens) {
    bool positional_only = false;
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        std::string token = std::move(tokens[index]);
        if (!positional_only && token == "--") {
            positional_only = true;
            continue;
        }
        if (positional_only || !is_option_token(token)) {
            positional_.push_back(std::move(token));
            continue;
        }

        const auto equal = token.find('=');
        const std::string name = canonical_name(
            equal == std::string::npos ? std::string_view(token)
                                       : std::string_view(token).substr(0, equal));
        Entry entry;
        if (equal != std::string::npos) {
            entry.value = token.substr(equal + 1);
        } else if (index + 1 < tokens.size() && !is_option_token(tokens[index + 1])) {
            entry.value = std::move(tokens[++index]);
        }
        options_[name].push_back(std::move(entry));
    }
}

std::vector<std::string> Arguments::tokenize_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw ParseError("cannot open argument file: " + path.string());

    std::vector<std::string> tokens;
    std::string current;
    char quote{};
    bool escaped = false;
    bool comment = false;
    char character{};
    while (input.get(character)) {
        if (comment) {
            if (character == '\n') comment = false;
            else continue;
        }
        if (escaped) {
            current.push_back(character);
            escaped = false;
            continue;
        }
        if (character == '\\') {
            escaped = true;
            continue;
        }
        if (quote != 0) {
            if (character == quote) quote = 0;
            else current.push_back(character);
            continue;
        }
        if (character == '\'' || character == '"') {
            quote = character;
        } else if (character == '#') {
            comment = true;
        } else if (std::isspace(static_cast<unsigned char>(character))) {
            if (!current.empty()) {
                tokens.push_back(std::move(current));
                current.clear();
            }
        } else {
            current.push_back(character);
        }
    }
    if (escaped) throw ParseError("dangling escape in argument file: " + path.string());
    if (quote != 0) throw ParseError("unterminated quote in argument file: " + path.string());
    if (!current.empty()) tokens.push_back(std::move(current));
    return tokens;
}

} // namespace genutils::cli
