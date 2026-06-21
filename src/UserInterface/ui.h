/******************************************************************************
 * Generic C++20 command-line and configuration-file parsing.
 * SPDX-License-Identifier: MIT
 ******************************************************************************/
#pragma once

#include <charconv>
#include <cstddef>
#include <filesystem>
#include <locale>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace genutils::cli {

class ParseError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class Arguments {
public:
    Arguments() = default;
    explicit Arguments(std::vector<std::string> tokens);

    [[nodiscard]] static Arguments from_argv(int argc, const char* const* argv,
                                             std::size_t first = 1);
    [[nodiscard]] static Arguments from_file(const std::filesystem::path& path);

    [[nodiscard]] bool contains(std::string_view option) const;
    [[nodiscard]] bool flag(std::string_view option) const;
    [[nodiscard]] std::optional<std::string_view> value(std::string_view option) const;
    [[nodiscard]] std::vector<std::string_view> values(std::string_view option) const;
    [[nodiscard]] const std::vector<std::string>& positional() const noexcept {
        return positional_;
    }

    template <typename T>
    [[nodiscard]] T require(std::string_view option) const {
        const auto raw = value(option);
        if (!raw) throw ParseError("missing required option: " + canonical_name(option));
        return parse<T>(*raw, option);
    }

    template <typename T>
    [[nodiscard]] T get_or(std::string_view option, T fallback) const {
        const auto raw = value(option);
        return raw ? parse<T>(*raw, option) : std::move(fallback);
    }

    template <typename T>
    [[nodiscard]] std::vector<T> values_as(std::string_view option) const {
        std::vector<T> result;
        const auto raw_values = values(option);
        result.reserve(raw_values.size());
        for (const auto raw : raw_values) result.push_back(parse<T>(raw, option));
        return result;
    }

private:
    struct Entry {
        std::optional<std::string> value;
    };

    std::unordered_map<std::string, std::vector<Entry>> options_;
    std::vector<std::string> positional_;

    [[nodiscard]] static std::string canonical_name(std::string_view option);
    [[nodiscard]] static bool is_option_token(std::string_view token);
    [[nodiscard]] static std::vector<std::string> tokenize_file(
        const std::filesystem::path& path);
    void parse_tokens(std::vector<std::string> tokens);

    template <typename T>
    [[nodiscard]] static T parse(std::string_view text, std::string_view option) {
        if constexpr (std::is_same_v<T, std::string>) {
            return std::string(text);
        } else if constexpr (std::is_same_v<T, bool>) {
            if (text == "1" || text == "true" || text == "yes" || text == "on") return true;
            if (text == "0" || text == "false" || text == "no" || text == "off") return false;
        } else if constexpr (std::is_integral_v<T>) {
            T result{};
            const auto* begin = text.data();
            const auto* end = begin + text.size();
            const auto [position, error] = std::from_chars(begin, end, result);
            if (error == std::errc{} && position == end) return result;
        } else if constexpr (std::is_floating_point_v<T>) {
            std::istringstream input{std::string(text)};
            input.imbue(std::locale::classic());
            T result{};
            input >> result;
            if (input && input.peek() == std::char_traits<char>::eof()) return result;
        } else {
            static_assert(std::is_same_v<T, void>, "unsupported CLI conversion type");
        }
        throw ParseError("invalid value '" + std::string(text) + "' for option " +
                         canonical_name(option));
    }
};

} // namespace genutils::cli
