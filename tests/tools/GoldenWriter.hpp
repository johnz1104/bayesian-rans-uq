#pragma once

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef CFDR_CXX_FLAGS
#define CFDR_CXX_FLAGS "unknown"
#endif

namespace cfdr::golden {

// ---- minimal ordered JSON writer -------------------------------------------

inline std::string jsonNumber(double v) {
    if (!std::isfinite(v))
        throw std::invalid_argument("cannot record a nonfinite golden number");
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return buf;
}

inline std::string jsonString(const std::string &s) {
    std::string out = "\"";
    for (char c : s) {
        const auto byte = static_cast<unsigned char>(c);
        if (byte < 0x20) {
            constexpr char hex[] = "0123456789abcdef";
            out += "\\u00";
            out += hex[byte >> 4];
            out += hex[byte & 0x0f];
        } else {
            if (c == '"' || c == '\\')
                out += '\\';
            out += c;
        }
    }
    return out + "\"";
}

struct JObj {
    std::vector<std::pair<std::string, std::string>> kv; // key -> encoded value

    JObj &num(const std::string &k, double v) {
        kv.emplace_back(k, jsonNumber(v));
        return *this;
    }
    JObj &integer(const std::string &k, long v) {
        kv.emplace_back(k, std::to_string(v));
        return *this;
    }
    JObj &str(const std::string &k, const std::string &v) {
        kv.emplace_back(k, jsonString(v));
        return *this;
    }
    JObj &boolean(const std::string &k, bool v) {
        kv.emplace_back(k, v ? "true" : "false");
        return *this;
    }
    JObj &arr(const std::string &k, const std::vector<double> &v) {
        std::string s = "[";
        for (size_t i = 0; i < v.size(); ++i)
            s += (i ? ", " : "") + jsonNumber(v[i]);
        kv.emplace_back(k, s + "]");
        return *this;
    }
    JObj &mat(const std::string &k, const std::vector<std::vector<double>> &m) {
        std::string s = "[";
        for (size_t i = 0; i < m.size(); ++i) {
            s += (i ? ",\n    [" : "\n    [");
            for (size_t j = 0; j < m[i].size(); ++j)
                s += (j ? ", " : "") + jsonNumber(m[i][j]);
            s += "]";
        }
        kv.emplace_back(k, s + "]");
        return *this;
    }
    JObj &obj(const std::string &k, const JObj &o) {
        kv.emplace_back(k, o.dump(1));
        return *this;
    }

    std::string dump(int depth = 0) const {
        std::string pad(2 * (depth + 1), ' '), padEnd(2 * depth, ' ');
        std::string s = "{\n";
        for (size_t i = 0; i < kv.size(); ++i)
            s += pad + jsonString(kv[i].first) + ": " + kv[i].second +
                 (i + 1 < kv.size() ? ",\n" : "\n");
        return s + padEnd + "}";
    }
};

struct RunInfo {
    std::string outDir, sha;
    bool dirty = false;
};

inline void writeCase(const RunInfo &ri, const std::string &name, JObj body) {
    if (!std::filesystem::is_directory(ri.outDir))
        throw std::runtime_error("golden output directory does not exist or is not a directory: " +
                                 ri.outDir);
    JObj code;
    code.str("git_sha", ri.sha).boolean("dirty", ri.dirty);
    JObj build;
    build.str("compiler", __VERSION__).str("cxx_flags", CFDR_CXX_FLAGS);
    JObj top;
    top.str("case", name).obj("code", code).obj("build", build);
    for (auto &e : body.kv)
        top.kv.push_back(e);
    const std::string filename = ri.outDir + "/" + name + ".json";
    try {
        std::ofstream os;
        os.exceptions(std::ios::failbit | std::ios::badbit);
        os.open(filename);
        os << top.dump() << "\n";
        // Buffered writes can fail only when close flushes the buffer.
        os.close();
    } catch (const std::ios_base::failure &error) {
        throw std::runtime_error("could not write golden " + filename + ": " + error.what());
    }
    std::printf("  wrote %s/%s.json\n", ri.outDir.c_str(), name.c_str());
}

} // namespace cfdr::golden
