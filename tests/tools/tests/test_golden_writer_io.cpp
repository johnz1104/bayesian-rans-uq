// Actual ordered writer I/O controls. POSIX children isolate real file-size
// limits and permission checks; no solver or stream implementation is mocked.
#include "GoldenWriter.hpp"
#include <csignal>
#include <filesystem>
#include <iostream>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;
using namespace cfdr::golden;

bool rejects(const RunInfo &info, const std::string &name, const JObj &body) {
    try {
        writeCase(info, name, body);
    } catch (const std::exception &) {
        return true;
    }
    return false;
}

bool childRejects(const RunInfo &info, const JObj &body, bool limitSize) {
    std::fflush(stdout);
    const auto child = fork();
    if (child < 0)
        throw std::runtime_error("fork failed");
    if (child == 0) {
        if (limitSize) {
            struct rlimit limit;
            if (getrlimit(RLIMIT_FSIZE, &limit) != 0)
                _exit(3);
            limit.rlim_cur = 0;
            if (setrlimit(RLIMIT_FSIZE, &limit) != 0 || std::signal(SIGXFSZ, SIG_IGN) == SIG_ERR)
                _exit(3);
        } else if (geteuid() == 0 && setuid(65534) != 0) {
            // A privileged runner must exercise permission denial as an
            // unprivileged child; inability to do so is a failed control.
            _exit(3);
        }
        _exit(rejects(info, "sample", body) ? 0 : 1);
    }
    int status = 0;
    if (waitpid(child, &status, 0) != child)
        throw std::runtime_error("waitpid failed");
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

int main() {
    auto pattern = (fs::temp_directory_path() / "cfdr-golden-io-XXXXXX").string();
    if (!mkdtemp(pattern.data()))
        throw std::runtime_error("mkdtemp failed");
    const fs::path root(pattern);
    JObj body;
    body.num("value", 1.25);
    int failures = 0;
    const auto check = [&](bool ok, const char *label) {
        if (!ok) {
            std::cerr << "FAIL: " << label << '\n';
            ++failures;
        }
    };
    const auto info = [&](const fs::path &directory) {
        return RunInfo{directory.string(), "synthetic-writer-control", false};
    };

    // JSON strings must escape every control byte, including embedded NUL.
    // Quotes and backslashes need escaping; valid UTF-8 must be preserved.
    std::string controls;
    for (int byte = 0; byte < 32; ++byte)
        controls.push_back(static_cast<char>(byte));
    controls += "\"\\\xcf\x89";
    const std::string expectedControls =
        "\"\\u0000\\u0001\\u0002\\u0003\\u0004\\u0005\\u0006\\u0007"
        "\\u0008\\u0009\\u000a\\u000b\\u000c\\u000d\\u000e\\u000f"
        "\\u0010\\u0011\\u0012\\u0013\\u0014\\u0015\\u0016\\u0017"
        "\\u0018\\u0019\\u001a\\u001b\\u001c\\u001d\\u001e\\u001f\\\"\\\\\xcf\x89\"";
    check(jsonString(controls) == expectedControls, "JSON string escaping is incomplete");
    JObj escaped;
    escaped.str(controls, controls);
    check(escaped.dump() == "{\n  " + expectedControls + ": " + expectedControls + "\n}",
          "object keys or values have invalid escaping");

    check(rejects(info(root / "missing"), "sample", body), "missing output directory accepted");
    {
        std::ofstream file(root / "regular-file");
        file << "existing";
    }
    check(rejects(info(root / "regular-file"), "sample", body), "file used as output directory");
    fs::create_directory(root / "sample.json");
    check(rejects(info(root), "sample", body), "directory used as JSON output file");
    fs::remove(root / "sample.json");

    // Ensure the parent is searchable even when the permission child drops uid.
    fs::permissions(root, fs::perms::owner_all | fs::perms::group_exec | fs::perms::others_exec);
    fs::create_directory(root / "blocked");
    fs::permissions(root / "blocked", fs::perms::none);
    check(childRejects(info(root / "blocked"), body, false), "unwritable directory accepted");
    fs::permissions(root / "blocked", fs::perms::owner_all);

    fs::create_directory(root / "flush");
    check(childRejects(info(root / "flush"), body, true), "buffered close failure accepted");
    fs::create_directory(root / "write");
    JObj large;
    large.str("value", std::string(65536, 'x'));
    check(childRejects(info(root / "write"), large, true), "large write failure accepted");

    writeCase(info(root), "sample", body);
    std::ifstream input(root / "sample.json");
    const std::string actual((std::istreambuf_iterator<char>(input)), {});
    const std::string expected =
        "{\n  \"case\": \"sample\",\n  \"code\": {\n"
        "    \"git_sha\": \"synthetic-writer-control\",\n    \"dirty\": false\n  },\n"
        "  \"build\": {\n    \"compiler\": \"" __VERSION__ "\",\n"
        "    \"cxx_flags\": \"" CFDR_CXX_FLAGS "\"\n  },\n  \"value\": 1.25\n}\n";
    check(actual == expected, "finite JSON bytes changed");
    fs::remove_all(root);
    if (failures == 0)
        std::cout
            << "golden writer escapes JSON, rejects six I/O failures and preserves finite output\n";
    return failures == 0 ? 0 : 1;
}
