#pragma once
#include <vector>

// Shared between incompressible and compressible forward models.

enum class EvaluationStatus { Converged, Unconverged, Diverged, InvalidParameters, Unknown };

inline const char *statusString(EvaluationStatus s) {
    switch (s) {
    case EvaluationStatus::Converged:
        return "Converged";
    case EvaluationStatus::Unconverged:
        return "Unconverged";
    case EvaluationStatus::Diverged:
        return "Diverged";
    case EvaluationStatus::InvalidParameters:
        return "InvalidParameters";
    case EvaluationStatus::Unknown:
        return "Unknown";
    }
    return "Unknown";
}

struct EvaluationResult {
    EvaluationStatus status = EvaluationStatus::Unknown;
    double loglik = -1e30;
    std::vector<double> predictions;
    int simpleIters = 0;
};
