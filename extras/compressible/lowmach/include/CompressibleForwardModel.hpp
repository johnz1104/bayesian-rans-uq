#pragma once
#include "Mesh.hpp"
#include "SSTModel.hpp"
#include "ObservationOperator.hpp"
#include "FlowFields.hpp"
#include "EvaluationTypes.hpp"
#include "CompressibleFlowFields.hpp"
#include "CompressibleBCs.hpp"
#include "CompressibleSIMPLESolver.hpp"
#include "IdealGasEOS.hpp"
#include "InferenceParameters.hpp"
#include <vector>

// Compressible RANS forward model for Bayesian SST calibration.
//
// Accepts a parameter vector θ (via InferenceParameterSet), sets SST
// coefficients, runs the compressible SIMPLE solver, and evaluates the
// ObservationOperator to produce a scalar log-likelihood.
//
// The interface mirrors the incompressible ForwardModel so existing
// BayesianInference machinery can be reused without modification.
class CompressibleForwardModel {
  public:
    CompressibleForwardModel(const Mesh &mesh, const InferenceParameterSet &paramSet,
                             const ObservationOperator &obsOp,
                             const CompressibleBoundaryConditions &bcs, const IdealGasEOS &eos,
                             const SolverSettings &settings = {},
                             const Vec3 &uInit = {1.0, 0.0, 0.0}, double pInit = 101325.0,
                             double TInit = 300.0, double kInit = 1e-4, double omegaInit = 100.0);

    EvaluationResult evaluate(const std::vector<double> &theta);
    double penalizedLogLikelihood(const std::vector<double> &theta);

    bool hasLastFields() const { return hasFields_; }

    // Return last solution as FlowFields-compatible dict (accessed from Python)
    const CompressibleFlowFields &lastCompressibleFields() const {
        if (!hasFields_)
            throw std::logic_error("No fields retained for the last compressible evaluation");
        return lastFields_;
    }
    const InferenceParameterSet &paramSet() const { return paramSet_; }

  private:
    Mesh mesh_;
    InferenceParameterSet paramSet_;
    ObservationOperator obsOp_;
    CompressibleBoundaryConditions bcs_;
    IdealGasEOS eos_;
    SolverSettings settings_;

    Vec3 uInit_;
    double pInit_, TInit_, kInit_, omegaInit_;

    CompressibleFlowFields lastFields_;
    bool hasFields_ = false;
};
