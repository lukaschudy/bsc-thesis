#include "generic_routines.hpp"

namespace generic_routines {

std::vector<std::vector<double>> ComputeRowSummaryStatistics(
    int m,
    int n,
    const std::vector<std::vector<double>>& x) {
    return runtime::compute_row_summary_statistics(m, n, x);
}

bool AreEqualReals(double a, double b) {
    return runtime::are_equal_reals(a, b);
}

std::vector<int> convertNumberBase(int n, int b, int l) {
    return runtime::convert_number_base(n, b, l);
}

double ran2(int& idum, std::vector<int>& iv, int& iy, int& idum2) {
    return runtime::ran2(idum, iv, iy, idum2);
}

void generateCombinations(
    int rows,
    int cols,
    const std::vector<int>& lengths,
    const std::vector<std::vector<int>>& x,
    int totrows,
    std::vector<std::vector<int>>& comb) {
    runtime::generate_combinations(rows, cols, lengths, x, totrows, comb);
}

void updateVectorAverage(int m, int n, const std::vector<double>& x, std::vector<double>& xbar) {
    runtime::update_vector_average(m, n, x, xbar);
}

void updateScalarAverage(int n, double x, double& xbar) {
    runtime::update_scalar_average(n, x, xbar);
}

void MaxLocBreakTies(
    int n,
    const std::vector<double>& x,
    int& idumQ,
    std::vector<int>& ivQ,
    int& iyQ,
    int& idum2Q,
    double& m,
    int& p) {
    runtime::maxloc_break_ties(n, x, idumQ, ivQ, iyQ, idum2Q, m, p);
}

}  // namespace generic_routines
