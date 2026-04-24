#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace runtime {

template <typename T>
using Vec1 = std::vector<T>;

template <typename T>
using Vec2 = std::vector<std::vector<T>>;

template <typename T>
using Vec3 = std::vector<std::vector<std::vector<T>>>;

template <typename T>
using Vec4 = std::vector<std::vector<std::vector<std::vector<T>>>>;

template <typename T>
inline Vec1<T> make1(int n, const T& value = T()) {
    return Vec1<T>(n + 1, value);
}

template <typename T>
inline Vec2<T> make2(int n1, int n2, const T& value = T()) {
    return Vec2<T>(n1 + 1, Vec1<T>(n2 + 1, value));
}

template <typename T>
inline Vec3<T> make3(int n1, int n2, int n3, const T& value = T()) {
    return Vec3<T>(n1 + 1, Vec2<T>(n2 + 1, Vec1<T>(n3 + 1, value)));
}

template <typename T>
inline Vec4<T> make4(int n1, int n2, int n3, int n4, const T& value = T()) {
    return Vec4<T>(n1 + 1, Vec3<T>(n2 + 1, Vec2<T>(n3 + 1, Vec1<T>(n4 + 1, value))));
}

bool are_equal_reals(double a, double b);

std::vector<int> convert_number_base(int n, int b, int l);

std::vector<std::vector<double>> compute_row_summary_statistics(
    int m,
    int n,
    const std::vector<std::vector<double>>& x);

// Numerical Recipes ran2 RNG; Fortran-compatible stateful interface.
double ran2(int& idum, std::vector<int>& iv, int& iy, int& idum2);

void update_vector_average(int m, int n, const std::vector<double>& x, std::vector<double>& xbar);

void update_scalar_average(int n, double x, double& xbar);

void maxloc_break_ties(
    int n,
    const std::vector<double>& x,
    int& idumQ,
    std::vector<int>& ivQ,
    int& iyQ,
    int& idum2Q,
    double& m,
    int& p);

void generate_combinations(
    int rows,
    int cols,
    const std::vector<int>& lengths,
    const std::vector<std::vector<int>>& x,
    int totrows,
    std::vector<std::vector<int>>& comb);

int maxloc_first(const std::vector<double>& x, int start = 1);
int minloc_first(const std::vector<double>& x, int start = 1);

double sum_range(const std::vector<double>& x, int lo, int hi);

std::vector<int> flatten_transpose_depth_agent(const std::vector<std::vector<int>>& p, int depth, int agents);
std::vector<std::vector<int>> reshape_to_depth_agent(const std::vector<int>& v, int depth, int agents);

namespace io {

void open_unit(int unit, const std::string& file_name, std::ios::openmode mode);
void close_unit(int unit);
bool unit_is_open(int unit);
std::fstream& unit(int unit);
void skip_line(int unit);

}  // namespace io

}  // namespace runtime
