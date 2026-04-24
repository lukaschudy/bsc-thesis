#include "fortran_runtime.hpp"

namespace runtime {

bool are_equal_reals(double a, double b) {
    return std::abs(a - b) <= std::numeric_limits<double>::epsilon() * std::max(1.0, std::max(std::abs(a), std::abs(b)));
}

std::vector<int> convert_number_base(int n, int b, int l) {
    std::vector<int> out(l + 1, 0);
    int tmp = n;
    for (int i = 1; i <= l; ++i) {
        out[l - i + 1] = (tmp % b) + 1;
        tmp /= b;
    }
    return out;
}

std::vector<std::vector<double>> compute_row_summary_statistics(
    int m,
    int n,
    const std::vector<std::vector<double>>& x) {
    std::vector<std::vector<double>> y(m + 1, std::vector<double>(10, 0.0));

    for (int i = 1; i <= m; ++i) {
        double s1 = 0.0;
        double s2 = 0.0;
        double mn = x[i][1];
        double mx = x[i][1];
        std::vector<double> tmp(n + 1, 0.0);

        for (int j = 1; j <= n; ++j) {
            const double v = x[i][j];
            s1 += v;
            s2 += v * v;
            mn = std::min(mn, v);
            mx = std::max(mx, v);
            tmp[j] = v;
        }

        const double mean = s1 / static_cast<double>(n);
        y[i][1] = mean;
        y[i][2] = std::sqrt(std::abs(s2 / static_cast<double>(n) - mean * mean));
        y[i][3] = mn;
        y[i][9] = mx;

        std::sort(tmp.begin() + 1, tmp.begin() + n + 1);

        auto idx = [n](double q) {
            int k = static_cast<int>(std::floor(q * static_cast<double>(n) + 0.5));
            if (k < 1) {
                k = 1;
            }
            if (k > n) {
                k = n;
            }
            return k;
        };

        y[i][4] = tmp[idx(0.025)];
        y[i][5] = tmp[idx(0.25)];
        y[i][6] = tmp[idx(0.5)];
        y[i][7] = tmp[idx(0.75)];
        y[i][8] = tmp[idx(0.975)];
    }

    return y;
}

double ran2(int& idum, std::vector<int>& iv, int& iy, int& idum2) {
    constexpr int IM1 = 2147483563;
    constexpr int IM2 = 2147483399;
    constexpr int IMM1 = IM1 - 1;
    constexpr int IA1 = 40014;
    constexpr int IA2 = 40692;
    constexpr int IQ1 = 53668;
    constexpr int IQ2 = 52774;
    constexpr int IR1 = 12211;
    constexpr int IR2 = 3791;
    constexpr int NTAB = 32;
    constexpr int NDIV = 1 + IMM1 / NTAB;
    constexpr double AM = 1.0 / static_cast<double>(IM1);
    constexpr double EPS = 1.2e-7;
    constexpr double RNMX = 1.0 - EPS;

    if (static_cast<int>(iv.size()) < NTAB + 1) {
        iv.assign(NTAB + 1, 0);
    }

    int j;
    int k;

    if (idum <= 0) {
        idum = std::max(-idum, 1);
        idum2 = idum;
        for (j = NTAB + 8; j >= 1; --j) {
            k = idum / IQ1;
            idum = IA1 * (idum - k * IQ1) - k * IR1;
            if (idum < 0) {
                idum += IM1;
            }
            if (j <= NTAB) {
                iv[j] = idum;
            }
        }
        iy = iv[1];
    }

    k = idum / IQ1;
    idum = IA1 * (idum - k * IQ1) - k * IR1;
    if (idum < 0) {
        idum += IM1;
    }

    k = idum2 / IQ2;
    idum2 = IA2 * (idum2 - k * IQ2) - k * IR2;
    if (idum2 < 0) {
        idum2 += IM2;
    }

    j = 1 + iy / NDIV;
    if (j < 1) {
        j = 1;
    }
    if (j > NTAB) {
        j = NTAB;
    }

    iy = iv[j] - idum2;
    iv[j] = idum;
    if (iy < 1) {
        iy += IMM1;
    }

    return std::min(AM * static_cast<double>(iy), RNMX);
}

void update_vector_average(int m, int n, const std::vector<double>& x, std::vector<double>& xbar) {
    const double w_old = static_cast<double>(n - 1) / static_cast<double>(n);
    const double w_new = 1.0 / static_cast<double>(n);
    for (int i = 1; i <= m; ++i) {
        xbar[i] = w_old * xbar[i] + w_new * x[i];
    }
}

void update_scalar_average(int n, double x, double& xbar) {
    const double w_old = static_cast<double>(n - 1) / static_cast<double>(n);
    const double w_new = 1.0 / static_cast<double>(n);
    xbar = w_old * xbar + w_new * x;
}

void maxloc_break_ties(
    int n,
    const std::vector<double>& x,
    int& idumQ,
    std::vector<int>& ivQ,
    int& iyQ,
    int& idum2Q,
    double& m,
    int& p) {
    m = x[1];
    for (int i = 2; i <= n; ++i) {
        m = std::max(m, x[i]);
    }

    std::vector<int> tied(n + 1, 0);
    int h = 0;
    for (int i = 1; i <= n; ++i) {
        if (are_equal_reals(x[i], m)) {
            ++h;
            tied[h] = i;
        }
    }

    if (h > 1) {
        const double u = ran2(idumQ, ivQ, iyQ, idum2Q);
        p = tied[1 + static_cast<int>(static_cast<double>(h) * u)];
    } else {
        p = tied[1];
    }
}

void generate_combinations(
    int rows,
    int cols,
    const std::vector<int>& lengths,
    const std::vector<std::vector<int>>& x,
    int totrows,
    std::vector<std::vector<int>>& comb) {
    (void)rows;
    for (int itotrows = 1; itotrows <= totrows; ++itotrows) {
        int itmp = itotrows - 1;
        for (int icol = cols; icol >= 1; --icol) {
            const int index = 1 + (itmp % lengths[icol]);
            itmp /= lengths[icol];
            comb[itotrows][icol] = x[index][icol];
        }
    }
}

int maxloc_first(const std::vector<double>& x, int start) {
    int idx = start;
    double best = x[start];
    for (int i = start + 1; i < static_cast<int>(x.size()); ++i) {
        if (x[i] > best) {
            best = x[i];
            idx = i;
        }
    }
    return idx;
}

int minloc_first(const std::vector<double>& x, int start) {
    int idx = start;
    double best = x[start];
    for (int i = start + 1; i < static_cast<int>(x.size()); ++i) {
        if (x[i] < best) {
            best = x[i];
            idx = i;
        }
    }
    return idx;
}

double sum_range(const std::vector<double>& x, int lo, int hi) {
    double s = 0.0;
    for (int i = lo; i <= hi; ++i) {
        s += x[i];
    }
    return s;
}

std::vector<int> flatten_transpose_depth_agent(const std::vector<std::vector<int>>& p, int depth, int agents) {
    std::vector<int> out(depth * agents + 1, 0);
    int k = 1;
    for (int d = 1; d <= depth; ++d) {
        for (int a = 1; a <= agents; ++a) {
            out[k++] = p[d][a];
        }
    }
    return out;
}

std::vector<std::vector<int>> reshape_to_depth_agent(const std::vector<int>& v, int depth, int agents) {
    std::vector<std::vector<int>> p(depth + 1, std::vector<int>(agents + 1, 0));
    int k = 1;
    for (int a = 1; a <= agents; ++a) {
        for (int d = 1; d <= depth; ++d) {
            p[d][a] = v[k++];
        }
    }
    return p;
}

namespace io {

namespace {
std::unordered_map<int, std::fstream> g_units;
}

void open_unit(int unit, const std::string& file_name, std::ios::openmode mode) {
    auto& f = g_units[unit];
    if (f.is_open()) {
        f.close();
    }
    f.clear();
    f.open(file_name, mode);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_name);
    }
}

void close_unit(int unit) {
    auto it = g_units.find(unit);
    if (it != g_units.end() && it->second.is_open()) {
        it->second.close();
    }
}

bool unit_is_open(int unit) {
    const auto it = g_units.find(unit);
    return it != g_units.end() && it->second.is_open();
}

std::fstream& unit(int unit) {
    auto it = g_units.find(unit);
    if (it == g_units.end() || !it->second.is_open()) {
        throw std::runtime_error("Requested unopened unit: " + std::to_string(unit));
    }
    return it->second;
}

void skip_line(int unit_number) {
    std::string line;
    std::getline(unit(unit_number), line);
}

}  // namespace io

}  // namespace runtime
