#pragma once

#include <vector>
#include <cstdint>
#include <armadillo>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <ensmallen.hpp>
#include <limits>
#include <random>
#include "../../../utils/type.hpp"
#include "../../../utils/common.hpp"


/*
 * Ranker
 * ------
 * This module provides an auxiliary implementation for ranking data dimensions
 * based on a set of descriptive statistical features captured in the
 * DimFeatures structure.
 *
 * For each dimension, several distribution-related characteristics are computed
 * (e.g., variance, entropy, skewness, kurtosis, flatness, and normalized range).
 * These features are then normalized across dimensions and combined into a
 * single scalar score using a learned linear weighting.
 *
 * The weights are optimized by minimizing a custom loss function that encourages
 * score variance across dimensions while applying a smooth regularization term
 * to ensure numerical stability and prevent degenerate solutions.
 *
 * The final output is a descending ranking of dimensions based on their
 * computed scores derived from the learned feature weights.
 *
 * NOTE:
 * This ranking mechanism is self-contained and independent of other components
 * of the skd-tree index. It can be used as an analysis
 * step without affecting the index construction or query logic.
 */

namespace ranker
{
    struct DimFeatures {
        double variance;
        double entropy;
        double skew;
        double excess_kurtosis;
        double flatness;
        double range_norm;
    };

    struct RankedDimension {
        uint32_t dimension;
        double score;
    };

    static inline double smooth_abs(double x, double eps = 1e-6) {
        return std::sqrt(x * x + eps);
    }

    static inline double smooth_abs_grad(double x, double eps = 1e-6) {
        return x / std::sqrt(x * x + eps);
    }

    struct ScoreLoss {
        const std::vector<DimFeatures>& features;
        const double lambda = 0.01;
        const double eps_reg = 1e-6;

        ScoreLoss(const std::vector<DimFeatures>& features) : features(features) {}

        double Evaluate(const arma::mat& w) const {
            const int d = static_cast<int>(features.size());
            const int f = 6;

            arma::vec scores(d);
            arma::mat FM(f, d);
            for (int i = 0; i < d; ++i) {
                FM(0, i) = features[i].variance;
                FM(1, i) = features[i].entropy;
                FM(2, i) = -features[i].skew;
                FM(3, i) = -features[i].excess_kurtosis;
                FM(4, i) = -features[i].flatness;
                FM(5, i) = features[i].range_norm;
                scores(i) = arma::dot(w.col(0), FM.col(i));
            }

            double mean_score = arma::mean(scores);
            double var = arma::mean(arma::square(scores - mean_score));

            double reg = 0.0;
            for (int i = 0; i < f; ++i) reg += smooth_abs(w(i, 0), eps_reg);
            reg *= lambda;

            return -var + reg;
        }

        void Gradient(const arma::mat& w, arma::mat& grad) const {
            const int d = static_cast<int>(features.size());
            const int f = 6;

            arma::mat F(f, d);
            arma::vec scores(d);
            for (int i = 0; i < d; ++i) {
                F(0, i) = features[i].variance;
                F(1, i) = features[i].entropy;
                F(2, i) = -features[i].skew;
                F(3, i) = -features[i].excess_kurtosis;
                F(4, i) = -features[i].flatness;
                F(5, i) = features[i].range_norm;
            }

            scores = F.t() * w;
            double mean_score = arma::mean(scores);

            grad.set_size(f, 1);
            grad.zeros();

            for (int j = 0; j < f; ++j) {
                double g = 0.0;
                double mean_fj = arma::mean(F.row(j));
                for (int i = 0; i < d; ++i) {
                    g += (scores(i) - mean_score) * (F(j, i) - mean_fj);
                }
                g *= (2.0 / d);
                double reg_grad = lambda * smooth_abs_grad(w(j, 0), eps_reg);
                grad(j, 0) = -g + reg_grad;
            }
        }
    };

    static inline double safe_std(double var, double floor = 1e-9) {
        return std::sqrt(std::max(var, floor));
    }

    static inline double mean(const std::vector<double>& v) {
        if (v.empty()) return 0.0;
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    }

    static inline double variance(const std::vector<double>& v, double m) {
        if (v.empty()) return 0.0;
        double sum = 0.0;
        for (auto x : v) sum += (x - m) * (x - m);
        return sum / v.size();
    }

    static inline double skewness(const std::vector<double>& v, double m, double stddev) {
        if (v.empty() || stddev < 1e-12) return 0.0;
        double sum = 0.0;
        for (auto x : v) sum += pow(x - m, 3);
        return sum / (v.size() * pow(stddev, 3));
    }

    static inline double kurtosis(const std::vector<double>& v, double m, double stddev) {
        if (v.empty() || stddev < 1e-12) return 0.0;
        double sum = 0.0;
        for (auto x : v) sum += pow(x - m, 4);
        return sum / (v.size() * pow(stddev, 4));
    }

    struct HistogramStats {
        double entropy;
        double flatness;
    };

    static HistogramStats computeHistogramStats(const std::vector<double>& v) {
        if (v.empty()) return {0.0, 1.0};

        int bins = std::clamp(static_cast<int>(std::sqrt(v.size())), 4, 100);
        std::vector<double> counts(bins, 1e-6);

        auto [min_v, max_v] = std::minmax_element(v.begin(), v.end());
        double span = *max_v - *min_v;
        double bin_size = (span < 1e-12) ? 1.0 : (span / bins);

        for (double x : v) {
            int idx = (span < 1e-12) ? 0 : std::min(bins - 1, int((x - *min_v) / bin_size));
            counts[idx] += 1.0;
        }

        double total = std::accumulate(counts.begin(), counts.end(), 0.0);
        double mean_count = total / bins;
        double var_counts = 0.0;
        double entropy = 0.0;

        for (double c : counts) {
            double p = c / total;
            entropy -= p * std::log2(p);
            var_counts += (c - mean_count) * (c - mean_count);
        }

        var_counts /= bins;
        double denom = mean_count * mean_count;
        double flatness = (denom < 1e-12) ? 1.0 : 1.0 - var_counts / denom;

        return {entropy, flatness};
    }

    template<size_t dim>
    class DimRanker {
        using Point = point_t<dim>;
        using Points = std::vector<Point>;
    public:
        DimRanker(Points& points) : points(points) {}

        std::vector<RankedDimension> rankDimensions() {
            computeFeatures();
            normalizeFeatures();
            learnWeights();

            std::vector<RankedDimension> result;
            for (size_t i = 0; i < features.size(); ++i) {
                double s = computeScore(features[i], weights);
                result.push_back({static_cast<uint32_t>(i), s});
            }
            std::sort(result.begin(), result.end(), [](const RankedDimension& a, const RankedDimension& b) {
                return a.score > b.score;
            });
            return result;
        }

        std::vector<RankedDimension> rankDimensions(double sample_ratio) {
            computeFeatures(sample_ratio);
            normalizeFeatures();
            learnWeights();

            std::vector<RankedDimension> result;
            for (size_t i = 0; i < features.size(); ++i) {
                double s = computeScore(features[i], weights);
                result.push_back({static_cast<uint32_t>(i), s});
            }
            std::sort(result.begin(), result.end(), [](const RankedDimension& a, const RankedDimension& b) {
                return a.score > b.score;
            });

            // computeCorrelationMatrixSampled();
            return result;
        }

        void printCorrelationMatrix()
        {
            const size_t d = correlationMatrix.n_rows;

            std::cout << "Correlation matrix (" << d << "x" << d << "):\n\n";

            // Header
            std::cout << "      ";
            for (size_t j = 0; j < d; ++j)
                std::cout << "d" << j << "      ";
            std::cout << "\n";

            // Rows
            for (size_t i = 0; i < d; ++i)
            {
                std::cout << "d" << i << "  ";
                for (size_t j = 0; j < d; ++j)
                {
                    double v = correlationMatrix(i, j);
                    if (std::abs(v) < 1e-12) v = 0.0; // remove tiny numerical noise
                    std::cout << std::setw(7) << v << " ";
                }
                std::cout << "\n";
            }
        }

    private:
        Points& points;
        std::vector<DimFeatures> features;
        arma::vec globalFeatureMeans;
        arma::vec globalFeatureStds;
        arma::vec weights;
        arma::vec meansPerDim;
        std::vector<size_t> corrSampleIdx;
        arma::mat correlationMatrix;
        
        // Optimized full dataset feature computation
        void computeFeatures() {
            const size_t d = dim;
            const size_t n = points.size();
            features.clear();
            features.reserve(d);
            meansPerDim.set_size(d);

            std::vector<double> vals;
            vals.reserve(n);

            for (size_t j = 0; j < d; ++j) {
                vals.clear();
                for (size_t i = 0; i < n; ++i)
                    vals.push_back(points[i][j]);

                double m = mean(vals);
                double var = variance(vals, m);
                double stddev = safe_std(var);
                double sk = skewness(vals, m, stddev);
                double kurt = kurtosis(vals, m, stddev) - 3.0;

                HistogramStats h = computeHistogramStats(vals);
                double ent = h.entropy;
                double flatness = h.flatness;

                double range_norm = ((*std::max_element(vals.begin(), vals.end())) -
                                     (*std::min_element(vals.begin(), vals.end()))) / (stddev + 1e-9);
                if (!std::isfinite(range_norm) || std::abs(range_norm) > 1e6)
                    range_norm = (range_norm > 0) ? 1e6 : -1e6;

                features.push_back({var, ent, std::abs(sk), std::abs(kurt), flatness, range_norm});
                meansPerDim(j) = m;
            }
        }

        // Optimized sampled feature computation
        void computeFeatures(double sample_ratio) {
            const size_t d = dim;
            const size_t total_n = points.size();
            const size_t n = static_cast<size_t>(total_n * sample_ratio);
            // std::cout << "ratio: " << sample_ratio << " -- size: " << total_n << " -- sample_size: " << n << std::endl;

            features.clear();
            features.reserve(d);
            meansPerDim.set_size(d);

            std::vector<size_t> sampleIdx;
            sampleIdx.reserve(n);
            if (sample_ratio < 0.5) {
                size_t step = std::max<size_t>(1, total_n / n);
                for (size_t i = 0; i < total_n && sampleIdx.size() < n; i += step)
                    sampleIdx.push_back(i);
            } else {
                sampleIdx.resize(total_n);
                std::iota(sampleIdx.begin(), sampleIdx.end(), 0);
                std::mt19937 gen(12345);
                std::shuffle(sampleIdx.begin(), sampleIdx.end(), gen);
                sampleIdx.resize(n);
            }

            std::vector<double> vals;
            vals.reserve(n);

            for (size_t j = 0; j < d; ++j) {
                vals.clear();
                for (size_t i = 0; i < n; ++i)
                    vals.push_back(points[sampleIdx[i]][j]);

                double m = mean(vals);
                double var = variance(vals, m);
                double stddev = safe_std(var);
                double sk = skewness(vals, m, stddev);
                double kurt = kurtosis(vals, m, stddev) - 3.0;

                HistogramStats h = computeHistogramStats(vals);
                double ent = h.entropy;
                double flatness = h.flatness;

                double range_norm = ((*std::max_element(vals.begin(), vals.end())) -
                                     (*std::min_element(vals.begin(), vals.end()))) / (stddev + 1e-9);
                if (!std::isfinite(range_norm) || std::abs(range_norm) > 1e6)
                    range_norm = (range_norm > 0) ? 1e6 : -1e6;

                features.push_back({var, ent, std::abs(sk), std::abs(kurt), flatness, range_norm});
                meansPerDim(j) = m;
            }

            corrSampleIdx = sampleIdx;

        }

        void normalizeFeatures() {
            const size_t d = features.size();
            constexpr size_t f = 6;
            arma::mat M(f, d);

            for (size_t j = 0; j < d; ++j) {
                M(0, j) = features[j].variance;
                M(1, j) = features[j].entropy;
                M(2, j) = features[j].skew;
                M(3, j) = features[j].excess_kurtosis;
                M(4, j) = features[j].flatness;
                M(5, j) = features[j].range_norm;
            }

            globalFeatureMeans = arma::mean(M, 1);
            globalFeatureStds = arma::stddev(M, 0, 1);

            for (size_t i = 0; i < f; ++i) {
                if (globalFeatureStds(i) > 1e-9)
                    M.row(i) = (M.row(i) - globalFeatureMeans(i)) / globalFeatureStds(i);
                else
                    M.row(i).zeros();
            }

            for (size_t j = 0; j < d; ++j) {
                features[j] = {M(0, j), M(1, j), M(2, j), M(3, j), M(4, j), M(5, j)};
            }
        }

        void learnWeights() {
            arma::mat w = arma::ones<arma::mat>(6, 1) / 6.0;
            ScoreLoss loss_fn(features);
            ens::L_BFGS optimizer;
            optimizer.Optimize(loss_fn, w);

            double nrm = arma::norm(w);
            if (nrm > 1e-9) w /= nrm;
            weights = w;
        }

        double computeScore(const DimFeatures& f, const arma::vec& w) const {
            return w(0)*f.variance + w(1)*f.entropy - w(2)*f.skew -
                   w(3)*f.excess_kurtosis + w(4)*f.flatness + w(5)*f.range_norm;
        }

        void computeCorrelationMatrixSampled()
        {
            const size_t d = dim;
            const size_t n = corrSampleIdx.size();

            correlationMatrix.set_size(d,d);

            arma::vec meanVec(d, arma::fill::zeros);
            arma::vec stdVec(d, arma::fill::zeros);

            // Compute mean and std (sample std, n-1)
            for (size_t j = 0; j < d; ++j)
            {
                double sum = 0.0;
                for (size_t idx : corrSampleIdx) sum += points[idx][j];
                meanVec(j) = sum / n;

                double var = 0.0;
                for (size_t idx : corrSampleIdx) var += (points[idx][j] - meanVec(j)) * (points[idx][j] - meanVec(j));
                var /= (n - 1);
                stdVec(j) = std::sqrt(var);
            }

            // Compute correlation
            for (size_t i = 0; i < d; ++i)
            {
                for (size_t j = i; j < d; ++j)
                {
                    double cov = 0.0;
                    for (size_t idx : corrSampleIdx)
                        cov += (points[idx][i] - meanVec(i)) * (points[idx][j] - meanVec(j));

                    cov /= (n - 1);
                    double corr = cov / (stdVec(i) * stdVec(j));

                    if (!std::isfinite(corr)) corr = 0.0;

                    correlationMatrix(i,j) = corr;
                    correlationMatrix(j,i) = corr;
                }
            }
        }

    };
}
