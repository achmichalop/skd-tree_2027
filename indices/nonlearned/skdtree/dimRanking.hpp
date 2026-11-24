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

// /*
//  * Ranker
//  * ------
//  * This module provides an auxiliary implementation for ranking data dimensions
//  * based on a set of descriptive statistical features captured in the
//  * DimFeatures structure.
//  *
//  * For each dimension, several distribution-related characteristics are computed
//  * (e.g., variance, entropy, skewness, kurtosis, flatness, and normalized range).
//  * These features are then normalized across dimensions and combined into a
//  * single scalar score using a learned linear weighting.
//  *
//  * The weights are optimized by minimizing a custom loss function that encourages
//  * score variance across dimensions while applying a smooth regularization term
//  * to ensure numerical stability and prevent degenerate solutions.
//  *
//  * The final output is a descending ranking of dimensions based on their
//  * computed scores derived from the learned feature weights.
//  *
//  * NOTE:
//  * This ranking mechanism is self-contained and independent of other components
//  * of the skd-tree index. It can be used as an analysis
//  * step without affecting the index construction or query logic.
//  */

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
        uint32_t distinct = 0;
        bool gated = false;
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

    static HistogramStats computeHistogramStats(const std::vector<double>& v,
                                                  double min_v, double max_v) {
        if (v.empty()) return {0.0, 1.0};

        int bins = std::clamp(static_cast<int>(std::sqrt(v.size())), 4, 100);
        std::vector<double> counts(bins, 1e-6);

        double span = max_v - min_v;
        double bin_size = (span < 1e-12) ? 1.0 : (span / bins);
        double inv_bin = (span < 1e-12) ? 0.0 : (1.0 / bin_size);

        for (double x : v) {
            int idx = (span < 1e-12) ? 0 : std::min(bins - 1, int((x - min_v) * inv_bin));
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

        DimRanker(Points& points, uint32_t min_distinct = 32)
            : points(points), min_distinct(min_distinct) {}

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

            applyDistinctGate(result, false);
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

            applyDistinctGate(result, true);

            return result;
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
        uint32_t min_distinct = 32;

 
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

                double sum = 0.0, min_v = vals[0], max_v = vals[0];
                for (double x : vals) 
                { 
                    sum += x; 
                    if (x < min_v) 
                        min_v = x; 
                    if (x > max_v) 
                        max_v = x; 
                }

                double m = sum / vals.size();
                double m2 = 0.0, m3 = 0.0, m4 = 0.0;
                for (double x : vals) 
                { 
                    double dd = x - m; 
                    double d2 = dd * dd; 
                    m2 += d2; 
                    m3 += d2 * dd; 
                    m4 += d2 * d2; 
                }

                double invn = 1.0 / vals.size();
                m2 *= invn; m3 *= invn; m4 *= invn;
                double var = m2;
                double stddev = safe_std(var);
                double sd3 = stddev * stddev * stddev;
                double sk = (stddev < 1e-12) ? 0.0 : m3 / sd3;
                double kurt = (stddev < 1e-12) ? 0.0 : (m4 / (sd3 * stddev)) - 3.0;

                HistogramStats h = computeHistogramStats(vals, min_v, max_v);
                double ent = h.entropy;
                double flatness = h.flatness;

                double range_norm = (max_v - min_v) / (stddev + 1e-9);
                if (!std::isfinite(range_norm) || std::abs(range_norm) > 1e6)
                    range_norm = (range_norm > 0) ? 1e6 : -1e6;

                features.push_back({var, ent, std::abs(sk), std::abs(kurt), flatness, range_norm});
                meansPerDim(j) = m;
            }
        }

        void computeFeatures(double sample_ratio) {
            size_t d = dim;
            size_t total_n = points.size();
            size_t n = static_cast<size_t>(total_n * sample_ratio);
            n = (n < 5000000) ? n : 5000000; 

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

                double sum = 0.0, min_v = vals[0], max_v = vals[0];
                for (double x : vals) 
                { 
                    sum += x; 
                    if (x < min_v) 
                        min_v = x; 
                    
                    if (x > max_v) 
                        max_v = x; 
                    
                }

                double m = sum / vals.size();
                double m2 = 0.0, m3 = 0.0, m4 = 0.0;
                
                for (double x : vals) 
                { 
                    double dd = x - m; 
                    double d2 = dd * dd; 
                    m2 += d2; 
                    m3 += d2 * dd; 
                    m4 += d2 * d2; 
                }
                
                double invn = 1.0 / vals.size();
                m2 *= invn; m3 *= invn; m4 *= invn;
                double var = m2;
                double stddev = safe_std(var);
                double sd3 = stddev * stddev * stddev;
                double sk = (stddev < 1e-12) ? 0.0 : m3 / sd3;
                double kurt = (stddev < 1e-12) ? 0.0 : (m4 / (sd3 * stddev)) - 3.0;

                HistogramStats h = computeHistogramStats(vals, min_v, max_v);
                double ent = h.entropy;
                double flatness = h.flatness;

                double range_norm = (max_v - min_v) / (stddev + 1e-9);
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

        uint32_t countDistinctDim(size_t j, bool sampled) const {
            std::unordered_set<double> seen;
            if (sampled && !corrSampleIdx.empty()) {
                seen.reserve(corrSampleIdx.size());
                for (size_t idx : corrSampleIdx)
                    seen.insert(static_cast<double>(points[idx][j]));
            } else {
                const size_t n = points.size();
                seen.reserve(n < 1000000 ? n : 1000000);
                for (size_t i = 0; i < n; ++i)
                    seen.insert(static_cast<double>(points[i][j]));
            }
            return static_cast<uint32_t>(seen.size());
        }

        void applyDistinctGate(std::vector<RankedDimension>& result, bool sampled) const {
            for (auto& r : result) {
                r.distinct = countDistinctDim(r.dimension, sampled);
                r.gated = (r.distinct < min_distinct);
            }
            std::stable_partition(result.begin(), result.end(),
                [](const RankedDimension& r) { return !r.gated; });
        }
    };
}