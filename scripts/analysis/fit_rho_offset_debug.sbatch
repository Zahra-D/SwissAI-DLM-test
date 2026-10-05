#!/usr/bin/env bash
#SBATCH --job-name=fit_rho_offset
#SBATCH --partition=debug
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=ab035
#SBATCH --output=logs/debug/fit_rho_offset_%j.out
#SBATCH --error=logs/debug/fit_rho_offset_%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"

srun --environment=uni-d2 --ntasks=1 \
  "${REPO_ROOT}/venv-docker/bin/python" - <<'PY'
import numpy as np
from scipy.optimize import least_squares

B = np.array([64, 128, 256, 512, 1024, 2048, 4096], dtype=np.float64)
x = B / B[0]
datasets = {
    "original_local_t": np.array([
        99.93999481201172, 100.22232055664062, 102.81481170654297,
        104.5498275756836, 102.60204315185547, 100.58169555664062,
        98.50177001953125,
    ]),
    "remeasured_global_t": np.array([
        100.28166198730469, 102.40760040283203, 103.20990753173828,
        103.95305633544922, 101.7281494140625, 102.46168518066406,
        98.85164642333984,
    ]),
}


def score(y, pred, k):
    residual = np.log(pred) - np.log(y)
    rss = float(residual @ residual)
    n = len(y)
    return rss, n * np.log(rss / n) + 2 * k


for name, y in datasets.items():
    # Pure power law: rho(B) = c * (B/64)^a.
    a_pure, log_c_pure = np.polyfit(np.log(x), np.log(y), 1)
    c_pure = float(np.exp(log_c_pure))
    pred_pure = c_pure * x ** a_pure
    rss_pure, aic_pure = score(y, pred_pure, 2)

    # Offset power law: rho(B) = c * (B/64)^a + b, c,b >= 0.
    # Multiple starts expose the weak identifiability when a is close to zero.
    best = None
    for frac_b in (0.01, 0.25, 0.5, 0.75, 0.95):
        for a0 in (-2.0, -0.5, -0.1, 0.1, 0.5, 2.0):
            b0 = float(y.mean() * frac_b)
            c0 = max(float(y.mean() - b0), 1e-8)

            def residual(theta):
                log_c, a, log_b = theta
                pred = np.exp(log_c) * x ** a + np.exp(log_b)
                return np.log(pred) - np.log(y)

            fit = least_squares(
                residual,
                np.array([np.log(c0), a0, np.log(b0)]),
                bounds=(np.array([-30.0, -5.0, -30.0]),
                        np.array([10.0, 5.0, 10.0])),
                max_nfev=100000,
            )
            if best is None or np.sum(fit.fun ** 2) < np.sum(best.fun ** 2):
                best = fit

    log_c, a_offset, log_b = best.x
    c_offset, b_offset = float(np.exp(log_c)), float(np.exp(log_b))
    pred_offset = c_offset * x ** a_offset + b_offset
    rss_offset, aic_offset = score(y, pred_offset, 3)

    # A constant is a useful null model because measured rho is almost flat.
    constant = float(np.exp(np.mean(np.log(y))))
    pred_constant = np.full_like(y, constant)
    rss_constant, aic_constant = score(y, pred_constant, 1)

    print(name)
    print("  constant: rho = %.8f; logRSS=%.10g; AIC_log=%.6f" %
          (constant, rss_constant, aic_constant))
    print("  pure:     rho = %.8f * (B/64)^(%.8f); logRSS=%.10g; AIC_log=%.6f" %
          (c_pure, a_pure, rss_pure, aic_pure))
    print("  offset:   rho = %.8f * (B/64)^(%.8f) + %.8f; logRSS=%.10g; AIC_log=%.6f" %
          (c_offset, a_offset, b_offset, rss_offset, aic_offset))
    print("  delta_AIC_offset_minus_pure=%.6f" % (aic_offset - aic_pure))
    print("  observed_range=[%.8f, %.8f]" % (y.min(), y.max()))
PY
