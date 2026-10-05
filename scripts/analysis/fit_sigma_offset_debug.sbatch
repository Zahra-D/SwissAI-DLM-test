#!/usr/bin/env bash
#SBATCH --job-name=fit_sigma_offset
#SBATCH --partition=debug
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=ab035
#SBATCH --output=logs/debug/fit_sigma_offset_%j.out
#SBATCH --error=logs/debug/fit_sigma_offset_%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"

srun --environment=uni-d2 --ntasks=1 \
  "${REPO_ROOT}/venv-docker/bin/python" - <<'PY'
import numpy as np
from scipy.optimize import least_squares

B = np.array([64, 128, 256, 512, 1024, 2048, 4096], dtype=np.float64)
datasets = {
    "original_local_t": np.array([
        8.282748749479651e-4,
        5.678054294548929e-4,
        3.8437204784713686e-4,
        2.4176618899218738e-4,
        1.5865560271777213e-4,
        9.68830063357018e-5,
        6.5342326706741e-5,
    ]),
    "remeasured_global_t": np.array([
        8.114480879157782e-4,
        5.602830788120627e-4,
        3.830034693237394e-4,
        2.374595933360979e-4,
        1.535890914965421e-4,
        1.0517425835132599e-4,
        6.970610411372036e-5,
    ]),
}


def metrics(y, pred, k):
    residual = y - pred
    rss = float(np.sum(residual ** 2))
    log_rss = float(np.sum((np.log(y) - np.log(pred)) ** 2))
    n = len(y)
    aic_log = n * np.log(log_rss / n) + 2 * k
    return rss, log_rss, aic_log


for name, y in datasets.items():
    x = np.log(B)
    a_pure, log_c_pure = np.polyfit(x, np.log(y), 1)
    c_pure = np.exp(log_c_pure)
    pred_pure = c_pure * B ** a_pure
    pure_metrics = metrics(y, pred_pure, 2)

    # Fit relative/log residuals. Parameterize c and b in log space so both
    # remain positive, and require b to stay below the smallest observation.
    b_upper = float(y.min() * (1.0 - 1e-9))
    initial_b = float(y.min() * 0.25)
    initial = np.array([np.log(c_pure), a_pure, np.log(initial_b)])
    lower = np.array([-50.0, -5.0, np.log(y.min() * 1e-8)])
    upper = np.array([50.0, 0.0, np.log(b_upper)])

    def residual(theta):
        log_c, a, log_b = theta
        pred = np.exp(log_c) * B ** a + np.exp(log_b)
        return np.log(pred) - np.log(y)

    result = least_squares(residual, initial, bounds=(lower, upper))
    log_c, a_offset, log_b = result.x
    c_offset = float(np.exp(log_c))
    b_offset = float(np.exp(log_b))
    pred_offset = c_offset * B ** a_offset + b_offset
    offset_metrics = metrics(y, pred_offset, 3)

    # Approximate local parameter uncertainty from the residual Jacobian. This
    # is diagnostic only; seven correlated measurements are not enough for a
    # reliable confidence interval.
    dof = len(y) - 3
    try:
        covariance = np.linalg.inv(result.jac.T @ result.jac)
        covariance *= np.sum(result.fun ** 2) / dof
        stderr = np.sqrt(np.diag(covariance))
        exponent_se = float(stderr[1])
    except np.linalg.LinAlgError:
        exponent_se = float("inf")

    print(name)
    print(
        "  pure:   sigma2 = %.8g * B^(%.6f); logRSS=%.8g; AIC_log=%.6f"
        % (c_pure, a_pure, pure_metrics[1], pure_metrics[2]))
    print(
        "  offset: sigma2 = %.8g * B^(%.6f) + %.8g; "
        "logRSS=%.8g; AIC_log=%.6f; exponent_local_SE=%.6f"
        % (c_offset, a_offset, b_offset, offset_metrics[1],
           offset_metrics[2], exponent_se))
    print("  delta_AIC_offset_minus_pure=%.6f" % (
        offset_metrics[2] - pure_metrics[2]))
PY
