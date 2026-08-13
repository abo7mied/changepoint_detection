r"""
\[
d_u=\left\lceil\frac d2\right\rceil,\qquad
d_v=\left\lfloor\frac d2\right\rfloor,\qquad
X_t=(U_t,V_t)\in\mathbb R^d.
\]

\[
U_t\overset{\mathrm{iid}}{\sim}
\operatorname{Uniform}([-1,1]^{d_u}),
\qquad
q(u)=u^2-\frac13.
\]

\[
s_t=
\begin{cases}
1, & t<\tau^\star,\\
-1, & t\geq\tau^\star.
\end{cases}
\]

\[
V_{t,j}
=
a\,s_t\,q(U_{t-1,j})
+\varepsilon_{t,j},
\qquad
\varepsilon_{t,j}
\overset{\mathrm{iid}}{\sim}
\mathcal N(0,\sigma^2),
\qquad
j=1,\ldots,d_v.
\]

\[
V_{0,j}=\varepsilon_{0,j}.
\]

\[
\mathbb E[U_{t,j}q(U_{t,j})]=0,
\qquad
\operatorname{Cov}(X_t,X_{t-1})=0.
\]

\[
\mathbb E[X_t\mid U_{t-1}]
=
\left(
0,\,
a\,s_t\,q(U_{t-1})
\right).
\]
"""

import numpy as np


def sign_flip_quadratic_dataset(
    num_variables,
    num_timepoints,
    change_point,
    coefficient,
    noise_std_dev,
    rng,
):
    """Generate a sign-flipped quadratic dataset."""
    if num_variables < 2:
        raise ValueError(
            "num_variables must be at least 2. "
            f"Got {num_variables}."
        )

    if num_timepoints < 2:
        raise ValueError(
            "num_timepoints must be at least 2. "
            f"Got {num_timepoints}."
        )

    if not 0 < change_point < num_timepoints:
        raise ValueError(
            "change_point must satisfy "
            "0 < change_point < num_timepoints. "
            f"Got change_point={change_point} "
            f"and num_timepoints={num_timepoints}."
        )

    if not np.isfinite(coefficient):
        raise ValueError(
            "coefficient must be finite. "
            f"Got {coefficient}."
        )

    if (
        not np.isfinite(noise_std_dev)
        or noise_std_dev < 0
    ):
        raise ValueError(
            "noise_std_dev must be finite and "
            "nonnegative. "
            f"Got {noise_std_dev}."
        )

    if not isinstance(
        rng,
        np.random.Generator,
    ):
        raise TypeError(
            "rng must be a "
            "numpy.random.Generator."
        )

    num_response_variables = (
        num_variables // 2
    )

    num_driver_variables = (
        num_variables
        - num_response_variables
    )

    drivers = rng.uniform(
        low=-1.0,
        high=1.0,
        size=(
            num_timepoints,
            num_driver_variables,
        ),
    )

    responses = rng.normal(
        loc=0.0,
        scale=noise_std_dev,
        size=(
            num_timepoints,
            num_response_variables,
        ),
    )

    quadratic_signal = (
        drivers[:-1, :num_response_variables]
        ** 2
        - 1.0 / 3.0
    )

    regime_signs = np.ones(
        num_timepoints - 1,
        dtype=float,
    )

    regime_signs[
        max(change_point - 1, 0):
    ] = -1.0

    responses[1:] += (
        coefficient
        * regime_signs[:, None]
        * quadratic_signal
    )

    time_series = np.concatenate(
        [
            drivers,
            responses,
        ],
        axis=1,
    )

    return time_series