"""Checks that decide whether the learned weights mean what they look like.

The gravity model's per-domain coefficients are the headline output, so the
question "are these individually identified, or is the design too collinear
to read them one at a time" is not optional. Two occupations similar in
skills are usually similar in abilities too, and with correlated predictors a
coefficient can flip sign without the underlying relationship reversing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pairs import similarity_columns

# Conventional thresholds. VIF above ~5 means a coefficient is substantially
# explained by the other predictors and shouldn't be read in isolation.
VIF_CAUTION = 5.0


def collinearity(pairs: pd.DataFrame) -> pd.DataFrame:
    """Variance inflation factor per O*NET domain similarity.

    VIF_d = 1 / (1 - R^2_d), where R^2_d is from regressing domain d's
    similarity on all the others. 1.0 means orthogonal; large means redundant.
    """
    cols = similarity_columns(pairs)
    corr = pairs[cols].corr().to_numpy()
    vif = np.diag(np.linalg.inv(corr))
    frame = pd.DataFrame(
        {
            "domain": [c.replace("sim__", "") for c in cols],
            "vif": vif,
            "max_corr_with_other": [
                max(abs(corr[i, j]) for j in range(len(cols)) if j != i)
                for i in range(len(cols))
            ],
        }
    )
    frame["individually_readable"] = frame["vif"] < VIF_CAUTION
    return frame.sort_values("vif", ascending=False).reset_index(drop=True)


def condition_number(pairs: pd.DataFrame) -> float:
    """Condition number of the similarity correlation matrix."""
    cols = similarity_columns(pairs)
    eigenvalues = np.linalg.eigvalsh(pairs[cols].corr().to_numpy())
    return float(eigenvalues.max() / eigenvalues.min())


def suppressed_domains(joint_weights: pd.DataFrame, marginal: dict) -> pd.DataFrame:
    """Domains whose joint coefficient contradicts their standalone one.

    A domain that predicts transitions positively on its own but takes a
    negative coefficient in the joint fit is *redundant*, not repellent --
    the other domains already carry its signal. Reporting the negative number
    without this distinction would say something false about the labour
    market.
    """
    joint = joint_weights.set_index("domain")["coefficient"]
    rows = []
    for domain, alone in marginal.items():
        together = joint.get(domain, np.nan)
        rows.append(
            {
                "domain": domain,
                "alone": alone,
                "in_joint_model": together,
                "sign_flipped": bool(np.sign(alone) != np.sign(together)),
            }
        )
    return pd.DataFrame(rows).sort_values("alone", ascending=False).reset_index(drop=True)
