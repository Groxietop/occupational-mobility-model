import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


DOMAINS = {
    'ability':        'ability__',
    'skill':          'skill__',
    'knowledge':      'knowledge__',
    'work_activity':  'work_activity__',
    'work_style':     'work_style__',
    'work_value':     'work_value__',
    'interest':       'interest__',
    'work_context':   'work_context__',
}


def _domain_cols(df, prefix):
    return [c for c in df.columns if c.startswith(prefix)]


def _feature_matrix(df, cols):
    """Extract, impute with column means, and z-score normalize a feature matrix."""
    X = df[cols].copy().astype(float)
    X = X.fillna(X.mean())
    X = StandardScaler().fit_transform(X)
    return X


def compute_similarity_matrices(master: pd.DataFrame) -> dict:
    """
    Returns a dict of {domain: similarity_matrix} plus 'cumulative'.
    Each matrix is a DataFrame indexed/columned by soc_code.
    """
    soc_codes = master['soc_code'].values
    matrices = {}

    domain_sims = []
    for domain, prefix in DOMAINS.items():
        cols = _domain_cols(master, prefix)
        if not cols:
            continue
        X = _feature_matrix(master, cols)
        sim = cosine_similarity(X)
        matrices[domain] = pd.DataFrame(sim, index=soc_codes, columns=soc_codes)
        domain_sims.append(sim)

    # Cumulative = average across all domains
    cumulative = np.mean(domain_sims, axis=0)
    matrices['cumulative'] = pd.DataFrame(cumulative, index=soc_codes, columns=soc_codes)

    return matrices


def most_similar(
    soc_code: str,
    matrices: dict,
    master: pd.DataFrame,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Return top_n most similar occupations to soc_code, with per-domain and cumulative scores.
    """
    results = {}
    for domain, mat in matrices.items():
        if soc_code not in mat.index:
            continue
        results[domain] = mat.loc[soc_code]

    scores = pd.DataFrame(results)
    scores.index.name = 'soc_code'

    # Exclude self
    scores = scores[scores.index != soc_code]

    # Sort by cumulative descending
    scores = scores.sort_values('cumulative', ascending=False).head(top_n)

    # Join title
    title_map = master.set_index('soc_code')['title']
    scores.insert(0, 'title', scores.index.map(title_map))
    scores.insert(1, 'job_zone', scores.index.map(master.set_index('soc_code')['job_zone']))

    # Round for readability
    score_cols = [c for c in scores.columns if c not in ('title', 'job_zone')]
    scores[score_cols] = scores[score_cols].round(4)

    return scores
