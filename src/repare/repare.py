from collections import deque
import itertools

import dcor
import networkx as nx
import numpy as np
from scipy.stats import chi2, chi2_contingency, ks_2samp

from .utils import SimpleCanCorr


class PartitionDagModelOracle(object):
    """A probabilistic graphical model in the form of a directed acyclic graph over parts of a partition."""

    def __init__(self, rng=np.random.default_rng(0)) -> None:
        self.dag = nx.DiGraph()
        self.rng = rng

    def fit(self, order, adj) -> object:
        self.order = order
        self.adj = adj
        # always use lexicographical order for node partitions
        init_partition = [set(order)]  # trivial coarsening
        self.dag.add_node(tuple(sorted(order)))
        self.refinable = deque(init_partition)
        while len(self.refinable) > 0:
            self._recurse()
        return self

    def _refine(self):
        to_refine = self.refinable.popleft()
        u_len = self.rng.choice(range(1, len(to_refine)))  # rest go to v
        u = []
        for el in self.order:
            if el in to_refine:
                u.append(el)
            if len(u) == u_len:
                break
        u = set(u)
        v = set((el for el in to_refine if el not in u))
        if len(u) > 1:
            self.refinable.append(u)
        if len(v) > 1:
            self.refinable.append(v)
        return to_refine, u, v

    def _is_adj(self, pa, ch):
        for el_pa in pa:
            for el_ch in ch:
                if el_ch in self.adj[el_pa]:
                    return True
        return False

    def _recurse(self):
        to_refine, u, v = self._refine()
        if not u or not v:
            return
        self.dag.add_node(tuple(u))
        self.dag.add_node(tuple(v))
        for pa in self.dag.predecessors(tuple(to_refine)):
            for ch in (u, v):
                if self._is_adj(set(pa), ch):
                    self.dag.add_edge(pa, tuple(ch))
        for pa in (u, v):
            for ch in self.dag.successors(tuple(to_refine)):
                if self._is_adj(pa, set(ch)):
                    self.dag.add_edge(tuple(pa), ch)
        if self._is_adj(u, v):
            self.dag.add_edge(tuple(u), tuple(v))
        self.dag.remove_node(tuple(to_refine))


class PartitionDagModelIvn(PartitionDagModelOracle):
    def __init__(self, rng=np.random.default_rng(0)) -> None:
        super().__init__(rng)
        self.score = 0.0
        self.fit_metadata = {}

    def fit(self, data_dict, alpha=0.05, beta=0.05, assume=None, refine_test="ks"):
        self.assume = assume
        self.refine_test = (refine_test or "ks").lower()
        data_dict = data_dict.copy()
        self.beta = beta
        obs = data_dict.pop("obs")

        def _split_env(env):
            env_type = None
            if isinstance(env, tuple):
                if len(env) == 2:
                    data, targets = env
                elif len(env) == 3:
                    data, targets, env_type = env
                else:
                    raise ValueError(
                        "Environment tuple must be (data, targets) or (data, targets, type)"
                    )
            elif isinstance(env, dict):
                if "data" not in env:
                    raise ValueError("Environment dict must contain a 'data' key")
                data = env["data"]
                targets = env.get("targets", ())
                env_type = env.get("type", env.get("intervention_type"))
            else:
                data = env
                targets = ()
            data_array = np.asarray(data)
            target_set = set(targets)
            if env_type is None:
                env_type = "obs" if len(target_set) == 0 else "hard"
            return data_array, target_set, env_type

        obs_array, obs_targets, obs_type = _split_env(obs)
        if obs_targets:
            raise ValueError("Observational dataset cannot have intervention targets")

        processed_envs = {}
        env_targets = {}
        env_types = {}
        for key, env in data_dict.items():
            env_array, targets, env_type = _split_env(env)
            processed_envs[key] = env_array
            env_targets[key] = targets
            env_types[key] = env_type

        self.obs = obs_array
        self.obs_type = obs_type
        self.data_dict = processed_envs
        self.env_targets = env_targets
        self.env_types = env_types
        obs_data = self.obs

        if self.assume is None:

            def p_val(post_ivn):
                baseline = np.asarray(obs_data)
                comparison = np.asarray(post_ivn)
                if baseline.ndim == 1:
                    baseline = baseline[:, None]
                if comparison.ndim == 1:
                    comparison = comparison[:, None]
                num_feats = baseline.shape[1]
                pvals = np.empty(num_feats)
                test = self.refine_test
                for j in range(num_feats):
                    x = baseline[:, j]
                    y = comparison[:, j]
                    if test == "ks":
                        for j in range(num_feats):
                            stat = ks_2samp(
                                baseline[:, j],
                                comparison[:, j],
                                alternative="two-sided",
                                mode="auto",
                            )
                            pvals[j] = stat.pvalue
                    elif test == "energy":
                        result = dcor.homogeneity.energy_test(
                            x[:, None], y[:, None], num_resamples=199
                        )
                        pvals[j] = result.pvalue
                    else:
                        raise ValueError(f"Unknown refine_test '{self.refine_test}'")
                return pvals

        # Gaussian LRT: vectorized over columns of post_ivn
        elif self.assume == "gaussian":

            def p_val(post_ivn):
                num_feats = obs_data.shape[1]
                pvals = np.empty(num_feats)
                for j in range(num_feats):
                    x = obs_data[:, j]
                    y = post_ivn[:, j]

                    def ll(data):
                        m = data.size
                        mu = data.mean()
                        s2 = float(np.mean((data - mu) ** 2))
                        s2 = max(s2, 1e-12)
                        return -0.5 * m * (np.log(2 * np.pi * s2) + 1)

                    ll_sep = ll(x) + ll(y)
                    ll_null = ll(np.concatenate([x, y]))
                    LR = 2.0 * (ll_sep - ll_null)
                    pvals[j] = chi2.sf(LR, df=2)
                return pvals

        # Discrete chi2: vectorized over columns of post_ivn
        elif self.assume == "discrete":

            def p_val(post_ivn):
                num_feats = obs_data.shape[1]
                pvals = np.empty(num_feats)
                for j in range(num_feats):
                    x = obs_data[:, j]
                    y = post_ivn[:, j]
                    vals = np.concatenate([x, y])
                    uniq, inv = np.unique(vals, return_inverse=True)
                    inv_x = inv[: x.size]
                    inv_y = inv[x.size :]
                    V = uniq.size
                    table = np.zeros((2, V), dtype=int)
                    for idx in inv_x:
                        table[0, idx] += 1
                    for idx in inv_y:
                        table[1, idx] += 1
                    _, pval, _, _ = chi2_contingency(table, correction=False)
                    pvals[j] = float(pval)
                return pvals
        else:
            raise ValueError(f"Unsupported assumption '{self.assume}'")

        partition_masks = {}
        for idx, post_ivn in self.data_dict.items():
            pvals = np.asarray(p_val(post_ivn), dtype=float)
            partition_masks[idx] = pvals < alpha
        self.partition = _get_totally_ordered_partition(partition_masks)
        init_partition = [set(range(self.obs.shape[1]))]
        self.dag.add_node(tuple(range(self.obs.shape[1])))
        self.refinable = deque(init_partition)
        self.edge_tests = []
        self.edge_score = 0.0
        while len(self.refinable) > 0:
            self._recurse()
        self.score = float("nan")
        self.fit_metadata = dict(
            alpha=alpha,
            beta=beta,
            score=self.score,
            num_parts=self.dag.number_of_nodes(),
            num_edges=self.dag.number_of_edges(),
        )
        return self

    def _refine(self):
        to_refine = self.refinable.pop()
        u = self.partition.popleft()
        v = to_refine - u
        if self.partition and v != self.partition[-1]:
            self.refinable.append(v)
        return to_refine, u, v

    def _recurse(self,):
        to_refine, u, v = self._refine()
        if not u or not v:
            return
        self.dag.add_node(tuple(u))
        self.dag.add_node(tuple(v))
        if self._is_adj(u, v, self.dag.predecessors(tuple(to_refine))):
            self.dag.add_edge(tuple(u), tuple(v))
        for pa in self.dag.predecessors(tuple(to_refine)):
            for ch in (u, v):
                conditioning_set = (c for c in self.dag.predecessors(tuple(to_refine)) if c != pa)
                if ch == v:
                    conditioning_set = itertools.chain(conditioning_set, (u,))
                if self._is_adj(set(pa), ch, conditioning_set):
                    self.dag.add_edge(pa, tuple(ch))
        for pa in (u, v):
            for ch in self.dag.successors(tuple(to_refine)):
                conditioning_set = (c for c in self.dag.predecessors(tuple(to_refine)) if c != pa)
                if self._is_adj(pa, set(ch), conditioning_set):
                    self.dag.add_edge(tuple(pa), ch)
        self.dag.remove_node(tuple(to_refine))

    def _is_adj(self, pa, ch, conditioning_set=None):
        # Flatten conditioning set indices
        z_indices = []
        if conditioning_set:
            for part in conditioning_set:
                z_indices.extend(part)
        z_indices = sorted(list(set(z_indices)))

        if self.assume is None:

            def p_val(x, y, z=None):
                test_result = dcor.independence.distance_correlation_t_test(x, y)
                return test_result.pvalue

        elif self.assume == "gaussian":

            def p_val(x, y, z=None):
                try:
                    cca = SimpleCanCorr(x, y, Z=z)
                    result = cca.wilks_lambda_test()
                    return result.loc[0, "Pr > F"]
                except (ValueError, np.linalg.LinAlgError):
                    return 1.0

        elif self.assume == "discrete":
            """For paired samples X (n×p) and Y (n×q) where both are
            discrete (categorical), give a concise Python function
            that tests independence parametrically: encode
            multivariate categories as joint labels, build the
            contingency table, run scipy.stats.chi2_contingency
            (Pearson chi-square) to return the asymptotic p-value, and
            handle sparse tables by warning when expected counts < 5
            (or fall back to Fisher/exact/permutation). Include
            imports and a minimal example ."""
            p_val = 1
        pa_indices = sorted(pa)
        ch_indices = sorted(ch)

        datasets = [(self.obs, set(), getattr(self, "obs_type", "obs"))] + [
            (
                env,
                self.env_targets.get(key, set()),
                self.env_types.get(key, "hard"),
            )
            for key, env in self.data_dict.items()
        ]

        p_values = []
        for env, targets, env_type in datasets:
            if env_type in {"hard", "do"} and (
                targets.intersection(pa) or targets.intersection(ch)
            ):
                continue
            x = env[:, pa_indices]
            y = env[:, ch_indices]
            if x.ndim == 1:
                x = x[:, None]
            if y.ndim == 1:
                y = y[:, None]
                
            z = None
            if z_indices:
                z = env[:, z_indices]
            
            p_values.append(p_val(x, y, z=z))

        if not p_values:
            self.edge_tests.append(
                {
                    "pa": tuple(pa_indices),
                    "ch": tuple(ch_indices),
                    "p_value": None,
                    "decision": False,
                    "reason": "insufficient_data",
                }
            )
            return False

        test_p = float(max(p_values))
        return test_p <= self.beta

    def expand_coarsened_dag(self, fully_connected=False):
        """Expand the coarse partition DAG into a full adjacency matrix.

        Parameters
        ----------
        fully_connected : bool, default=False
            When True, connect every part internally according to the
            part ordering (i.e., each part becomes a fully connected DAG).
        """
        if not hasattr(self, "obs"):
            raise ValueError("Observational data must be provided before expansion.")
        num_features = self.obs.shape[1]
        adjacency = np.zeros((num_features, num_features), dtype=int)

        if fully_connected:
            for part in self.dag.nodes:
                ordered = sorted(part)
                for idx, src in enumerate(ordered):
                    for dst in ordered[idx + 1 :]:
                        adjacency[src, dst] = 1

        for src_part, dst_part in self.dag.edges:
            for src in src_part:
                for dst in dst_part:
                    adjacency[src, dst] = 1

        return adjacency

    def chain_gaussian_score(self, datasets):
        """Negative log-likelihood under the Gaussian AMP chain-graph model.

        Parameters
        ----------
        datasets : iterable of array-like
            Each element is (n_samples, num_features). All datasets must share the
            same feature dimension/order used during fitting.

        Returns
        -------
        float
            Negative log-likelihood summed over datasets and parts (lower is better).
        """

        if not hasattr(self, "obs"):
            raise ValueError("Model must be fit before scoring.")
        nodes = list(nx.topological_sort(self.dag))
        if not nodes:
            return 0.0

        loglik = 0.0
        for data in datasets:
            arr = np.asarray(data, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != self.obs.shape[1]:
                raise ValueError("Dataset feature dimension mismatch.")
            if arr.shape[0] == 0:
                continue
            centered = arr - arr.mean(axis=0, keepdims=True)
            cov = centered.T @ centered / float(centered.shape[0])
            n = centered.shape[0]

            for part in nodes:
                part_idx = sorted(part)
                parents = list(self.dag.predecessors(part))
                parent_atoms = sorted({atom for pa in parents for atom in pa})

                Scc = cov[np.ix_(part_idx, part_idx)]
                if parent_atoms:
                    Spp = cov[np.ix_(parent_atoms, parent_atoms)]
                    Scp = cov[np.ix_(part_idx, parent_atoms)]
                    Spc = Scp.T
                    inv_Spp = np.linalg.pinv(Spp)
                    beta = Scp @ inv_Spp
                    residual = (
                        Scc
                        - beta @ Spc
                        - Scp @ beta.T
                        + beta @ Spp @ beta.T
                    )
                else:
                    residual = Scc.copy()

                residual = 0.5 * (residual + residual.T)
                omega = np.linalg.pinv(residual)
                sign, invlogdet = np.linalg.slogdet(omega)
                if sign <= 0:
                    raise np.linalg.LinAlgError("Residual covariance not positive definite.")
                
                block_dim = len(part_idx)
                loglik += 0.5 * n * (invlogdet - block_dim)

        # Return negative log-likelihood so lower is better (for existing selection logic).
        return float(-loglik)


def _get_totally_ordered_partition(ivn_biadj):
    num_atoms = len(next(iter(ivn_biadj.values())))
    partition = [list(range(num_atoms))]
    while len(ivn_biadj) > 0:
        idx, ivned = ivn_biadj.popitem()
        new_partition = []
        for part in partition:
            first_part = []
            second_part = []
            for atom in part:
                if not ivned[atom]:
                    first_part.append(atom)
                else:
                    second_part.append(atom)
            if len(first_part) > 0 and len(second_part) > 0:
                new_partition.append(first_part)
                new_partition.append(second_part)
            else:
                new_partition.append(part)
        partition = new_partition
    return deque((set(part) for part in partition))


# from collections import OrderedDict, deque

# def get_totally_ordered_partition(ivn_biadj):
#     # ivn_biadj: dict -> iterable of bool arrays (same length)
#     n = len(next(iter(ivn_biadj.values())))
#     pattern_to_nodes = OrderedDict()
#     for node in range(n):
#         # build pattern as tuple of booleans in a stable order of keys
#         pattern = tuple(ivn_biadj[k][node] for k in ivn_biadj)
#         pattern_to_nodes.setdefault(pattern, []).append(node)
#     return deque(set(nodes) for nodes in pattern_to_nodes.values())
