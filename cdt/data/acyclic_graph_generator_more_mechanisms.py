import random
from typing import Any, Callable, Dict, Optional, Sequence, Union

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

from .acyclic_graph_generator import AcyclicGraphGenerator
from .causal_mechanisms import (
    Binary_Mechanism,
    GaussianProcessAdd_Mechanism,
    GaussianProcessMix_Mechanism,
    LinearMechanism,
    NN_Mechanism,
    Polynomial_Mechanism,
    SigmoidAM_Mechanism,
    SigmoidMix_Mechanism,
    bernoulli_cause,
    gaussian_cause,
    gmm_cause,
)

MECH_MAP = {
    "linear": LinearMechanism,
    "polynomial": Polynomial_Mechanism,
    "sigmoid_add": SigmoidAM_Mechanism,
    "sigmoid_mix": SigmoidMix_Mechanism,
    "gp_add": GaussianProcessAdd_Mechanism,
    "gp_mix": GaussianProcessMix_Mechanism,
    "nn": NN_Mechanism,
    "binary": Binary_Mechanism,
}
ROOT_GEN_MAP = {
    "gmm": gmm_cause,
    "gaussian": gaussian_cause,
    "bernoulli": bernoulli_cause,
}


class AcyclicGraphGeneratorMoreMechanisms(AcyclicGraphGenerator):
    def __init__(
        self,
        mechanism_pool: Optional[Sequence[str]] = None,
        mechanism_prob: Optional[Sequence[float]] = None,
        force_binary_root: bool = True,
        binary_root_p: float = 0.5,
        root_pool: Optional[Sequence] = None,
        root_prob: Optional[Sequence[float]] = None,
        rescale_binary: bool = False,
        random_state: Optional[int] = None,
        noise: Union[str, Callable] = "gaussian",
        noise_coeff: float = 0.4,
        initial_variable_generator=gmm_cause,
        npoints: int = 500,
        nodes: int = 20,
        parents_max: int = 5,
        expected_degree: int = 3,
        dag_type: str = "default",
    ):
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        self.mech_pool = list(mechanism_pool) if mechanism_pool else ["linear"]
        self.mech_prob = (
            np.array(mechanism_prob)
            if mechanism_prob is not None
            else np.full(len(self.mech_pool), 1 / len(self.mech_pool))
        )
        self.force_bin_root = force_binary_root
        self.bin_root_p = binary_root_p
        self.root_pool = root_pool or ["gmm"]
        self.root_prob = (
            np.array(root_prob)
            if root_prob is not None
            else np.full(len(self.root_pool), 1 / len(self.root_pool))
        )
        self.rescale_bin = rescale_binary
        self.chosen_binary_root = None

        super().__init__(
            causal_mechanism="linear",  # Placeholder
            noise=noise,
            noise_coeff=noise_coeff,
            initial_variable_generator=initial_variable_generator,  # Placeholder?
            npoints=npoints,
            nodes=nodes,
            parents_max=parents_max,
            expected_degree=expected_degree,
            dag_type=dag_type,
        )

    def _choose_binary_root(self) -> int:
        roots = np.where(~self.adjacency_matrix.any(axis=0))[0]
        bin_root = np.random.choice(roots)
        self.chosen_binary_root = f"V{bin_root}"
        return bin_root

    def init_variables(self, verbose=False):
        self.init_dag(verbose)
        N = self.nodes
        self.mechanism_assign = [None] * N
        self.variable_types = ["cont"] * N
        self.cfunctions = [None] * N
        bin_root = self._choose_binary_root() if self.force_bin_root else None
        for i in range(N):
            n_par = int(self.adjacency_matrix[:, i].sum())
            if n_par == 0:
                if i == bin_root:
                    root_name = "bernoulli"
                    gen = lambda pts, p=self.bin_root_p: bernoulli_cause(pts, p)
                    self.variable_types[i] = "binary"
                else:
                    root_name = np.random.choice(self.root_pool, p=self.root_prob)
                    gen = ROOT_GEN_MAP[root_name]
                self.cfunctions[i] = gen
                self.mechanism_assign[i] = gen.__name__ if callable(gen) else str(gen)
                if root_name == "bernoulli":
                    self.variable_types[i] = "binary"

            else:
                mech_name = np.random.choice(self.mech_pool, p=self.mech_prob)
                mech_cls = MECH_MAP[mech_name]
                self.cfunctions[i] = mech_cls(
                    n_par,
                    self.npoints,
                    self.noise,
                    noise_coeff=self.noise_coeff,
                )
                self.mechanism_assign[i] = mech_name
                if mech_name == "threshold_binary":
                    self.variable_types[i] = "binary"

    def generate(
        self, rescale=True, return_metadata=False
    ) -> tuple[pd.DataFrame, nx.DiGraph, dict]:
        if self.cfunctions is None:
            self.init_variables()
        self.data = pd.DataFrame(
            index=range(self.npoints),
            columns=[f"V{i}" for i in range(self.nodes)],
        )
        for i in nx.topological_sort(nx.DiGraph(self.adjacency_matrix)):
            parents = np.where(self.adjacency_matrix[:, i])[0]
            if len(parents) == 0:
                col = self.cfunctions[i](self.npoints)
            else:
                X = self.data.iloc[:, list(parents)].values
                if X.ndim == 1:
                    X = X[:, None]
                col = self.cfunctions[i](X)
            col = np.squeeze(col)
            if rescale and not (
                self.variable_types[i] == "binary" and not self.rescale_bin
            ):
                col = scale(col)
            self.data[f"V{i}"] = col
        graph = nx.relabel_nodes(
            nx.DiGraph(self.adjacency_matrix),
            {i: f"V{i}" for i in range(self.nodes)},
            copy=True,
        )
        if return_metadata:
            meta = dict(
                adjacency_matrix=self.adjacency_matrix.copy(),
                mechanisms=self.mechanism_assign,
                variable_types=self.variable_types,
                binary_root=self.chosen_binary_root,
            )
            return self.data, graph, meta
        return self.data, graph
