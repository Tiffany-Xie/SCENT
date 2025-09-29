import gin
import abc
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from typing import List

from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionState,
    ReactionStateEarlyTerminal,
)
from rgfn.shared.proxies.cached_proxy import CachedProxyBase


@gin.configurable()
class TanimotoSimilarityProxy(CachedProxyBase[ReactionState]):
    def __init__(self, reference_mol: str, radius: int=2, n_bits: int=2048):
        super().__init__()
    
        self.cache = {ReactionStateEarlyTerminal(None): 0.0}
        self.radius = radius
        self.n_bits = n_bits

        ref = Chem.MolFromSmiles(reference_mol)
        if ref == None:
            raise ValueError(f"Something wrong with the reference mol (SMILES): {reference_mol}")
        self.ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref, radius, nBits = n_bits)
    
    @property
    def is_non_negative(self) -> bool:
        return True
    
    @property
    def higher_is_better(self) -> bool:
        return True
    
    def _compute_proxy_output(self, states: List[ReactionState]) -> list[float]: 
        scores = []
        for s in states:
            mol = getattr(s.molecule, "rdkit_mol", None) if getattr(s, "molecule", None) else None
            if mol == None:
                scores.append(0.0)
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
            score = DataStructs.TanimotoSimilarity(self.ref_fp, fp)
            scores.append(score)
        return scores
        



