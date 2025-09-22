import numpy as np
import matplotlib.pyplot as plt
import sys


def eigen_to_G(evals, evecs, efermi, energy):
    return (
        np.einsum("ij, j-> ij", evecs, 1.0 / (-evals + (energy + efermi)))
        @ evecs.conj().T
    )

class GreenFunction:

    def __init__(self): 
        self.irvec = None
        self.ndegen = None
        self.HmnR_np_iR = None
        self.Ham_bulk_k = None


    def eigen_to_G(evals, evecs, efermi, energy):
        return (
        np.einsum("ij, j-> ij", evecs, 1.0 / (-evals + (energy + efermi)))
        @ evecs.conj().T
    )

def get_Gk(self, ik, energy):
    """Green's function G(k) for one energy
    G(\epsilon)= (\epsilon I- H)^{-1}
    :param ik: indices for kpoint
    :returns: Gk
    :rtype:  a matrix of indices (nbasis, nbasis)
    """
    Gk = eigen_to_G(
        evals=self.get_evalue(ik),
        evecs=self.get_evecs(ik),
        efermi=self.efermi,
        energy=energy,
    )
    # A slower version. For test.
    # Gk = np.linalg.inv((energy+self.efermi)*self.S[ik,:,:] - self.H[ik,:,:])
    return Gk

def get_GR(self, Rpts, energy, get_rho=False):
    """calculate real space Green's function for one energy, all R points.
    G(R, epsilon) = G(k, epsilon) exp(-2\pi i R.dot. k)
    :param Rpts: R points
    :param energy:
    :returns:  real space green's function for one energy for a list of R.
    :rtype:  dictionary, the keys are tuple of R, values are matrices of nbasis*nbasis
    """
    Rpts = [tuple(R) for R in Rpts]
    GR = defaultdict(lambda: 0.0j)
    rhoR = defaultdict(lambda: 0.0j)
    for ik, kpt in enumerate(self.kpts):
        Gk = self.get_Gk(ik, energy)
        if get_rho:
            if self.is_orthogonal:
                rhok = Gk
            else:
                rhok = self.get_Sk(ik) @ Gk
        for iR, R in enumerate(Rpts):
            phase = np.exp(self.k2Rfactor * np.dot(R, kpt))
            tmp = Gk * (phase * self.kweights[ik])
            GR[R] += tmp
            # change this if need full rho
            if get_rho and R == (0, 0, 0):
                rhoR[R] += rhok * (phase * self.kweights[ik])
    if get_rho:
        return GR, rhoR
    else:
        return GR



    def fermi(e, mu, width=0.01):
        x = (e - mu) / width
        return np.where(x < np.log(sys.float_info.max), 1 / (1.0 + np.exp(x)), 0.0)

    def get_density_matrix(self):
        rho = np.zeros((self.nbasis, self.nbasis), dtype=complex)
        if self.is_orthogonal:
            for ik, _ in enumerate(self.kpts):
                rho += (
                    (self.get_evecs(ik) * fermi(self.evals[ik], self.efermi))
                    @ self.get_evecs(ik).T.conj()
                    * self.kweights[ik]
                )
        else:
            for ik, _ in enumerate(self.kpts):
                rho += (
                    (self.get_evecs(ik) * fermi(self.evals[ik], self.efermi))
                    @ self.get_evecs(ik).T.conj()
                    @ self.get_Sk(ik)
                    * self.kweights[ik]
                )
        return rho
    