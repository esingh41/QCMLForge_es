import psi4
import numpy as np

mol = psi4.geometry("""
0 1
C 0.04747 -0.925821 0.699421
C 0.310425 0.351818 -0.270987
O -0.620672 0.944778 -0.815336
N 1.600253 0.794383 -0.312761
C -1.361818 -1.28657 0.768556
H 0.507237 -0.647797 1.653275
H 1.782805 1.505333 -0.913962
H 2.320161 0.286132 0.065534
H 0.647904 -1.734136 0.242649
H -1.544467 -2.137687 1.440215
H -2.032038 -0.528731 1.14436
H -1.788329 -1.517923 -0.231068

  symmetry c1
  no_reorient
  no_com
""")

e, wfn = psi4.energy('hf/sto-3g', return_wfn=True)
psi4.oeprop(wfn, 'MBIS_CHARGES')
print(wfn.variables())

print(wfn.variable("MBIS RADIAL MOMENTS <R^3>").np)

charges = np.array(wfn.variable('MBIS CHARGES'))
dipoles = np.array(wfn.variable('MBIS DIPOLES'))
n_at = len(charges)
quadrupoles = np.reshape(np.array(wfn.variable('MBIS QUADRUPOLES')), (n_at, 9))
multipoles = np.concatenate([charges, dipoles, quadrupoles], axis=1)
