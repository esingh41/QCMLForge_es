import psi4

mol = psi4.geometry("""
0 1 
S                    -0.879500001408    -2.083200002882    -0.553100000005
N                    -0.295899999723    -1.817700000929     1.031199999697
N                     0.544699999716    -0.720099998443     1.040099998366
C                     0.708899999822    -0.138000002480    -0.126900002476
C                     0.009299998430    -0.724900000876    -1.172200003037
H                     1.354100001861     0.729100001021    -0.198900001930
H                    -0.034099999535    -0.452300001829    -2.219600001023
units angstrom
""")

psi4.set_options(
    {
        "d_convergence": 8,
        "dft_radial_points": 99,
        "dft_spherical_points": 590,
        "e_convergence": 10,
        "guess": "sad",
        "mbis_d_convergence": 9,
        "mbis_maxiter": 1000,
        "mbis_radial_points": 99,
        "mbis_spherical_points": 590,
        "scf_properties": ["mbis_charges", "MBIS_VOLUME_RATIOS"],
        "scf_type": "df",
    }
)
psi4.energy("pbe0/sto-3g")
psi4.oeprop("MBIS_VOLUME_RATIOS")
