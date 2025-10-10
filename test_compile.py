import torch
from src.apnet_pt.AtomPairwiseModels.apnet2_fused import APNet2_AM_MPNN
from src.apnet_pt.AtomPairwiseModels.mtp_mtp import DimerProp, AtomTypeParamNN
from src.apnet_pt.AtomModels.ap2_atom_model import AtomMPNN
from src.apnet_pt.AtomModels.ap3_atom_model import AtomHirshfeldMPNN
from torch_geometric.data import Data

print("Testing torch.compile compatibility...")

print("\n1. Creating APNet2 model...")
am = AtomMPNN(n_message=2, n_rbf=8, n_neuron=64, n_embed=8, r_cut=5.0)
model = APNet2_AM_MPNN(
    n_message=2,
    n_rbf=8,
    n_neuron=64,
    n_embed=8,
    r_cut=5.0,
    r_cut_im=8.0,
    atom_model=am
)

print("2. Compiling APNet2 model with torch.compile()...")
try:
    compiled_model = torch.compile(model)
    print("✓ APNet2 model compiled successfully!")
except Exception as e:
    print(f"✗ Failed to compile APNet2: {e}")

print("\n3. Creating DimerProp model with AtomTypeParamNN...")
am_for_atp = AtomMPNN(n_message=2, n_rbf=8, n_neuron=64, n_embed=8, r_cut=5.0)
atp = AtomTypeParamNN(
    atom_model=am_for_atp,
    n_message=2,
    n_neuron=64,
    n_embed=8,
)
dp_model = DimerProp(
    ATParam=atp,
    dimer_eval="elst_damping"
)

print("4. Compiling DimerProp model with torch.compile()...")
try:
    compiled_dp = torch.compile(dp_model)
    print("✓ DimerProp model compiled successfully!")
except Exception as e:
    print(f"✗ Failed to compile DimerProp: {e}")

print("\n✓ All models are torch.compile compatible!")
