import torch
from src.apnet_pt.AtomPairwiseModels.apnet2_fused import APNet2_AM_MPNN
from src.apnet_pt.AtomPairwiseModels.mtp_mtp import DimerProp, AtomTypeParamNN
from src.apnet_pt.AtomModels.ap2_atom_model import AtomMPNN
from torch_geometric.data import Data, Batch
import time

print("Testing torch.compile with actual inference...")

print("\n1. Creating test batch for APNet2...")
ZA = torch.tensor([1, 6, 8], dtype=torch.long)
RA = torch.randn(3, 3)
ZB = torch.tensor([1, 7], dtype=torch.long)
RB = torch.randn(2, 3)

e_AA_source = torch.tensor([0, 1, 2], dtype=torch.long)
e_AA_target = torch.tensor([1, 2, 0], dtype=torch.long)
e_BB_source = torch.tensor([0, 1], dtype=torch.long)
e_BB_target = torch.tensor([1, 0], dtype=torch.long)
e_ABsr_source = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
e_ABsr_target = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)

batch_atomic_A = Data(
    x=ZA,
    edge_index=torch.vstack((e_AA_source, e_AA_target)),
    R=RA,
    molecule_ind=torch.zeros(3, dtype=torch.long),
    total_charge=torch.zeros(1),
    natom_per_mol=torch.tensor([3], dtype=torch.long)
)

batch_atomic_B = Data(
    x=ZB,
    edge_index=torch.vstack((e_BB_source, e_BB_target)),
    R=RB,
    molecule_ind=torch.zeros(2, dtype=torch.long),
    total_charge=torch.zeros(1),
    natom_per_mol=torch.tensor([2], dtype=torch.long)
)

e_ABlr_source = torch.tensor([0, 1, 2], dtype=torch.long)
e_ABlr_target = torch.tensor([0, 1, 0], dtype=torch.long)
dimer_ind = torch.zeros(6, dtype=torch.long)
dimer_ind_lr = torch.zeros(3, dtype=torch.long)

class TestBatch:
    def __init__(self):
        self.ZA = ZA
        self.RA = RA
        self.ZB = ZB
        self.RB = RB
        self.e_AA_source = e_AA_source
        self.e_AA_target = e_AA_target
        self.e_BB_source = e_BB_source
        self.e_BB_target = e_BB_target
        self.e_ABsr_source = e_ABsr_source
        self.e_ABsr_target = e_ABsr_target
        self.e_ABlr_source = e_ABlr_source
        self.e_ABlr_target = e_ABlr_target
        self.dimer_ind = dimer_ind
        self.dimer_ind_lr = dimer_ind_lr
        self.batch_atomic_A = batch_atomic_A
        self.batch_atomic_B = batch_atomic_B
        self.total_charge_A = torch.zeros(1)
        self.total_charge_B = torch.zeros(1)
        self.molecule_ind_A = torch.zeros(3, dtype=torch.long)
        self.molecule_ind_B = torch.zeros(2, dtype=torch.long)
        self.natom_per_mol_A = torch.tensor([3], dtype=torch.long)
        self.natom_per_mol_B = torch.tensor([2], dtype=torch.long)

batch = TestBatch()

print("\n2. Testing APNet2...")
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
model.eval()

print("   Running eager mode inference...")
with torch.no_grad():
    output_eager = model(batch)
E_output_eager = output_eager[0]
print(f"   Eager output shape: {E_output_eager.shape}, value: {E_output_eager.sum().item():.6f}")

print("   Compiling model with torch.compile()...")
compiled_model = torch.compile(model)

print("   Running compiled inference...")
with torch.no_grad():
    output_compiled = compiled_model(batch)
E_output_compiled = output_compiled[0]
print(f"   Compiled output shape: {E_output_compiled.shape}, value: {E_output_compiled.sum().item():.6f}")

diff = torch.abs(E_output_eager - E_output_compiled).max().item()
print(f"   Difference: {diff:.2e}")
assert diff < 1e-4, f"Outputs don't match! Difference: {diff}"
print("   ✓ APNet2: Eager and compiled outputs match!")

print("\n3. Testing DimerProp...")
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
dp_model.eval()

print("   Running eager mode inference...")
with torch.no_grad():
    elst_eager, v_A_eager, v_B_eager = dp_model(batch)
print(f"   Eager output sum: {elst_eager.sum().item():.6f}")

print("   Compiling model with torch.compile()...")
compiled_dp = torch.compile(dp_model)

print("   Running compiled inference...")
with torch.no_grad():
    elst_compiled, v_A_compiled, v_B_compiled = compiled_dp(batch)
print(f"   Compiled output sum: {elst_compiled.sum().item():.6f}")

diff = torch.abs(elst_eager - elst_compiled).max().item()
print(f"   Difference: {diff:.2e}")
assert diff < 1e-5, f"Outputs don't match! Difference: {diff}"
print("   ✓ DimerProp: Eager and compiled outputs match!")

print("\n4. Performance benchmark (100 iterations)...")
n_iters = 100

print("   Warming up compiled model...")
with torch.no_grad():
    for _ in range(5):
        _ = compiled_model(batch)[0]

torch.manual_seed(42)
start = time.time()
with torch.no_grad():
    for _ in range(n_iters):
        _ = model(batch)[0]
eager_time = time.time() - start

torch.manual_seed(42)
start = time.time()
with torch.no_grad():
    for _ in range(n_iters):
        _ = compiled_model(batch)[0]
compiled_time = time.time() - start

print(f"   Eager mode: {eager_time:.4f}s ({n_iters/eager_time:.1f} iter/s)")
print(f"   Compiled:   {compiled_time:.4f}s ({n_iters/compiled_time:.1f} iter/s)")
print(f"   Speedup:    {eager_time/compiled_time:.2f}x")
print("   Note: Small batch size may show slowdown. Larger batches typically benefit from compilation.")

print("\n✓✓✓ All torch.compile tests passed! ✓✓✓")
