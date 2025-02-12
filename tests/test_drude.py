import pytest
import apnet_pt
"""
%%%%%%%%%%% STARTING WATER U_IND CALCULATION %%%%%%%%%%%%
-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
=-=-=-=-=-=-=-=-=-=-=-=-OpenMM Output-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
total Energy = -24.14799427986145 kJ/mol
<class 'openmm.openmm.NonbondedForce'>-26.1219482421875 kJ/mol
<class 'openmm.openmm.DrudeForce'>1.9739539623260498 kJ/mol
<class 'openmm.openmm.CMMotionRemover'>0.0 kJ/mol
Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: libtpu.so: cannot open shared object file: No such file or directory
JAXOPT.BFGS Minimizer completed in 9.746 seconds!!
OpenMM U_ind = -24.1480 kJ/mol
Python U_ind = -24.1479 kJ/mol
0.00% Error
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
"""
