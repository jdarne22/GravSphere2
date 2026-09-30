'''
Set up the Segue 2 data and the parameters and priors for GravSphere2. Adapted
from initialise_Seg1.py; every number below is Kirby et al. (2013), ApJ 770, 16
unless stated otherwise.

    d          35 +/- 2 kpc          heliocentric
    R_half     3.4' +/- 0.2' = 34.6 pc projected  (46 +/- 3 pc deprojected)
    L_V        900 +/- 200 Lsun,  M_V = -2.5
    M*         1000 +/- 300 Msun
    vsys       -40.2 +/- 0.9 km/s
    sigma_v    UNRESOLVED: < 2.2 (2.6) km/s at 90% (95%)
    M_1/2      < 1.5 (2.1) e5 Msun,  (M/L_V)_1/2 < 360 (500)

READ THIS BEFORE INTERPRETING A RUN. Segue 2's dispersion is an upper limit,
not a measurement: Kirby+13 cannot resolve it with 25 stars whose velocity
errors (2-22 km/s) are comparable to any plausible sigma. GravSphere2 will
still return a posterior on M200, but at the low-mass end that posterior is the
prior, not the data. Treat every mass as an upper limit and check whether the
posterior is pressed against logM200low before quoting anything.

There is no keep/drop variant here as there is for Segue 1. Segue 2 has no
ambiguous star -- the one removal, the RR Lyrae J021900.06+200635.2, is a
confirmed pulsator (50-70 km/s velocity swings) that Kirby+13 also exclude, so
get_S2_data.py drops it unconditionally and writes a single file.
'''

import numpy as np
from constants import *
import os

dirf = 'Data/Seg2/'

kinematic_file = 'Seg2_Rproj_vlos_vloserr.txt'   # 25 stars, RR Lyrae removed


# True  = Option A, fit the member positions directly            (ndims = 18)
# False = Option B, fit a binned light profile                   (ndims = 19)
#
# Option B is NOT available for Segue 2: Data/Seg2/ holds only the Kirby+13
# spectroscopic catalogue, with no binned star-count profile of the kind
# SegIcut.txt provides for Segue 1, so there is nothing for the binned
# likelihood to fit. Setting this False raises rather than silently loading
# Segue 1's photometry.
individual = True

name = 'Segue2'
diro = 'Output/' + name + '/'
for newpath in [diro + 'Plots', diro + 'Sampler Chains']:
    if not os.path.exists(newpath):
        os.makedirs(newpath)


rLOS, vLOS, vLOS_err = np.loadtxt(str(dirf + kinematic_file), unpack=True)
rPM = rLOS
propermotion = False  # Gaia PM errors dwarf an unresolved <2.2 km/s dispersion
esc_check = False     # turn on only with a vetted member list

Rhalf = 0.0346  # kpc; projected half-light radius, 3.4' at d = 35 kpc

print(f'[initialise_Seg2] N = {len(rLOS)} tracers from {kinematic_file}, '
      f'individual = {individual}, output -> {diro}')

if individual:
    # NB: this uses the *spectroscopic* sample as the photometric tracer
    # sample. Kirby+13 selected targets on the CMD and prioritised by distance
    # from the centre, so this is not a fair draw from the light profile.
    # gravsphere2.py builds the light-profile priors from Rhalf in this branch
    # and drops rho0 entirely (its normalisation cancels in the unbinned
    # likelihood), so pfits/tracertol are deliberately not defined here.
    R = rLOS

else:
    raise NotImplementedError(
        'Option B needs a binned surface-density profile for Segue 2; '
        'Data/Seg2/ has none -- see the note above individual = True.')

Rall = np.append(R, rLOS)
rmin = np.min([Rhalf/100.0, np.min(Rall)/10.0])
rmax = np.max([Rhalf*100.0, np.max(Rall)*2.0])

# Guard: rmin == 0 would make log10(rmin) = -inf and NaN the whole Jeans grid.
if rmin <= 0.0:
    raise ValueError('rmin <= 0: check for R = 0 tracers in the input catalogue')

rcn = np.max(Rall)*1.5  # enforce positive 4th moment out to ~1.5x the data extent

Mstar = 0.0  # M* = 1000 +/- 300 Msun, vs M_1/2 < 1.5e5 -- negligible
barrad_min = rmin
barrad_max = rmax
bar_pnts = 300

bet0min = -1.0
bet0max = 1.0
betinfmin = -1.0
betinfmax = 1.0
betr0min = -2  
betr0max = 1 
betnmin = 1.0
betnmax = 3.0

# --- coreNFWtides ---
# LOWERED 7.0 -> 6.0 relative to initialise_Seg1.py. Segue 1 has a resolved
# dispersion, so its posterior sits well inside a prior starting at 10^7. Segue
# 2 does not: an unresolved sigma is consistent with arbitrarily small M200, so
# the posterior runs down onto whatever lower edge is set and piles up there.
# 6.0 does not fix that -- nothing can, the data are an upper limit -- but it
# keeps the edge far enough below the M_1/2 < 1.5e5 Msun constraint that the
# pile-up is visibly prior-driven rather than looking like a detection.
# If the posterior still stacks against logM200low, say so; do not quote a mass.
logM200low = 5.5
logM200high = 10

clow = 1         
chigh = 60.0

rclow = 1.0e-3      # 1 pc -- must go BELOW R_half = 34.6 pc
rchigh = 0.1        # 1 kpc
logrclow = np.log10(rclow)
logrchigh = np.log10(rchigh)

nlow = 0.0          # 0 = NFW cusp, 1 = full core
nhigh = 1.0

# Kirby+13 put Segue 2's tidal radius at 1.1 kpc for M_MW = 1e11 Msun inside 41
# kpc, and conclude the MW cannot be disrupting it at its present position --
# so unlike Segue 1 there is no reason to expect strong truncation inside the
# data. The range is left wide on both sides: rt below the data extent (98 pc)
# is allowed so the fit CAN find truncation if the stars demand it, and the
# upper edge sits above 1.1 kpc so the prior never binds.
rtlow = 0.05        # 50 pc
rthigh = 5.0
logrtlow = np.log10(rtlow)
logrthigh = np.log10(rthigh)

dellow = 3.01
delhigh = 6.0
