'''
Python script to set up Leo II data and set parameters and priors for GravSphere2.
This is a modified version of the Segue 1 initialisation script, with the
following changes:
- The kinematics come from Spencer+17 via get_LeoII_data.py: a single 175-star
  member sample, with none of the keep/drop variant machinery Segue 1 needed.
- Option B (individual = False) is removed. No binned star-count profile for
  Leo II is shipped with the repo, so there is nothing to fit; the tracer
  profile is fitted to the member positions themselves.
- The half-light radius is set to 0.176 kpc (Munoz+18 / McConnachie+12 at
  d = 233 kpc), six times Segue 1's.
- The stellar mass is switched ON. Segue 1's M* ~ 1e3 Msun was genuinely
  negligible; Leo II's is ~1e6 Msun against M_1/2 ~ 9e6 Msun, i.e. of order
  10% of the mass inside the half-light radius, so leaving it out would push
  that mass into the halo. This adds one dimension (ndims 18 -> 19).
- The coreNFWtides priors are moved from ultra-faint to classical-dSph ranges:
  a heavier and less concentrated halo, a core radius that can reach past
  R_half, and a tidal radius out at the Jacobi radius of a galaxy 233 kpc away
  rather than 23 kpc.
- Proper motions are optional (propermotion below). They come from the HST
  catalogue of Lepine+11 via get_LeoII_data.py (DATA_MODE = 'pm'), a separate
  sample from the Spencer+17 stars. Errors are ~0.3 mas/yr even for the bright
  stars, i.e. ~300 km/s at 233 kpc against a ~7 km/s internal dispersion.

NB gravsphere2.py imports its initialisation file at the top by name. Switch
`from initialise_Seg1 import *` to `from initialise_LeoII import *` there
before running this.
'''

import numpy as np
from constants import *
import os


dirf = 'Data/LeoII/'

# Written by get_LeoII_data.py from Data/LeoII/LeoII.tsv (Spencer+17, VizieR
# J/AJ/153/254). All 175 stars flagged Mm = 'Y' are kept
#
# The one systematic NOT folded in: Spencer+17's own result is that ~30-40% of
# Leo II's stars are binaries, and the raw dispersion (7.37 km/s from
# get_LeoII_data.py) is therefore an upper bound. Their binary-corrected value
# is lower, so the mass here should be read as an upper bound too.
kinematic_file = 'LeoII_Rproj_vlos_vloserr.txt'


# True = Option A, fit the member positions directly.
#
# Segue 1 had this as a switch because SegIcut.txt gave it a binned light
# profile to fall back on. Leo II has no such file, so Option B has no input
# and this is fixed rather than a choice. If a real Leo II surface-brightness
# profile turns up, add the photometry block back to get_LeoII_data.py, write
# a leoII_surfden.txt, and restore the else branch from initialise_Seg1.py.
individual = True

if not individual:
    raise NotImplementedError(
        'Option B needs a binned Leo II surface-density profile; none is '
        'shipped. See the note above.')

# True  = fit LOS velocities + HST proper motions
# False = LOS velocities only
propermotion = True
pm_file = 'LeoII_Rproj_vR_vRerr_vT_vTerr.txt'

# name stays 'LeoII' (run_LeoII.py checks it); only the output directory is
# tagged, so a PM run never resumes from or overwrites the LOS-only chains.
name = 'LeoII'
diro = 'Output/' + name + ('_pm' if propermotion else '') + '/'
for newpath in [diro + 'Plots', diro + 'Sampler Chains']:
    if not os.path.exists(newpath):
        os.makedirs(newpath)

rLOS, vLOS, vLOS_err = np.loadtxt(str(dirf + kinematic_file), unpack=True)

if propermotion:
    rPM, vPMR, vPMR_err, vPMt, vPMt_err = np.loadtxt(str(dirf + pm_file),
                                                     unpack=True)
    print(f'[initialise_LeoII] N = {len(rPM)} proper-motion tracers from {pm_file}')
else:
    rPM = rLOS

esc_check = False     # turn on only with a vetted member list

# Projected half-light radius. Munoz+18's Plummer fit to MegaCam photometry
# gives ~2.6 arcmin, which is 176 pc at d = 233 kpc (Bellazzini+05) -- the same
# distance get_LeoII_data.py uses to convert the member radii, so the two must
# be changed together.
Rhalf = 0.176  # kpc

print(f'[initialise_LeoII] N = {len(rLOS)} tracers from {kinematic_file}, '
      f'individual = {individual}, output -> {diro}')

# NB: this uses the *spectroscopic* sample as the photometric tracer sample.
# Spencer+17's targets were colour/magnitude selected, so it is not a fair draw
# from the light profile -- the median member radius is 3.0 arcmin against
# R_half = 2.6 arcmin, i.e. mildly biased outward, as expected when the faint
# centre is under-targeted. gravsphere2.py builds the light-profile priors from
# Rhalf in this branch and drops rho0 entirely (its normalisation cancels in
# the unbinned likelihood), so pfits/tracertol are deliberately not defined.
R = rLOS

# Kept in the Seg1 form: under Option A R and rLOS are the same array, but
# appending costs nothing and keeps the grid correct if a photometric sample
# with a different radial extent is ever supplied. rPM is included because the
# HST proper-motion stars are a separate sample from the Spencer+17 ones; with
# propermotion = False it is just rLOS again.
Rall = np.concatenate([R, rLOS, rPM])
rmin = np.min([Rhalf/100.0, np.min(Rall)/10.0])
rmax = np.max([Rhalf*100.0, np.max(Rall)*2.0])

# Guard: rmin == 0 would make log10(rmin) = -inf and NaN the whole Jeans grid.
if rmin <= 0.0:
    raise ValueError('rmin <= 0: check for R = 0 tracers in the input catalogue')

rcn = np.max(Rall)*1.5  # enforce positive 4th moment out to ~1.5x the data extent


# --- stellar mass ---
# Segue 1 set Mstar = 0.0 and was right to: L_V ~ 340 Lsun. Leo II is four
# orders of magnitude brighter -- M_V = -9.8 gives L_V ~ 7e5 Lsun, and an old
# metal-poor population at M/L_V ~ 1.5 puts M* at ~1e6 Msun. Against
# M_1/2 ~ 9e6 Msun (sigma = 7.4 km/s, r_1/2 = 4/3 R_half) that is ~10% of the
# enclosed mass, so it is not something to drop into the dark matter.
#
# gravsphere2.py assumes the baryonic mass follows the tracer light, which is
# fine for a dSph with no gas and no measured gradient. It samples Mstar as a
# free parameter inside [Mstar_min, Mstar_max], so the +/-25% band below is the
# prior on M/L, not a fixed value. Setting Mstar = 0.0 turns the whole thing
# off and returns ndims to 18.
INCLUDE_STELLAR_MASS = True

if INCLUDE_STELLAR_MASS:
    Mstar = 1.0e6                      # Msun; L_V ~ 7e5 Lsun at M/L_V ~ 1.5
    Mstar_err = Mstar * 0.25           # spans M/L_V ~ 1.1 - 1.9
    Mstar_min = Mstar - Mstar_err
    Mstar_max = Mstar + Mstar_err
else:
    Mstar = 0.0

barrad_min = rmin
barrad_max = rmax
bar_pnts = 300


# --- anisotropy ---
bet0min = -1.0
bet0max = 1.0
betinfmin = -1.0
betinfmax = 1.0
# Widened from Segue 1's [-2.5, -0.5] to track the six-times-larger R_half:
# r_beta now runs from R_half/18 to R_half*5.7, so the transition can sit well
# inside or well outside the tracers rather than being forced near them.
betr0min = -2.0  # log10 r_beta [kpc]: 10 pc
betr0max = 0.0   # 1 kpc
betnmin = 1.0
betnmax = 3.0


# --- coreNFWtides ---
# Leo II is a classical dSph, not an ultra-faint: abundance matching puts it at
# M200 ~ 10^9 - 10^9.5, so the Segue 1 window is shifted up at the bottom end.
logM200low = 8.0
logM200high = 11.0

clow = 5.0          # c(M) at 10^9-10^10 Msun is ~13 with a factor ~2 scatter
chigh = 50.0

rclow = 1.0e-3      # 1 pc -- must go BELOW R_half = 176 pc
rchigh = 2.0        # 2 kpc -- ~11x R_half, so a core can exceed the tracers
logrclow = np.log10(rclow)
logrchigh = np.log10(rchigh)

nlow = 0.0          # 0 = NFW cusp, 1 = full core
nhigh = 1.0

# At 233 kpc Leo II is far outside the region where the MW strips anything:
# r_J = D*(M/3M_MW)^(1/3) ~ 16 kpc for a 10^9 Msun halo. Segue 1's [50 pc,
# 5 kpc] would truncate the halo inside the tracer sample, which is why this
# moves out by more than an order of magnitude. rtlow is still left just below
# the outermost tracer (742 pc) so the data can argue for truncation if it
# really wants to, rather than the prior forbidding it.
rtlow = 0.5         # 500 pc
rthigh = 20.0       # ~ the Jacobi radius at 233 kpc
logrtlow = np.log10(rtlow)
logrthigh = np.log10(rthigh)

# 5.0 rather than Segue 1's 6.0: delta is the outer tidal falloff, and a very
# steep truncation has no motivation for an unstressed galaxy this far out.
dellow = 3.01
delhigh = 5.0
