''' 
Python script to set up Segue1 data and set parameters and priors for GravSphere2. This is a modified version of the Fornax initialisation script, with the following changes:
- The photometric profile is set to the binned SegIcut.txt profile, rather than the Fornax profile.
- The half-light radius is set to 0.0295 kpc, based on Martin+08.
- The proper motion fitting is turned off, as Gaia PM errors are much larger than the internal dispersion of Segue1.
- The escape velocity check is turned off, as the member list is already vetted.
- The prior limits for the coreNFWtides model are adjusted to reflect the expected properties of ultra-faint dwarf galaxies, with lower mass and concentration ranges, and smaller core and tidal radii.
'''

import numpy as np
from constants import *
import os


dirf = 'Data/Seg1/'

# Which sample to run. Set by job_Seg1.sh from the keep=true/false toggle near
# the top of that script; defaults to 'keep' for an interactive run here.
#
#   keep   68 stars   J100704.35+160459.4 kept
#   drop   67 stars   J100704.35+160459.4 removed
#
# Both come from Simon+11 table 3 via get_S1_data.py, with the two RR Lyraes
# and the RGB binary already removed. They differ in exactly one star: the
# ambiguous one at 38" from the centre with a Bayesian membership probability
# of 0.49. Nobody can say whether it is a member, and it alone carries a large
# factor in mass and the entire preference for radial anisotropy, so the two
# runs ARE the systematic -- quote both rather than picking one.

variant = os.environ.get('GS2_VARIANT', 'keep')

if variant not in ('keep', 'drop'):
    raise ValueError(f"GS2_VARIANT = {variant!r}; expected 'keep' or 'drop'")

kinematic_file = f'Seg1_Rproj_vlos_vloserr_{variant}.txt'


# True  = Option A, fit the member positions directly            (ndims = 18)
# False = Option B, fit the binned SegIcut.txt light profile     (ndims = 19)
#
# Left as a plain switch rather than a variant because it is an independent
# axis from the keep/drop question. Option B was run once against the old
# 71-star sample and moved rho(R_half) by only -0.08 dex, i.e. the mass is
# insensitive to the tracer profile, so Option A is the default. Flip this to
# re-check that on the current sample; the output directory is tagged so the
# two cannot overwrite each other.
individual = True

name = 'Segue1_' + variant + ('' if individual else '_phot')
diro = 'Output/' + name + '/'
for newpath in [diro + 'Plots', diro + 'Sampler Chains']:
    if not os.path.exists(newpath):
        os.makedirs(newpath)


rLOS, vLOS, vLOS_err = np.loadtxt(str(dirf + kinematic_file), unpack=True)
rPM = rLOS
propermotion = False  # Gaia PM errors ~30x the internal dispersion
esc_check = False     # turn on only with a vetted member list

Rhalf = 0.0295  # kpc; projected half-light radius (Martin+08 at d = 23 kpc)

print(f'[initialise_Seg1] variant = {variant}: N = {len(rLOS)} tracers from '
      f'{kinematic_file}, individual = {individual}, output -> {diro}')

if individual:
    # NB: this uses the *spectroscopic* sample as the photometric tracer
    # sample. Simon+11 targets were colour/magnitude selected, so this is not
    # a fair draw from the light profile -- see the caveat in the notes.
    # gravsphere2.py builds the light-profile priors from Rhalf in this branch
    # and drops rho0 entirely (its normalisation cancels in the unbinned
    # likelihood), so pfits/tracertol below are deliberately not defined here.
    R = rLOS

else:
    R, surfden, surfdenerr = np.loadtxt(str(dirf + 'segue1_surfden.txt'), unpack=True)

    # alpha-beta-gamma prior centre: Plummer with a = R_half and unit total
    # mass. For a Plummer the scale radius IS the projected half-light radius,
    # so a_plum is not an independent number. rho0 = 3M/(4 pi a^3) with M = 1
    # matches the normalisation get_S1_data.py imposes on segue1_surfden.txt --
    # the binned likelihood in gravsphere2.py compares Sigma_model to Sigma_data
    # directly, with no free amplitude, so the absolute scale has to agree.
    a_plum = Rhalf

    # NB gam = 0.0 exactly makes nupars_min[4] == nupars_max[4] == 0, so the
    # photometric inner slope is FROZEN cored rather than merely constrained.
    # That is fine for a Plummer-like UFD, but it is a choice, not a prior --
    # initialise_PlumCoreOm/PlumCuspOm use 0.1 to keep gam free.
    pfits = np.array([3.0 / (4.0 * np.pi * a_plum**3), a_plum, 2.0, 5.0, 0.0])

    # Half-width of the flat prior box gravsphere2.py builds around pfits. It is
    # applied to each entry INDEPENDENTLY, but unit mass is a curve (rho0 ~
    # a^-3), so the box does not contain the unit-mass locus over the full
    # nominal range: +/-30% on rho0 admits only a/a_plum in
    # [(1+tol)^-1/3, (1-tol)^-1/3] = [0.92, 1.13], i.e. ~27-33 pc rather than
    # 21-38 pc. get_S1_data.py checks the fitted scale against that effective
    # window, not the nominal one.
    #
    # WIDENED 0.3 -> 0.4. At 0.3 the effective window is [27.0, 33.2] pc and the
    # background-subtracted SegIcut.txt profile fits a = 33.4 pc, i.e. just
    # OUTSIDE it -- get_S1_data.py's guard fires, and the sampler would pin the
    # light profile against the prior edge, making Option B a measurement of the
    # prior rather than of the photometry. 0.4 gives [26.4, 35.0] pc, which
    # admits both Martin+08's 29.5 pc and the 33.4 pc the binned profile itself
    # prefers, and lets the data choose between them. It also loosens alp to
    # [1.2, 2.8] and bet to [3.0, 7.0], so the shape can flex away from an exact
    # Plummer; gam stays frozen at 0 either way, since the box is multiplicative
    # and pfits[4] = 0 (see the note above).
    tracertol = 0.4

# NB: use both the photometric (R) and kinematic (rLOS) radii. With
# individual = True they are the same array, but under Option B the photometric
# profile stops at 57 pc while the outermost member sits at 90 pc, so keying the
# grid on R alone would leave the outer stars outside it.
#rmin = np.min([Rhalf/100.0, np.min(R)/10.0])
#rmax = np.max([Rhalf*100.0, np.max(R)*2.0])
Rall = np.append(R, rLOS)
rmin = np.min([Rhalf/100.0, np.min(Rall)/10.0])
rmax = np.max([Rhalf*100.0, np.max(Rall)*2.0])

# Guard: rmin == 0 would make log10(rmin) = -inf and NaN the whole Jeans grid.
if rmin <= 0.0:
    raise ValueError('rmin <= 0: check for R = 0 tracers in the input catalogue')

rcn = np.max(Rall)*1.5  # enforce positive 4th moment out to ~1.5x the data extent

Mstar = 0.0  # L_V ~ 340 Lsun -> M* ~ 1e3 Msun, negligible vs the halo
barrad_min = rmin
barrad_max = rmax
bar_pnts = 300

bet0min = -1.0
bet0max = 1.0
betinfmin = -1.0
betinfmax = 1.0
betr0min = -2.5  # log10 r_beta [kpc]: 3 pc 
betr0max = -0.5  # 316 pc
betnmin = 1.0
betnmax = 3.0

# --- coreNFWtides ---
logM200low = 7.0    # UFD haloes: ~10^8.5 - 10^9.5
logM200high = 10.5

clow = 5.0          # c < 5 is unphysical at 10^9 Msun
chigh = 60.0

rclow = 1.0e-3      # 1 pc -- must go BELOW R_half = 29.5 pc
rchigh = 1.0        # 1 kpc
logrclow = np.log10(rclow)
logrchigh = np.log10(rchigh)

nlow = 0.0          # 0 = NFW cusp, 1 = full core
nhigh = 1.0

rtlow = 0.05        # 50 pc -- Segue 1 may be strongly truncated
rthigh = 5.0        # Jacobi radius at 23 kpc is ~2 kpc
logrtlow = np.log10(rtlow)
logrthigh = np.log10(rthigh)

dellow = 3.01
delhigh = 6.0
