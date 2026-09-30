'''
Build the Segue 1 input files for GravSphere2 from the published Simon et al.
(2011) catalogue: ApJ 733, 46, table 3 (VizieR J/ApJ/733/46).

Three stars are removed from every sample. They vary in velocity for reasons
unrelated to the potential of Segue 1, so they carry no mass information:

    J100644.58+155953.9   RR Lyrae; P = 0.50 d, confirmed photometrically
    J100705.60+160422.0   RR Lyrae; HB star, velocity varies between epochs
    J100652.33+160235.8   red giant in a binary. Simon+11: 216.1 +/- 2.9 km/s
                          on 2007 Nov 12, falling to 203.0 +/- 2.3 on 2009 Feb
                          27, back up to 210.8 +/- 2.3 on 2010 Feb 13 -- about
                          a 1 yr period with a ~0.65 Msun companion.

All three are named in Simon+11, which removes them before quoting any
dispersion (their sections 4.2-4.3).

A fourth star is genuinely ambiguous

    J100704.35+160459.4   38" from the centre, 231.6 +/- 3.0 km/s from two
                          epochs, ~6 sigma from systemic. Simon+11 give it a
                          Bayesian membership probability of 0.49 -- a coin
                          flip -- and note it alone drives a factor ~2 in mass
                          and is the only star producing a preference for
                          radial anisotropy.
'''

import os

import numpy as np
from scipy.optimize import minimize

import constants


arcmin   = constants.arcmin      # arcmin -> rad
dgal_kpc = 23.0                  # distance to Segue 1 [kpc]

Simon_data = 'Data/Seg1/Simon2011_table3_velocities.tsv'

RR_LYRAE   = ['J100644.58+155953.9', 'J100705.60+160422.0']
RGB_BINARY = 'J100652.33+160235.8'
AMBIGUOUS  = 'J100704.35+160459.4'

ALWAYS_REMOVE = RR_LYRAE + [RGB_BINARY]

# Radii are published to 0.1 arcmin, so the innermost star is listed as exactly
# 0.0. R = 0 is fatal downstream: it sets rmin = 0 (log10 -> -inf on the Jeans
# grid) and makes the individual-star term log(R*Sigma) equal -inf, so every
# likelihood call returns -inf. Floor at half the quantisation step.
R_FLOOR_ARCMIN = 0.05

# True  -> one inverse-variance weighted velocity per star, over all epochs.
# False -> the single flagged epoch, i.e. what the old gravsphere1 file had.
USE_MULTIEPOCH = True


# --- reading the catalogue ---------------------------------------------------

def read_simon_table(path):

    header = None
    data_rows = []

    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        fields = line.rstrip('\n').split('\t')
        if fields[0].strip() == 'recno':
            header = [f.strip() for f in fields]
        elif fields[0].strip().isdigit():
            data_rows.append(fields)

    if header is None:
        raise ValueError(f'no header row found in {path}')

    def column(name):
        i = header.index(name)
        return np.array([float(r[i]) if r[i].strip() else np.nan for r in data_rows])

    star_id = np.array([r[header.index('SDSS')].strip() for r in data_rows])

    return dict(star_id=star_id, velocity=column('Vel'), error=column('e_Vel'),
                radius=column('Rad'), i_mag=column('imag'),
                is_member=column('Mm'), bayes_prob=column('Bpr'))


# --- combining repeat measurements of one star -------------------------------

def weighted_mean(velocity, error):
    '''Inverse-variance mean of one star's epochs, and its error.'''
    weight = 1.0/error**2
    mean = np.sum(weight*velocity)/np.sum(weight)
    return mean, np.sqrt(1.0/np.sum(weight))


def first_finite(values):
    '''
    First non-blank entry of a per-star column. i_mag and Bpr are properties of
    the star, so every epoch repeats the same value -- but some rows leave them
    blank, and the two RR Lyraes have no Bpr at all, so this returns nan rather
    than warning about an all-NaN slice.
    '''
    finite = values[np.isfinite(values)]
    return finite[0] if len(finite) else np.nan


# --- systemic velocity and intrinsic dispersion ------------------------------

def neg_lnlike(params, velocity, error):
    vsys, ln_sigma = params
    variance = error**2 + np.exp(2*ln_sigma)
    return 0.5*np.sum(np.log(2*np.pi*variance) + (velocity - vsys)**2/variance)


def fit_vsys_and_sigma(velocity, error):
    '''
    Joint MLE for the systemic velocity and the intrinsic dispersion.

    A plain mean is wrong because the per-star errors span 2-38 km/s. 
    '''
    fit = minimize(neg_lnlike, [np.mean(velocity), np.log(3.0)],
                   args=(velocity, error), method='Nelder-Mead',
                   options={'xatol': 1e-6, 'fatol': 1e-8})
    return fit.x[0], np.exp(fit.x[1])


# --- one row per member star -------------------------------------------------

catalogue = read_simon_table(Simon_data)

members = catalogue['star_id'][catalogue['is_member'] == 1]
if len(members) != len(set(members)):
    raise ValueError('a star is flagged as a member on more than one row')

print(f'{Simon_data}: {len(catalogue["star_id"])} rows, '
      f'{len(set(catalogue["star_id"]))} stars, {len(members)} flagged members')

velocity   = np.zeros(len(members))
error      = np.zeros(len(members))
radius     = np.zeros(len(members))
n_epochs   = np.zeros(len(members), dtype=int)
bayes_prob = np.zeros(len(members))

for k, star in enumerate(members):
    epochs = catalogue['star_id'] == star
    v, e = catalogue['velocity'][epochs], catalogue['error'][epochs]

    if USE_MULTIEPOCH:
        velocity[k], error[k] = weighted_mean(v, e)
    else:
        flagged = epochs & (catalogue['is_member'] == 1)
        velocity[k] = catalogue['velocity'][flagged][0]
        error[k] = catalogue['error'][flagged][0]

    radius[k]     = catalogue['radius'][epochs][0]      # constant for a star
    n_epochs[k]   = epochs.sum()
    bayes_prob[k] = first_finite(catalogue['bayes_prob'][epochs])

epoch_counts = dict(sorted(zip(*np.unique(n_epochs, return_counts=True))))
print(f'epochs per member: {epoch_counts}  '
      f'-> {int((n_epochs > 1).sum())} stars have repeats')
print('velocities used: '
      + ('inverse-variance mean over all epochs' if USE_MULTIEPOCH
         else 'single flagged epoch'))


# --- build and write the two samples -----------------------------------------

n_floored = int((radius < R_FLOOR_ARCMIN).sum())
if n_floored:
    print(f'\nflooring {n_floored} star(s) with R < {R_FLOOR_ARCMIN} arcmin')

R_proj = np.maximum(radius, R_FLOOR_ARCMIN) * dgal_kpc/arcmin   # kpc

removed = np.isin(members, ALWAYS_REMOVE)
print(f'\nremoved from every sample ({removed.sum()} stars):')

for star in ALWAYS_REMOVE:
    match = np.where(members == star)[0]
    if not len(match):
        raise ValueError(f'{star} is not among the flagged members -- check the ID')
    k = match[0]
    why = 'RR Lyrae' if star in RR_LYRAE else 'RGB binary'
    print(f"   {star}  R = {radius[k]:4.1f}'  "
          f'v = {velocity[k]:6.1f} +/- {error[k]:4.1f}  '
          f'({n_epochs[k]} epochs)  <- {why}')

amb = np.where(members == AMBIGUOUS)[0][0]
print('\nambiguous star, switched between the two samples:')
print(f"   {AMBIGUOUS}  R = {radius[amb]:4.1f}'  "
      f'v = {velocity[amb]:6.1f} +/- {error[amb]:4.1f}  '
      f'({n_epochs[amb]} epochs)  Bayesian p = {bayes_prob[amb]:.3f}')

samples = {}
print()
for tag, keep_ambiguous in (('keep', True), ('drop', False)):
    selected = ~removed
    if not keep_ambiguous:
        selected = selected & (members != AMBIGUOUS)

    v, e, R = velocity[selected], error[selected], R_proj[selected]
    vsys, sigma_int = fit_vsys_and_sigma(v, e)

    path = f'Data/Seg1/Seg1_Rproj_vlos_vloserr_{tag}.txt'
    np.savetxt(path, np.column_stack([R, v - vsys, e]),
               header=f'R_proj[kpc]  v_los[km/s]  v_los_err[km/s]   '
                      f'Simon+11 table 3, ambiguous star {tag}, '
                      f'N={selected.sum()}, vsys={vsys:.3f}, '
                      f'sigma_int={sigma_int:.3f}')

    samples[tag] = dict(selected=selected, vsys=vsys, sigma=sigma_int)
    print(f'{tag:5s}: N = {selected.sum():2d}, vsys = {vsys:7.2f} km/s, '
          f'sigma_int = {sigma_int:.2f} km/s, '
          f'R = {R.min()*1e3:.1f} - {R.max()*1e3:.1f} pc  -> {path}')

print('\nNB Simon+11 quote 3.7 +1.4/-1.1 km/s AFTER correcting for undetected')
print('   binaries (Martinez+11); that correction is not applied here.')


# --- diagnostic plot ---------------------------------------------------------
# Written before the photometry section so the figure always describes exactly
# the samples just saved. Agg because this runs headless under job_Seg1.sh.

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plotdir = 'Output/Segue1/Plots/'
os.makedirs(plotdir, exist_ok=True)

vsys, sigma_int = samples['keep']['vsys'], samples['keep']['sigma']
kept = (~removed) & (members != AMBIGUOUS)

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.errorbar(radius[kept], velocity[kept], yerr=error[kept], fmt='o', ms=4,
            color='k', ecolor='0.55', elinewidth=1, capsize=2, zorder=3,
            label='kept in both samples')
ax.errorbar(radius[removed], velocity[removed], yerr=error[removed], fmt='s',
            ms=7, mfc='none', color='C3', elinewidth=1, capsize=2, zorder=4,
            label='RR Lyrae / RGB binary (always removed)')
ax.errorbar(radius[amb], velocity[amb], yerr=error[amb], fmt='D', ms=8,
            mfc='none', color='C0', elinewidth=1.5, capsize=3, zorder=5,
            label=f'ambiguous (Bayesian p = {bayes_prob[amb]:.2f})')
ax.axhline(vsys, color='C3', lw=1.2, zorder=2)
ax.axhspan(vsys - sigma_int, vsys + sigma_int, color='C3', alpha=0.12, zorder=1)
ax.set_xlabel(r"$R_{\rm proj}$ [arcmin]")
ax.set_ylabel(r'$v_{\rm los}$ [km/s]')
ax.set_title('Segue 1 members (Simon+11 table 3, epoch-averaged)')
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(plotdir + 'data_vlos_R_segue1.pdf')
plt.close(fig)

print(f'\nplot written to {plotdir}data_vlos_R_segue1.pdf')


# Photometry for Option B (individual = False)
#
# SegIcut.txt is the binned star-count profile shipped with gravsphere1:
#     R[arcmin]   Sigma[arbitrary]   err_low(negative)   err_high
#
# GravSphere2 expects a profile of unit total mass, 2 pi Int Sigma R dR = 1.
# The five bins only span 0.5-8.5 arcmin, so integrating them directly would
# miss the centre and the tail. Fit a Plummer instead -- its total is analytic
# -- and divide by that.
#
# The shipped profile is NOT background subtracted. A flat contaminant floor
# props up the outer bins and nearly doubles the fitted scale radius (56 pc vs
# Martin+08's 29.5 pc), which would flatten nu_star at large R and propagate
# all the way into M(r).

SUBTRACT_BACKGROUND = True      # False = use SegIcut.txt exactly as shipped
MARTIN08_RHALF_PC   = 29.5      # published half-light radius, for comparison

NELDER_MEAD = {'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 60000, 'maxfev': 60000}


phot_table  = np.loadtxt('Data/Seg1/SegIcut.txt')

R_phot      = phot_table[:, 0] * dgal_kpc/arcmin    # arcmin -> kpc
surfden     = phot_table[:, 1]
surfden_err = 0.5*(np.abs(phot_table[:, 2]) + phot_table[:, 3])   # symmetrise

outermost_bin = phot_table[-1, 1]   # kept for the background print-out


def plummer_surfden(R, total, scale):
    '''Projected Plummer profile normalised so 2 pi Int Sigma R dR = total.'''
    return total * scale**2 / (np.pi * (scale**2 + R**2)**2)


def chi2_of(total, scale, background):
    '''Chi2 of the binned profile against Plummer + a flat floor.'''
    model = plummer_surfden(R_phot, total, scale) + background
    return np.sum(((surfden - model)/surfden_err)**2)


def fit_plummer(background=None):
    '''
    Fit (total, scale) to the binned profile, and the background too unless a
    value is given. Fitted in log space so the parameters stay positive.
    Returns (total, scale, background, chi2).
    '''
    fit_background = background is None

    guess = [np.log(np.sum(surfden)), np.log(0.03)]
    if fit_background:
        guess.append(np.log(0.2))

    def objective(log_params):
        params = np.exp(log_params)
        bg = params[2] if fit_background else background
        return chi2_of(params[0], params[1], bg)

    result = minimize(objective, guess, method='Nelder-Mead', options=NELDER_MEAD)

    total, scale = np.exp(result.x[0]), np.exp(result.x[1])
    if fit_background:
        background = np.exp(result.x[2])

    return total, scale, background, result.fun


if SUBTRACT_BACKGROUND:
    total, scale, background, chi2_best = fit_plummer()
    n_dof = len(R_phot) - 3

    # 1-sigma on the background: step through it, re-fitting total and scale at
    # each step, and keep whatever stays within delta-chi2 = 1 of the best fit.
    bg_grid  = np.linspace(1e-4, 3.0*background, 60)
    bg_chi2  = np.array([fit_plummer(background=b)[3] for b in bg_grid])
    bg_allowed = bg_grid[bg_chi2 < chi2_best + 1.0]

    background_err = (0.5*(bg_allowed.max() - bg_allowed.min())
                      if len(bg_allowed) > 1 else 0.0)

    # The outer bins are almost pure background, so folding this into the bin
    # errors is what stops Option B looking far tighter than it really is.
    surfden     = surfden - background
    surfden_err = np.hypot(surfden_err, background_err)

    print(f'\nphotometry: background = {background:.4f} +/- {background_err:.4f} '
          f'({100*background/outermost_bin:.0f}% of the outermost bin) -- SUBTRACTED')

else:
    total, scale, background, chi2_best = fit_plummer(background=0.0)
    n_dof = len(R_phot) - 2

    print('\nphotometry: background NOT subtracted')


# Normalise to the unit-total-mass convention GravSphere2 assumes.
surfden     = surfden / total
surfden_err = surfden_err / total

scale_pc = scale*1e3


# --- diagnostics ---
# For a Plummer the scale radius is also the projected half-light radius.
print(f'photometry: Plummer fit a = {scale_pc:.1f} pc '
      f'(Martin+08 R_half = {MARTIN08_RHALF_PC} pc), '
      f'chi2 = {chi2_best:.2f} for {n_dof} dof')

R_fine = np.logspace(-5, 2, 4000)
norm_check = 2*np.pi*np.trapezoid(plummer_surfden(R_fine, 1.0, scale)*R_fine, R_fine)
print(f'normalisation check: 2*pi*Int Sigma R dR = {norm_check:.4f} (target 1)')
print(f'R range = {R_phot.min()*1e3:.1f} - {R_phot.max()*1e3:.1f} pc')


# --- does the fit sit inside the prior initialise_Seg1.py will impose? ---
# gravsphere2.py puts a +/- tracertol box on rho0 and on r0 separately, but the
# unit-mass Plummer the normalised data represent lives on rho0 = 3/(4 pi a^3).
# Keeping that curve inside the rho0 box is the tighter of the two:
#     a/a_plum in [(1+tol)^(-1/3), (1-tol)^(-1/3)]
# Outside it, the sampler pins to the prior edge and the light profile comes
# from the prior rather than the data.
PRIOR_SCALE = 0.0295    # a_plum in initialise_Seg1.py [kpc]
PRIOR_TOL   = 0.4       # tracertol in initialise_Seg1.py

prior_lo = PRIOR_SCALE * max(1.0 - PRIOR_TOL, (1.0 + PRIOR_TOL)**(-1.0/3.0))
prior_hi = PRIOR_SCALE * min(1.0 + PRIOR_TOL, (1.0 - PRIOR_TOL)**(-1.0/3.0))

print(f'pfits prior on a: nominal [{PRIOR_SCALE*(1 - PRIOR_TOL)*1e3:.1f}, '
      f'{PRIOR_SCALE*(1 + PRIOR_TOL)*1e3:.1f}] pc, '
      f'effective [{prior_lo*1e3:.1f}, {prior_hi*1e3:.1f}] pc')

if not prior_lo <= scale <= prior_hi:
    print(f'WARNING: fitted a = {scale_pc:.1f} pc is OUTSIDE the effective prior '
          f'[{prior_lo*1e3:.1f}, {prior_hi*1e3:.1f}] pc -- re-centre a_plum or '
          'widen tracertol in initialise_Seg1.py')


np.savetxt('Data/Seg1/segue1_surfden.txt',
           np.column_stack([R_phot, surfden, surfden_err]),
           header='R[kpc]  Sigma[2pi.Int.Sigma.R.dR=1]  Sigma_err')
