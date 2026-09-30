'''
Build the Leo II input file for GravSphere2 from the published Spencer et al.
(2017) catalogue: AJ 153, 254 (VizieR J/AJ/153/254), Data/LeoII/LeoII.tsv.

The table is one row per *observation*, not per star. Stars are identified by
their (RAJ2000, DEJ2000) pair, which repeats exactly across epochs. For each
star with Nobs > 1 the catalogue carries Nobs rows flagged Comb = 'S' (the
individual epochs) plus one row flagged Comb = 'C' holding Spencer+17's own
combined velocity; stars with Nobs = 1 have a single 'S' row and no 'C' row.
This script rebuilds the combined velocity itself from the 'S' rows and checks
it against the published 'C' value (they agree to <0.05 km/s).

Membership is the Mm column, which is constant across a star's rows: 175 of the
222 stars are flagged 'Y'. Unlike Segue 1, no individual stars are removed here
-- the Mm flag is taken as the vetted member list, and Spencer+17 name no RR
Lyraes or confirmed binaries to excise. Their whole point is that undetected
binaries inflate the dispersion of *every* dwarf; that is a correction to sigma,
not a list of stars to drop (see the note at the end).

Unlike get_S1_data.py there is no photometry section: no binned star-count
profile for Leo II is shipped with the repo, so Option B (individual = False)
has no input and only the kinematic file is written. Add a photometry block
here if a real Leo II surface-brightness profile turns up.

With DATA_MODE = 'pm' the script instead reads the HST proper-motion catalogue
(Data/LeoII/LeoII_proper_mot.tsv), removes the bulk motion, converts to km/s
and splits each star's motion into radial and tangential components about the
centre. That table has no membership flag and is not cross-matched to
Spencer+17; every star with a complete proper motion is kept.
'''

import os

import numpy as np
from scipy.optimize import minimize

import matplotlib
matplotlib.use('Agg')            # headless on the cluster
import matplotlib.pyplot as plt

import constants


# 'vlos' -> line-of-sight velocities from Spencer+17
# 'pm'   -> HST proper motions
DATA_MODE = 'pm'


arcmin   = constants.arcmin      # arcmin -> rad
dgal_kpc = 233.0                 # distance to Leo II [kpc], Bellazzini+05

# Centre from Munoz+18 (structural fits to MegaCam photometry). The member
# centroid computed below lands within ~0.3' of this, which is the check that
# matters -- an offset centre would bias every R_proj outward.
RA0_DEG  = 168.3627
DEC0_DEG =  22.1529

Spencer_data = 'Data/LeoII/LeoII.tsv'

proper_mot_data = 'Data/LeoII/LeoII_proper_mot.tsv'

# Guard only. Leo II coordinates are published to 1e-6 deg and the innermost
# member sits at 0.52 arcmin, so this never bites -- but R = 0 is fatal
# downstream (rmin = 0 -> log10 -> -inf on the Jeans grid, and log(R*Sigma)
# = -inf in the individual-star term, so every likelihood call returns -inf).
R_FLOOR_ARCMIN = 0.01

# True  -> one inverse-variance weighted velocity per star, over all epochs.
# False -> the first epoch only, i.e. one measurement per star with no averaging.
USE_MULTIEPOCH = True

# mas/yr at 1 kpc -> km/s
K_PM = 4.740470463


# --- reading the catalogue ---------------------------------------------------

def read_spencer_table(path):
    '''
    Parse the VizieR tab-separated dump. The first three lines are the column
    names, the units, and a rule of dashes; everything after that is data.
    '''
    lines = [l.rstrip('\n').split('\t') for l in open(path)]

    header = [f.strip() for f in lines[0]]
    data_rows = [r for r in lines[3:] if len(r) >= len(header) and r[0].strip()]

    if 'RAJ2000' not in header or 'Comb' not in header:
        raise ValueError(f'{path} does not look like the Spencer+17 VizieR dump')

    def column(name, cast=float):
        i = header.index(name)
        if cast is float:
            return np.array([float(r[i]) if r[i].strip() else np.nan
                             for r in data_rows])
        return np.array([r[i].strip() for r in data_rows])

    # (RA, Dec) as printed is the star ID: the strings repeat byte-for-byte
    # across a star's epochs, so no float tolerance is needed.
    star_id = np.array([f'{r[header.index("RAJ2000")].strip()}'
                        f'{r[header.index("DEJ2000")].strip()}'
                        for r in data_rows])

    return dict(star_id=star_id,
                ra=column('RAJ2000'), dec=column('DEJ2000'),
                velocity=column('RVel'), error=column('e_RVel'),
                n_obs=column('Nobs'), feh=column('[Fe/H]'),
                is_member=column('Mm', str), combined=column('Comb', str))

def read_proper_mot_table(path):
    '''HST proper-motion catalogue (VizieR dump): one row per star, no membership flag.'''
    lines = [l.rstrip('\n').split('\t') for l in open(path)]
    header = [f.strip() for f in lines[0]]
    data_rows = [r for r in lines[3:] if len(r) >= len(header) and r[0].strip()]

    if 'RAJ2000' not in header or 'pmRA' not in header:
        raise ValueError(f'{path} does not look like the Leo II proper-motion dump')

    def column(name):
        i = header.index(name)
        return np.array([float(r[i]) if r[i].strip() else np.nan for r in data_rows])

    return dict(ra=column('RAJ2000'), dec=column('DEJ2000'),
                pmra=column('pmRA'), pmra_err=column('e_pmRA'),
                pmdec=column('pmDE'), pmdec_err=column('e_pmDE'))



# --- combining repeat measurements of one star -------------------------------

def weighted_mean(velocity, error):
    '''Inverse-variance mean of one star's epochs, and its error.'''
    weight = 1.0/error**2
    mean = np.sum(weight*velocity)/np.sum(weight)
    return mean, np.sqrt(1.0/np.sum(weight))


def angular_separation_arcmin(ra, dec, ra0, dec0):
    '''
    True angular separation on the sphere, in arcmin. Leo II spans only ~0.4
    deg so the flat tangent-plane approximation would do, but this costs
    nothing and cannot go wrong at large radii.
    '''
    ra, dec = np.radians(ra), np.radians(dec)
    ra0, dec0 = np.radians(ra0), np.radians(dec0)
    cos_sep = (np.sin(dec0)*np.sin(dec)
               + np.cos(dec0)*np.cos(dec)*np.cos(ra - ra0))
    return np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0))) * 60.0


# --- systemic velocity and intrinsic dispersion ------------------------------

def neg_lnlike(params, velocity, error):
    vsys, ln_sigma = params
    variance = error**2 + np.exp(2*ln_sigma)
    return 0.5*np.sum(np.log(2*np.pi*variance) + (velocity - vsys)**2/variance)


def fit_vsys_and_sigma(velocity, error):
    '''
    Joint MLE for the systemic velocity and the intrinsic dispersion.

    A plain mean is wrong because the per-star errors span a wide range.
    '''
    fit = minimize(neg_lnlike, [np.mean(velocity), np.log(7.0)],
                   args=(velocity, error), method='Nelder-Mead',
                   options={'xatol': 1e-6, 'fatol': 1e-8})
    return fit.x[0], np.exp(fit.x[1])


plotdir = 'Output/LeoII/Plots/'


# --- line-of-sight velocities (DATA_MODE = 'vlos') ---------------------------

def build_vlos_sample():
    catalogue = read_spencer_table(Spencer_data)

    all_stars = list(dict.fromkeys(catalogue['star_id']))   # order-preserving unique

    # Mm is a property of the star, so it must not disagree between a star's rows.
    for star in all_stars:
        flags = set(catalogue['is_member'][catalogue['star_id'] == star])
        if len(flags) > 1:
            raise ValueError(f'{star} is flagged {flags} on different rows')

    members = np.array([s for s in all_stars
                        if catalogue['is_member'][catalogue['star_id'] == s][0] == 'Y'])

    print(f'{Spencer_data}: {len(catalogue["star_id"])} rows, '
          f'{len(all_stars)} stars, {len(members)} flagged members')

    velocity   = np.zeros(len(members))
    error      = np.zeros(len(members))
    ra         = np.zeros(len(members))
    dec        = np.zeros(len(members))
    n_epochs   = np.zeros(len(members), dtype=int)
    feh        = np.zeros(len(members))
    published  = np.full(len(members), np.nan)   # Spencer+17's own combined value

    for k, star in enumerate(members):
        rows = catalogue['star_id'] == star
        epochs = rows & (catalogue['combined'] == 'S')
        combined = rows & (catalogue['combined'] == 'C')

        if epochs.sum() != catalogue['n_obs'][rows][0]:
            raise ValueError(f'{star}: {epochs.sum()} epoch rows but Nobs = '
                             f'{catalogue["n_obs"][rows][0]:.0f}')

        v, e = catalogue['velocity'][epochs], catalogue['error'][epochs]

        if USE_MULTIEPOCH:
            velocity[k], error[k] = weighted_mean(v, e)
        else:
            velocity[k], error[k] = v[0], e[0]

        if combined.any():
            published[k] = catalogue['velocity'][combined][0]

        ra[k]       = catalogue['ra'][rows][0]          # constant for a star
        dec[k]      = catalogue['dec'][rows][0]
        n_epochs[k] = epochs.sum()
        feh[k]      = catalogue['feh'][combined][0] if combined.any() else \
                      catalogue['feh'][epochs][0]

    epoch_counts = dict(sorted(zip(*np.unique(n_epochs, return_counts=True))))
    print(f'epochs per member: {epoch_counts}  '
          f'-> {int((n_epochs > 1).sum())} stars have repeats')
    print('velocities used: '
          + ('inverse-variance mean over all epochs' if USE_MULTIEPOCH
             else 'first epoch only'))

    # Cross-check against the catalogue's own combined column. A large discrepancy
    # would mean the 'S'/'C' rows are not being paired up the way this script assumes.
    has_published = np.isfinite(published)
    if USE_MULTIEPOCH and has_published.any():
        offset = np.abs(velocity[has_published] - published[has_published])
        print(f'cross-check vs the published Comb="C" velocities: '
              f'max |difference| = {offset.max():.3f} km/s '
              f'over {has_published.sum()} stars')
        if offset.max() > 0.5:
            raise ValueError('recomputed velocities disagree with the published '
                             'combined values -- check the S/C row pairing')

    # --- build and write the sample ------------------------------------------

    radius = angular_separation_arcmin(ra, dec, RA0_DEG, DEC0_DEG)

    # Centroid check on the adopted centre. Weighted by nothing -- a plain mean of
    # the member positions, which is all this is meant to catch.
    ra_bar, dec_bar = ra.mean(), dec.mean()
    centroid_offset = angular_separation_arcmin(ra_bar, dec_bar, RA0_DEG, DEC0_DEG)
    print(f'\ncentre: Munoz+18 ({RA0_DEG:.4f}, {DEC0_DEG:+.4f}) deg; '
          f'member centroid ({ra_bar:.4f}, {dec_bar:+.4f}) deg '
          f'-> offset {centroid_offset:.2f} arcmin')
    if centroid_offset > 1.0:
        print('WARNING: centroid is more than 1 arcmin from the adopted centre -- '
              'check RA0_DEG/DEC0_DEG')

    n_floored = int((radius < R_FLOOR_ARCMIN).sum())
    if n_floored:
        print(f'flooring {n_floored} star(s) with R < {R_FLOOR_ARCMIN} arcmin')

    R_proj = np.maximum(radius, R_FLOOR_ARCMIN) * dgal_kpc/arcmin   # kpc

    vsys, sigma_int = fit_vsys_and_sigma(velocity, error)

    path = 'Data/LeoII/LeoII_Rproj_vlos_vloserr.txt'
    np.savetxt(path, np.column_stack([R_proj, velocity - vsys, error]),
               header=f'R_proj[kpc]  v_los[km/s]  v_los_err[km/s]   '
                      f'Spencer+17 (VizieR J/AJ/153/254), Mm=Y members, '
                      f'N={len(members)}, vsys={vsys:.3f}, '
                      f'sigma_int={sigma_int:.3f}')

    print(f'\nN = {len(members)}, vsys = {vsys:7.2f} km/s, '
          f'sigma_int = {sigma_int:.2f} km/s, '
          f'R = {R_proj.min()*1e3:.1f} - {R_proj.max()*1e3:.1f} pc  -> {path}')

    print('\nNB Spencer+17 measure a 30-40% binary fraction for Leo II and quote a')
    print('   binary-corrected dispersion below the raw one; that correction is not')
    print('   applied here, so sigma_int above is an upper bound.')

    # --- diagnostic plot -----------------------------------------------------

    os.makedirs(plotdir, exist_ok=True)

    repeats = n_epochs > 1

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.errorbar(radius[~repeats], velocity[~repeats], yerr=error[~repeats],
                fmt='o', ms=4, mfc='none', color='0.35', ecolor='0.7',
                elinewidth=1, capsize=2, zorder=3, label='single epoch')
    ax.errorbar(radius[repeats], velocity[repeats], yerr=error[repeats],
                fmt='o', ms=4, color='k', ecolor='0.55',
                elinewidth=1, capsize=2, zorder=4, label='epoch-averaged')
    ax.axhline(vsys, color='C3', lw=1.2, zorder=2)
    ax.axhspan(vsys - sigma_int, vsys + sigma_int, color='C3', alpha=0.12, zorder=1)
    ax.set_xlabel(r"$R_{\rm proj}$ [arcmin]")
    ax.set_ylabel(r'$v_{\rm los}$ [km/s]')
    ax.set_title('Leo II members (Spencer+17, epoch-averaged)')
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(plotdir + 'data_vlos_R_leoII.pdf')
    plt.close(fig)

    print(f'\nplot written to {plotdir}data_vlos_R_leoII.pdf')


# --- proper motions (DATA_MODE = 'pm') ---------------------------------------

def build_pm_sample():
    '''
    Bulk-subtracted proper motions in km/s, split into radial (v_R) and
    tangential (v_T) components about (RA0_DEG, DEC0_DEG). pmRA is taken to be
    pmRA*cos(Dec), as VizieR publishes it. The field is a few arcmin across, so
    directions are taken on the flat tangent plane and perspective effects from
    Leo II's bulk motion are ignored.
    '''
    pm = read_proper_mot_table(proper_mot_data)

    good = (np.isfinite(pm['pmra']) & np.isfinite(pm['pmdec'])
            & np.isfinite(pm['pmra_err']) & np.isfinite(pm['pmdec_err']))
    print(f'{proper_mot_data}: {len(good)} stars, {good.sum()} with complete '
          f'proper motions')

    to_kms = K_PM*dgal_kpc
    v_ra, e_ra = pm['pmra']*to_kms, pm['pmra_err']*to_kms
    v_de, e_de = pm['pmdec']*to_kms, pm['pmdec_err']*to_kms

    # Bulk motion per component: joint MLE of mean and intrinsic spread.
    keep = good
    mean_ra, _ = fit_vsys_and_sigma(v_ra[keep], e_ra[keep])
    mean_de, _ = fit_vsys_and_sigma(v_de[keep], e_de[keep])

    print(f'bulk proper motion: pmRA = {mean_ra/to_kms:+.4f}, '
          f'pmDE = {mean_de/to_kms:+.4f} mas/yr')

    ra, dec = pm['ra'][keep], pm['dec'][keep]
    dv_ra, dv_de = v_ra[keep] - mean_ra, v_de[keep] - mean_de
    e_ra, e_de = e_ra[keep], e_de[keep]

    # Tangent-plane offsets (arcmin); only their direction is used.
    x = (ra - RA0_DEG)*np.cos(np.radians(DEC0_DEG))*60.0
    y = (dec - DEC0_DEG)*60.0

    radius = angular_separation_arcmin(ra, dec, RA0_DEG, DEC0_DEG)
    n_floored = int((radius < R_FLOOR_ARCMIN).sum())
    if n_floored:
        print(f'flooring {n_floored} star(s) with R < {R_FLOOR_ARCMIN} arcmin')
    radius = np.maximum(radius, R_FLOOR_ARCMIN)

    r_xy = np.maximum(np.hypot(x, y), R_FLOOR_ARCMIN)
    cos_t, sin_t = x/r_xy, y/r_xy

    v_R = cos_t*dv_ra + sin_t*dv_de
    v_T = cos_t*dv_de - sin_t*dv_ra
    v_R_err = np.hypot(cos_t*e_ra, sin_t*e_de)
    v_T_err = np.hypot(cos_t*e_de, sin_t*e_ra)

    R_proj = radius * dgal_kpc/arcmin   # kpc

    path = 'Data/LeoII/LeoII_Rproj_vR_vRerr_vT_vTerr.txt'
    np.savetxt(path, np.column_stack([R_proj, v_R, v_R_err, v_T, v_T_err]),
               header=f'R_proj[kpc]  v_R[km/s]  v_R_err[km/s]  v_T[km/s]  '
                      f'v_T_err[km/s]   HST proper motions, bulk motion '
                      f'(pmRA={mean_ra/to_kms:+.4f}, pmDE={mean_de/to_kms:+.4f} '
                      f'mas/yr) subtracted, D={dgal_kpc} kpc, '
                      f'N={keep.sum()}')

    print(f'\nN = {keep.sum()}, median error v_R = {np.median(v_R_err):.1f}, '
          f'v_T = {np.median(v_T_err):.1f} km/s, '
          f'R = {R_proj.min()*1e3:.1f} - {R_proj.max()*1e3:.1f} pc  -> {path}')

    # --- diagnostic plot -----------------------------------------------------

    os.makedirs(plotdir, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True)
    for ax, v, e, label in [(axes[0], v_R, v_R_err, r'$v_R$ [km/s]'),
                            (axes[1], v_T, v_T_err, r'$v_T$ [km/s]')]:
        ax.errorbar(radius, v, yerr=e, fmt='o', ms=3, color='k', ecolor='0.7',
                    elinewidth=1, capsize=0)
        ax.axhline(0.0, color='C3', lw=1.2)
        ax.set_ylabel(label)
    axes[1].set_xlabel(r"$R_{\rm proj}$ [arcmin]")
    axes[0].set_title('Leo II HST proper motions (bulk-subtracted)')
    fig.tight_layout()
    fig.savefig(plotdir + 'data_pm_R_leoII.pdf')
    plt.close(fig)

    print(f'\nplot written to {plotdir}data_pm_R_leoII.pdf')


if DATA_MODE == 'vlos':
    build_vlos_sample()
elif DATA_MODE == 'pm':
    build_pm_sample()
else:
    raise ValueError(f"DATA_MODE must be 'vlos' or 'pm', not {DATA_MODE!r}")
