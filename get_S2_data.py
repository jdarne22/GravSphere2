'''
Build the Segue 2 input files for GravSphere2 from the published Kirby et al.
(2013) catalogue: ApJ 770, 16, table 2 (VizieR J/ApJ/770/16).

The table is laid out nothing like Simon+11's Segue 1 one, so this is not a
line-for-line copy of get_S1_data.py:

  * one row per star, not one per epoch -- Nm counts the exposures Kirby+13
    already combined, so there is nothing here to inverse-variance average;
  * no Rad column, so the projected radius is computed from _RA/_DE against
    the adopted centre rather than read off;
  * Mm is a letter flag, not 0/1: Y = member, B = the four blue horizontal
    branch members (g - i < 0, all within a few km/s of systemic), N = rejected,
    ? = spectrum unusable;
  * no Bayesian membership probability column (Simon+11's Bpr), so the
    keep/drop split below is made on the catalogue's own reason flags.

Y + B is 26 members: 21 red giants/subgiants, 4 blue horizontal branch, and

    J021900.06+200635.2   an RR Lyrae. Kirby+13 exclude it from the dispersion
                          because it pulsates -- its radial velocity varies by
                          50-70 km/s -- so the -52.4 km/s in this table is a
                          phase, not a systemic offset, and it carries no
                          information about the potential. It is the only member
                          carrying a "reason" flag (CMD,v_r_), which is how it
                          can be picked out of the table.

Removing it leaves the 25-star kinematic sample Kirby+13 analyse. Unlike Segue
1 there is no ambiguous star here, so no keep/drop variants: one file out.

NB there is no binned star-count profile for Segue 2 in Data/Seg2/, so the
photometry block at the end of get_S1_data.py has no counterpart here and
Option B (individual = False) is not available -- see the note printed at the
end of this script.
'''

import os

import numpy as np
from scipy.optimize import minimize

import constants


arcmin   = constants.arcmin      # arcmin -> rad
dgal_kpc = 35.0                  # distance to Segue 2 [kpc]       (Kirby+13)

Kirby_data = 'Data/Seg2/SegueII.tsv'

# Adopted centre, Belokurov+09 as used by Kirby+13: 02h19m16s, +20d10'31".
RA0_DEG  = (2.0 + 19.0/60.0 + 16.0/3600.0) * 15.0
DEC0_DEG = 20.0 + 10.0/60.0 + 31.0/3600.0

# Which Mm flags count as members. 'B' are the blue horizontal branch stars;
# they carry the same kinematic information as the giants, so they are in.
MEMBER_FLAGS = ('Y', 'B')

RR_LYRAE      = 'J021900.06+200635.2'     # pulsating; see the header note
ALWAYS_REMOVE = [RR_LYRAE]

# Radii here are computed from coordinates rather than read from a column
# quantised to 0.1 arcmin, so an exact 0.0 is unlikely -- but R = 0 is still
# fatal downstream (it sets rmin = 0, log10 -> -inf on the Jeans grid, and makes
# the individual-star term log(R*Sigma) equal -inf, so every likelihood call
# returns -inf), so keep the floor as a guard.
R_FLOOR_ARCMIN = 0.05


# --- reading the catalogue ---------------------------------------------------

def read_kirby_table(path):
    '''
    VizieR tab-separated dump of table 2: a header row of column names, a units
    row, a row of dashes, then the data. Data rows are recognised by the SDSS
    name in column 0; blank cells are common and become nan.
    '''
    header = None
    data_rows = []

    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        fields = line.rstrip('\n').split('\t')
        if fields[0].strip() == 'SDSS':
            header = [f.strip() for f in fields]
        elif fields[0].strip().startswith('J'):
            data_rows.append(fields)

    if header is None:
        raise ValueError(f'no header row found in {path}')

    def column(name):
        i = header.index(name)
        return np.array([float(r[i]) if r[i].strip() else np.nan for r in data_rows])

    def text_column(name):
        i = header.index(name)
        return np.array([r[i].strip() for r in data_rows])

    return dict(star_id=text_column('SDSS'), velocity=column('HRV'),
                error=column('e_HRV'), ra=column('_RA'), dec=column('_DE'),
                g_mag=column('gmag'), i_mag=column('imag'),
                n_meas=column('Nm'), member_flag=text_column('Mm'),
                reason=text_column('reason'), feh=column('[Fe/H]'))


# --- projected radius --------------------------------------------------------

def projected_radius_arcmin(ra_deg, dec_deg):
    '''
    Angular separation from the adopted centre, in arcmin. The field is only
    ~20 arcmin across so the tangent-plane form is good to much better than the
    0.1 arcmin Simon+11 published for Segue 1.
    '''
    dx = (ra_deg - RA0_DEG) * np.cos(np.radians(0.5*(dec_deg + DEC0_DEG)))
    dy = dec_deg - DEC0_DEG
    return np.hypot(dx, dy) * 60.0


# --- systemic velocity and intrinsic dispersion ------------------------------

def neg_lnlike(params, velocity, error):
    vsys, ln_sigma = params
    variance = error**2 + np.exp(2*ln_sigma)
    return 0.5*np.sum(np.log(2*np.pi*variance) + (velocity - vsys)**2/variance)


def fit_vsys_and_sigma(velocity, error):
    '''
    Joint MLE for the systemic velocity and the intrinsic dispersion.

    A plain mean is wrong because the per-star errors span 2-22 km/s. Segue 2's
    dispersion is unresolved (Kirby+13 quote sigma < 2.2 km/s at 90%), so the
    likelihood is flat towards ln_sigma -> -inf and the returned sigma can run
    to essentially zero: read it as an upper limit, not a measurement.
    '''
    fit = minimize(neg_lnlike, [np.mean(velocity), np.log(2.0)],
                   args=(velocity, error), method='Nelder-Mead',
                   options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 20000})
    return fit.x[0], np.exp(fit.x[1])


# --- one row per member star -------------------------------------------------

catalogue = read_kirby_table(Kirby_data)

is_member = np.isin(catalogue['member_flag'], MEMBER_FLAGS)

members = catalogue['star_id'][is_member]
if len(members) != len(set(members)):
    raise ValueError('a star appears on more than one row')

flags, counts = np.unique(catalogue['member_flag'], return_counts=True)
flag_counts = {str(f): int(n) for f, n in zip(flags, counts)}
print(f'{Kirby_data}: {len(catalogue["star_id"])} rows, '
      f'Mm flags {flag_counts}')
print(f'members = Mm in {MEMBER_FLAGS}: {is_member.sum()} stars '
      f'({int((catalogue["member_flag"][is_member] == "B").sum())} blue '
      f'horizontal branch)')

velocity    = catalogue['velocity'][is_member]
error       = catalogue['error'][is_member]
radius      = projected_radius_arcmin(catalogue['ra'][is_member],
                                      catalogue['dec'][is_member])
n_meas      = catalogue['n_meas'][is_member]
reason      = catalogue['reason'][is_member]
colour      = catalogue['g_mag'][is_member] - catalogue['i_mag'][is_member]

if not np.all(np.isfinite(velocity) & np.isfinite(error)):
    bad = members[~(np.isfinite(velocity) & np.isfinite(error))]
    raise ValueError(f'member(s) with no velocity or no error: {list(bad)}')

repeats = int(np.nansum(n_meas > 1))
n_vals, n_counts = np.unique(n_meas[np.isfinite(n_meas)].astype(int),
                             return_counts=True)
print(f'exposures per member: {dict(zip(n_vals.tolist(), n_counts.tolist()))}'
      f'  -> {repeats} stars combine more than one, done by Kirby+13')


# --- build and write the two samples -----------------------------------------

n_floored = int((radius < R_FLOOR_ARCMIN).sum())
if n_floored:
    print(f'\nflooring {n_floored} star(s) with R < {R_FLOOR_ARCMIN} arcmin')

R_proj = np.maximum(radius, R_FLOOR_ARCMIN) * dgal_kpc/arcmin   # kpc

removed = np.isin(members, ALWAYS_REMOVE)
for star in ALWAYS_REMOVE:
    match = np.where(members == star)[0]
    if not len(match):
        raise ValueError(f'{star} is not among the members -- check the ID')
    k = match[0]
    print(f'\nremoved ({removed.sum()} star, pulsating -- carries no mass '
          f'information):')
    print(f"   {star}  R = {radius[k]:4.1f}'  "
          f'v = {velocity[k]:6.1f} +/- {error[k]:4.1f}  '
          f'(g-i = {colour[k]:+.2f}, catalogue reason = "{reason[k]}")')

rrl = np.where(members == RR_LYRAE)[0][0]

# What the RR Lyrae would do if it were left in. Not written out -- this is
# only here to show that the removal is not cosmetic: one pulsating star is
# the whole difference between an unresolved dispersion and an apparent 2.4
# km/s detection, which is the entire mass measurement for this galaxy.
vsys_with, sigma_with = fit_vsys_and_sigma(velocity, error)

selected = ~removed
v, e, R = velocity[selected], error[selected], R_proj[selected]
vsys, sigma_int = fit_vsys_and_sigma(v, e)

path = 'Data/Seg2/Seg2_Rproj_vlos_vloserr.txt'
np.savetxt(path, np.column_stack([R, v - vsys, e]),
           header=f'R_proj[kpc]  v_los[km/s]  v_los_err[km/s]   '
                  f'Kirby+13 table 2, RR Lyrae removed, '
                  f'N={selected.sum()}, vsys={vsys:.3f}, '
                  f'sigma_int={sigma_int:.3f}')

print(f'\nwith the RR Lyrae:    N = {len(velocity):2d}, '
      f'vsys = {vsys_with:7.2f} km/s, sigma_int = {sigma_with:.2f} km/s')
print(f'without (written):    N = {selected.sum():2d}, '
      f'vsys = {vsys:7.2f} km/s, sigma_int = {sigma_int:.2f} km/s, '
      f'R = {R.min()*1e3:.1f} - {R.max()*1e3:.1f} pc\n  -> {path}')

print('\nNB Kirby+13 measure vsys = -40.2 +/- 0.9 km/s and do NOT resolve the')
print('   dispersion: sigma_v < 2.2 (2.6) km/s at 90% (95%) confidence, on the')
print('   same 25 stars written above. The sigma_int printed here is a maximum-')
print('   likelihood point estimate on a likelihood that is flat towards zero,')
print('   so it is an upper limit, not a detection -- expect GravSphere2 to')
print('   return upper limits on M(<r) rather than a measured mass.')


# --- diagnostic plot ---------------------------------------------------------
# Agg because this runs headless under a PBS job.

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plotdir = 'Output/Segue2/Plots/'
os.makedirs(plotdir, exist_ok=True)

kept = ~removed
is_bhb = catalogue['member_flag'][is_member] == 'B'

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.errorbar(radius[kept & ~is_bhb], velocity[kept & ~is_bhb],
            yerr=error[kept & ~is_bhb], fmt='o', ms=4,
            color='k', ecolor='0.55', elinewidth=1, capsize=2, zorder=3,
            label='red giants / subgiants')
ax.errorbar(radius[kept & is_bhb], velocity[kept & is_bhb],
            yerr=error[kept & is_bhb], fmt='^', ms=6,
            color='C2', ecolor='0.55', elinewidth=1, capsize=2, zorder=4,
            label='blue horizontal branch (Mm = B)')
ax.errorbar(radius[rrl], velocity[rrl], yerr=error[rrl], fmt='D', ms=8,
            mfc='none', color='C3', elinewidth=1.5, capsize=3, zorder=5,
            label='RR Lyrae (removed)')
ax.axhline(vsys, color='C3', lw=1.2, zorder=2)
ax.axhspan(vsys - 2.2, vsys + 2.2, color='C3', alpha=0.12, zorder=1)
ax.set_xlabel(r"$R_{\rm proj}$ [arcmin]")
ax.set_ylabel(r'$v_{\rm los}$ [km/s]')
ax.set_title(r'Segue 2 members (Kirby+13 table 2); band is $\sigma_v < 2.2$ km/s')
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(plotdir + 'data_vlos_R_segue2.pdf')
plt.close(fig)

print(f'\nplot written to {plotdir}data_vlos_R_segue2.pdf')


# --- photometry --------------------------------------------------------------
# get_S1_data.py ends by fitting a Plummer to the binned star-count profile in
# SegIcut.txt and writing segue1_surfden.txt, which initialise_Seg1.py uses for
# Option B (individual = False). Data/Seg2/ has no such profile -- the only file
# there is the spectroscopic catalogue -- so there is nothing to fit and no
# surface-density file is written. Run Segue 2 with individual = True, and use
# the half-light radius below for the light-profile priors.

RHALF_ARCMIN = 3.4                              # Kirby+13, +/- 0.2
RHALF_KPC    = RHALF_ARCMIN * dgal_kpc/arcmin   # = 34 +/- 3 pc

print(f'\nno binned photometry in Data/Seg2/: Option B (individual = False) is '
      f'not available for Segue 2.\nUse individual = True with '
      f'Rhalf = {RHALF_KPC:.4f} kpc ({RHALF_KPC*1e3:.1f} pc, '
      f"Kirby+13 R_half = {RHALF_ARCMIN}' at d = {dgal_kpc:.0f} kpc).")
