#Runs the parallelised dynesty sampler for Leo II.
#Usage:  python run_LeoII.py [ncpus]      (resume with: python run_LeoII.py [ncpus] resume)
#
#NB gravsphere2.py chooses its initialise file from GS2_OBJECT, which is pinned
#below before that import runs, so no file has to be edited before submitting.
#The guard further down stays as a belt-and-braces check; without it a Segue 1
#run would quietly write its chains into Output/LeoII/.

import os

os.environ['GS2_OBJECT'] = 'LeoII'

from gravsphere2 import *

from multiprocessing import Pool
import sys
import dynesty

if name != 'LeoII':
    raise RuntimeError(
        f"gravsphere2.py is set up for {name!r}, not 'LeoII'. Uncomment "
        "'from initialise_LeoII import *' at the top of gravsphere2.py.")

ncpus = int(sys.argv[1]) if len(sys.argv) > 1 else len(os.sched_getaffinity(0))
resume = len(sys.argv) > 2 and sys.argv[2] == 'resume'

fsname = str(diro + 'Sampler Chains/leoII_chk')

if __name__ == '__main__':

    print('Running dynesty on %d tracers with %d CPUs, ndims = %d' %
          (len(rLOS), ncpus, ndims))
    print('Checkpoint file: ' + fsname)

    pool = Pool(ncpus)

    if resume and os.path.exists(fsname):
        dns = dynesty.DynamicNestedSampler.restore(fsname, pool=pool)
        dns.run_nested(checkpoint_file=fsname, use_stop=False, resume=True)
    else:
        dns = dynesty.DynamicNestedSampler(
            lnprob, ptform, ndims, pool=pool, queue_size=ncpus, nlive=500)
        dns.run_nested(checkpoint_file=fsname, use_stop=True)

    res = dns.results
    np.save(diro + 'Sampler Chains/leoII_samples.npy', res.samples)
    np.save(diro + 'Sampler Chains/leoII_logwt.npy', res.logwt)
    np.save(diro + 'Sampler Chains/leoII_logz.npy', res.logz)
    print('Done. logZ = %.2f +/- %.2f' % (res.logz[-1], res.logzerr[-1]))
