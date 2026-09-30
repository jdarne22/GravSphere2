#Runs the parallelised dynesty sampler for Seg II.
#Usage:  python run_Seg2.py [ncpus]      (resume with: python run_Seg2.py [ncpus] resume)
#
#NB gravsphere2.py chooses its initialise file from GS2_OBJECT, which is pinned
#below before that import runs, so no file has to be edited before submitting.
#The guard further down stays as a belt-and-braces check; without it a Segue 1
#run would quietly write its chains into Output/LeoII/.

import os

os.environ['GS2_OBJECT'] = 'Seg2'

from gravsphere2 import *

from multiprocessing import Pool
import sys
import dynesty

#NB 'name' is the output-tree name from initialise_Seg2.py ('Segue2'), not the
#GS2_OBJECT key ('Seg2') -- comparing against the key always fails.
if name != 'Segue2':
    raise RuntimeError(
        f"gravsphere2.py imported the initialise file for {name!r}, not "
        "'Segue2'; check GS2_OBJECT is not being overridden in the environment.")

ncpus = int(sys.argv[1]) if len(sys.argv) > 1 else len(os.sched_getaffinity(0))
resume = len(sys.argv) > 2 and sys.argv[2] == 'resume'

fsname = str(diro + 'Sampler Chains/seg2_chk')

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
    np.save(diro + 'Sampler Chains/seg2_samples.npy', res.samples)
    np.save(diro + 'Sampler Chains/seg2_logwt.npy', res.logwt)
    np.save(diro + 'Sampler Chains/seg2_logz.npy', res.logz)
    print('Done. logZ = %.2f +/- %.2f' % (res.logz[-1], res.logzerr[-1]))
