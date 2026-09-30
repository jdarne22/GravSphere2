#Runs the parallelised dynesty sampler for Segue 1.
#Usage:  python run_Seg1.py [ncpus]      (resume with: python run_Seg1.py [ncpus] resume)

import os

#Pin the object before gravsphere2 chooses one, so this script cannot be made
#to sample another galaxy into Output/Segue1_*/ by whatever GS2_OBJECT happens
#to be set in the submitting shell.
os.environ['GS2_OBJECT'] = 'Seg1'

from gravsphere2 import *

from multiprocessing import Pool
import sys
import dynesty

ncpus = int(sys.argv[1]) if len(sys.argv) > 1 else len(os.sched_getaffinity(0))
resume = len(sys.argv) > 2 and sys.argv[2] == 'resume'

fsname = str(diro + 'Sampler Chains/segue1_chk')

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
    np.save(diro + 'Sampler Chains/segue1_samples.npy', res.samples)
    np.save(diro + 'Sampler Chains/segue1_logwt.npy', res.logwt)
    np.save(diro + 'Sampler Chains/segue1_logz.npy', res.logz)
    print('Done. logZ = %.2f +/- %.2f' % (res.logz[-1], res.logzerr[-1]))
