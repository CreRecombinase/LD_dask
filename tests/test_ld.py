import allel
import h5py as h5
import numpy as np

import lddask


class Testmap(object):

    def test_theta(self):
        assert (np.allclose(np.array([0.001029112]), lddask.ld.calc_theta(85)))

    def test_ld(self):
        ''' unit test for ldshrink '''

        input_hdf = "/home/nwknoblauch/Dropbox/Repos/LD_dask/test_data/reference_genotype.h5"
        callset = h5.File(input_hdf, mode='r')
        ref_geno = allel.GenotypeDaskArray(callset['calldata/GT'])
        vt = allel.VariantChunkedTable(callset['variants'])
        map_data = vt['MAP']
        geno_ac = ref_geno.to_n_alt().T.compute()
        m = 85
        Ne = 11490.672741
        cutoff = 0.001
        test_R_file = "test_data/reference_ld.txt"
        sub_X = geno_ac[:, :4]
        sub_map = map_data[:4]
        est_r = lddask.ld.ldshrink(sub_X, sub_map, m, Ne, cutoff)
        true_r = np.loadtxt(test_R_file, delimiter="\t")
        sub_est_r = true_r[:4, :4]
        assert (np.allclose(true_r[:4, :4], est_r))
