#!/mnt/home/chenj159/anaconda3/bin/python
import sys, os
import numpy as np
from structure import get_structure
from protein import protein
from protein import construct_features_PH0
from protein import construct_features_PH12
from protein import construct_feature_aux

# 1DVF AB CD D D 52 A 7.0
#          arguement    example
PDBid    = sys.argv[1] # 1KBH
Chains   = sys.argv[2] # A
Chain    = sys.argv[3] # A
resWT    = sys.argv[4] # L
resID    = sys.argv[5] # 37
resMT    = sys.argv[6] # W
pH       = sys.argv[7] # 7.0

flag_BLAST = False
flag_MIBPB = False

s = get_structure(PDBid, Chains, Chain, resWT, resID, resMT, pH=pH)
s.generateMutedPDBs()
s.generateMutedPQRs()
s.readFASTA()
#########################################################################################
c_WT = protein(s, 'WT')
c_WT_Lap_b = c_WT.rips_complex_spectra()
#c_WT_Lap_m = c_WT.rips_complex_spectra(c_WT.atoms_m_m, c_WT.atoms_m_o)
#c_WT_Lap_s = c_WT.rips_complex_spectra(c_WT.atoms_m_m, c_WT.atoms_m_s)
#----------------------------------------------------------------------------------------
c_MT = protein(s, 'MT')
c_MT_Lap_b = c_MT.rips_complex_spectra()
#c_MT_Lap_m = c_MT.rips_complex_spectra(c_MT.atoms_m_m, c_MT.atoms_m_o)
#c_MT_Lap_s = c_MT.rips_complex_spectra(c_MT.atoms_m_m, c_MT.atoms_m_s)
#----------------------------------------------------------------------------------------
feature_Lap_b = np.concatenate((c_MT_Lap_b, c_WT_Lap_b), axis=0)
feature_Lap_b = np.concatenate((feature_Lap_b, c_MT_Lap_b-c_WT_Lap_b), axis=0)
#feature_Lap_m = np.concatenate((c_MT_Lap_m, c_WT_Lap_m), axis=0)
#feature_Lap_m = np.concatenate((feature_Lap_m, c_MT_Lap_m-c_WT_Lap_m), axis=0)
#feature_Lap_s = np.concatenate((c_MT_Lap_s, c_WT_Lap_s), axis=0)
#feature_Lap_s = np.concatenate((feature_Lap_s, c_MT_Lap_s-c_WT_Lap_s), axis=0)
#feature_Lap   = np.concatenate((feature_Lap_b, feature_Lap_m), axis=0)
#feature_Lap   = np.concatenate((feature_Lap,   feature_Lap_s), axis=0)
#----------------------------------------------------------------------------------------
feature_Lap_b_inv = np.concatenate((c_WT_Lap_b, c_MT_Lap_b), axis=0)
feature_Lap_b_inv = np.concatenate((feature_Lap_b_inv, c_WT_Lap_b-c_MT_Lap_b), axis=0)
#feature_Lap_m_inv = np.concatenate((c_WT_Lap_m, c_MT_Lap_m), axis=0)
#feature_Lap_m_inv = np.concatenate((feature_Lap_m_inv, c_WT_Lap_m-c_MT_Lap_m), axis=0)
#feature_Lap_s_inv = np.concatenate((c_WT_Lap_s, c_MT_Lap_s), axis=0)
#feature_Lap_s_inv = np.concatenate((feature_Lap_s_inv, c_WT_Lap_s-c_MT_Lap_s), axis=0)
#feature_Lap_inv   = np.concatenate((feature_Lap_b_inv, feature_Lap_m_inv), axis=0)
#feature_Lap_inv   = np.concatenate((feature_Lap_inv,   feature_Lap_s_inv), axis=0)
#########################################################################################
print('Lap feature size: ', feature_Lap_b.shape)

filename = PDBid+'_'+Chain+'_'+resWT+'_'+resID+'_'+resMT
OutFile = open(filename+'_Lap_b.npy', 'wb')
np.save(OutFile, feature_Lap_b)
OutFile.close()

filename_inv = PDBid+'_'+Chain+'_'+resMT+'_'+resID+'_'+resWT
OutFile = open(filename_inv+'_Lap_b.npy', 'wb')
np.save(OutFile, feature_Lap_b_inv)
OutFile.close()
#########################################################################################
