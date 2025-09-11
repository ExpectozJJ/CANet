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
p_WT = protein(s, 'WT')
c_WT_vr_curves, c_WT_vr_rates = p_WT.rips_complex_sr()
c_WT_alpha_curves, c_WT_alpha_rates = p_WT.alpha_complex_sr()
c_WT_vr_fcurves, c_WT_vr_frates = p_WT.rips_complex_fvector()
c_WT_alpha_fcurves, c_WT_alpha_frates = p_WT.alpha_complex_fvector()
#----------------------------------------------------------------------------------------
p_MT = protein(s, 'MT')
c_MT_vr_curves, c_MT_vr_rates = p_MT.rips_complex_sr()
c_MT_alpha_curves, c_MT_alpha_rates = p_MT.alpha_complex_sr()
c_MT_vr_fcurves, c_MT_vr_frates = p_MT.rips_complex_fvector()
c_MT_alpha_fcurves, c_MT_alpha_frates = p_MT.alpha_complex_fvector()
#----------------------------------------------------------------------------------------
feature_SR_curves = np.concatenate((c_MT_vr_curves.flatten(), c_WT_vr_curves.flatten()), axis=0)
feature_SR_curves = np.concatenate((feature_SR_curves, c_MT_vr_curves.flatten()-c_WT_vr_curves.flatten()), axis=0)
feature_SR_curves = np.concatenate((feature_SR_curves, c_MT_alpha_curves.flatten()), axis=0)
feature_SR_curves = np.concatenate((feature_SR_curves, c_WT_alpha_curves.flatten()), axis=0)
feature_SR_curves = np.concatenate((feature_SR_curves, c_MT_alpha_curves.flatten()-c_WT_alpha_curves.flatten()), axis=0)

feature_SR_rates = np.concatenate((c_MT_vr_rates.flatten(), c_WT_vr_rates.flatten()), axis=0)
feature_SR_rates = np.concatenate((feature_SR_rates, c_MT_vr_rates.flatten()-c_WT_vr_rates.flatten()), axis=0)
feature_SR_rates = np.concatenate((feature_SR_rates, c_MT_alpha_rates.flatten()), axis=0)
feature_SR_rates = np.concatenate((feature_SR_rates, c_WT_alpha_rates.flatten()), axis=0)
feature_SR_rates = np.concatenate((feature_SR_rates, c_MT_alpha_rates.flatten()-c_WT_alpha_rates.flatten()), axis=0)

feature_SR_fcurves = np.concatenate((c_MT_vr_fcurves.flatten(), c_WT_vr_fcurves.flatten()), axis=0)
feature_SR_fcurves = np.concatenate((feature_SR_fcurves, c_MT_vr_fcurves.flatten()-c_WT_vr_fcurves.flatten()), axis=0)
feature_SR_fcurves = np.concatenate((feature_SR_fcurves, c_MT_alpha_fcurves.flatten()), axis=0)
feature_SR_fcurves = np.concatenate((feature_SR_fcurves, c_WT_alpha_fcurves.flatten()), axis=0)
feature_SR_fcurves = np.concatenate((feature_SR_fcurves, c_MT_alpha_fcurves.flatten()-c_WT_alpha_fcurves.flatten()), axis=0)

feature_SR_frates = np.concatenate((c_MT_vr_frates.flatten(), c_WT_vr_frates.flatten()), axis=0)
feature_SR_frates = np.concatenate((feature_SR_frates, c_MT_vr_frates.flatten()-c_WT_vr_frates.flatten()), axis=0)
feature_SR_frates = np.concatenate((feature_SR_frates, c_MT_alpha_frates.flatten()), axis=0)
feature_SR_frates = np.concatenate((feature_SR_frates, c_WT_alpha_frates.flatten()), axis=0)
feature_SR_frates = np.concatenate((feature_SR_frates, c_MT_alpha_frates.flatten()-c_WT_alpha_frates.flatten()), axis=0)
#----------------------------------------------------------------------------------------
feature_SR_curves_inv = np.concatenate((c_WT_vr_curves.flatten(), c_MT_vr_curves.flatten()), axis=0)
feature_SR_curves_inv = np.concatenate((feature_SR_curves, c_WT_vr_curves.flatten()-c_MT_vr_curves.flatten()), axis=0)
feature_SR_curves_inv = np.concatenate((feature_SR_curves, c_WT_alpha_curves.flatten()), axis=0)
feature_SR_curves_inv = np.concatenate((feature_SR_curves, c_MT_alpha_curves.flatten()), axis=0)
feature_SR_curves_inv = np.concatenate((feature_SR_curves, c_WT_alpha_curves.flatten()-c_MT_alpha_curves.flatten()), axis=0)

feature_SR_rates_inv = np.concatenate((c_WT_vr_rates.flatten(), c_MT_vr_rates.flatten()), axis=0)
feature_SR_rates_inv = np.concatenate((feature_SR_rates, c_WT_vr_rates.flatten()-c_MT_vr_rates.flatten()), axis=0)
feature_SR_rates_inv = np.concatenate((feature_SR_rates, c_WT_alpha_rates.flatten()), axis=0)
feature_SR_rates_inv = np.concatenate((feature_SR_rates, c_MT_alpha_rates.flatten()), axis=0)
feature_SR_rates_inv = np.concatenate((feature_SR_rates, c_WT_alpha_rates.flatten()-c_MT_alpha_rates.flatten()), axis=0)

feature_SR_fcurves_inv = np.concatenate((c_WT_vr_fcurves.flatten(), c_MT_vr_fcurves.flatten()), axis=0)
feature_SR_fcurves_inv = np.concatenate((feature_SR_fcurves, c_WT_vr_fcurves.flatten()-c_MT_vr_fcurves.flatten()), axis=0)
feature_SR_fcurves_inv = np.concatenate((feature_SR_fcurves, c_WT_alpha_fcurves.flatten()), axis=0)
feature_SR_fcurves_inv = np.concatenate((feature_SR_fcurves, c_MT_alpha_fcurves.flatten()), axis=0)
feature_SR_fcurves_inv = np.concatenate((feature_SR_fcurves, c_WT_alpha_fcurves.flatten()-c_MT_alpha_fcurves.flatten()), axis=0)

feature_SR_frates_inv = np.concatenate((c_WT_vr_frates.flatten(), c_MT_vr_frates.flatten()), axis=0)
feature_SR_frates_inv = np.concatenate((feature_SR_frates, c_WT_vr_frates.flatten()-c_MT_vr_frates.flatten()), axis=0)
feature_SR_frates_inv = np.concatenate((feature_SR_frates, c_WT_alpha_frates.flatten()), axis=0)
feature_SR_frates_inv = np.concatenate((feature_SR_frates, c_MT_alpha_frates.flatten()), axis=0)
feature_SR_frates_inv = np.concatenate((feature_SR_frates, c_WT_alpha_frates.flatten()-c_MT_alpha_frates.flatten()), axis=0)
#########################################################################################
print('SR-curves feature size: ', feature_SR_curves.shape)

filename = PDBid+'_'+Chain+'_'+resWT+'_'+resID+'_'+resMT
OutFile = open(filename+'_SR_curves.npy', 'wb')
np.save(OutFile, feature_SR_curves)
OutFile.close()

filename_inv = PDBid+'_'+Chain+'_'+resMT+'_'+resID+'_'+resWT
OutFile = open(filename_inv+'_SR_curves.npy', 'wb')
np.save(OutFile, feature_SR_curves_inv)
OutFile.close()
#########################################################################################
print('SR-rates feature size: ', feature_SR_rates.shape)

filename = PDBid+'_'+Chain+'_'+resWT+'_'+resID+'_'+resMT
OutFile = open(filename+'_SR_rates.npy', 'wb')
np.save(OutFile, feature_SR_rates)
OutFile.close()

filename_inv = PDBid+'_'+Chain+'_'+resMT+'_'+resID+'_'+resWT
OutFile = open(filename_inv+'_SR_rates.npy', 'wb')
np.save(OutFile, feature_SR_rates_inv)
OutFile.close()
#########################################################################################
print('SR-fcurves feature size: ', feature_SR_fcurves.shape)

filename = PDBid+'_'+Chain+'_'+resWT+'_'+resID+'_'+resMT
OutFile = open(filename+'_SR_fcurves.npy', 'wb')
np.save(OutFile, feature_SR_fcurves)
OutFile.close()

filename_inv = PDBid+'_'+Chain+'_'+resMT+'_'+resID+'_'+resWT
OutFile = open(filename_inv+'_SR_fcurves.npy', 'wb')
np.save(OutFile, feature_SR_fcurves_inv)
OutFile.close()
#########################################################################################
print('SR-frates feature size: ', feature_SR_frates.shape)

filename = PDBid+'_'+Chain+'_'+resWT+'_'+resID+'_'+resMT
OutFile = open(filename+'_SR_frates.npy', 'wb')
np.save(OutFile, feature_SR_frates)
OutFile.close()

filename_inv = PDBid+'_'+Chain+'_'+resMT+'_'+resID+'_'+resWT
OutFile = open(filename_inv+'_SR_frates.npy', 'wb')
np.save(OutFile, feature_SR_frates_inv)
OutFile.close()
#########################################################################################
