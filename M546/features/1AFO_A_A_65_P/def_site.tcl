mol new {1AFO_MT_old.pdb} type {pdb} first 0 last 0 step 1 waitfor 1
set prot2 [atomselect top "within 12.0 of resid 65 and chain A"]
$prot2 writepdb 1AFO_MT.pdb
exit