cd "$(dirname "$0")/.."
M="S1 S2 CL_entropy DbS_entropy MLBA_entropy SR_looks TLS_looks LBA MLBA_LBA Wald RWald SR_timing Ceiling S2+TLS S2+SR S2+CLH TLS_tfit Ceiling2 LBA_n Wald_n RWald_n MLBA_LBA_n S2+TLS_tfit S2+Wald_n"
D=/mnt/user-data/uploads/Updated_Choice_Overload.csv
for run in 1 2; do
python3 -I scripts/time_models_run.py --dev $D --stage vault --round final --sets snack dev_file krajbich2010 folke2016 --models $M --out ../runs/tm/vault$run > ../runs/tm/vault${run}a.log 2>&1 &
python3 -I scripts/time_models_run.py --dev $D --stage vault --round final --sets reutskaja cdm dev_laptop krajbich2011 --models $M --out ../runs/tm/vault$run > ../runs/tm/vault${run}b.log 2>&1 &
wait
done
wait
echo VAULT_DONE > ../runs/tm/status
