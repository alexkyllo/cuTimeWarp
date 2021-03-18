ncu --csv --section "MemoryWorkloadAnalysis" --section "Compute.*" \
    --section "Instruction.*" --section "Launch.*" \
    --section "Occupancy" --section "Warp.*" \
    --details-all ./bin/soft_dtw_perf_multi random 100 2 > output/ncu_100_2.csv
