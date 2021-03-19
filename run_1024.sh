for i in {1..20}
do
    ./bin/soft_dtw_perf_multi random 1024 $i >> output/random_1024.txt
done

for i in {1..20}
do
    ./bin/soft_dtw_perf_cpu random 1024 $i >> output/random_1024.txt
done
