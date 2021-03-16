for i in {10..200..10}
do
    for j in {10..200..10}
    do
        ./bin/soft_dtw_perf_multi random $i $j >> output/random.txt
    done
done
