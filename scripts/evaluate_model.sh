
python mare/evaluation/evaluation_runner.py --model-path $1 --test-data https://fh-aachen.sciebo.de/s/9ghU4Qi1azUMFPW/download --inc mare --output-dir $2 --predictor $3

python mare/evaluation/extract_stats.py $2/store.json