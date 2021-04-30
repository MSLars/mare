
python mare/evaluation/evaluation_runner.py --model-path $1 --test-data data/smart_data_test.jsonl --inc mare --inc spart --output-dir $2 --predictor $3

python mare/evaluation/extract_stats.py $2/store.json