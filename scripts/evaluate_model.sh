
python mare/evaluation/evaluation_runner.py --model-path $1 --test-data data/smart_data_test.jsonl --inc $4 --inc spart --output-dir $2 --predictor $3 -f

python mare/evaluation/extract_stats.py $2/store.json