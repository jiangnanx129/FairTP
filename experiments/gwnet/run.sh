# 全数据
python /home/data/xjn/23largest_baseline/LargeST/experiments/gwnet/main_fairS_all.py --device cuda:3 --dataset HKALL --years 202010 --model_name gwnet --seed 2018 --bs 64 --mode test
python /home/data/xjn/23largest_baseline/LargeST/experiments/gwnet/main_fairS_SD_all.py --device cuda:3 --dataset HKALLSD --years 20191w --model_name gwnet --seed 2018 --bs 64 --mode test

# our methods
python /home/data/xjn/23largest_baseline/LargeST/experiments/gwnet/main_fair3_T5.py --device cuda:3 --dataset HK2 --years 202010 --model_name gwnet --seed 2018 --bs 64 --patience 15 --mode test
python /home/data/xjn/23largest_baseline/LargeST/experiments/gwnet/main_fair3_T5_SD.py --device cuda:3 --dataset SD --years 20191w --model_name gwnet --seed 2018 --bs 64 --mode test
