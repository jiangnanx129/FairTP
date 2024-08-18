# python experiments/gwnet/main.py --device cuda:2 --dataset CA --years 2019 --model_name gwnet --seed 2018 --bs 32

# python experiments/gwnet/main.py --device cuda:2 --dataset GLA --years 2019 --model_name gwnet --seed 2018 --bs 64

# python experiments/gwnet/main.py --device cuda:2 --dataset GBA --years 2019 --model_name gwnet --seed 2018 --bs 64

# python experiments/gwnet/main.py --device cuda:2 --dataset SD --years 2019 --model_name gwnet --seed 2018 --bs 64


python /home/data/xjn/23largest_baseline/LargeST/experiments/gwnet/main_fair.py --device cuda:2 --dataset HK --years 202010 --model_name gwnet --seed 2018 --bs 64

6000

python /home/data/xjn/23largest_baseline/LargeST/experiments/gwnet/main_fairS.py --device cuda:0 --dataset HK2 --years 202010 --model_name gwnet --seed 2018 --bs 64
python /home/data/xjn/23largest_baseline/LargeST/experiments/gwnet/main_fairS_SD.py --device cuda:0 --dataset SD --years 20191w --model_name gwnet --seed 2018 --bs 64

python /home/data/xjn/23largest_baseline/LargeST/experiments/gwnet/main_fair3_T5.py --device cuda:0 --dataset HK2 --years 202010 --model_name gwnet --seed 2018 --bs 64
python /home/data/xjn/23largest_baseline/LargeST/experiments/gwnet/main_fair3_T5_SD.py --device cuda:0 --dataset SD --years 20191w --model_name gwnet --seed 2018 --bs 64

python /home/data/xjn/23largest_baseline/LargeST/experiments/gwnet/main_fairS_all.py --device cuda:2 --dataset HKALL --years 202010 --model_name gwnet --seed 2018 --bs 64
python /home/data/xjn/23largest_baseline/LargeST/experiments/gwnet/main_fairS_SD_all.py --device cuda:2 --dataset HKALLSD --years 20191w --model_name gwnet --seed 2018 --bs 64
