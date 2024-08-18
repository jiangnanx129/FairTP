# python experiments/stgode/main.py --device cuda:2 --dataset CA --years 2019 --model_name stgode --seed 2018 --bs 12

# python experiments/stgode/main.py --device cuda:2 --dataset GLA --years 2019 --model_name stgode --seed 2018 --bs 28

# python experiments/stgode/main.py --device cuda:2 --dataset GBA --years 2019 --model_name stgode --seed 2018 --bs 48

# python experiments/stgode/main.py --device cuda:2 --dataset SD --years 2019 --model_name stgode --seed 2018 --bs 64


python /home/data/xjn/23largest_baseline/LargeST/experiments/stgode/main_fair.py --device cuda:0 --dataset HK --years 202010 --model_name stgode --seed 2018 --bs 64


python /home/data/xjn/23largest_baseline/LargeST/experiments/stgode/main.py --device cuda:2 --dataset HK2 --years 202010 --model_name stgode --seed 2018 --bs 48

python /home/data/xjn/23largest_baseline/LargeST/experiments/stgode/main_fairS.py --device cuda:1 --dataset HK2 --years 202010 --model_name stgode --seed 2018 --bs 48
python /home/data/xjn/23largest_baseline/LargeST/experiments/stgode/main_fairS_SD.py --device cuda:2 --dataset SD --years 20191w --model_name stgode --seed 2018 --bs 48

python /home/data/xjn/23largest_baseline/LargeST/experiments/stgode/main_fair3_T5.py --device cuda:2 --dataset HK2 --years 202010 --model_name stgode --seed 2018 --bs 48  # 21569
python /home/data/xjn/23largest_baseline/LargeST/experiments/stgode/main_fair3_T5_SD.py --device cuda:1 --dataset SD --years 20191w --model_name stgode --seed 2018 --bs 48  # 21569

python /home/data/xjn/23largest_baseline/LargeST/experiments/stgode/main_fairS_all.py --device cuda:2 --dataset HKALL --years 202010 --model_name stgode --seed 2018 --bs 48
python /home/data/xjn/23largest_baseline/LargeST/experiments/stgode/main_fairS_SD_all.py --device cuda:3 --dataset HKALLSD --years 20191w --model_name stgode --seed 2018 --bs 48
