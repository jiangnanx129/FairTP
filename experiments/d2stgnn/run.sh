# python experiments/d2stgnn/main.py --device cuda:2 --dataset GLA --years 2019 --model_name d2stgnn --seed 2018 --max_epochs 80 --patience 80 --bs 4 --layer 1

# python experiments/d2stgnn/main.py --device cuda:2 --dataset GBA --years 2019 --model_name d2stgnn --seed 2018 --max_epochs 80 --patience 80 --bs 4

# python experiments/d2stgnn/main.py --device cuda:2 --dataset SD --years 2019 --model_name d2stgnn --seed 2018 --max_epochs 80 --patience 80 --bs 36

python /home/data/xjn/23largest_baseline/LargeST/experiments/d2stgnn/main_fair.py --device cuda:3 --dataset SD --years 2019 --model_name d2stgnn --seed 2018 --patience 80 --bs 36
python /home/data/xjn/23largest_baseline/LargeST/experiments/d2stgnn/main_fair.py --device cuda:3 --dataset SD --years 20191w --model_name d2stgnn --seed 2018 --patience 80 --bs 64

python /home/data/xjn/23largest_baseline/LargeST/experiments/d2stgnn/main_fair.py --device cuda:3 --dataset HK2 --years 202010 --model_name d2stgnn --seed 2018 --patience 80 --bs 36

python /home/data/xjn/23largest_baseline/LargeST/experiments/d2stgnn/main_fairS.py --device cuda:3 --dataset HK2 --years 202010 --model_name d2stgnn --seed 2018 --patience 80 --bs 24
python /home/data/xjn/23largest_baseline/LargeST/experiments/d2stgnn/main_fairS_SD.py --device cuda:0 --dataset SD --years 20191w --model_name d2stgnn --seed 2018 --patience 80 --bs 24


python /home/data/xjn/23largest_baseline/LargeST/experiments/d2stgnn/main_fair3_T5.py --device cuda:2 --dataset HK2 --years 202010 --model_name d2stgnn --seed 2018 --patience 80 --bs 36
python /home/data/xjn/23largest_baseline/LargeST/experiments/d2stgnn/main_fair3_T5_SD.py --device cuda:0 --dataset SD --years 20191w --model_name d2stgnn --seed 2018 --patience 80 --bs 36


python /home/data/xjn/23largest_baseline/LargeST/experiments/d2stgnn/main_fairS_all.py --device cuda:0 --dataset HKALL --years 202010 --model_name d2stgnn --seed 2018 --patience 80 --bs 24
python /home/data/xjn/23largest_baseline/LargeST/experiments/d2stgnn/main_fairS_SD_all.py --device cuda:1 --dataset HKALLSD --years 20191w --model_name d2stgnn --seed 2018 --patience 80 --bs 24
