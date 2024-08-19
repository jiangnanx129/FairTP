# python experiments/astgcn/main.py --device cuda:2 --dataset GLA --years 2019 --model_name astgcn --seed 2018 --bs 16 --patience 50 --wdecay 0

# python experiments/astgcn/main.py --device cuda:2 --dataset GBA --years 2019 --model_name astgcn --seed 2018 --bs 40 --patience 50

# python experiments/astgcn/main.py --device cuda:2 --dataset SD --years 2019 --model_name astgcn --seed 2018 --bs 64 --patience 50

python /home/data/xjn/23largest_baseline/LargeST/experiments/astgcn/main_fair.py --device cuda:2 --dataset SD --years 2019 --model_name astgcn --seed 2018 --bs 64 --patience 50
python /home/data/xjn/23largest_baseline/LargeST/experiments/astgcn/main_fair.py --device cuda:2 --dataset SD --years 20191w --model_name astgcn --seed 2018 --bs 64 --patience 50

python /home/data/xjn/23largest_baseline/LargeST/experiments/astgcn/main_fair.py --device cuda:0 --dataset HK --years 202010 --model_name astgcn --seed 2018 --bs 64 --patience 50

python /home/data/xjn/23largest_baseline/LargeST/experiments/astgcn/main_fair.py --device cuda:1 --dataset HK2 --years 202010 --model_name astgcn --seed 2018 --bs 64 --patience 50

python /home/data/xjn/23largest_baseline/LargeST/experiments/astgcn/main_fairS.py --device cuda:2 --dataset HK2 --years 202010 --model_name astgcn --seed 2018 --bs 64 --patience 50
python /home/data/xjn/23largest_baseline/LargeST/experiments/astgcn/main_fairS_SD.py --device cuda:0 --dataset SD --years 20191w --model_name astgcn --seed 2018 --bs 64 --patience 50




python /home/data/xjn/23largest_baseline/LargeST/experiments/astgcn/main_fair3_T5.py --device cuda:2 --dataset HK2 --years 202010 --model_name astgcn --seed 2018 --bs 64 --mode test --patience 50
python /home/data/xjn/23largest_baseline/LargeST/experiments/astgcn/main_fair3_T5_SD.py --device cuda:2 --dataset SD --years 20191w --model_name astgcn --seed 2018 --bs 64 --mode test --patience 50

python /home/data/xjn/23largest_baseline/LargeST/experiments/astgcn/main_fairS_all.py --device cuda:1 --dataset HKALL --years 202010 --model_name astgcn --seed 2018 --bs 64 --patience 50 --mode test
python /home/data/xjn/23largest_baseline/LargeST/experiments/astgcn/main_fairS_SD_all.py --device cuda:1 --dataset HKALLSD --years 20191w --model_name astgcn --seed 2018 --bs 64 --patience 50

