# python experiments/dgcrn/main.py --device cuda:2 --dataset GLA --years 2019 --model_name dgcrn --seed 2018 --bs 5 --gcn_depth 1

# python experiments/dgcrn/main.py --device cuda:2 --dataset GBA --years 2019 --model_name dgcrn --seed 2018 --bs 12

# python experiments/dgcrn/main.py --device cuda:2 --dataset SD --years 2019 --model_name dgcrn --seed 2018 --bs 64

python /home/data/xjn/23largest_baseline/LargeST/experiments/dgcrn/main_fair.py --device cuda:1 --dataset SD --years 2019 --model_name dgcrn --seed 2018 --bs 64
python /home/data/xjn/23largest_baseline/LargeST/experiments/dgcrn/main_fair.py --device cuda:2 --dataset SD --years 20191w --model_name dgcrn --seed 2018 --bs 64

python /home/data/xjn/23largest_baseline/LargeST/experiments/dgcrn/main_fair.py --device cuda:0 --dataset HK2 --years 202010 --model_name dgcrn --seed 2018 --bs 64


python /home/data/xjn/23largest_baseline/LargeST/experiments/dgcrn/main_fairS.py --device cuda:0 --dataset HK2 --years 202010 --model_name dgcrn --seed 2018 --bs 42
python /home/data/xjn/23largest_baseline/LargeST/experiments/dgcrn/main_fairS_SD.py --device cuda:0 --dataset SD --years 20191w --model_name dgcrn --seed 2018 --bs 42


python /home/data/xjn/23largest_baseline/LargeST/experiments/dgcrn/main_fair3_T5.py --device cuda:2 --dataset HK2 --years 202010 --model_name dgcrn --seed 2018 --bs 42 --mode test
python /home/data/xjn/23largest_baseline/LargeST/experiments/dgcrn/main_fair3_T5_SD.py --device cuda:2 --dataset SD --years 20191w --model_name dgcrn --seed 2018 --bs 42 --mode test


python /home/data/xjn/23largest_baseline/LargeST/experiments/dgcrn/main_fairS_all.py --device cuda:1 --dataset HKALL --years 202010 --model_name dgcrn --seed 2018 --bs 30 # --lrate 0.005
python /home/data/xjn/23largest_baseline/LargeST/experiments/dgcrn/main_fairS_SD_all.py --device cuda:2 --dataset HKALLSD --years 20191w --model_name dgcrn --seed 2018 --bs 42 --mode test
