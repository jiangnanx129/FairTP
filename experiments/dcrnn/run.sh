# python experiments/dcrnn/main.py --device cuda:2 --dataset CA --years 2019 --model_name dcrnn --seed 2018 --bs 16

# python experiments/dcrnn/main.py --device cuda:2 --dataset GLA --years 2019 --model_name dcrnn --seed 2018 --bs 32

# python experiments/dcrnn/main.py --device cuda:2 --dataset GBA --years 2019 --model_name dcrnn --seed 2018 --bs 64

# python experiments/dcrnn/main.py --device cuda:2 --dataset SD --years 2019 --model_name dcrnn --seed 2018 --bs 64

# fsample replace dcrnn !! 

python /home/data/xjn/23largest_baseline/LargeST/experiments/dcrnn/main_fair.py --device cuda:0 --dataset SD --years 2019 --model_name dcrnn --seed 2018 --bs 64
python /home/data/xjn/23largest_baseline/LargeST/experiments/dcrnn/main_fair.py --device cuda:0 --dataset SD --years 20191w --model_name dcrnn --seed 2018 --bs 64

python /home/data/xjn/23largest_baseline/LargeST/experiments/dcrnn/main_fair.py --device cuda:0 --dataset HK --years 202010 --model_name dcrnn --seed 2018 --bs 64





python /home/data/xjn/23largest_baseline/LargeST/experiments/fsample/main_fair.py --device cuda:3 --dataset SD --years 20191w --model_name fsample --seed 2018 --bs 64


python /home/data/xjn/23largest_baseline/LargeST/experiments/fsample/main_fair.py --device cuda:2 --dataset SD --years 20191w --model_name fsample --seed 2018 --bs 64

python /home/data/xjn/23largest_baseline/LargeST/experiments/fsample/main_fair_noDis.py --device cuda:1 --dataset SD --years 20191w --model_name fsample --seed 2018 --bs 64

python /home/data/xjn/23largest_baseline/LargeST/experiments/fsample/main_fair_HK2.py --device cuda:1 --dataset HK --years 202010 --model_name fsample --seed 2018 --bs 64



python /home/data/xjn/23largest_baseline/LargeST/experiments/fsample/main_fair_HK_d.py --device cuda:3 --dataset HK --years 202010 --model_name fsample --seed 2018 --bs 64




python /home/data/xjn/23largest_baseline/LargeST/experiments/fairgnn/main_fair.py --device cuda:2 --dataset HK --years 202010 --model_name fairgnn --seed 2018 --bs 64


python /home/data/xjn/23largest_baseline/LargeST/experiments/fsample/main_fair_HK_base_nodis.py --device cuda:2 --dataset HK --years 202010 --model_name fsample --seed 2018 --bs 64



# NEW
python /home/data/xjn/23largest_baseline/LargeST/experiments/fsample/main_fair_HK_3.py --device cuda:2 --dataset HK --years 202010 --model_name fsample --seed 2018 --bs 64


python /home/data/xjn/23largest_baseline/LargeST/experiments/fairgnn/main_fair.py --device cuda:2 --dataset HK --years 202010 --model_name fairgnn --seed 2018 --bs 64






python /home/data/xjn/23largest_baseline/LargeST/experiments/dcrnn/main.py --device cuda:0 --dataset HKALL --years 202010 --model_name dcrnn --seed 2018 --bs 48 --mode test