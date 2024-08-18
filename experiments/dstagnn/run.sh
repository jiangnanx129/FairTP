# python experiments/dstagnn/main.py --device cuda:2 --dataset GLA --years 2019 --model_name dstagnn --seed 2018 --bs 10 --input_dim 1

# python experiments/dstagnn/main.py --device cuda:2 --dataset GBA --years 2019 --model_name dstagnn --seed 2018 --bs 24 --input_dim 1

# python experiments/dstagnn/main.py --device cuda:2 --dataset SD --years 2019 --model_name dstagnn --seed 2018 --bs 64 --input_dim 1

python /home/data/xjn/23largest_baseline/LargeST/experiments/dstagnn/main_fair.py --device cuda:2 --dataset SD --years 2019 --model_name dstagnn --seed 2018 --bs 64 --input_dim 1
python /home/data/xjn/23largest_baseline/LargeST/experiments/dstagnn/main_fair.py --device cuda:2 --dataset SD --years 20191w --model_name dstagnn --seed 2018 --bs 64 --input_dim 1

python /home/data/xjn/23largest_baseline/LargeST/experiments/dstagnn/main_fair.py --device cuda:3 --dataset HK --years 202010 --model_name dstagnn --seed 2018 --bs 64 --input_dim 1

python /home/data/xjn/23largest_baseline/LargeST/experiments/dstagnn/main_fairS.py --device cuda:1 --dataset HK2 --years 202010 --model_name dstagnn --seed 2018 --bs 48 --input_dim 1
python /home/data/xjn/23largest_baseline/LargeST/experiments/dstagnn/main_fairS_SD.py --device cuda:2 --dataset SD --years 20191w --model_name dstagnn --seed 2018 --bs 48 --input_dim 1

python /home/data/xjn/23largest_baseline/LargeST/experiments/dstagnn/main_fair3_T5.py --device cuda:3 --dataset HK2 --years 202010 --model_name dstagnn --seed 2018 --bs 48 --input_dim 1
python /home/data/xjn/23largest_baseline/LargeST/experiments/dstagnn/main_fair3_T5_SD.py --device cuda:1 --dataset SD --years 20191w --model_name dstagnn --seed 2018 --bs 64 --input_dim 1


python /home/data/xjn/23largest_baseline/LargeST/experiments/dstagnn/main_fairS_all.py --device cuda:2 --dataset HKALL --years 202010 --model_name dstagnn --seed 2018 --bs 64 --input_dim 1
python /home/data/xjn/23largest_baseline/LargeST/experiments/dstagnn/main_fairS_SD_all.py --device cuda:2 --dataset HKALLSD --years 20191w --model_name dstagnn --seed 2018 --bs 64 --input_dim 1
