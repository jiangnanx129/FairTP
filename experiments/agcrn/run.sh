# python experiments/agcrn/main.py --device cuda:2 --dataset GLA --years 2019 --model_name agcrn --seed 2018 --bs 32

# python experiments/agcrn/main.py --device cuda:2 --dataset GBA --years 2019 --model_name agcrn --seed 2018 --bs 64

# python experiments/agcrn/main.py --device cuda:2 --dataset SD --years 2019 --model_name agcrn --seed 2018 --bs 64

python /home/data/xjn/23largest_baseline/LargeST/experiments/agcrn/main_fair.py --device cuda:3 --dataset SD --years 2019 --model_name agcrn --seed 2018 --bs 64
python /home/data/xjn/23largest_baseline/LargeST/experiments/agcrn/main_fair.py --device cuda:2 --dataset SD --years 20191w --model_name agcrn --seed 2018 --bs 64

python /home/data/xjn/23largest_baseline/LargeST/experiments/agcrn/main_fair.py --device cuda:1 --dataset HK2 --years 202010 --model_name agcrn --seed 2018 --bs 64

python /home/data/xjn/23largest_baseline/LargeST/experiments/agcrn/main_fair3.py --device cuda:3 --dataset HK --years 202010 --model_name agcrn --seed 2018 --bs 64

python /home/data/xjn/23largest_baseline/LargeST/experiments/agcrn/main_fair3_T_450.py --device cuda:1 --dataset HK --years 202010 --model_name agcrn --seed 2018 --bs 64

python /home/data/xjn/23largest_baseline/LargeST/experiments/agcrn/main_fair3_T.py --device cuda:3 --dataset HK2 --years 202010 --model_name agcrn --seed 2018 --bs 64 --patience 200

python /home/data/xjn/23largest_baseline/LargeST/experiments/agcrn/main_fair3_T4.py --device cuda:2 --dataset HK2 --years 202010 --model_name agcrn --seed 2018 --bs 64

python /home/data/xjn/23largest_baseline/LargeST/experiments/agcrn/main_fair3_T5.py --device cuda:0 --dataset HK2 --years 202010 --model_name agcrn --seed 2018 --bs 64 --mode test --patience 200




python /home/data/xjn/23largest_baseline/LargeST/experiments/agcrn/main_fairS.py --device cuda:3 --dataset HK2 --years 202010 --model_name agcrn --seed 2018 --bs 64 --mode test

python /home/data/xjn/23largest_baseline/LargeST/experiments/agcrn/main_fairS_SD.py --device cuda:0 --dataset SD --years 20191w --model_name agcrn --seed 2018 --bs 64
python /home/data/xjn/23largest_baseline/LargeST/experiments/agcrn/main_fair3_T5_SD.py --device cuda:0 --dataset SD --years 20191w --model_name agcrn --seed 2018 --bs 64 --mode test --patience 200


python /home/data/xjn/23largest_baseline/LargeST/experiments/agcrn/main_fairS_all.py --device cuda:2 --dataset HKALL --years 202010 --model_name agcrn --seed 2018 --bs 64 
python /home/data/xjn/23largest_baseline/LargeST/experiments/agcrn/main_fairS_SD_all.py --device cuda:0 --dataset HKALLSD --years 20191w --model_name agcrn --seed 2018 --bs 64


python /home/data/xjn/23largest_baseline/LargeST/experiments/agcrn/main_fair3_T5_samcf.py --device cuda:3 --dataset HK100 --years 202010 --model_name agcrn --seed 2018 --bs 64 --mode test --patience 200
python /home/data/xjn/23largest_baseline/LargeST/experiments/agcrn/main_fair3_T5_samcf250.py --device cuda:1 --dataset HK250 --years 202010 --model_name agcrn --seed 2018 --bs 64
python /home/data/xjn/23largest_baseline/LargeST/experiments/agcrn/main_fair3_T5_samcf300.py --device cuda:2 --dataset HK300 --years 202010 --model_name agcrn --seed 2018 --bs 64
python /home/data/xjn/23largest_baseline/LargeST/experiments/agcrn/main_fair3_T5_samcf150.py --device cuda:0 --dataset HK150 --years 202010 --model_name agcrn --seed 2018 --bs 64

python /home/data/xjn/23largest_baseline/LargeST/experiments/agcrn/main_fair3_T5_SD_samcf.py --device cuda:2 --dataset SD100 --years sd100 --model_name agcrn --seed 2018 --bs 64 
python /home/data/xjn/23largest_baseline/LargeST/experiments/agcrn/main_fair3_T5_SD_samcf250.py --device cuda:0 --dataset SD250 --years sd250 --model_name agcrn --seed 2018 --bs 64 
python /home/data/xjn/23largest_baseline/LargeST/experiments/agcrn/main_fair3_T5_SD_samcf300.py --device cuda:0 --dataset SD300 --years sd300 --model_name agcrn --seed 2018 --bs 64 
python /home/data/xjn/23largest_baseline/LargeST/experiments/agcrn/main_fair3_T5_SD_samcf150.py --device cuda:1 --dataset SD150 --years sd150 --model_name agcrn --seed 2018 --bs 64 




python /home/data/xjn/23largest_baseline/LargeST/experiments/agcrn/main_fair_fairst_1.py --device cuda:3 --dataset HKALL --years 202010 --model_name agcrn --seed 2018 --bs 64 
