

# NEW
python /home/data/xjn/23largest_baseline/LargeST/experiments/FairST_baseline/main_fair_HK_3.py --device cuda:2 --dataset HK --years 202010 --model_name fsample --seed 2018 --bs 64


python /home/data/xjn/23largest_baseline/LargeST/experiments/FairST_baseline/main_fair.py --device cuda:2 --dataset HK --years 202010 --model_name fairgnn --seed 2018 --bs 64


python /home/data/xjn/23largest_baseline/LargeST/experiments/FairST_baseline/main_fairS.py --device cuda:0 --dataset HK2 --years 202010 --model_name fsample --seed 2018 --bs 48
python /home/data/xjn/23largest_baseline/LargeST/experiments/FairST_baseline/main_fairS_SD.py --device cuda:0 --dataset SD --years 20191w --model_name fsample --seed 2018 --bs 48

python /home/data/xjn/23largest_baseline/LargeST/experiments/FairST_baseline/main_fair3_T5.py --device cuda:0 --dataset HK2 --years 202010 --model_name fsample --seed 2018 --bs 48
python /home/data/xjn/23largest_baseline/LargeST/experiments/FairST_baseline/main_fair3_T5_SD.py --device cuda:0 --dataset SD --years 20191w --model_name fsample --seed 2018 --bs 64






python /home/data/xjn/23largest_baseline/LargeST/experiments/FairST_baseline/main_fairS_all.py --device cuda:2 --dataset HKALL --years 202010 --model_name FairST_baseline --seed 2018 --bs 48



python /home/data/xjn/23largest_baseline/LargeST/experiments/FairST_baseline/main_fairS_SD_all.py --device cuda:2 --dataset HKALLSD --years 20191w --model_name fsample --seed 2018 --bs 48
