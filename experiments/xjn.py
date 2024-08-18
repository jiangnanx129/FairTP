import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import torch
import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# # 从文件中加载两个Tensor
# str ="1.3866	1.777	2.2223	2.0888	1.6516	2.0458	1.7282	1.9266	1.5865	1.6743	1.5508	3.5814	1.9322"
# result = str.split()
# result = [float(s) for s in result]
# sum1 = 0
# for i in result:
#      sum1 += i

# print(result)
# print(len(result))
# print(sum1 / len(result))
#lst = [1.8791,2.1117,2.085,2.2791,2.0557,2.4202,1.9889,3.6691,2.4433,1.4891,1.6357,4.6622,3.1021] # 4.209107692307694
# lst = result
# # 计算均值
# mean = sum(lst) / len(lst)

# # 计算每个值减去均值的绝对值，并相加
# result = sum([abs(x - mean) for x in lst])

# print(result)

# num =[32.9725,
# 32.1022
# ]
# minn = num[0]
# maxn = num[1]
# if minn > maxn:
#     tmp = minn
#     minn = maxn
#     maxn = tmp
# print(minn)
# print(maxn)
# print((maxn-minn)/ minn)

# str = "1.9348	3.2356	0.0386	32.1022	0.0549"
# resstr = str.split()
# result =""
# for r in resstr:
#     result = result + " & "+r
# print(result)

nums1 ="1.9213 & 3.0363 & 0.0378 & 4.8695 &-& 17.8687 & 26.3632 & 0.0721 & 57.5929 &-"
nums2 ="1.9432 & 3.0964 & 0.0391 & 3.8091 & 0.1266 & 13.4876 & 19.1271 & 0.0707 & 33.1559 & 1.067"
num1res =nums1.split("&")
num2res = nums2.split("&")
print(num1res)
print(num2res)
num1ans = ""
num2ans = ""
for i in range(len(num1res)):
    if num1res[i] == "-" or float(num1res[i]) > float(num2res[i]):
        num1ans = num1ans+"&"+num1res[i]
        num2ans = num2ans+"&"+"\\textbf{" +num2res[i] +"}"
    else:
         num1ans = num1ans+"&"+"\\textbf{" +num1res[i] +"}"
         num2ans = num2ans+"&"+num2res[i]

print(num1ans)
print("***************************************")
print(num2ans)



