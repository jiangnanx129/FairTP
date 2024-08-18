import torch

def process_tensor(tensor):
    if torch.all(tensor == tensor[0]):
        # 所有元素一致，直接进行 sigmoid 操作
        result = torch.sigmoid(tensor)
        print("aaaa")
    else:
        # 元素不一致，先进行归一化，再进行 sigmoid 操作
        normalized_tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        result = torch.sigmoid(normalized_tensor)
        print("bbb")
    
    return result


def process_tensor2(tensor):
    
    # 元素不一致，先进行归一化，再进行 sigmoid 操作
    normalized_tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    result = torch.sigmoid(normalized_tensor)
    print("bbb")
    
    return result




# 示例张量
tensor1 = torch.tensor([32, 32, 32, 32, 32])
tensor2 = torch.tensor([0, 2, 4, 6, 8])

# 处理张量
output1 = process_tensor2(tensor1)
output2 = process_tensor2(tensor2)

print(output1)
print(output2)


# 假设有两个等长张量，名为tensor1和tensor2
tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([4, 5, 6])

# 使用 * 运算符对应位置元素相乘
result = tensor1 * tensor2

print(result)

import numpy as np
def count_values(dictionary):
    values = np.array(list(dictionary.values()))

    greater_than_zero = np.sum(values > 0)
    less_than_zero = np.sum(values < 0)
    equal_to_zero = np.sum(values == 0)

    return greater_than_zero, less_than_zero, equal_to_zero

# 举例
my_dictionary = {'a': 5, 'b': -3, 'c': 0, 'd': 2, 'e': -1, 'f': -1, 'g': -0}
greater, less, equal = count_values(my_dictionary)
print("大于零的值数量:", greater)
print("小于零的值数量:", less)
print("等于零的值数量:", equal)
