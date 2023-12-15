import random


'''
file_to_array() - helps to convert the dataset from .data file to list type for the preprocessing and calculation for KNN
'''
kvalue=[3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
def file_to_array(Data_Set):
    result = []
    with open(Data_Set, 'r') as content:
        i = content.readline()
        while i:
            row = i.strip().split(',')
            result.append(row)
            i = content.readline()
    return result


'''
euclidean_distance() - calculate the distance between two points using euclidian formula.
'''
def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        try:
            val1 = float(point1[i])
            val2 = float(point2[i])
            distance += (val1 - val2) ** 2
        except ValueError:
            pass
    return distance ** 0.5


'''
minkowski_distance() - calculate the distance between two points using minkowshi formula.
'''
def minkowski_distance(x, y):
    result=0
    if len(x)!=len(y):
        pass
    i = 0
    while i < len(x):
        x_point = x[i]
        y_point = y[i]
        if (type(x_point) is int or type(x_point) is float):
            if (type(y_point) is int or type(y_point) is float):
                result+=(x_point-y_point)**2
        i += 1
    return result ** (1 / 2)

'''
hayenRothDatasetPreprocess() - Preprocessing and load data for the KNN classifier based on their header label
'''
def hayenRothDatasetPreprocess(filepath):
    value_x_list = []
    value_y_list = []
    for i in file_to_array(filepath):
        value_x_list.append(i[:-1])
        value_y_list.append(i[-1])
    return value_x_list, value_y_list

'''
breastCancerDatasetPreprocess() - Preprocessing and load data for the KNN classifier based on their respective range and string values.
'''
def breastCancerDatasetPreprocess(filepath):
    dataset=file_to_array(filepath)
    tumor={'0-4':2,'5-9':7,'10-14':12,'15-19':17,'20-24':22,'25-29':27,'30-34':32,'35-39':37,'40-44':42,'45-49':47,'50-54':52,'55-59':57}
    nodes={'0-2':1,'3-5':4,'6-8':7,'9-11':10,'12-14':13,'15-17':16,'18-20':19,'21-23':22,'24-26':25,'27-29':28,'30-32':31,'33-35':34,'36-39':37}
    i=0
    while i<len(dataset):
        dataset[i][1]=[dataset[i][1]]
        dataset[i][3]=tumor[dataset[i][3]]
        dataset[i][4]=nodes[dataset[i][4]]
        i+=1
    value_x_list = []
    value_y_list = []
    for j in dataset:
        value_x_list.append(j[:-1])
        value_y_list.append(j[-1])
    return value_x_list,value_y_list

'''
carDatasetPreprocess() - Preprocessing and load data for the KNN classifier based on their respective range and string values.
'''
def carDatasetPreprocess(filepath):
    value_x_list = []
    value_y_list = []
    for i in file_to_array(filepath):
        value_x_list.append(i[:-1])
        value_y_list.append(i[-1])
    return value_x_list,value_y_list

'''
KNN_function() - Calculating the KNN classifier using Minkowshi distance between points from the test and train dataset.
It calculates the distance between the points. K value will considered to check with the test data points and count it. 
Based on the max value result will be returned to its parent function.
'''
def KNN_function(x_point, y_point, test_set_x, k, distance_method):
    result_list = []
    for i in test_set_x:
        distance_list = []
        for x_axis, y_axis in enumerate(x_point):
            if distance_method == 'Minkowski':
                distance_value = minkowski_distance(i, y_axis)
            else:
                distance_value = euclidean_distance(i, y_axis)
            distance_list.append((x_axis, distance_value))
        index_list = []
        w=0
        while w < k:
            min = 1e10
            min_i = None
            for index, distance_value in distance_list:
                if distance_value < min:
                    min = distance_value
                    min_i = index
            index_list.append(min_i)
            distance_list = [(index, distance) for index, distance in distance_list if index != min_i]
            w += 1
        index_header = [y_point[index] for index in index_list]
        result = max(index_header, key=index_header.count)
        result_list.append(result)

    return result_list

'''
shuffle_data() - shuffle the data given data
'''
def shuffle_data(i, j):
    combined = list(zip(i, j))
    random.shuffle(combined)
    i[:], j[:] = zip(*combined)

'''
k_fold_cross_validation() -  based on the given k value it will split the x and y set and it will fold the test and training dataset iteratively. 
And calculate the accuracy of KNN for each fold and it's calculating the average accuracy.
'''
def k_fold_cross_validation(x_value, y_value, k, distance_measure):
    length = len(x_value) // k
    list_acc_value = []
    indices = list(range(len(x_value)))
    shuffle_list = []
    while indices:
        shuffle_list.append(random.choice(indices))
        indices.remove(random.choice(indices))
    for i in range(k):
        set_x_test,set_y_test,set_x_train,set_y_train=[],[],[],[]
        start = i * length
        end = (i + 1) * length
        for j in shuffle_list[start:end]:
            set_x_test.append(x_value[j])
        for j in shuffle_list[start:end]:
            set_y_test.append(y_value[j])
        for j in shuffle_list[:start] + shuffle_list[end:]:
            set_x_train.append(x_value[j])
        for j in shuffle_list[:start] + shuffle_list[end:]:
            set_y_train.append(y_value[j])
        knn_value = KNN_function(set_x_train, set_y_train, set_x_test, k, distance_measure)
        knn_pred = 0
        for a, b in zip(set_y_test, knn_value):
            if a == b:
                knn_pred += 1
        list_acc_value.append(knn_pred / len(set_y_test))
        result = 0
        for z in list_acc_value:
            result+=z
    return result/k

# calculating knn accuracy for car data and saving it in file
mean_car_dataset = 0
with open('car_from_scratch.txt', 'w') as file:
    for k in kvalue:
        set_x, set_y = carDatasetPreprocess('/Users/aravindh/Downloads/car+evaluation/car.data')
        combined_list = list(zip(set_x, set_y))
        random.shuffle(combined_list)
        set_x, set_y = zip(*combined_list)
        temp = k_fold_cross_validation(set_x, set_y, k, "Minkowski") * 100
        mean_car_dataset += temp
        file.write(str(temp/100)+"\n")
print("mean Car Dataset Accuracy:", mean_car_dataset / 10)

# calculating knn accuracy for breast cancer data and saving it in file
mean_cancer_dataset = 0
with open('breast_cancer_from_scratch.txt', 'w') as file:
    for k in kvalue:
        set_x, set_y = breastCancerDatasetPreprocess('/Users/aravindh/Downloads/breast+cancer/breast-cancer.data')
        combined_list = list(zip(set_x, set_y))
        random.shuffle(combined_list)
        set_x, set_y = zip(*combined_list)
        temp = k_fold_cross_validation(set_x, set_y, k, "Minkowski") * 100
        mean_cancer_dataset += temp
        file.write(str(temp/100)+"\n")
print("Mean Breast Cancer Dataset Accuracy:", mean_cancer_dataset / 10)

# calculating knn accuracy for hayes roth data and saving it in file
mean_hayesroth_dataset = 0
with open('hayes_roth_from_scratch.txt', 'w') as file:
    for k in kvalue:
        set_x, set_y = hayenRothDatasetPreprocess('/Users/aravindh/Downloads/hayes+roth/hayes-roth.data')
        combined_list = list(zip(set_x, set_y))
        random.shuffle(combined_list)
        set_x, set_y = zip(*combined_list)
        temp = k_fold_cross_validation(set_x, set_y, k, "Minkowski") * 100
        mean_hayesroth_dataset += temp
        file.write(str(temp/100)+"\n")
print("Hays-Roth Dataset Accuracy:", mean_hayesroth_dataset / 10)
