import pandas as pd
import random
import math

# Firstly we should read the data from the csv file
data = pd.read_csv("Data\\csv_result-messidor_features_normalize.csv")

# We should choose initial centroids from our data set
c1 = random.randint(0,1151)
c2 = random.randint(0,1151)

while data.iloc[c2]["id"] == data.iloc[c1]["id"]:
    c2 = random.randint(0,1151)
        
c1 = data.iloc[c1]
c2 = data.iloc[c2]

# We should store sample which is belong to which cluster
centroid1 = c1
centroid2 = c2

members1 = []
members2 = []

# We need stop criteria
STOP = 5
iteration = 0
STOP2 = True

columns = list(data.columns)
columns.remove("Class")
columns.remove("id")

while iteration < STOP and STOP2:
    print("############################################################################################")
    print("-----------------------------Iteration"+str(iteration)+"-----------------------------------")
    print("############################################################################################")
    iteration += 1

    changes = 0
    
    # After that, we assign each samples to these clusters
    for index1,sample in data.iterrows():
        # We have to find the distance from that sample to each cluster's centroid
        dist_C1 = 0
        dist_C2 = 0
        # Let's find the distance to cluster 1
        for col in columns:
            dist_C1 = dist_C1 + (sample[col]-centroid1[col])**2
            dist_C1 = math.sqrt(dist_C1)
        # Now, we should look at the distance to cluster 2
        for col in columns:
            dist_C2 = dist_C2 + (sample[col]-centroid2[col])**2
            dist_C2 = math.sqrt(dist_C2)

        if dist_C1 < dist_C2:
            dataList = list(sample)
            id = sample["id"]
            i = 0
            flag = True
            # Data may be already exist in the cluster
            while i < len(members1) and flag:
                if members1[i][0] == id:
                    flag = False
                i += 1
            if flag == True:
                changes += 1
                members1.append(dataList)

            i = 0
            flag = True
            while i < len(members2) and flag:
                if members2[i][0] == id:
                    flag = False
                i += 1
            if flag == False:
                members2.remove(dataList)
        else:
            dataList = list(sample)
            id = sample["id"]
            flag = True
            i = 0
            # Data may be already exist in the cluster
            while i < len(members2) and flag:
                if members2[i][0] == id:
                    flag = False
                i += 1
            if flag == True:
                changes += 1
                members2.append(dataList)        
            
            i = 0
            flag = True
            while i < len(members1)  and flag:
                if members1[i][0] == id:
                    flag = False
                i += 1
            if flag == False:
                members1.remove(dataList)

    if changes < 20:
        print("changes < 20,",changes)
        STOP2 = False

    print("Changes--->",changes)
    # We should update the centroids of the clusters
    for i,col in zip(range(18),columns):
        sumOfFeature = 0
        for j in range(len(members1)):
            sumOfFeature += members1[j][i+1]
        centroid1[col] = sumOfFeature / len(members1)
    
    for i,col in zip(range(18),columns):
        sumOfFeature = 0
        for j in range(len(members2)):
            sumOfFeature += members2[j][i+1]
        centroid2[col] = sumOfFeature / len(members2)

    print("New Centroid 1---->",centroid1)
    print("New Centroid 2---->",centroid2)

np = 0
nf = 0

if len(members1) > len(members2):
    class_c1 = 1
    class_c2 = 0
else:
    class_c1 = 0
    class_c2 = 1


for each in members1:
    if each[len(each)-1] == class_c1:
        np += 1
    else:
        nf += 1

for each in members2:
    if each[len(each)-1] == class_c2:
        np += 1
    else:
        nf += 1

print("------------------------------------------------------------")
print("Number of Sample in Cluster 1 :",len(members1))
print("Number of Sample in Cluster 2 :",len(members2))

print("------------------------------------------------------------")
print("Number of True Clustering: ",np)
print("Number of False Clustering: ",nf)

print("Accuracy----->",(np/(np+nf)))
