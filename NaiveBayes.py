import pandas as pd
import math

# Firstly we should read the data from the csv file
real_data = pd.read_csv("Data\\csv_result-messidor_features_normalize.csv")

# We drop the column which name is 0, because all values in this column is one, this feature doesn't help us to classify the samples
real_data.drop(columns=["0"],axis=1,inplace=True)
real_data.drop(columns=["id"],axis=1,inplace=True)

# We use the Kfold Cross Validation, K is selected 10
K = 10

# We have 1151 samples

start = 0
end = 115
index = 0

# We should store the accuracy values
sumOfTrue = 0
sumOfFalse = 0
sumOfAvg = 0
maxAcc = 0
minAcc = 1

tp = 0
tn = 0
fp = 0
fn = 0

while index < 10:
    print("######################################################################################################")
    print("--------------------------------------------"+str(index)+"th iteration----------------------------------------------")
    print("######################################################################################################")         
    if index == 9:   
        test_data = real_data[start:end]
    else:
        test_data = real_data[start:end]

    train_data = real_data.drop(list(range(start,end+1)),axis = 0)
    index += 1
    start = end
    end += 115

    # We divide data to train and test data
    data = train_data
    
    # Let's start the creating to model
    
    #P(h|x) = P(x|h)*P(h)/ P(x)---> Bayes Theorem

    class_no = 0 # no = 0
    class_yes = 0 # yes = 1

    # Let's find the P(class=0) and P(class=1)
    for sample in data["Class"]:
        if sample == 0:
            class_no += 1
        else:
            class_yes += 1

    prob_No = class_no / (class_no + class_yes)
    prob_Yes = class_yes / (class_no + class_yes)


    # Now we should find the P(X|class = 0) and P(X|class = 1),To do this, we need features for each class
    data_features_class_No = data[data["Class"]==0]
    data_features_class_Yes = data[data["Class"]==1]

    data_features_class_No.drop(columns=["Class"],axis=1,inplace=True)
    data_features_class_Yes.drop(columns=["Class"],axis=1,inplace=True)


    # Let's test our model--->
    numberOfTrue = 0
    numberOfFalse = 0

    # Firstly, we should find P(X|YES) and P(X|NO)
    for index1,row in test_data.iterrows():
        columns = list(data.columns)
        columns.remove("Class")
        test_rows = row
        probs = []
        
        #Let's start with P(X|YES)
        # We should predict the class label of this sample
        for col,feature in zip(columns,test_rows):
            # We have two nominal features, we should find their probabilities different than others
            if col == "1":
                cnt = 0
                size = 0
                for each in data_features_class_Yes["1"]:
                    if each == feature:
                        cnt += 1
                    size += 1
                prob_feature1 = (cnt / size)
                probs.append(prob_feature1)
            # Our second nominal feature is the 18 th columns
            elif col== "18":
                cnt = 0
                size = 0
                for each in data_features_class_Yes["18"]:
                    if each == feature:
                        cnt += 1
                    size += 1
                prob_feature18 = cnt / size
                probs.append(prob_feature18)
            else:
                # Other features of us, are numeric values, we should use the Gauss Formula to compute their probability
                mean = data_features_class_Yes[col].mean()
                std = data_features_class_Yes[col].std()
                power = ((feature-mean)*(feature-mean))
                power *= (-1)
                power = power / (2*std*std)
                e = math.exp(power)
                pi_2 = 2 * math.pi
                prob_feature = e / (math.sqrt(pi_2)*std )
                # We calculate the probability of this feature for Yes class
                probs.append((prob_feature))

        # Let's find the P(Yes|X), P(YES|X) = P(X|YES) *P(YES) / P(X)
        p_X = 1
        for p in probs:
            p_X *= p

        probYesForThatSample = p_X * prob_Yes

        #################################################################################################
        
        # After that, we should find P(X|NO)
        probs = []
        # We should predict class label of this sample
        for col,feature in zip(columns,test_rows):
            # We have two nominal features, we should find their probabilities different than others
            if col == "1":
                cnt = 0
                size = 0
                for each in data_features_class_No["1"]:
                    if each == feature:
                        cnt += 1
                    size += 1
                prob_feature1 = cnt / size
                probs.append(prob_feature1)
            # Our second nominal feature is the 18 th columns
            elif col== "18":
                cnt = 0
                size = 0
                for each in data_features_class_No["18"]:
                    if each == feature:
                        cnt += 1
                    size += 1
                prob_feature18 = cnt / size
                probs.append(prob_feature18)
            else:
                # Other features of us, are numeric values, we should use the Gauss Formula to compute their probability
                mean = data_features_class_No[col].mean()
                std = data_features_class_No[col].std()
                power = (-1)*(((feature-mean))**2)
                power = power / (2*std*std)
                e = math.exp(power)
                prob_feature = e / (math.sqrt((2*math.pi))*std )
                # We calculate the probability of this feature for Yes class
                probs.append(prob_feature)


        # Let's find the P(NO|X), P(NO|X) = P(X|NO) *P(NO) / P(X)
        p_X = 1
        for p in probs:
            p_X *= p

        # We should compare Yes and No probability for that sample
        probNoForThatSample = p_X * prob_No

        if probNoForThatSample < probYesForThatSample:
            if test_rows["Class"] == 1:
                numberOfTrue += 1
                tp += 1
            else:
                numberOfFalse += 1
                fp += 1
        else:
            if test_rows["Class"] == 0:
                numberOfTrue += 1
                tn += 1
            else:
                numberOfFalse += 1
                fn += 1
        
    print("------------------------------------------------------")

    print("--->Number of True Classification---->",numberOfTrue)
    print("--->Number of False Classification---->",numberOfFalse)
    acc =(numberOfTrue/(numberOfTrue+numberOfFalse))
    print("--->Accuracy --->",acc)
    if acc > maxAcc:
        maxAcc = acc
    elif acc < minAcc:
        minAcc = acc
    
    print("------------------------------------------------------")

    sumOfTrue += numberOfTrue
    sumOfFalse += numberOfFalse
    sumOfAvg += float(numberOfTrue/(numberOfTrue + numberOfFalse))


print("----------------------------------------------------------------------------------------------------")

print("###########################----RESULT OF THE CLASSIFICATON----###################################")
print("---------------------------------------------------------------------------------------------------")

print("Max accuracy --->",maxAcc)
print("Mean accuracy --->",(sumOfAvg/10))
print("Min accuracy --->",minAcc)
print("------------------------------------------------------")

tp = tp / 10
fp = fp / 10
fn = fn / 10
tn = tn / 10

print("True Positive-->",tp)
print("False Positive-->",fp)
print("True Negative--->",tn)
print("False Negative--->",fn)
print("------------------------------------------------------")

precision = tp / (tp+fp)
recall = tp / (tp+ fn)

print("Precision-->",precision)
print("Recall-->",recall)





