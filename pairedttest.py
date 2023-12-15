import numpy as np
from scipy.stats import t

#ttest() function will help to do the T-test using pre-defined t function from scipy.stats
def ttest(sk_learn,scratch):
    if len(sk_learn) != len(scratch):
        print("The number of values in file1 and file2 is not the same.")
    else:
        for i in range(len(sk_learn)):
            n = len(sk_learn)
            tvalue = (np.mean(sk_learn) - np.mean(scratch)) / np.sqrt(np.var(sk_learn, ddof=1) / n + np.var(scratch, ddof=1) / n)
            pvalue = 2 * (1 - t.cdf(abs(tvalue), df=n - 1))
            print("t-value: ",tvalue)
            print("P-value: ",pvalue)
            if pvalue > 0.05:
                print("Accept the hypothesis and difference is significant for the mean.\n")
            else:
                print("Reject the hypothesis and difference is not significant for the mean.\n")


#T-Test for the Breast Cancer dataset
with open('/Users/aravindh/PycharmProjects/MLProgrammingAssignment/breast_cancer_sklearn.txt', 'r') as file1,\
     open('/Users/aravindh/PycharmProjects/MLProgrammingAssignment/breast_cancer_from_scratch.txt', 'r') as file2:
    sklearn_accuracy = [float(value.strip()) for value in file1]
    scratch_accuracy = [float(value.strip()) for value in file2]
print("t-test for the breast cancer dataset:")
ttest(sklearn_accuracy, scratch_accuracy)
print("\n")


#T-Test for the Car dataset
with open('/Users/aravindh/PycharmProjects/MLProgrammingAssignment/car_sklearn.txt', 'r') as file1,\
     open('/Users/aravindh/PycharmProjects/MLProgrammingAssignment/car_from_scratch.txt', 'r') as file2:
    sklearn_accuracy = [float(value.strip()) for value in file1]
    scratch_accuracy = [float(value.strip()) for value in file2]
print("t-test for the car dataset:")
ttest(sklearn_accuracy, scratch_accuracy)
print("\n")


#T-Test for the Hayens Roth dataset
with open('/Users/aravindh/PycharmProjects/MLProgrammingAssignment/hayens_roth_sklearn.txt', 'r') as file1,\
     open('/Users/aravindh/PycharmProjects/MLProgrammingAssignment/hayes_roth_from_scratch.txt', 'r') as file2:
    sklearn_accuracy = [float(value.strip()) for value in file1]
    scratch_accuracy = [float(value.strip()) for value in file2]
print("t-test for the hayens roth dataset:")
ttest(sklearn_accuracy, scratch_accuracy)
print("\n")



