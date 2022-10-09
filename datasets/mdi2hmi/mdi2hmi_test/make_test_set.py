import numpy as np
import pandas as pd
import os

def main():
    testA = pd.read_csv('../mdi2hmi_small/testA.csv')
    testB = pd.read_csv('../mdi2hmi_small/testB.csv')
    trainA = pd.read_csv('../mdi2hmi_small/trainA.csv')
    trainB = pd.read_csv('../mdi2hmi_small/trainB.csv')

    testA[0:5].to_csv('testA.csv', header=False, index=False)
    testB[0:5].to_csv('testB.csv', header=False, index=False)
    trainA[0:20].to_csv('trainA.csv', header=False, index=False)
    trainB[0:20].to_csv('trainB.csv', header=False, index=False)

    return

if __name__ == "__main__":
    main()