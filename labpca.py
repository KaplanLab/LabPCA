

import numpy as np
import sklearn.decomposition
import argparse
import pdb
import matplotlib.pyplot as plt

def main():
    # argparse 
    parser = argparse.ArgumentParser(description='Lab member PCA')
    parser.add_argument('-f',help='tab delimited input file, 1 header col with questions, 1 header row with names',dest='infile',type=str,required=True)
    args = parser.parse_args()

    # load and parse file
    with open(args.infile,'r') as fh:

        names = fh.next().rstrip("\r\n").split("\t")[1:]

        data = []
        for i in fh:
            data.append(i.rstrip("\r\n").split("\t")[1:])

    # PCA
    data = np.array(data).T
    model = sklearn.decomposition.PCA(n_components=2)
    result = model.fit_transform(data)

    # plot
    plt.scatter(result[:,0],result[:,1])
    for i in zip(result,names):
        plt.annotate(i[1],i[0]+0.1)

    plt.xlabel('PC1 (exp var = {0:.2f}%)'.format(model.explained_variance_ratio_[0]))
    plt.ylabel('PC2 (exp var = {0:.2f}%)'.format(model.explained_variance_ratio_[1]))

    plt.show()

if __name__ == "__main__":
    main()
