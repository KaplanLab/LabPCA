import numpy as np
import sklearn.decomposition
import argparse
import pdb
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as m3d

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
    model = sklearn.decomposition.PCA(n_components=3)
    result = model.fit_transform(data)

    # plot by first 2 PC
    plt.scatter(result[:,0],result[:,1])
    for i in zip(result[:,:2],names):
        plt.annotate(i[1],i[0]+0.1)

    plt.xlabel('PC1 (exp var = {0:.2f}%)'.format(model.explained_variance_ratio_[0]))
    plt.ylabel('PC2 (exp var = {0:.2f}%)'.format(model.explained_variance_ratio_[1]))

    plt.show()

    # 3D plot first 3 PC
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(result[:,0],result[:,1],result[:,2])
    for i in zip(result,names):
        ax.text(i[0][0],i[0][1],i[0][2],i[1])

    ax.set_xlabel('PC1 (exp var = {0:.2f}%)'.format(model.explained_variance_ratio_[0]))
    ax.set_ylabel('PC2 (exp var = {0:.2f}%)'.format(model.explained_variance_ratio_[1]))
    ax.set_zlabel('PC3 (exp var = {0:.2f}%)'.format(model.explained_variance_ratio_[2]))

    # generate frame images for animated gif
    for i in range(360):
        ax.view_init(elev=30.,azim=i)
        plt.savefig('frame'+str(i)+'.jpg')

    # images can be converted into animated gif by using:
    # ffmpeg -f image2 -framerate 20 -i frame%d.jpg oufile.gif

if __name__ == "__main__":
    main()
