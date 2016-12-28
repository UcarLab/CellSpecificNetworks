
import sys
import pandas as pd
import numpy as np
import scipy
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import preprocessing

args = sys.argv

#arg1: MCF-7 Orbit File
mcf7f = args[1]
#arg2: K562 Orbit File
k562f = args[2]

#arg3: GM Orbit File
gmf = args[3]

#arg4 output directory
outdir = args[4]
if not(outdir.endswith("/")):
    outdir = outdir+"/"

def getOrbitClusterScores(data):
    orbits = data.iloc[:, 22:95].values
    stdorbits = preprocessing.StandardScaler().fit_transform(orbits)
    
    c1 = stdorbits[:,[23,7,21,47,33,30,11,26,2,5,28,17,16,20]]
    c2 = stdorbits[:,[25,3,10,29,32,43,52,65,54,59,46,12,40]]
    c3 = stdorbits[:,[34,51,36,8,37,38,53,49,62,50,63]]
    c4 = stdorbits[:,[72,70,71,66,14,57,58,67,64,68,42,60,41,13,48,55,61,44,69]]
    c5 = stdorbits[:,[4,15,18]]
    c6 = stdorbits[:,[19,6,22]]
    c7 = stdorbits[:,[27,35,39,31,9,24,45,56]]
    
    fun = np.max
    c1m = fun(c1, axis=1)
    c2m = fun(c2, axis=1)
    c3m = fun(c3, axis=1)
    c4m = fun(c4, axis=1)
    c5m = fun(c5, axis=1)
    c6m = fun(c6, axis=1)
    c7m = fun(c7, axis=1)
    
    return [c1m, c2m, c3m, c4m, c5m, c6m, c7m]
    
    
def subset(clust, idx):
    return clust[idx]

def clusterfeatures(data, cs, outfile):
    prevdata = data.iloc[:,range(0,22)].values
    classes = data.iloc[:, range(95,100)].values
    prevcols = list(data.columns.values[range(0,22)])
    sufcols = list(data.columns.values[range(95,100)])
    alldata = np.concatenate((prevdata, cs[0][np.newaxis].transpose(), cs[1][np.newaxis].transpose(), cs[2][np.newaxis].transpose(), cs[3][np.newaxis].transpose(), cs[4][np.newaxis].transpose(), cs[5][np.newaxis].transpose(), cs[6][np.newaxis].transpose(), classes), axis=1)
    prevcols.extend(["C0", "C1", "C2", "C3", "C4", "C5", "C6"])
    prevcols.extend(sufcols)
    pd.DataFrame(alldata, columns=prevcols).to_csv(outfile, sep="\t")
      
mcf7d = data = pd.read_csv(mcf7f, sep="\t")
k562d = data = pd.read_csv(k562f, sep="\t")
gmd = data = pd.read_csv(gmf, sep="\t")


#TODO specify the columns in another way using arguments, for now these will work with the data provided
mpidx = np.where(mcf7d['MCF7_P.txt'] == 1)
kpidx = np.where(k562d['K562_P.txt'] == 1)
gpidx = np.where(gmd['GM_P.txt'] == 1)

mbdidx = np.where(mcf7d['MCF7_BD.txt'] == 1)
kbdidx = np.where(k562d['K562_BD.txt'] == 1)
gbdidx = np.where(gmd['GM_BD.txt'] == 1)

meidx = np.where(mcf7d['MCF7_E.txt'] == 1)
keidx = np.where(k562d['K562_E.txt'] == 1)
geidx = np.where(gmd['GM_E.txt'] == 1)

msteidx = np.where(mcf7d['MCF7_StE.txt'] == 1)
ksteidx = np.where(k562d['K562_StE.txt'] == 1)
gsteidx = np.where(gmd['GM_StE.txt'] == 1)

mseidx = np.where(mcf7d['MCF7_SE.txt'] == 1)
kseidx = np.where(k562d['K562_SE.txt'] == 1)
gseidx = np.where(gmd['GM_SE.txt'] == 1)

moidx = np.where(mcf7d['MCF7_O.txt'] == 1)
koidx = np.where(k562d['K562_O.txt'] == 1)
goidx = np.where(gmd['GM_O.txt'] == 1)

mcf7 = getOrbitClusterScores(mcf7d)
k562 = getOrbitClusterScores(k562d)
gm = getOrbitClusterScores(gmd)

clusterfeatures(mcf7d, mcf7, outdir+"MCF7NetworkClusterFeatures.txt")
clusterfeatures(k562d, k562, outdir+"K562NetworkClusterFeatures.txt")
clusterfeatures(gmd, gm, outdir+"GMNetworkClusterFeatures.txt")


def trimmean(vals):
    std = np.std(vals)
    median = np.median(vals)
    return scipy.stats.mstats.tmean(vals, (median-1.5*std,median+1.5*std))

for i in range(0,7):    
    fig = plt.figure()
    fun = trimmean
    pmedians = [fun(subset(mcf7[i],mpidx)),fun(subset(k562[i],kpidx)), fun(subset(gm[i],gpidx))]
    bdmedians = [fun(subset(mcf7[i],mbdidx)),fun(subset(k562[i],kbdidx)),fun(subset(gm[i],gbdidx))]
    emedians = [fun(subset(mcf7[i],meidx)), fun(subset(k562[i],keidx)),fun(subset(gm[i],geidx))]
    stemedians = [fun(subset(mcf7[i],msteidx)),fun(subset(k562[i],ksteidx)), fun(subset(gm[i],gsteidx))]
    semedians = [fun(subset(mcf7[i],mseidx)),fun(subset(k562[i],kseidx)),fun(subset(gm[i],gseidx))]
    omedians = [fun(subset(mcf7[i],moidx)),fun(subset(k562[i],koidx)),fun(subset(gm[i],goidx))]

    plt.plot((pmedians, bdmedians, emedians, stemedians, semedians, omedians), "o", label=["MCF7", "K562", "GM"], alpha=0.5, markersize=9, markeredgewidth=0.0)
    plt.xlim(-0.5,5.5)
    plt.ylim((-1,4))
    plt.xticks([0,1,2,3,4,5], ["P","BD","E","StE","SE","O"])
    plt.title("Cluster"+str(i))

    
    pdfplot = PdfPages(outdir+"tmean_cluster"+str(i)+".pdf");
    pdfplot.savefig(fig)
    pdfplot.close()