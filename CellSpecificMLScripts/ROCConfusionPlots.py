import sys
import pandas as pd
import numpy as np

from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix,roc_curve, auc
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

args = sys.argv

#arg1: MCF-7 True Labels
mcf7true = args[1]
#arg2: MCF-7 Probabilities
mcf7proba = args[2]

#arg3: K562 True Labels
k562true = args[3]
#arg4: K562 Probabilities
k562proba = args[4]

#arg5: GM True Labels
gmtrue = args[5]
#arg6: GM Probabilities
gmproba = args[6]

#arg7
directory = args[7]

MTrue = pd.read_csv(mcf7true, sep="\t", header=None).values
MProba = pd.read_csv(mcf7proba, sep="\t", header=None).values


KTrue = pd.read_csv(k562true, sep="\t", header=None).values
KProba = pd.read_csv(k562proba, sep="\t", header=None).values

GTrue = pd.read_csv(gmtrue, sep="\t", header=None).values
GProba = pd.read_csv(gmproba, sep="\t", header=None).values


roctitle = "5-Fold Cross Validation"
bestthresh = []
def plotROCCurves(test, predprob, labels, outdir):
    fig = plt.figure(figsize=(10, 10))
    
    ci = 0
    for i in range(0, len(labels)):
        try:
            fpr, tpr, thresh = roc_curve(test[i], predprob[i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=labels[i]+' (AUC = %0.2f)' % roc_auc)
            ci = ci+1
            edist = []
            for k in range(0, len(thresh)):
                dx = 1-tpr[k]
                dy = 0-fpr[k]
                edist.append(dx*dx+dy*dy)
            bestthresh.append(thresh[np.argmin(edist)])
        except(KeyError):
            pass

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(roctitle)
    plt.legend(loc="lower right")
    pdfplot = PdfPages(outdir+"_"+roctitle+"_roccurves.pdf");
    pdfplot.savefig(fig)
    pdfplot.close()

Y = [MTrue,KTrue, GTrue]    
proba = [MProba,KProba, GProba]
flabels = ["MCF-7","K562","GM12878"]
plotROCCurves(Y, proba, flabels, directory)

def plotConfusionMatrix(ncm, title, cm, mcc, accuracy, labels,outdir, cmap=plt.cm.Blues):
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(ncm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    plt.title(title+" (MCC: "+str(mcc)+" Acc: "+str(accuracy)+")")
    plt.colorbar()
    for i in range(0,len(labels)):
        for j in range(0,len(labels)):
            plt.text(j,i,cm[i,j],va='center',ha='center')
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    pdfplot = PdfPages(outdir+title+"_t1.pdf");
    pdfplot.savefig(fig)
    pdfplot.close()

pred = dict()
for i in range(0,len(bestthresh)):
    predi = []
    for j in range(0,len(proba[i])):
        if(proba[i][j] >= bestthresh[i]):
            predi.append(1)
        else:
            predi.append(0)
    pred[i] = predi
    

print(bestthresh)
for i in range(0,len(bestthresh)):
    print(np.shape(Y[i]))
    print(np.shape(pred[i]))
    cm = confusion_matrix(Y[i],  pred[i]);
    ncm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    mcc = matthews_corrcoef(Y[i],  pred[i])
    acc = accuracy_score(Y[i],  pred[i])
    plotConfusionMatrix(ncm, flabels[i], cm, mcc, acc, ["P","BD"], directory+str(i))