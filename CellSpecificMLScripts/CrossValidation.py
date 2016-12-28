import sys
import pandas as pd
import numpy as np

from sklearn.cross_validation import StratifiedKFold

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix,roc_curve, auc
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

args = sys.argv

#arg1: File containing list of files to process (two column label, filepath)
filedata = pd.read_csv(args[1], sep="\t", header=None)
flabels = filedata.iloc[:,0].values
files = filedata.iloc[:,1].values

#arg2: File for feature columns (two column specifying ranges [start,end) )
featurecoldata = pd.read_csv(args[2], sep="\t", header=None).values

features = []
for i in range(0, len(featurecoldata)):
    features.extend(range(featurecoldata[i,0], featurecoldata[i,1]))

#arg3: File for classes (three column: column, value, conversion) 
#conversion must be 0, 1, or -1 where -1 means the data element is not used
classconversion = pd.read_csv(args[3], sep="\t", header=None).values

#arg4: 'svm' or 'rf' to select ML algorithm
issvm = args[4] == 'svm'

#arg5: File containing CM parameters eg.
#l1 Labels
#2-Number of Files, two columns representing the parameters
cmparams = pd.read_csv(args[5], sep="\t", header=None)
labels = cmparams.iloc[0,:].values
p1 = cmparams.iloc[0:,0].values
p2 = cmparams.iloc[0:,1].values

#arg6: RState
rstate = int(args[6])

#arg7 output directory
outdir = args[7]
if not(outdir.endswith("/")):
    outdir = outdir+"/"

roctitle = args[8]

rfeatures = pd.read_csv(args[9], sep="\t")
rf_labels = rfeatures.columns.values
rfeatures = rfeatures.values
maxfeature = int(np.nanmax(rfeatures))+1

#Feature indexes each column corresponds to a single dataset, each row corresponds to a feature included for that dataset
rfidata = pd.read_csv(args[10], sep="\t", header=None).values
rfi = dict()
rfi[0] = list(rfidata[~np.isnan(rfidata[:,0]),0])
rfi[1] = list(rfidata[~np.isnan(rfidata[:,1]),1])
rfi[2] = list(rfidata[~np.isnan(rfidata[:,2]),2])

labelencoders = []
def getData(filepath, featurecols):
    data = pd.read_csv(filepath, sep="\t")
    #Replace String features with numeric values
    for i in range(0, len(labelencoders)):
        le = preprocessing.LabelEncoder()
        le.fit(labelencoders[i][1:])
        data.iloc[:, labelencoders[i][0]] = le.transform(data.iloc[:, labelencoders[i][0]].values)
        
    #select the appropriate columns for the features
    features = data.iloc[:, featurecols].values 
    features = features.astype(float)
    
    featurelabels = data.columns[featurecols].values
    rids = data.iloc[:, [0,1,2,3]].values 
    
    
    #set classes
    classes = np.empty(len(features))
    classes[:] = -1
    for i in range(0, len(classconversion)):
        classcol = data.iloc[:, classconversion[i,0]].values
        classes[classcol == classconversion[i,1]] = classconversion[i,2]
    classes = classes.astype(int)

    #return features classes and labels, removing unclassified (-1)
    return [features[classes[:] > -1,:], classes[classes[:] > -1], featurelabels, rids[classes[:] > -1,:]]

def plotROCCurves(test, predprob, labels, suffix):
    fig = plt.figure(figsize=(10, 10))
    
    colors = ["blue", "red", "green"]
    for i in range(0, len(labels)):
        fpr, tpr, _ = roc_curve(test[i], predprob[i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=labels[i]+' (area = %0.2f)' % roc_auc, color=colors[i])
 
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(roctitle)
    plt.legend(loc="lower right")
    plt.show()
    pdfplot = PdfPages(outdir+"_"+roctitle+"_roccurves_"+suffix+".pdf");
    pdfplot.savefig(fig)
    pdfplot.close()


#Set up the imputer and scaler
imputer = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
scaler = preprocessing.StandardScaler()


fally = dict()
fallproba = dict()
fdata = dict()
for i in range(0, len(files)):    
    included = rfi[i]   
    
    if issvm: 
        la = SVC(class_weight='balanced',random_state=rstate, C=float(p1[i]), gamma=float(p2[i]), probability=True)
    else:
        param1 = p1[i]
        param2 = p2[i]
        if param1 != 'auto' and param1 != 'sqrt' and param1 != 'log2':
            try:
                param1 = int(param1)
            except (ValueError, TypeError):
                try:
                    param1 = float(param1)
                except (ValueError):
                    param1 = None
        la = RandomForestClassifier(class_weight='balanced',random_state=rstate, n_estimators=int(param2), max_features=param1, probability=True)
    
    clf = Pipeline(steps=[('imp', imputer), ('std', scaler), ('la', la)])
    
    X, y, curfl, rids = getData(files[i], features);
    X = X[:,included]
    cv = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=rstate)
    
    allpredict = [None]*len(y)
    fallproba[i] = [None]*len(y)

    for train, test in cv:
        clf.fit(X[train,:], y[train])
        predict = clf.predict(X[test,:])
        proba = clf.predict_proba(X[test,:])
        for k in range(0, len(test)):
            allpredict[test[k]] = predict[k]
            fallproba[i][test[k]] = proba[k,1]

    fdata[i] = rids
    fally[i] = y;
    cm = confusion_matrix(y,  allpredict);
    ncm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    mcc = matthews_corrcoef(y, allpredict)
    acc = accuracy_score(y, allpredict)
    

for i in range(0, len(flabels)):
    pd.DataFrame(fally[i]).to_csv(outdir+flabels[i]+"_besttrue.csv", index=False, header=None)
    pd.DataFrame(fallproba[i]).to_csv(outdir+flabels[i]+"_bestproba.csv", index=False, header=None)
    pd.DataFrame(fdata[i]).to_csv(outdir+flabels[i]+"_bestdata.csv", index=False, header=None)

plotROCCurves(fally, fallproba, flabels, "best")
