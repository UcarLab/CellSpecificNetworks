OrbitClustering.py
This script takes as input the files obtained after calculating orbit scores for each node annotation.  
Currently this script only works with the network node annotation file which includes the orbits.  
Data is available in the following repository: 

Running Example: 
python orbitclustering.py "<path to containing directory>/MCF7Network_NodeAnnotations_Orbits.txt" "<path>/K562Network_NodeAnnotations_Orbits.txt" "<path>/GMNetwork_NodeAnnotations_Orbits.txt" "<output directory>"


CrossValidation.py
This script runs the 5-fold cross validation.  It takes as arguments the orbit cluster features and a number of other files which select the best features and parameters we've identified using parallel computing.

Arguments (More information in CrossValidation.py):
1 - Path to the file defining the datasets to use.
2 - Path to the file defining the features to use from these datasets.
3 - Path to the file that define the classes on each node
4 - Specifies whether to use SVM or Random Forest
5 - Path to the file containing the hyperparameters specified for the model.
6 - Random state number.  929 was used in our analyses
7 - Output Directory
8 - Path to the file defining all of the features used in feature selection
9 - Path to the file defining the specific features to use for the current comparisons.

Running Example:
python CrossValidation.py "<path>/datasets.txt" "<path>/features.txt" "<path>/classes_EvSE.txt" svm "<path>/best_EvSE.txt" 929 "<outputdirectory" EvSE "<path>/featureselectionfeatures.txt" "<path>/EvSEFeatureIndices.txt"


ROCConfusionPlots.py
The output of CrossValidation provides files for the true labels as well as the probabilities.  Using these datasets, this script will generate the ROC Curves as well as the Confusion matrix plots.

Running Example:
python ROCConfusionPlots.py "<path>/MCF7_besttrue.csv" "<path>/MCF7_bestproba.csv" "<path>/K562_besttrue.csv" "<path>/CTS2/K562_bestproba.csv" "<path>/GM12878_besttrue.csv" "<path>/GM12878_bestproba.csv" "<output directory>"