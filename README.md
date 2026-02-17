Base paper :

Unawareness framework, regression (and classification) achieving demographic parity (Divol V., Gaucher S.):
https://hal.science/hal-04684789v1/document 

Useful papers :

Awareness method for fair regression in the sense of demographic parity (Chzhen Evgenii et al.):
https://proceedings.neurips.cc/paper_files/paper/2020/file/51cdbd2611e844ece5d80878eb770436-Paper.pdf
and code
https://github.com/lucaoneto/NIPS2020_Fairness 
TO DO : adapt this awareness method to an unawareness framework by plugging in instead of the true S value, its estimate delta, and compare this method with Divol and Gaucher. 

Regression under demographic parity constraints via
unlabeled post-processing
https://proceedings.neurips.cc/paper_files/paper/2024/file/d5c3ecf397fff63419bb5f5f2d8afe33-Paper-Conference.pdf 
TO DO : implement this method which did not come with theoretical guarantees but seems to be answering the same problem that Divol and Gaucher tackle

Mapping Estimation for Discrete Optimal Transport
https://papers.nips.cc/paper_files/paper/2016/file/26f5bd4aa64fdadf96152ca6e6408068-Paper.pdf
IDEA : this could be useful for out-of-sample prediction within our Divol and Gaucher implementation but R. Flamary deemed it a bit overkill -> we should rather test simple regression methods such as k-nn, random forests, gradient boosting, neural networks.

Jeux de données :

https://archive.ics.uci.edu/dataset/183/communities+and+crime

https://archive.ics.uci.edu/dataset/2/adult 

https://www.kaggle.com/datasets/danofer/compass 

Slides (beamer) :

https://www.overleaf.com/project/698b63cb383f80060ab40e24







