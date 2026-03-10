## Code outline :

In OTUnawareFairRegressor.py we implemented the Divol and Gaucher optimal unaware fair regressor

In OTAwareFairRegressor.py we implemented the awareness method of Chzhen et al. as well as an adapted version of it which works in an unawareness framework by plugging in instead of the true S value, its estimate delta.

The unaware-fair-reg-third-method folder comes from the code of the paper by Taturyan et al. "Regression under demographic parity constraints via
unlabeled post-processing" and contains the files necessary (in particular FairReg.py) to use their method, which relaxes the demographic parity constraint to use convex optimization.

We obtained the visualizations in our slides with the following files : visualization_unawareness.py, visualization_routine.py, toy_example_visualization.ipynb, comparison.py, comparison_alpha.py for our generated data experiments as well as data_prep.py for preparation of "real world" datasets and the notebook real_world_data_visualization.ipynb for different model performance comparisons on the Communities and Crime and the Adult datasets. We explain in more details the use of the file at the beginning of the file.

A naive extension of Divol and Gaucher's method to the case of a non-binary sensitive attribute and a comparison of all models on the Communities and Crime dataset with the four different communities as the sensitive attribute S is given in extension_to_non_binary_sensitive_attribute.py

Some of the code is a bit duplicated between different files since we wanted to avoid conflicts when working at the same time on the git repository so we used different files for some tasks where we could have used only one.


## References and resources :

Base paper :

Unawareness framework, regression (and classification) achieving demographic parity (Divol V., Gaucher S.):
https://hal.science/hal-04684789v1/document

Useful papers :

Awareness method for fair regression in the sense of demographic parity (Chzhen Evgenii et al.):
https://proceedings.neurips.cc/paper_files/paper/2020/file/51cdbd2611e844ece5d80878eb770436-Paper.pdf
only with MATLAB code
https://github.com/lucaoneto/NIPS2020_Fairness
We adapted this awareness method to an unawareness framework by plugging in instead of the true S value, its estimate delta, and compared this method with Divol and Gaucher.

Regression under demographic parity constraints via
unlabeled post-processing
https://proceedings.neurips.cc/paper_files/paper/2024/file/d5c3ecf397fff63419bb5f5f2d8afe33-Paper-Conference.pdf
We also compared the previous models with this method which solves a relaxed version of the problem that Divol and Gaucher tackle

Mapping Estimation for Discrete Optimal Transport
https://papers.nips.cc/paper_files/paper/2016/file/26f5bd4aa64fdadf96152ca6e6408068-Paper.pdf
IDEA : this could be useful for out-of-sample prediction within our Divol and Gaucher implementation maybe a bit overkill -> we should rather test simple regression methods such as k-nn, random forests, gradient boosting, neural networks.

Jeux de données :

https://archive.ics.uci.edu/dataset/183/communities+and+crime

https://archive.ics.uci.edu/dataset/2/adult

https://www.kaggle.com/datasets/danofer/compass

Slides (beamer) :

https://www.overleaf.com/project/698b63cb383f80060ab40e24
