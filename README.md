Machine Learning Challenges

Description: 

This program is to showcase adaptation to label shifts via Lipton et al's BBSC implementation. The classifiers used in the code are as follows:

rf: Random Forest

gpc: Gaussian Process

3nn: 3 Nearest Neighbors

9nn: 9 Nearest Neighbors

d-freq: Dummy classifier using most frequent strategy (Baseline 1)

d-strat: Dummy classifier using stratified strategy (Baseline 2)

The csv files are sourced and then modified from the following website: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents

The program will read in the csv files and convert the data to dataframe format. It will then train the above 6 models on train-tx. The program will then use the trained
classifiers to predict the class labels of the items in the datasets val-tx, test1-tx, test2-fl and test3-fl. It will then apply Lipton's BBSC label shift adaptation on
the 3 test sets. The program will output the following: 

1. Feature Importance as determined by the random forest classifier
2. Confusion matrices as obtained by each classifier on each test set
3. The new accuracy of the predictions by each classifier after BBSC is used
4. The list of weights for each class as obtained by the BBSC method

Instructions:

To run the program, clone the repository and open the directory. Run the program in zsh using python3 main.py


Packages required: 

Numpy, Pandas, Scikit-Learn, MatPlotLib, Scipy

Links to install these packages:

Numpy: https://numpy.org/install/

Pandas: https://pandas.pydata.org/docs/getting_started/install.html

Scikit-Learn: https://scikit-learn.org/stable/install.html

MatPlotLib: https://matplotlib.org/stable/users/installing/index.html

Scipy: https://scipy.org/install/


Credits: 

This was one of the homework assignment associated with the Machine Learning Challenges (Winter 2024 iteration) course at Oregon State University. All credits belong to the course developer, Dr. Kiri Wagstaff and the lecturer-in-charge, Prof. Rebecca Hutchinson. All code is written by myself solely for the purpose of the assignment. For more information on the course developer and lecturer-in-charge:

Dr. Kiri Wagstaff: https://www.wkiri.com/

Prof. Rebecca Hutchinson: https://hutchinson-lab.github.io/

The label_shift_adaptation.py file was proposed by Lipton et al. and written by Dr. Kiri Wagstaff. The link to the paper by Lipton et al is: https://arxiv.org/pdf/1802.03916.pdf


Use: 

The code shall be used for personal educational purposes only. Students of current (Winter 2024) and future iterations of Machine Learning Challenges at Oregon State University may not use any code in this repo for this assignment should this or a similar assignment be assigned. If any Oregon State University student is found to have plagarized any code in this repo, the author of the repository cannot be held responsible for the incident of plagarism. The author promises to cooperate in any investigations regarding plagarism pertaining to this repo if required. If any of the code in this repo is reused for strictly personal projects, please credit this repository. 
