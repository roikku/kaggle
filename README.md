# Kaggle's Problems

Playground where some machine learning problems from [Kaggle](https://www.kaggle.com) are addressed.  

## Titanic: Machine Learning from Disaster

Two different approaches to the [Titanic problem](https://www.kaggle.com/c/titanic) are presented. The first one follows a *typical* supervised learning process to craft a binary classifier using a random forest algorithm: Feature tinkering, feature selection and grid search. The second one somewhat falls in the semi-supervised learning realm: after preparing adequate feature, an autoencoder---unsupervised learning technique aiming at reconstructing its input signal---is trained from the entire data set (including the unlebeled test set); then, the autoencoder is used as pre-trained layers to a neural network trained---using data from the training set only---to classify its input. Although the first approach seem to perform better, the second one yields a surprisingly good outcome, somwehat comparable to the first approach. 

To run the R scripts, first download the data sets on Kaggle's [dedicated page](https://www.kaggle.com/c/titanic/data), and copy the two files `train.csv` and `test.csv` in the folder `data` (located in the working directory where the scripts are).

#### Random Forest

- [Kaggle's Titanic Toy Problem with Random Forest](http://www.619.io/blog/2017/06/20/kaggle-s-titanic-toy-problem-with-random-forest/) <small>**[**blog post**]**</small>
- [Random Forest, Forward Feature Selection & Grid Search](https://goo.gl/CqfMg4) <small>**[**Kaggle Kernels**]**</small>

#### Autoencoder

- [Kaggle's Titanic Toy Problem Revisited](http://www.619.io/blog/2017/06/24/kaggle-s-titanic-toy-problem-revisited/) <small>**[**blog post**]**</small>
- [Diabolo Trick: Pre-train, Train and Solve](https://goo.gl/JXxc6n) <small>**[**Kaggle Kernels**]**</small>

## License and Source Code  

&copy; 2017 Loic Merckel, [Apache v2](https://www.apache.org/licenses/LICENSE-2.0.html) licensed. The source code is available on [GitHub](https://github.com/roikku/kaggle/).
