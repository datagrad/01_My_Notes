


A Comparative study of various ML Models:

| Model                  | Supervised/Unsupervised | Data Requirement                                  | Performance Metrics                 | Performance Tuning                                                                                              |
|------------------------|------------------------|---------------------------------------------------|-------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| Linear Regression      | Supervised             | Continuous numerical data                         | MSE, R-squared                      | Feature engineering, regularization (L1/L2), handling multicollinearity                                    |
| Logistic Regression    | Supervised             | Binary or multiclass classification               | Accuracy, Precision, Recall         | Regularization (C parameter), feature engineering                                                            |
| Decision Trees         | Supervised             | Categorical or numerical data                     | Accuracy, Gini Impurity             | Pruning, maximum depth, minimum samples per leaf                                                              |
| Random Forest          | Supervised             | Categorical or numerical data                     | Accuracy, Gini Impurity             | Number of trees, maximum depth, minimum samples per leaf                                                     |
| Support Vector Machines| Supervised             | Binary or multiclass classification               | Accuracy, Precision, Recall         | Choice of kernel, regularization parameter (C), kernel-specific parameters                                   |
| k-Nearest Neighbors    | Supervised (or unsupervised) | Numerical data                               | Accuracy, Precision, Recall         | Number of neighbors (k), distance metric                                                                      |
| Naive Bayes            | Supervised             | Text classification, categorical data            | Accuracy, Precision, Recall         | Laplace smoothing, feature engineering                                                                       |
| Principal Component Analysis (PCA) | Unsupervised | Numerical data                              | Variance explained, Reconstruction error | Number of components, data scaling                                                                             |
| K-Means Clustering     | Unsupervised           | Numerical data                                  | Inertia, Silhouette Score           | Number of clusters, initialization method                                                                    |
| Hierarchical Clustering| Unsupervised           | Numerical or categorical data                   | Linkage, Cophenetic correlation     | Linkage method, distance metric                                                                               |
| Gaussian Mixture Model| Unsupervised           | Numerical data                                  | Log-likelihood                     | Number of components, covariance type, initialization method                                                |
| Support Vector Clustering| Unsupervised         | Numerical data                                  | Margin, Silhouette Score           | Kernel choice, regularization parameter (C)                                                                  |
| Neural Networks        | Supervised             | Numerical data                                  | Accuracy, Loss                      | Architecture design, activation functions, optimization algorithm                                           |
| Gradient Boosting      | Supervised             | Categorical or numerical data                   | Various metrics (e.g., RMSE, Log-loss) | Learning rate, number of estimators, depth of trees                                                          |
| XGBoost                | Supervised             | Categorical or numerical data                   | Various metrics (e.g., RMSE, Log-loss) | Learning rate, number of estimators, maximum depth, regularization                                           |
| LightGBM               | Supervised             | Categorical or numerical data                   | Various metrics (e.g., RMSE, Log-loss) | Learning rate, number of leaves, boosting type, maximum bin, feature fraction                               |

Remember that this table is not exhaustive, and there are many other machine learning models and algorithms out there. Each model comes with its own characteristics, strengths, and limitations, so choosing the right model depends on your specific problem and data.
