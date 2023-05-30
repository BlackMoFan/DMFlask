from flask import Flask, render_template, request, send_file
import pandas as pd
from io import StringIO
from collections import Counter
# from flask import Flask, render_template, request, jsonify, Response
#import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn import linear_model 
import numpy as np
import random as rd
import os
# import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
import csv

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/regression')
# def clustering():
#     X = df.iloc[:, 0:2] # the X variable is the first and second columns
#     y = df.iloc[:, -1] # the y variable is the last column
#     df = pd.read_csv('Jobstreet.csv', usecols=['Experience'])

#     # Find the mean of the column.
#     mean = df['Experience'].mean()

#     # Replace the empty rows with the mean.
#     df['Experience'].fillna(mean, inplace=True)

#     X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
#     X_train = np.array(X_train).reshape((len(X_train),2))
#     y_train = np.array(y_train).reshape((len(y_train),1))

#     X_test = np.array(X_test).reshape((len(X_test),2))
#     y_test = np.array(y_test).reshape((len(y_test),1))

#     model = linear_model.LinearRegression()
#     model.fit(X_train,y_train) 
#     #this snippet of code will be used to train the linearRegression model using the X and y variables.

#     y_pred = model.predict(X_test)
#     plt.figure(10.5)
#     sns.regplot(x=X_test[:,1], y = y_pred, scatter_kws={'color':'red'})


# @app.route('/Act0LoadData/')
# def activity0():
#     return render_template('Activity0.html')

#--------------- KNN PART
def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

class KNN:
    def __init__(self, k=3):
        self.k = k

    def Fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def Predict(self, X):
        predictions = [self._Predict(x) for x in X]
        return predictions

    def _Predict(self, x):
        # compute the distance
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
    
        # get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority vote
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]

@app.route('/KNNPage')
def activity1KNN():
    return render_template('activity1knn.html')

@app.route('/get_inputknn', methods=['POST'])
def get_inputknn():
    #data from form
    knnexperience = int(request.form['knnexperience'])
    knnsalary = int(request.form['knnsalary'])
    #processing of data
    data = pd.read_csv('Jobstreet.csv')
    X = data[['Experience', 'Salary']].values
    y = data['Job'].values
    # Euclidean Distance with scikitlearn
    # from sklearn.neighbors import KNeighborsClassifier
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
    # knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')# (n_neighbors=5) original
    # knn.fit(X_train, y_train)
    # knnpredicted = knn.predict(np.array([knnexperience, knnsalary]))
    # Euclidean Distance Manually
    from sklearn.model_selection import train_test_split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = KNN(k=2)#(k=5) original
    clf.Fit(X_train, y_train)
    knnpredicted = clf.Predict(np.array([knnexperience, knnsalary]))

    # html_list = "<ol>"
    # html_list = ""
    # for item in list(knnpredicted):
    #     html_list += "<li>{}</li><br />".format(item)
    # html_list += "</ol>"
    # return html_list

    return render_template('activity1knn.html', knnexperience=knnexperience, knnsalary=knnsalary, knnprediction=knnpredicted)

#--------------- KNN PART


#--------------- KMEANS PART
@app.route('/KMeansPage')
def activity2KMeans():
    return render_template('activity2kmeans.html')

# @app.route('/get_inputkmeans/<cropzonekey>')
# def images(cropzonekey):
#     return render_template('activity2kmeans.html', title=cropzonekey)

def draw_plot():
    print("Hello")
    #processing of dataimport pandas as pd
    data = pd.read_csv('Jobstreet.csv', encoding='utf-8')
    jobs = data['Job'].tolist()
    # print(jobs)
        
    # Create a dictionary to store the jobs and integers
    jobs_dictionary = {}

    # Iterate over the list of jobs
    for job in jobs:
        # Assign each job to an integer
        jobs_dictionary[jobs.index(job)] = job

    # Print the dictionary
    # print(jobs_dictionary)

    data['jobToInt'] = 0

    for i in range(len(data)):
        data['jobToInt'][i] = i
        
    # print(data)

    X = data.iloc[:, [4,3]].values # use Job and Salary
    # number of training samples
    m = X.shape[0]
    # number of features
    n = X.shape[1]
    # choosing the number of iteration which guarantee convergence
    n_iterations = 1000
    # number of clusters
    K = 10
    # Step 1. Initialize the centroids randomly
    centroids = np.array([]).reshape(n,0)
    for i in range(10):
        rand = rd.randint(0, m-1) # random
        centroids = np.c_[centroids, X[rand]] # randomize centroids

    # Main K-Means Clustering Part
    from collections import defaultdict
    Output = defaultdict()
    # output = {}

    for i in range(n_iterations):
        #step 2a.
        lst1 = [] # create list
        lst = np.array(lst1) # convert list to numpy array
        ED = np.array(lst).reshape(m, 0)
        for k in range(K):
            temporary_distance = np.sum((X-centroids[:, k])**2, axis=1)
            ED = np.c_[ED, temporary_distance]
        
        C = np.argmin(ED, axis=1)+1
        #step 2b.
        
        Y = {}
        lst1 = [] # create list
        lst = np.array(lst1) # convert list to numpy array
        for k in range(K):
            Y[k+1] = np.array(lst.reshape(2,0))
            #Horizontal Concatenation, regrouping -> clustered index C
        for i in range(m):
            Y[C[i]] = np.c_[Y[C[i]], X[i]]
            #Transpose
        for k in range(K):
            Y[k+1] = Y[k+1].T
            #Mean Computation & New Assigned centroid
        for k in range(K):
            centroids[:, k] = np.mean(Y[k+1], axis=0)

        Output = Y
        # print("Output:", output)

    color = ['red', 'blue', 'green', 'cyan', 'magenta', 'grey', 'yellow', 'pink', 'brown', 'orange']
    labels = ['cluster#1', 'cluster#2', 'cluster#3', 'cluster#4', 'cluster#5', 'cluster#6', 'cluster#7', 'cluster#8', 'cluster#9', 'cluster#10']

    for k in range(K):
        plt.scatter(Output[k+1][:,0],
            Output[k+1][:,1],
            c=color[k],
            label=labels[k])
        
    plt.scatter(centroids[0,:], centroids[1,:], s=150, c='yellow', label='centroid')
    plt.xlabel('Job')
    plt.ylabel('Salary')
    # plt.legend()
    plt.title('Plot of data points')
    # plt.show()
    # Save the figure in the static directory 
    plt.savefig(os.path.join('static', 'images', 'plot', 'plot.png'))
    # plt.close()
    # return plt

@app.route('/get_inputkmeans', methods=['POST'])
def get_inputkmeans():
    draw_plot()
    print("Hi")
    return render_template('activity2kmeans.html')

# @app.route('/fig/<cropzonekey>')
# def fig(cropzonekey):
#     fig = draw_plot(cropzonekey)
#     img = StringIO()
#     fig.savefig(img)
#     img.seek(0)
#     return send_file(img, mimetype='image/png')
#--------------- KMEANS PART

#--------------- NBayes PART
@app.route('/NBayesPage')
def activity3NBayes():
    return render_template('activity3nbayes.html')
#--------------- NBayes PART

#--------------- Regression PART
@app.route('/RegressionPage')
def activity4Regression():
    return render_template('activity4regression.html')
#--------------- Regression PART

#--------------- TextGen PART
@app.route('/TextGenPage')
def activity5TextGen():
    return render_template('activity5textgen.html')
#--------------- TextGen PART

#--------------- Classification PART
@app.route('/ClassificationPage')
def activity6Classification():
    return render_template('activity6classification.html')
#--------------- Classification PART

if __name__ == '__main__':
    app.run(debug=True)