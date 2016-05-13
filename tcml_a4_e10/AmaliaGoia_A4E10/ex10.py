import numpy as np
import sklearn
from matplotlib.pyplot import savefig


def read_data(filename):
    from bokeh.models import pd
    df = pd.read_csv(
        filepath_or_buffer=filename,
        header=None,
        sep=',')
    data = np.array(df.ix[:, :].values)
    data = np.hstack([np.ones((data.shape[0], 1)), data])
    return data

def split_randomly(data):
    from math import floor
    import numpy as np

    np.random.shuffle(data)
    # test = data[:, :]
    test = data[floor(len(data)/2):,:]
    train = data[:floor(len(data)/2),:]
    # train = data[:, :]
    return test, train


def separate_labels_from_features(data):
    x = data[:, :-1].T  # data matrix: features as lines, samples as columns
    y = data[:, -1]  # labels row vector
    return x, y


def split_by_class(x, y):
    x_plus1_coordX1 = [x[1, i] for i in range(y.shape[0]) if y[i] == 1]
    x_plus1_coordX2 = [x[2, i] for i in range(y.shape[0]) if y[i] == 1]
    x_minus1_coordX1 = [x[1, i] for i in range(y.shape[0]) if y[i] == 0]
    x_minus1_coordX2 = [x[2, i] for i in range(y.shape[0]) if y[i] == 0]
    return x_plus1_coordX1, x_plus1_coordX2, x_minus1_coordX1, x_minus1_coordX2


############################################################################################################

def sigmoid(x):
    '''
    :return: the value of the sigmoid function for the given x (aka the prob. to be a positive example)
    '''
    import numpy as np
    return 1.0 / (1.0 + np.exp(-x))


def compute_error(w, x, y):
    error = y.T * sigmoid(w.T * x).T + (1 - y.T) * (1 - sigmoid(w.T * x).T)
    return -error


def compute_log_likelihood(w, x, y):  # the cost function
    # compute the probability for class 1
    prob_1 = sigmoid(np.dot(w.T, x))
    # compute the value of the log likelihood vector (aka cross entropy error)
    log_likelihood = (y) * np.log(prob_1) + (1 - y) * np.log(1 - prob_1)
    # this shows how likely is each data observation to appear - the log likelihood of the whole ds is the mean
    return -log_likelihood.sum()


def logistic_gradient(w, x, y):
    """
    :param w: parameter vector
    :param x: data matrix
    :param y: label vector
    :return: a vector representing the gradient dL/dw
    """
    gradi = -np.dot(y - sigmoid(np.dot(w.T, x)), x.T)
    return gradi[0].reshape(len(w), 1)


def numerical_gradient(w, x, y):
    """
    :param w: parameter vector
    :param x: data matrix
    :param y: label vector
    :return: the gradient dL/dw computed using the central difference quotient
    """
    eps = 1 / 1000

    grad_approx = np.zeros(w.shape)

    for i in range(len(w)):
        current_ei = np.zeros(w.size).reshape(w.size, 1)
        current_ei[i] = eps
        L_plus_eps = compute_log_likelihood(w + current_ei, x, y)
        L_minus_eps = compute_log_likelihood(w - current_ei, x, y)
        grad_approx[i] = (L_plus_eps - L_minus_eps) / (2 * eps)
    return grad_approx


def gradient_descent(learning_rate, w, x, y):
    iteration = 0
    epsilon = 0.01
    error = 0

    while True:
        # compute the gradients
        grad = logistic_gradient(w, x, y)
        w = w - learning_rate * grad
        prev_error = error
        error = np.sum(abs(y - sigmoid(np.dot(w.T, x))), axis=1)
        # print("Iteration {0}, error {1}, w = {2}".format(iteration, error, w))
        if abs(error - prev_error) < epsilon or iteration > 500:
            break
        iteration += 1

    return w


########################################################################################

def plot_points_and_separation_line(x, y, w, title):
    import matplotlib.pyplot as plt

    x_plus1_coordX1, x_plus1_coordX2, x_minus1_coordX1, x_minus1_coordX2 = split_by_class(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_plus1_coordX1, x_plus1_coordX2, s=30, c='red', marker='s')
    ax.scatter(x_minus1_coordX1, x_minus1_coordX2, s=30, c='blue')

    plt.xlabel("X1 coord")
    plt.ylabel("X2 coord")
    plt.title(title)

    # plot separation line
    coord_x_line = np.linspace(-1, 1, 100)

    # the weights are the coordinates of the line
    # w0*x+w1*y+w2= 0
    # solve for y: => y = (-w2-wo*x)/w1
    coord_y_line = (-w[0] - w[1] * coord_x_line) / w[2]
    ax.plot(coord_x_line, coord_y_line.T, 'g')

    savefig(title+".png", bbox_inches='tight')
    # plt.show()


###################################################################################

def compute_prediction(w, xtest):
    prediction_fract = sigmoid(np.dot(w.T, xtest))
    prediction_m11 = []
    for i in range(prediction_fract.shape[1]):
        if prediction_fract[0][i] > 0.5:
            prediction_m11.append(1)
        else:
            prediction_m11.append(0)
    return prediction_m11


def compute_classification_based_confusion_matrix(predictions, true_labels):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for i in range(len(predictions)):
        if predictions[i] == true_labels[i]:
            if predictions[i] == 1:
                true_positive += 1
            else:
                true_negative += 1
        else:
            if predictions[i] == 1:
                false_positive += 1
            else:
                false_negative += 1
    return true_positive, true_negative, false_positive, false_negative


# formulas from http://people.inf.ethz.ch/bkay/talks/Brodersen_2010_06_21.pdf
def compute_accuracy(tp, tn, n):
    return (tp + tn) / n


def compute_balanced_accuracy(tp, tn, fp, fn):
    return (1 / 2) * (tp / (tp + fn) + tn / (fp + tn))


###################################################################################################

def plot_roc_curve(ytest, y_prediction_as_prob_of_being_1, filename):
    import matplotlib.pyplot as plt
    import sklearn.metrics
    # Plot of a ROC curve

    fpr, tpr, _ = sklearn.metrics.roc_curve(ytest, y_prediction_as_prob_of_being_1)
    roc_auc = roc_auc_score(ytest, y_prediction_as_prob_of_being_1)

    fig = plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic ' + str(filename))
    print("The ROC AUC for {0} is {1}.".format(filename, roc_auc))
    plt.legend(loc="lower right")
    txt = '''
            The ROC (receiver operating characteristic) curve provides a visualization of
            the performance of a binary classifier (like logistic regresion here). The AUC
            (area under curve) summarizes it in a single number. It plots the True positive
            rate against the False positive rate for every possible classification threshold.
            For example, a threshold of 0.2 would classify 30/100 points as 1 and 70/100 points as 0.
            '''
    # fig.text(.1, .1, txt)

    savefig("ROC-"+str(filename)+".png", bbox_inches='tight')


def plot_precision_recall_curve(ytest, y_prediction_as_prob_of_being_1, filename):
    import matplotlib.pyplot as plt

    precision, recall, _ = sklearn.metrics.precision_recall_curve(ytest, y_prediction_as_prob_of_being_1)
    average_precision = sklearn.metrics.average_precision_score(ytest, y_prediction_as_prob_of_being_1)

    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve' + str(filename) + ": AUC={0:0.2f}".format(average_precision))
    print("The average precision (AUC) for {0} is {1}.".format(filename,average_precision))
    plt.legend(loc="lower left")
    # plt.show()
    savefig("Prec-Recall-" + str(filename)+".png", bbox_inches='tight')


##############################################################################
if __name__ == "__main__":
    datasetnames = ["DataSet3.csv", "DataSet4.csv"]
    for filename in datasetnames:
        # read data and split it in train and test
        data = read_data("DataSet4.csv")
        test, train = split_randomly(data)
        # extract the labels
        xtrain, ytrain = separate_labels_from_features(train)
        xtest, ytest = separate_labels_from_features(test)

        # initialize the weights vector with 1
        # each weight for a coefficient of the separation line: ax+by+c=0
        nr_weights = xtrain.shape[0]
        w = np.ones((nr_weights, 1))

        # perform gradient checking by computing the gradient and its numerical approximation
        # print(logistic_gradient(w, xtrain, ytrain))
        # print(numerical_gradient(w, xtrain, ytrain))

        # compute the final weights using gradient descent on the training set with a learning
        # rate of 0.01
        w = gradient_descent(0.01, w, xtrain, ytrain)

        # for visualization purposes
        plot_points_and_separation_line(xtrain, ytrain, w, "Training set " + str(filename))
        plot_points_and_separation_line(xtest, ytest, w, "Testing set " + str(filename))

        # compute the predictions on the test set
        predictions = compute_prediction(w, xtest)
        # compute true positive, true neg, false pos, false neg based on true labels and predictions
        tp, tn, fp, fn = compute_classification_based_confusion_matrix(predictions, ytest)

        # compute accuracy and balanced accuracy
        """
        Given a confusion matrix of classification results, the
        accuracy can be a misleading performance measure. Specifically,
        it may falsely suggest above-chance generalizability.

        It is a well-known phenomenon in binary classification
        that a training set consisting of different numbers of representatives
        from either class may result in a classifier that
        is biased towards the more frequent class. When applied
        to a test set that is imbalanced in the same direction, this
        classifier may yield an optimistic accuracy estimate. In an
        extreme case, the classifier might assign every single test
        case to the large class, thereby achieving an accuracy equal
        to the fraction of the more frequent labels in the test set.

        This observation
        motivates the use of a different generalizability
        measure: the balanced accuracy, which can be defined as
        the average accuracy obtained on either class.
        (via http://ong-home.my/papers/brodersen10post-balacc.pdf)
        """
        accuracy = compute_accuracy(tp, fn, len(ytest))
        balanced_accuracy = compute_balanced_accuracy(tp, tn, fp, fn)

        print("The accuracy for "+str(filename)+" is {0} and the balanced accuracy is {1}".format(accuracy, balanced_accuracy))

        # Compute ROC curve and ROC area
        from sklearn.metrics import roc_auc_score

        y_prediction_as_prob_of_being_1 = sigmoid(np.dot(w.T, xtest))[0]

        plot_roc_curve(ytest, y_prediction_as_prob_of_being_1, filename)

        # Compute Precision/Recall curve
        plot_precision_recall_curve(ytest, y_prediction_as_prob_of_being_1, filename)

'''
Comments:
The accuracy for DataSet3.csv is 0.26285714285714284 and the balanced accuracy is 0.7578361981799797
The ROC AUC for DataSet3.csv is 0.8690596562184024.
The average precision (AUC) for DataSet3.csv is 0.7011633213103383.

The accuracy for DataSet4.csv is 0.2571428571428571 and the balanced accuracy is 0.7102564102564103
The ROC AUC for DataSet4.csv is 0.8905982905982907.
The average precision (AUC) for DataSet4.csv is 0.7203145041300328.

One can notice that the simple accuracy is not a very good indicator of the performance of the classifier,
the balanced accuracy coming closer to it.
'''