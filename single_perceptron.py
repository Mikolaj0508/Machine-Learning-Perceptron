import numpy as np
import yaml
import matplotlib.pyplot as plt
import csv
import sys
from timeit import default_timer as timer

class Simple_perceptron:

    def __init__(self,activation="tanh",epochs=100,learning_rate=0.1):
        self.activation = activation
        self.epochs = epochs
        self.learning_rate = learning_rate
        random_seed = 100
        np.random.seed(random_seed)

    def f(self, x):
        if self.activation == "tanh":
            f =  np.tanh(x)
        elif self.activation == "relu":
            if x > 0:
                f=x
            else:
                f=0
        elif self.activation == "sigmoid":
            f = 1/(1+np.exp(-x))
        elif self.activation == "softplus":
            f = 1/(np.log(1+np.exp(x)))
        elif self.activation == "gaussian":
            f = np.exp(-(x**2))
        else:
            sys.exit("No such activation function")
        return f

    def f_derivative(self,x):
        if self.activation == "tanh":
            fp =  1 - np.tanh(x)*np.tanh(x)
        elif self.activation == "relu":
            if x > 0:
                fp=1
            else:
                fp=0
        elif self.activation == "sigmoid":
            fp = np.exp(-x)/(np.exp(-x)+1)**2
        elif self.activation == "softplus":
            fp = 1/(1+np.exp(-x))
        elif self.activation == "gaussian":
            fp = -2*x*np.exp(-(x**2))
        else:
            sys.exit("No such activation function")
        return fp

    def read_input_data(self, filename, normalize=True):
        '''
        Reads input data (train or test) from the CSV file.
        Parameters:
            filename - CSV file name (string)
                CSV file format:
                    input1, input2, ..., output
                                    ...
                                    ...
            normalize - flag for data normalization (bool, optional)
        Sets:
            self.Nin = number of inputs of the d (int)
        Returns:
            X - input training data (list)
            Y - output (expected) training data (list)
        '''

        # Read CSV data
        try:
            file = open(filename, 'rt')
        except FileNotFoundError:
            sys.exit('Error: data file does not exists.')

        dataset = csv.reader(file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)

        # Construct the X and Y lists. This is a simple perceptron with only one output,
        # so X should contain all data from all columns except the last one,
        # and Y - data from the last column only.
        X = []
        Y = []
        try:
            for line in dataset:
                X.append(line[0:-1])
                Y.append(line[-1])
        except ValueError:
            sys.exit('Error: Wrong format of the CSV file.')

        file.close()

        # Store the size of the input vector (Nin) as a class property
        self.Nin = len(X[0])

        if self.Nin == 0:
            sys.exit('Error: zero-length training vector.')

        # Normalize data, if requested
        if normalize:
            X,Y = self.normalize(X, Y)

        return X,Y

    def initalize_weights(self):

        self.weights = np.random.random(self.Nin)

    def normalize(self,X,Y):
        '''
        Normalizes the data and stores normalization parameters as properties.
        Parameters:
            X - X-vector to normalize (list)
            Y - Y-vector to normalize (list)
        Sets:
            self.min_val - minimum value used in normalization
            self.max_val - maximum value used in normalization
        Returns:
            normalized vectors X, Y (lists)
        '''

        if not hasattr(self,"min_val"):
            self.min_val = min(np.amin(X),np.amin(Y))
        
        if not hasattr(self,"max_val"):
            self.max_val = max(np.amax(X),np.amax(Y))

        X = (X - self.min_val) / (self.max_val - self.min_val)
        Y = (Y - self.min_val) / (self.max_val - self.min_val)

        return X,Y

    def denormalize(self,*X):
        '''
        "Unnormalizes" vector(s), using previously determined minimum and maximum values.
        Parameters:
            X - tuple of vector(s) to normalize (lists)
        Returns:
            tuple of vectors of "unnormalized" vector(s) (lists)
        '''

        if hasattr(self,"min_val") and hasattr(self,"max_val"):
            Xout = []
            for x in X:
                Xout.append([i*(self.max_val - self.min_val)+self.min_val for i in x])
        else:
            print("Cannot denormalize data")
            Xout = X
        
        return Xout

    def train_validation_split(self,X,Y,split=0.2, shuffle=False):
        '''
        Splits the input vectors into the train and validation ones.
        Parameters:
            X - X-vector to be splitted (list)
            Y - Y-vector to be splitted (list)
            split - splitting factor (float in range [0.1-0.9], optional)
            shuffle - data shuffling flag (bool, optional)
        Returns:
            splitted Xtrain, Ytrain, Xvalid, Yvalid (lists)
        '''

        if split>0.9:
            print("Wrong split value, adjusted to 0.9")
            split=0.9
        if split<0.1:
            print("Wrong split value, adjusted to 0.1")
            split = 0.1

        data_size = len(X)
        valid_data_size = int(split*data_size)

        valid_random_indexes = sorted(list(np.random.choice(
            data_size,
            size=valid_data_size,
            replace=False)))

        if shuffle:
            randomize = np.arange(len(X))
            np.random.shuffle(randomize)
            X = X[randomize]
            Y = Y[randomize]

        Xvalid, Yvalid, Xtrain, Ytrain = [],[],[],[]

        for i in range(data_size):
            if i in valid_random_indexes:
                Xvalid.append(X[i])
                Yvalid.append(Y[i])
            else:
                Xtrain.append(X[i])
                Ytrain.append(Y[i])

        return Xtrain, Ytrain, Xvalid, Yvalid

    def train(self,Xtrain,Ytrain,Xvalid,Yvalid):
        '''
        Trains the simple perceptron using the gradient method.
        Parameters:
            Xtrain - training (input) vector (list)
            Ytrain - training (output) vector (list)
            Xvalid - validating (input) vector (list)
            Yvalid - validating (output) vector (list)
        Returns:
            None
        '''
        start_time = timer()

        self.initalize_weights()

        RMSE_train=[]
        RMSE_valid=[]

        for epoch in range(self.epochs):
            print(f"Epochs: {epoch+1}")

            sumRMSE_train=0
            for i in range(len(Xtrain)):
                sumWeighted=0
                for j in range(self.Nin):
                    sumWeighted += self.weights[j]*Xtrain[i][j]
                Yout = self.f(sumWeighted)

                for j in range(self.Nin):
                    self.weights[j] += -1*self.learning_rate*self.f_derivative(sumWeighted)*(self.f(sumWeighted)-Ytrain[i])*Xtrain[i][j]

                sumRMSE_train += (Yout-Ytrain[i])**2
            
            RMSE_train.append(np.sqrt(sumRMSE_train/len(Xtrain)))
            print(f'RMSE (training set) = {RMSE_train[epoch]}')

            if len(Xvalid)>0:
                sumRMSE_valid=0
                for i in range(len(Xvalid)):
                    sumWeighted = 0
                    for j in range(self.Nin):
                        sumWeighted += self.weights[j]*Xvalid[i][j]
                    Yout = self.f(sumWeighted)
                    sumRMSE_valid += (Yout-Yvalid[i])**2

                RMSE_valid.append(np.sqrt(sumRMSE_valid/len(Xvalid)))
                print(f'RMSE (validating set) = {RMSE_valid[epoch]}')
        print('\nTraining completed in {:.2f} seconds.'.format(timer()-start_time))
 
        self.save_plot(RMSE_train,RMSE_valid)

    def test(self, Xtest):
        '''
        Test of the trained perceptron.
        Parameters:
            Xtest - test vector (list)
        Returns:
            Y - output from the perceptron (list)
        '''

        Y = []
        for i in range(len(Xtest)):
            sum_weighted = 0
            for j in range(self.Nin):
                sum_weighted += self.weights[j]*Xtest[i][j]
            Y.append(self.f(sum_weighted))

        return  Y
            
    def save_plot(self, RMSE_train, RMSE_valid, filename='loss.png',show=True):

        plt.plot(RMSE_train, label="RMS (train data set")
        plt.plot(RMSE_valid, label="RMS (validation data set")
        plt.legend()
        plt.title("Results of training of single perceptron")
        plt.xlabel("Epochs")
        plt.ylabel("RMSE")
        plt.savefig(fname=filename)
        if show:
            plt.show()
        print('RMSE plot has been saved to the file', filename)

    def save_model(self,filename):

        weights = self.weights.tolist()

        data ={
            "epochs" : self.epochs,
            "activation" : self.activation,
            "learning_rate": self.learning_rate,
            "weights": weights
        }

        with open(filename,'w') as file:
            yaml.dump(data,file)

    def load_model(self,filename):

        try:
            with open(filename,'r') as file:
                data = yaml.safe_load(file)
                self.activation = data["activation"]
                self.epochs = data["epochs"]
                self.learning_rate = data["learning_rate"]
                self.weights = np.array(data["weights"])
        except:
            sys.exit('Error: model file does not exists.')

        print('Model loaded from file', filename)
