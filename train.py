from single_perceptron import Simple_perceptron
import argparse

if __name__ == "__main__":

    EPOCHS = 100
    LEARNING_RATE = 0.1
    ACTIVATION = "tanh"
    SPLIT = 0.2
    SHUFFLE = False
    NORMALIZE = False

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("-d", "--dataset", help="Path to train dataset", metavar="filename", required=True)
    parser.add_argument("-e", "--epochs", help="Number of epochs", metavar='int', default=EPOCHS, required=False, type=int)
    parser.add_argument("-a", "--activation", help="Activation function", choices=['tanh','relu','sigmoid','softplus','gaussian'], default=ACTIVATION, metavar='function',required=False)
    parser.add_argument("-l", "--learning_rate", help="Learning rate", default=LEARNING_RATE, metavar='float', type=float, required=False)
    parser.add_argument("-s", "--split", help="Train/validation Split", default=SPLIT, metavar='int', type=float, required=False)
    parser.add_argument("-sh", "--shuffle", help="Shuffle dataset",default=SHUFFLE, metavar="boolean", type=bool, required=False )
    parser.add_argument("-n", "--normalize", help="Normalize dataset", default=True, metavar="boolean", type=bool, required=False)

    args = vars(parser.parse_args())

    input_filename = args["dataset"]
    epochs = args["epochs"]
    activation = args["activation"]
    learning_rate = args["learning_rate"]
    split = args["split"]
    shuffle = args["shuffle"]
    normalize = args["normalize"]
    
    p = Simple_perceptron(activation=activation, epochs=epochs,learning_rate=learning_rate)
    
    X,Y = p.read_input_data(input_filename)
    
    Xtrain, Ytrain, Xvalid, Yvalid = p.train_validation_split(X,Y, split=split, shuffle=shuffle)
    
    p.train(Xtrain,Ytrain,Xvalid,Yvalid)
    
    # Save model to a file (with .model extension)
    p.save_model(input_filename[:-3] + 'model')
