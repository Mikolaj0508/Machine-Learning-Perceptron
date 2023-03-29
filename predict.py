from single_perceptron import Simple_perceptron
import argparse
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('-t', '--testset', help="Path to test dataset",metavar="filename", required=True)
    parser.add_argument('-m', '--model', help="Path to model file",metavar="filename", required=True)

    args = vars(parser.parse_args())
    testfile = args["testset"]
    modelfile = args["model"]

    p = Simple_perceptron()

    p.load_model(modelfile)

    Xtest, Yexpected = p.read_input_data(filename=testfile)
    Yout = p.test(Xtest=Xtest)

    Xtest, Yout, Yexpected = p.denormalize(Xtest, Yout, Yexpected)

    print("Test results: ")

    for i in range(len(Yout)):
        if "sum" in testfile:
            print(f"{Xtest[i][0]:.3f} + {Xtest[i][1]:.3f} = {Yout[i]:.3f} (expected {Yexpected[i]:.3f})")
        else:
            print(f"obtained: {Yout[i]}, expected {Yexpected[i]}")

    #Scores: RMSE and R squared score
    sse = sum((np.array(Yexpected) - np.array(Yout))**2)
    tse = (len(Yexpected) - 1) * np.var(Yexpected, ddof=1)
    rmse = np.sqrt(sse / len(Yout))
    r2_score = 1 - (sse / tse)
    print("\nRMSE score      = {:.2f}".format(rmse))
    print("R squared score = {:.2f}".format(r2_score))



