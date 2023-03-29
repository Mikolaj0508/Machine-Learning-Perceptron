# Machine Learning Perceptron

## Microcomputer part

The objective of this project is to better understand the fundamentals of machine learning by implementing a simple neural network called the Perceptron from scratch. The project will involve developing a basic Perceptron algorithm using Python without using any pre-built machine learning packages. By building the Perceptron from scratch, we will gain a deeper understanding of the core principles behind machine learning, such as how data is transformed into useful features, how weights are updated during training, and how a model makes predictions.

The scope of the project will be limited to developing a simple Perceptron algorithm that can handle multiple input features but can only produce a single output variable. This will involve designing and implementing the core components of the algorithm, including the activation function, the weight update rule, and the decision boundary. By focusing on this specific task, we will be able to develop a clear and concise understanding of how the Perceptron works.

We may also experiment with different activation functions, such as sigmoid or ReLU, to improve the model's performance. Next step might be implement from scratch multi-layer Perceptron (MLP)


## Installation

To set the things up, you have to do following steps:

Create virtual environment
```bash
  virtualenv venv
```

Clone the project

```bash
  git clone https://github.com/Mikolaj0508/Solar-Tracker.git
```

Go to the project directory

```bash
  cd my-project
```
Activate virtual environment

```bash
  source venv/bin/activate
```

Install all required packages

```bash
  pip install -r requirements.txt
```
## How to use it

To use Perceptron first of all ensure data in csv format to train, validate and test your model. Then call file train.py. It might look like this:

```bash
  python3 train.py -d {path/to/your/data.csv} 
```

Train file has several option to customize your model such as activation functions, number of epochs etc. To look at it just type:

```bash
  python3 train.py -h
```

It will create model file in your directory, which contains hyperparameters of your model. Also .png file will be created to show you how good your model is.

To predict some data type this:

```bash
  python3 predict.py -t {path/to/your/data.csv} -m {path/to/your/model.model}
```

Enjoy results!
