# Full python-model to tfjs-node

This is an example project to show how you can:

- Train a simple model with python (doc-classification with imdb-reviews)
- Save the model
- Load the model with tfjs-node to use it in a node-backend

## Setup

- Python 3.10
- Node 18.16.1 (18.17 does throw errors)
- g++/gdb (Linux) or C++ Build Tools (Windows)

Feel free to run the [cpp.main](./cpp-test/main.cpp) file to check if C++ is correctly configured.

## Ongoin conclusion

So far the difficulty in setting up this project was solely the setup of the C++ compiler.
The rest was pretty straight forward.

I will continue to work on this project to extend examples and possibly compare setting up a basic Backend with Express/FastAPI to serve the Model and provide a simple database application to manage the data, predicting, editing labels and retraining the model.
