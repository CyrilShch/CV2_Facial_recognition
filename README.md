# Assignment 2. Facial emotion recognition using CNN.
All experiments were performed in [Google Colaboratory service](https://colab.research.google.com/) in order to use GPUs power.

## Dependencies
- Python 3
- NVIDIA GPU + CUDA cuDNN
- [NumPy](http://www.numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Scipy](https://www.scipy.org/)
- [Collections](https://docs.python.org/3/library/collections.html)
- [PyTorch](https://pytorch.org/)
- [Albumentations 0.4.6](https://albumentations.ai/)
- [Sklearn](https://scikit-learn.org/stable/)

## Dataset info
The data consists of 48x48 pixel grayscale images of faces. The training set consists of 28,709 examples. The public test set used for the leaderboard consists of 3,589 examples. The final test set, which was used to determine the winner of the competition, consists of another 3,589 examples. The data is available on [kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).

## How to launch the code?
- Firstly, you should open the corresponding notebook in [Google Colaboratory service](https://colab.research.google.com/) or locally on your computer with [Jupyter Notebook](https://jupyter.org/install.html), installing all dependencies.

### 1. Training the model.
In order to perform training:
- run the section "Downloading the data and data preparation functions initialization"
- run the section "Models"
- in the section "Model training" specify the training hyperparameters (batch size, epochs)
- choose (Net1_1, Net1_2, Net1_3, Net2, or UpgradedNet) and initialize the model 
- define training parameters: criterion, optimizer, scheduler, scheduler mode
- run the training process specifying flags for saving the model (if True - then the best model will be automatically saved on the Google Colab)
- after training is completed, visualize the validation and training losses

### 2. Testing the model.
In order to perform testing:
- run the section "Downloading the data and data preparation functions initialization"
- run the section "Models"
- in the section "Model training" specify the training hyperparameters (batch size, epochs) and run this cell
- in the section "Model training" run the cell creating dataloaders
- in the section "Testing the model on the test set" upload the trained model weights and initalize the model with the learned weights
- in the section "Testing the model on the test set" run the testing process.

### 3. Predict labels.
In order to perform testing:
- run the section "Downloading the data and data preparation functions initialization"
- run the section "Models"
- in the section "Model training" specify the training hyperparameters (batch size, epochs) and run this cell
- in the section "Model training" run the cell creating dataloaders
- in the section "Testing the model on the test set" upload the trained model weights and initalize the model with the learned weights
- in the section "Predict labels" run the predicting script.

### 4. Visualizing filters and weights.
In order to visualize the learned filters and the feature maps of the convolutional layers:
- run the section "Downloading the data and data preparation functions initialization"
- run the section "Models"
- in the section "Model training" specify the training hyperparameters (batch size, epochs) and run this cell
- in the section "Model training" run the cell creating dataloaders
- in the section "Testing the model on the test set" upload the trained model weights and initalize the model with the learned weights
- run the section "Visualizing filters and weights" specifying the exact layer to visualize and to perform the featuring mapping.
