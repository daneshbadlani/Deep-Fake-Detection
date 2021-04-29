# Deep Fake Detection

This repository consists of all the work for the Deep Fake Detection project.

# Getting Started

Make sure you have the dataset under **./celeba/img_align_celeba**, otherwise the code will automatically download and save it there.

# Running the Model

The **./saved** directory (if it exists) contains the trained models at 10,000 epochs. If you run as is, it will load the models and will not train. If you want to train the model yourself, delete the **./saved** folder and rerun. This will train the models, create the folder and save them.

```
python faceGAN.py
```

Once ran, the script will output generated images to a **./fakeImages** folder. If training occurs, it will save an image in the **./generated** folder every 100 epochs, as well as a GIF showing the progress in the root folder called **./visual.gif**.

# Requirements

- tensorflow
- numpy
- matplotlib
- pandas
- gdown
