# Image classification: Recognizing MNIST handwritten digits

This goal of the project is to compares the effort of implementing popular machine learning algorithms from scratch to simply importing ready to use machine learning library scikit learn in order to classify hand written digits from the MNIST dataset. The project compares training time and accuracy, and discusses the pros and cons of implementing the algorithms from scratch. The conclusion is that it takes longer time to write the algortihms from scratch, it introduces more room for error. It is also discovered that the imported libraries make some greedy choices on how to handle the dataset which allows for faster training.

This project was create by Håvard Farestveit and Tobias Skjelvik as a course project for  Distributed Database Systems at Tsinghua University

## Installation 

```bash
cd Tsinghua-ML
pip install -r requirements.txt
```

## Usage

Each model is in separte folders in in ~/Tsinghua-ML/code. Each has a runnable pyhton file so just hit run or run in terminal/cmd

```bash
python Tsinghua-ML/code/KNN/KNN.py
```
