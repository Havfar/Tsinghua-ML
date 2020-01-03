# Image classification: Recognizing MNIST handwritten digits

This goal of the project is to compares the effort of implementing popular machine learning algorithms from scratch to simply importing ready to use machine learning library scikit learn in order to classify hand written digits from the MNIST dataset. The project compares training time and accuracy, and discusses the pros and cons of implementing the algorithms from scratch. The conclusion is that it takes longer time to write the algortihms from scratch, it introduces more room for error. It is also discovered that the imported libraries make some greedy choices on how to handle the dataset which allows for faster training.

This project was create by Håvard Farestveit and Tobias Skjelvik as a course project for Machine Learning at Tsinghua University fall 2019. 



## Installation 

```bash
cd Tsinghua-ML
pip install -r requirements.txt
```



## Usage

Each model is in separte folders in in ~/Tsinghua-ML/code. Each has a runnable python file so just hit run or run in terminal/cmd

```bash
python Tsinghua-ML/code/KNN/KNN.py
```



# Result

| Machine Learning Algorithm | Accuracy (us) | Accuracy (Scikit) |
| -------------------------- | ------------- | ----------------- |
| Desiscion tree             | 69.44* %      | 85.57 %           |
| Random forests             | 51.70* %      | 96.5 %            |
| Naive Bayes                | 47.61 %       | 55.34 %           |
| KNN (K = 1)                | 97.11 %       | 97.11 %           |
| CNN                        | 97.15 %       | 99 + %            |

*Our implementation of Desiscion Tree and Random Forest is only testet on 5000 images due to too long running time