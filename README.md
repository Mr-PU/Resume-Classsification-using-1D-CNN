# Resume Classsification using 1D-CNN

Resume classification is the task of automatically categorizing resumes into pre-defined classes or job positions based on the information contained in the resumes. This is typically accomplished using machine learning algorithms such as Support Vector Machines (SVMs), Decision Trees, or Neural Networks. The input to the classification algorithm is typically the text of the resume, which may be preprocessed to extract relevant features such as skills, education, and work experience. The output of the classification algorithm is the predicted job position for the resume. Resume classification is commonly used in the human resources and recruitment industries to streamline the resume screening and shortlisting process.

 
# 1D-Convolutional Neural Network

1D Convolutional Neural Network (1D CNN) is a type of Convolutional Neural Network that is used for processing 1-dimensional data, such as signals and sequences. It is commonly used for tasks such as speech recognition, audio classification, and natural language processing. 1D CNN applies filters to local regions of the input, and slides the filters along the input to learn local patterns. The use of convolutional layers allows 1D CNN to maintain the spatial information in the input data while reducing the number of parameters in the model, resulting in a more efficient and effective learning process.


# 1D-CNN and NLP

In Natural Language Processing (NLP), 1D Convolutional Neural Network (1D CNN) is used for tasks such as text classification and sentiment analysis. In these tasks, the input to the 1D CNN is typically a sequence of word embeddings, which represent the words in a text as high-dimensional vectors. The 1D CNN then applies filters to the word embeddings to learn local patterns in the text. This allows the 1D CNN to effectively capture the relationships between words and their context, which is important for NLP tasks such as sentiment analysis. Additionally, 1D CNN can be combined with other NLP techniques, such as Recurrent Neural Networks (RNNs) and Transformers, to create powerful models for NLP tasks.


### Files:
1. **Model.ipynb:**
Includes code for data preprocessing, 1D-CNN model and code to save the model in h5py.

### Dataset used:
1. **ResumeDataSet.csv** (main)
2. **cv.pdf** (test)

**Model accuracy is:**  0.9584774971008301


![image](https://user-images.githubusercontent.com/86981617/215507085-62b52bed-428a-4006-86ee-29f69b317ca7.png)

![image](https://user-images.githubusercontent.com/86981617/215506697-adf02c67-c365-4b7f-877a-5fd5f4042b9f.png)
