# Machine Learning Reverse Engineering

### Disclaimer 
This repo. is no longer updated and maintained (May 2019). <br/>
It was a tiny side-project developed with the UROP, Detecting Deep Learning Software Defects (Spring 2019). 

### Is it possible to conduct reverse-engineering on a black-box neural network?
If yes, reverse-engineering on a black-box neural network must be taken into considerations while implementing a neural network system since it will be a vulnerability exploited by malicious attackers in the future. Once malicious attackers have high confidence for the internal information of a neural network system such as weights and architectures, many present white-box attacks, like [1], against neural network systems can be applied and compromise the security of neural network systems.  

Assumptions and Measurements
----------------
To make the reverse-engineering framework more concrete and understandable, it is significant to precisely specify the exact target and related assumptions for the reverse-engineering attack. The description will be further split into three parts in details as below. <br/>

<details><summary><b>Knowledge of Attackers</b></summary>
<p>
It's assumed that the architecture, including activation functions, of black-box neural networks is known but the training process is unknown. For instance, malicious attackers do not have the knowledge about optimizer and training dataset used. This assumption is based on the contribution of [2]. This work has manifested how it is potential for malicious attackers to discover the architecture applied by a black-box neural network. Simply speaking, for malicious attackers, information of layer and corresponding activation functions is distinguished but none of the training process and weights values of black-box neural networks is known. Therefore, the reverse-engineering of black-box neural networks becomes a grey-box problem to solve, and the current objective of this project is to resolve this grey-box problem efficiently and intelligently. 
</p>
</details>

<details><summary><b>The Objective of the Reverse-engineering Attack</b></summary>
<p>
Given the architecture and activation functions of a black-box neural network, a malicious attacker attempts to retrieve the weight values as close as possible. For instance, if the weight value of a certain node in a black-box neural network is 1, the objective of the reverse-engineering framework is to generate predicted weight value as close as to 1. Generally, the ultimate goal of the reverse-engineering framework is to reproduce the exact same set of weight values as the target black-box neural network. 
</p>
</details>

<details><summary><b>Methods for Measuring the Effectiveness of Reverse-engineering Attack</b></summary>
<p>
As illustrated above, the objective is to predict the weight values of black-box neural networks as close as possible. In this paragraph, three ways of measurements will be introduced, including the reason for adoptions and corresponding baseline values. It's likely to replace or introduce additional measurement through the progress of the research. <br/>
 <br/>
<b>- Average Absolute Error of weight values:</b></br>
This measurement calculates the average value based on the absolute difference of all weight values in a black-box neural network. For example, if a black-box neural network solely contains 3 nodes with value 3 and predicted weights given by the reverse-engineering framework is 1, 3, and 5. The Average Absolute Error would be 1.333.<br/>
<br/>
<b>-    Average Absolute Percentage Error of weight values:</b></br>
This measurement is similar to the Average Absolute Error. Instead of Absolute Difference Error, Absolute Percentage Error is used for calculation, which aims to utilize the ratio for estimating the closeness. For instance, if a black-box neural network solely contains 3 nodes with 3 and predicted weights given by the reverse-engineering framework is 1, 3, and 5. The Average Relative Error would be 0.444.<br/>
<br/>
<b>-    The difference of prediction accuracy between predicted weights and ground-truth weights:</b></br>
If the reverse-engineering framework can actually reproduce an identical model as a black-box neural network framework, it should possess identical prediction accuracy. For instance, if a black-box neural network can achieve 92% accuracy on a certain test dataset, a reproduced model with identical weight values should also obtain 92% accuracy on the same test dataset. In our framework, we compute the difference of accuracy given by predicted weights and ground-truth weights. For a simple demonstration, if the prediction accuracy of a black-box neural network is 92% and that of predicted weights is 50%, this measurement should produce error value with 0.42.<br/>
<br/>
Through experiments, for the target black-box neural network in the research, its baseline values are around 0.05, 3.65, 0.83 respectively to above three measurements. Please refer to <b>baseline_values_computation.ipynb</b> for the more specific implementation. Note that these measurements are only meaningful while comparing to the measurements obtained after conducting reverse-engineering on the target black-box neural network. The baseline values should be different based on the architecture, activation functions, training process, and initialization method of the target black-box neural network.
</p>
</details>


The First Version of the Reverse-engineering Framework   
----------------
<p align="center">
  <img src="https://github.com/KuoTzu-yang/ML-reverse/blob/master/pictures_for_README/first_version_framework_1.png">
</p>

<details><summary><b>Introduction</b></summary>
<p>
The only difference between a neural network system and a mathematical function is that the inputs and outputs of a neural network system can be of any format. When we consider neural network systems as several approximated functions, given sufficient inputs with a slight difference, we should be able to observe different ways of variation in the outputs, which can be utilized to distinguish different neural network systems. This concept is mainly my first design principle behind the first version of the framework.  
</p>
</details>

<details><summary><b>Training Process Explanation</b></summary>
<p>
In the training process, we would like to train a model called reverse model R by leveraging abundant white-box neural networks, which have the same architecture but diverse weights from each other. The input of reverse model R is sensitivity maps of corresponding white-box neural networks and the output of R is the predicted weights for individual white-box neural networks. The purpose of the reverse model R is to leverage features (e.g. sensitivity maps) of neural networks and map these features with weights of neural networks via the training process. 
<div align="center">
  <img width="500" height="500" src="https://github.com/KuoTzu-yang/ML-reverse/blob/master/pictures_for_README/first_version_framework_2.png">
</div>
More specifically, for each white-box neural network, we feed a sufficient amount of input samples from a fixed set and receive the corresponding outputs through a white-box model. Then we generate a sensitive map for each neural network, which acts the representation of attention of each neural network. For instance, if a neural network M takes a 28 * 28 image as input, the corresponding sensitivity map for M is also a 28 * 28 image. Each pixel in the sensitivity map describes how important a pixel is for the contribution of decisions made by M. The concept of the sensitivity map is quite similar to derivative, such as how small change in each pixel of an input image can affect the final decision. However, there is a difference between the derivative and the sensitivity map. For instance,  if there are 10 classes for a classification prediction task, a change on a certain pixel, p1, may change prediction scores of all classes significantly but change on a p1 could not the neural network to generate a different prediction result, which indicates the prediction score of a certain class, let’s denote it as C_h, remains highest among all classes. However, there is a change on another pixel, p2, which will slightly decrease the prediction score of C_h without affecting prediction scores on all other classes. The change on p2 will produce a variation in the prediction result easily for a given image. Under this situation, p2 is of higher sensitivity than p1 in the sensitivity map but smaller derivatives for the prediction scores.<br/>
<br/>
Finally, after collection of sensitivity maps and ground-truth weights of white-box neural networks, these data are utilized as the training dataset for the reverse model. 
</p>
</details>

<details><summary><b>A Numerical Example</b></summary>
<p>
<div align="center">
  <img src="https://github.com/KuoTzu-yang/ML-reverse/blob/master/pictures_for_README/first_version_framework_3.png">
</div>
To facilitate the comprehension for readers, a real-life example is given to illustrate the concept of the training process. Assume we would like to reverse a black-box neural network used for image classification and its architecture and activation functions are known, we build 5000 white-box neural networks with the same architecture and activation functions as the target black-box neural network. Next, for each white-box neural network, 1000 images are forwarded through an individual white-box neural network to obtain 1000 corresponding outputs. Subsequently, based on 1000 image inputs and 1000 corresponding outputs, a sensitivity map is computed for an individual white-box neural network, where totally 5000 sensitivity maps are formed as the inputs for the reverse model R. Eventually, by leveraging these sensitivity maps, we train the reverse model R to approximate the mapping function from a sensitivity map to a set of weights.
</p>
</details>

<details><summary><b>Testing Process Explanation</b></summary>
<p>
<div align="center">
  <img src="https://github.com/KuoTzu-yang/ML-reverse/blob/master/pictures_for_README/first_version_framework_4.png">
</div>
As reverse model R is trained, we can apply the same principle for a target black-box neural network. By feeding the same fixed input set and computation, a sensitivity map for a black-box model is derived. Then, this sensitivity map serves as the input for the reverse model for producing predicted weights of the black-box model. The effectiveness of reverse-engineering is measured by the similarity between predicted weights and ground-truth weights, which can be designed by practitioners. In this project, I used the Average Absolute Percentage Error and cross entropy loss. 
</p>
</details>


The Second Version of the Reverse-engineering Framework   
----------------
<p align="center">
  <img src="https://github.com/KuoTzu-yang/ML-reverse/blob/master/pictures_for_README/second_version_framework_1.png">
</p>

<details><summary><b>Introduction</b></summary>
<p>
After experiments, I had discovered some problematic issues in the first version of the framework. For instance, it is computational inefficiency and it can solely reverse to a certain level.

The failure of the first approach drives me to reconsider from the design perspective. I had studied several papers related to either reverse-engineering or attacks on neural network systems. [2] and [3] delivered transcendent motivations to me. In [2], this research work relaxed the restriction, it paves the foundation of reverse-engineering towards black-box neural networks by classifying architecture among black-box neural networks. While in [3], it deliberately illustrates how perturbation of selected internal nodes of neural networks can effectively be retrained the model, which provides a potential direction to improve the effectiveness of reverse-engineering. 

Note that the concept mentioned in the first version will be omitted in the description of the second version. 
</p>
</details>

<details><summary><b>Primary Improvements</b></summary>
 
1.	Instead of utilizing sensitivity maps, in fact, direct utilizing fixed input set and corresponding outputs can achieve a similar purpose. This alternative consumes more memory space but it is more computationally efficient, which saves a huge amount of time for computing sensitivity maps.

2.	For the architecture of the reverse model, after experiences from the first approach, both fully-connected neural networks and convolutional neural networks are not suitable for this task. The training process of reverse-engineering requires extremely huge computational capability since input size and output size can be around 100K and 60K for conducting reverse-engineering on a naive three-layer fully-connected neural network. 
To cope with this issue, I adopt variational autoencoder (VAE) architecture [8] for the reverse model, which has the nice property to downsample input data for efficient training and computation and further upsample to generate final predictions.

3.	Another improvement is that single loss like the Average Absolute Percentage Error (AAPE) or cross entropy loss does not perform well in this task. No matter which loss function is applied to the reverse model R, during the testing process, the AAPE on the testing dataset is just slightly less than 100%. For instance, if a ground-truth value of a node is 1, the predicted value may be around 0.01. Through experiments, it is demonstrated the performance on the testing dataset can hardly be improved with the increase in the number of training epochs. To overcome this problem, another loss is introduced. Temporarily, I denote this loss as prediction similarity loss. The principle behind is that solely utilizing single loss like AAPE may generate a huge portion of reasonable weights. It indicates many predicted weights generated by the reverse model do not preserve the capability for predictions. If we feed some input images through predicted weights, we will receive accuracy of around 10%, which is similar to random guessing. Intuitively, these predicted weights are trapped in a local minimum. By introducing prediction similarity loss, training process avoids being easily trapped by arbitrary local minimums, where the training process towards a more reasonable direction. The training process preserves the same prediction capability while minimizing the loss from the weight difference. This technique demonstrates surprising results, where the difference between ground-truth weights and predicted weights can be significantly reduced. 
</details>

<details><summary><b>Training Process Explanation</b></summary>
<p>
For the training process of the second approach, in the beginning, we would like to extract useful information from white-box neural networks. In this approach, a fixed set of inputs and a corresponding set of outputs are directly utilized as the reverse model. Through experiments, mapping relation between inputs and outputs can reach similar performance level as using sensitivity maps but with the less computational requirement. 

Subsequently, we train a customized VAE model for learning to reverse-engineer from mapping the relation between inputs and outputs to the ground-width weights. An additional loss, measuring the difference of prediction capability between the original white-box neural network and predicted weighted is introduced to improve the training process.
</p>
</details>

<details><summary><b>Testing Process Explanation</b></summary>
<p>
For the testing process, it is just simply forwarding in the training process, where details are omitted. The second approach is still under development and I will continuously update more related information in the future. 
</p>
</details>

(Unofficial) References
----------------
[1] C. Xiao, B. Li, J.-Y. Zhu, W. He, M. Liu, and D. Song, “Generating Adversarial Examples with Adversarial Networks,” Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence, 2018. <br/>
[2] Seong Joon Oh, Max Augustin, Mario Fritz, and Bernt Schiele. Towards reverse-engineering black-box neural networks. In ICLR, 2018. <br/>
[3] Y. Liu, S. Ma, Y. Aafer, W.-C. Lee, J. Zhai, W. Wang, and X. Zhang, “Trojaning Attack on Neural Networks,” Proceedings 2018 Network and Distributed System Security Symposium, 2018. <br/>
[4] Yunchen Pu, Zhe Gan, Ricardo Hena, Xin Yuan, Chunyuan Li, Andrew Stevens and Lawrence Carin, “Variational Autoencoder for Deep Learning of Images, Labels and Captions”, In NIPS, 2016. <br/>
