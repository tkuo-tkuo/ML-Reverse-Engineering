# ML Reverse

### Is it possible to conduct reverse-engineering on a black-box neural network?
If yes, reverse-engineering on a black-box neural network must be taken into considerations while implementing a neural network system since it will be a vulnerability exploited by malicious attackers in the future. Once malicious attackers have high confidence for the internal information of a neural network system such as weights and architectures, many present white-box attacks, like [1], against neural network systems can be applied and compromise the security of neural network systems.  


Settings and assumptions
----------------
To make the reverse-engineering framework more concrete and understandable, it is significant to precisely specify the exact target and related assumptions for the reverse-engineering attack. The description will be further split into three parts in details as below. <br/>

<b>Knowledge of attackers:</b><br/>
It's assumed that the architecture, including activation functions, of black-box neural networks is known but the training process is unknown. For instance, malicious attackers do not have the knowledge about optimizer and training dataset used. This assumption is based on the contribution of [6]. This work has manifested how it is potential for malicious attackers to discover the architecture applied by a black-box neural network. Simply speaking, for malicious attackers, information of layer and corresponding activation functions is distinguished but none of the training process and weights values of black-box neural networks is known. Therefore, the reverse-engineering of black-box neural networks becomes a grey-box problem to solve, and the current objective of this project is to resolve this grey-box problem efficiently and intelligently. 

<b>The objective of the reverse-engineering attack:</b><br/>
Given the architecture and activation functions of a black-box neural network, a malicious attacker attempts to retrieve the weight values as close as possible. For instance, if the weight value of a certain node in a black-box neural network is 1, the objective of the reverse-engineering framework is to generate predicted weight value as close as to 1. Generally, the ultimate goal of the reverse-engineering framework is to reproduce the exact same set of weight values as the target black-box neural network.

<b>Methods for measuring the effectiveness of reverse-engineering attack: </b><br/>
As illustrated above, the objective is to predict the weight values of black-box neural networks as close as possible. In this paragraph, three ways of measurements will be introduced, including the reason for adoptions and corresponding baseline values. It's likely to replace or introduce additional measurement through the progress of the research.
-	Average Absolute Error of weight values: This measurement calculates the average value based on the absolute difference of all weight values in a black-box neural network. For example, if a black-box neural network solely contains 3 nodes with value 3 and predicted weights given by the reverse-engineering framework is 1, 3, and 5. The Average Absolute Error would be 1.333.
-	Average Absolute Percentage Error of weight values: This measurement is similar to Average Absolute Error. Instead of Absolute Difference Error, Absolute Percentage Error is used for calculation, which aims to utilize the ratio for estimating the closeness. For instance, if a black-box neural network solely contains 3 nodes with 3 and predicted weights given by the reverse-engineering framework is 1, 3, and 5. The Average Relative Error would be 0.444.
-	The difference of prediction accuracy between predicted weights and ground-truth weights: If the reverse-engineering framework can actually reproduce an identical model as a black-box neural network framework, it should possess identical prediction accuracy. For instance, if a black-box neural network can achieve 92% accuracy on a certain test dataset, a reproduced model with identical weight values should also obtain 92% accuracy on the same test dataset. In our framework, we compute the difference of accuracy given by predicted weights and ground-truth weights. For a simple demonstration, if the prediction accuracy of a black-box neural network is 92% and that of predicted weights is 50%, this measurement should produce error value with 0.42.
 
Through experiments, for the target black-box neural network in the research, its baseline values are around 0.05, 3.65, 0.83 respectively to above three measurements. Please refer to <b>baseline_values_computation.ipynb</b> for the more specific implementation. Note that these measurements are only meaningful while comparing to the measurements obtained after conducting reverse-engineering on the target black-box neural network. The baseline values should be different based on the architecture, activation functions, training process, and initialization method of the target black-box neural network.


Inspirations
----------------
The only difference between a neural network system and a mathematical function is that the inputs and outputs of a neural network system can be of any format. When we consider neural network systems as several approximated functions, given sufficient inputs with a slight difference, we should be able to observe different ways of variation in the outputs, which can be utilized to distinguish different neural network systems. This concept is mainly my first design principle behind the first version of architecture.    

After discussing with Victor and Professor Cheung, I had discovered some problematic issues in the first approach. For instance, the approach itself is computational inefficiency and it can reverse the model to a certain level. The measurement for the effectiveness of the reverse-engineering will be introduced in the following paragraphs.  

The failure of the first approach drives me to reconsider from the design perspective. I had studied several papers related to either reverse-engineering or attacks on neural network systems. [6] and [7] delivered transcendent motivations to me. In [6], this research work relaxed the restriction, it paves the foundation of reverse-engineering towards black-box neural networks by classifying architecture among black-box neural networks. While in [7], it deliberately illustrates how perturbation of selected internal nodes of neural networks can effectively be retrained the model, which provides a potential direction to improve the effectiveness of reverse-engineering. 

 
 
First version of the reverse-enginnering framework   
----------------
The only difference between a neural network system and a mathematical function is that the inputs and outputs of a neural network system can be of any format. When we consider neural network systems as several approximated functions, given sufficient inputs with a slight difference, we should be able to observe different ways of variation in the outputs, which can be utilized to distinguish different neural network systems. This concept is mainly my first design principle behind the first version of framework.  
   
Second version of the reverse-enginnering framework   
----------------
After experiments, I had discovered some problematic issues in the first version of framework. For instance, it is computational inefficiency and it can solely reverse to a certain level.

The failure of the first approach drives me to reconsider from the design perspective. I had studied several papers related to either reverse-engineering or attacks on neural network systems. [2] and [3] delivered transcendent motivations to me. In [2], this research work relaxed the restriction, it paves the foundation of reverse-engineering towards black-box neural networks by classifying architecture among black-box neural networks. While in [3], it deliberately illustrates how perturbation of selected internal nodes of neural networks can effectively be retrained the model, which provides a potential direction to improve the effectiveness of reverse-engineering. 
   
   
Purpose & content of each file 
----------------
Files below are ordered in alphabetical order.  
-  <b>first file</b>  
   description
   
   
Configuration
----------------
  Python: 3.6.7  
  Pytorch: 1.0.1  
  NumPy: 1.14.2 


References
----------------
[1] C. Xiao, B. Li, J.-Y. Zhu, W. He, M. Liu, and D. Song, “Generating Adversarial Examples with Adversarial Networks,” Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence, 2018. <br/>
[2] Seong Joon Oh, Max Augustin, Mario Fritz, and Bernt Schiele. Towards reverse-engineering black-box neural networks. In ICLR, 2018. <br/>
[3] Y. Liu, S. Ma, Y. Aafer, W.-C. Lee, J. Zhai, W. Wang, and X. Zhang, “Trojaning Attack on Neural Networks,” Proceedings 2018 Network and Distributed System Security Symposium, 2018. <br/>
[4] Yunchen Pu, Zhe Gan, Ricardo Hena, Xin Yuan, Chunyuan Li, Andrew Stevens and Lawrence Carin, “Variational Autoencoder for Deep Learning of Images, Labels and Captions”, In NIPS, 2016. <br/>

