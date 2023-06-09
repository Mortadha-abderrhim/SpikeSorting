# Automated Hardware-efficient spike sorting algorithm for implantable BMIs

In this project, we implement a high-performance hardware-efficient online lagorith for neural spike sorting. Towards this end, we first develop a feature selection methodology to enhance discrimination between different spike classes. The spikes are then clusterd in the low-dimensional feature space based on an auto-updated l2 template-matching algorithm.


## Methodology

In this section, we present a short summary of the proposed methodology.

### Feature Selection

<p>In order to enhance the discrimination between different spike classes, we perform a feature extraction step where the detected spikes are mapped into a feature space. Different approaches for feature extraction have been proposed throughout the literature to achieve high discrimination between the spike classes while reducing the computational complexity.  
In this project, we select the maximum and minimum of the first and second derivative of the signal, also known as First and Second Derivative Extrema (FSDE). We also add the timing of the first positive peak of the signal. All the features are computed over a time window of 25 samples. </p>
<p>The proposed approach yields a five dimensional feature space. The figure below represents the selected features for a typical action potential waveform with its first and second derivatives:
![image](/Results/featureSelection.png) </p>
<p>This approach offers several advantages over alternative methods. It enables detailed analysis of spike shapes, capturing important characteristics such as peak timings and rate of change. Thus, the proposed features provide discriminatory power, enabling the effective separation of neurons of different types. Besides, the technique exhibits robustness to noise, as it focuses on detecting specific points of interest that are less prone to noise interference. Finally, the computational efficiency of this feature extraction methodology makes it
more suitable for real-time spike sorting applications than other more computationally intensive approaches, such as PCA.</p>

### Automated L-2 norm Template Matching

<p>The proposed algorithm could be summarized by the following flowchart:
![image](/FlowChart/AlgorithmFlowChart.png)

<ul>
  <li>**Distance Metric :** In order to compute distances between the input and the cluster centroids in the feature space, we employ the square of the l2 norm </li>
  <li>** Threshold :** The threshold is calculated as a function of the variance of the spike based on the following formula:  
      $$ Threshold = C<sub>1</sub> * \sigma + C<sub>0</sub> $$
    Where C<sub>1</sub> and C<sub>0</sub> are hyperparameters that have been finetuned based on the algorithm performance on the different WaveClus datasets. </li>
  <li>** Cluster centroids update :** The cluster centroids are updated based on a hardware efficient formula that approximates the traditional weighted average for centroid update.  
    $$ \mu_l = \Vec{\mu_l} + \frac{-\Vec{\mu_l}  + \Vec{X}}{2^{\left \lceil \log_2(N_{l}+1)\right \rceil}} $$</li>
</ul>
</p>

## Results

We evaluate the proposed approach on different datasets from WaveClus benchmark.  
The evaluation process is divided into two steps:   
In the first step, we evaluate the performance of our algorithm on the groundtruth spikes provided in the datasets. That is we consider the input signals based on the peak times provided in the dataset.  
In a second step, we evaluate the performance of our algorithm on the peak times returned by the Dual spike detector.
For the evaluation metric, we employ classification accuracy, that is the percentage of correctly classified spikes from the total number of the spikes (we only consider the True Positives of the spike detector in the second step).

### On the Dataset

The evaluation is done for the four datasets of Waveclus benchmark (Easy1, Easy2, Difficult1, Difficult2), at different noise levels, sampling rates and data resolutions. The results are shown in the figure below:
![image](/FlowChart/Accuracy.png)
On average, our algorithm achieves an accuracy of 96.1% for the WaveClus datasets at different noise levels which is superior to the known state-of-the-art methods. Furthermore, it achieves an accuracy higher than 95% for a resolution larger than 6 at 0.1 noise level. The high classification accuracy is alson ensured at a sampling rate larger than 18kS/s.

### On the output of the Spike Detector

The hyperparamets as finetuned for the first step lead a drop of the performance when applied to the output of the spike detector, achieving only 87% classification accuracy compared to 96.1% when applied to the groundtruth data. Thus, further hypertuning is currently being done to adapt the hyperparameters to the output of the dual spike detector.


## Contents of the Repo

This Repo contains the implementation of the proposed methodology as well as the notebooks to finetune the hyperparameters and test the implementation for different noise levels, sampling rates and resolutions, and is organized as follows

    |
    |- Results: Contains the results and feature selection figures
    |- FlowChart: Contains the figures used for the readme
    |- model.py: Contains the implementation of the algorithm and the different utility functions
    |- plot_resolution.ipynb: Evaluate and plot results of the algorithm on different data resolution
    |- plot_samping_rate.ipynb : Evaluate and plot results of the algorithm on different sampling rates
    |- plot_noise.ipynb: Evaluate and plot results of the algorithm on different noise levels
    |- FeaturesViz.ipynb: Vizualise the feature selection
    |- playground.ipynb: Notebook for finetuning and trying different things
    
