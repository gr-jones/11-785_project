# To Approximate or Not to Approximate:Backpropagation in Spiking Neural Networks
Repository for 11-785 Project

Abstract: 

Spiking neural networks (SNNs) are a class of neural networks that use event-times, the times at which spikes occur, to encode and process information. Traditional artificial neural networks (ANNs) differ from SNNs in that they use event-rates, the number of spikes within a given time window, to encode and process information. One main advantage of SNNs over ANNs is their lower energy use, this is due to the fact SNNs require less memory storage than ANNs. Because SNNs encode information using binary spike trains, modeled using dirac-delta functions, the derivatives for individual neurons do not always exist making error backpropagation impossible. To handle this problem, researchers have created SNN backpropagation algorithms that either 1) roughly approximate or 2) calculate exactly the derivative of the output at each neuron at an event-time. In our final report, we will present a complete comparison between an approximate and exact SNN backpropagation algorithm, known as Spiking Neural Units (SNUs) and EventProp, respectively. In this report, however, we provide further context about SNNs, a literature review of the two SNN algorithms and all other items outlined in the rubric.

To run this code please run:

    python main.py

To generate plots comparing the two backpropagation algorithms run:

    python main.py plot

EventProp implementation is based on:

Wunderlich, Timo. Pehle, Christian. (2020). EventProp: Backpropagation For Exact Gradients In Spiking Neural Networks. arXiv:2009.083782 

SNU implementation is based on:

Wozniak, S. et al. "Deep learning incorporating biologically-inspired neural dynamics" from https://www.nature.com/articles/s42256-020-0187-0
