# BP-LDA
Backpropagation Latent Dirichlet Allocation (reimplementation of paper "End-to-end Learning of LDA by Mirror-Descent Back Propagation over a Deep Architecture" by Jianshu Chen et al.)

This paper is accepted by NIPS 2015.
Link to this paper: http://papers.nips.cc/paper/5967-end-to-end-learning-of-lda-by-mirror-descent-back-propagation-over-a-deep-architecture.pdf

Here is a brief explanation on command line arguments:
--nHid: Number of topics

--nHidLayer: Number of layers

--nInput: Vocabulary size

--nOutput: Number of output classes

--OutputType: “softmaxCE” means classification with softmax and cross entropy

--alpha: Dirichlet parameter of the topics

--beta: Dirichlet parameter of the topic-word probability

--nEpoch: number of training epochs

--BatchSize: Minibatch size (number of documents at each mini-batch)

--BatchSize_Test: Minibatch size for testing

--flag_DumpFeature: if true, then save the topic distribution of each document. If false, then do not save

--mu_Phi: learning rate for the topic-word probability matrix

--mu_U: learning rate for the topic to output matrix

--LearnRateSchedule “Constant” for constant learning rate

--nSamplesPerDisplay: “10000” means displaying the progress of training after every 10000 documents

--nEpochPerSave: The frequency of model saving

--nEpochPerTest: The frequency of testing the model

--nEpochPerDump: The frequency of dumping the topic distribution for each document (if –flag_DumpFeature is true)

--TrainLabelFile: Label file for training set (one column, each row represents the index of the class, starting from 0. For example, 0 means the 0-th class, 3 means the 3rd class)

--TestLabelFile: Label file for test set

--ValidLabelFile: Label file for validation set

--TrainInputFile: Input file for training set (tab separated file, each row means the bag-of-words vector of the document. For example, 0:3 \t 5:7 means that this document has the 0-th word occurred three times and has the 5th word occurred 7 times)

--TestInputFile: Input file for the test set

--ValidInputFile Input file for the validation set

--ResultFile: Name of the result file. (Name for the model file, log file, etc, will be the same as this one but with different extensions)

--ThreadNum: Number of threads

--MaxThreadDeg: Threads parameter

--T_value: Initial step-size for mirror descent (Chosen to be one if alpha>1, and chosen to be 0.01 if alpha<1)

--DebugLevel: high/low/medium

--flag_AdaptivenHidLayer: Choose to be false for alpha>1 and choose to be true for alpha<1  

--flag_RunningAvg: true to smooth the training process (more stable model)

--flag_SaveAllModels: true means saving models at different epochs separately, and false means only saving the model at the last epoch

--ExternalEval: external batch file for evaluating AUC etc. (The code will automatically evaluate the test error. If you want to evaluate other metrics, then you have to specify the exe/bat file for evaluation.)
