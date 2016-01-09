# BP-LDA
Backpropagation Latent Dirichlet Allocation (reimplementation of paper "End-to-end Learning of LDA by Mirror-Descent Back Propagation over a Deep Architecture" by Jianshu Chen et al.)

This paper is accepted by NIPS 2015.
Link to this paper: http://papers.nips.cc/paper/5967-end-to-end-learning-of-lda-by-mirror-descent-back-propagation-over-a-deep-architecture.pdf

To run the codes, there are two executables (one depends on the other). You need to copy both executables to the same working directory.

bp-lda/SupLDA_UnfoldBP/bin/x64/Release/SupLDA_UnfoldBP.exe # for supervised task

bp-lda/SupLDA_UnfoldBP/bin/x64/Release/UnsupLDA_UnfoldBP.exe # for unsupervised task

For Windows users, simply open a command prompt window and run "SupLDA_UnfoldBP.exe" or "UnsupLDA_UnfoldBP.exe"

For Linux/Mac users, you need to install mono (http://www.mono-project.com/). Then, open a terminal and run "mono SupLDA_UnfoldBP.exe" or "mono UnsupLDA_UnfoldBP.exe"

## Regression demo using Amazon Movie Review data set, 1% data, vocabulary size 5000
### Data
.label: 1~5 star rating, shifted to zero mean
.feature: each line is a document in bag-of-words representation. Colon is used to delimit word index and word counts. Tab is used to delimit different words.
The data folder is bp-lda/data_AmazonMovieReview_1percent/, in which you can find label files and feature files for train/test, respectively.

### Command
alpha = 1.001:

SupLDA_UnfoldBP.exe --nHid 5 --nHidLayer 10 --nInput 5000 --nOutput 1 --OutputType linearQuad --alpha 1.001 --beta 1.0001 --nEpoch 50 --BatchSize 1000 --BatchSize_Test 10000 --flag_DumpFeature false --mu_Phi 0.01 --mu_U 1 --nSamplesPerDisplay 10000 --nEpochPerSave 1 --nEpochPerTest 1 --nEpochPerDump 5 --TrainLabelFile train.label --TestLabelFile test.label --TrainInputFile train.feature --TestInputFile test.feature --ResultFile result_Voc5000 --ThreadNum 32 --MaxThreadDeg 32 --T_value 1 --DebugLevel high --flag_AdaptivenHidLayer false --flag_RunningAvg true

alpha = 0.1:

SupLDA_UnfoldBP.exe --nHid 5 --nHidLayer 10 --nInput 5000 --nOutput 1 --OutputType linearQuad --alpha 0.1 --beta 1.0001 --nEpoch 50 --BatchSize 1000 --BatchSize_Test 10000 --flag_DumpFeature false --mu_Phi 0.0001 --mu_U 1 --nSamplesPerDisplay 10000 --nEpochPerSave 1 --nEpochPerTest 1 --nEpochPerDump 5 --TrainLabelFile train.label --TestLabelFile test.label --TrainInputFile train.feature --TestInputFile test.feature --ResultFile result_Voc5000 --ThreadNum 32 --MaxThreadDeg 32 --T_value 0.01 --DebugLevel high --flag_AdaptivenHidLayer true --flag_RunningAvg true

### Output files (if you follow the above example)
result_Voc5000.model.Phi and result_Voc5000.model.U: model files with model parameters Phi and U, as described in the paper

result_Voc5000.perf and result_Voc5000.testscore: performance file and test score file

#### Here is a brief explanation on command line arguments:

--nHid: Number of topics

--nHidLayer: Number of layers

--nInput: Vocabulary size

--nOutput: Number of output classes

--OutputType: “softmaxCE” means classification with softmax and cross entropy, "linearQuad" means linear quadratic (L2) loss

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
