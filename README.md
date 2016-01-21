# BP-LDA
Backpropagation Latent Dirichlet Allocation (a third-party reimplementation of paper "End-to-end Learning of LDA by Mirror-Descent Back Propagation over a Deep Architecture" by Jianshu Chen et al.)

The paper is accepted by NIPS 2015.
Link to this paper: http://papers.nips.cc/paper/5967-end-to-end-learning-of-lda-by-mirror-descent-back-propagation-over-a-deep-architecture.pdf

To run the codes, there are two executables (one depends on the other). You need to copy both executables to the same working directory.

bp-lda/BP_sLDA/bin/Release/BP_sLDA.exe # for supervised task

bp-lda/BP_sLDA/bin/Release/BP_LDA.exe # for unsupervised task

For Windows users, simply open a command prompt window and run "BP_sLDA.exe" or "BP_LDA.exe".

For Linux/Mac users, you need to install mono (http://www.mono-project.com/). Then, open a terminal and run "mono BP_sLDA.exe" or "mono BP_LDA.exe".

-------------------------------------------------------

### Regression demo using Amazon Movie Review data set, 1% data, vocabulary size 5000
Raw data can be downloaded from https://snap.stanford.edu/data/web-Movies.html
#### Data: data_AmazonMovieReview_1percent/
.label: 1~5 star rating, shifted to zero mean

.feature: each line is a document in bag-of-words representation. Colon is used to delimit word index and word counts. Tab is used to delimit different words.

#### Command (for simplicity, we omit the paths to .exe and data files. Make sure you specify the correct paths to you executables and data files when you experiment)
Supervised (alpha = 1.001):

    BP_sLDA.exe --nHid 5 --nHidLayer 10 --nInput 5000 --nOutput 1 --OutputType linearQuad --alpha 1.001 --nEpoch 50 --BatchSize 1000 --mu_Phi 0.01 --nSamplesPerDisplay 10000 --TrainLabelFile train.label --TestLabelFile test.label --TrainInputFile train.feature --TestInputFile test.feature --ResultFile result_Voc5000 --ThreadNum 32 --MaxThreadDeg 32

Supervised (alpha = 0.1):

    BP_sLDA.exe --nHid 5 --nHidLayer 10 --nInput 5000 --nOutput 1 --OutputType linearQuad --alpha 0.1 --nEpoch 50 --BatchSize 1000 --mu_Phi 0.0001 --nSamplesPerDisplay 10000 --TrainLabelFile train.label --TestLabelFile test.label --TrainInputFile train.feature --TestInputFile test.feature --ResultFile result_Voc5000 --ThreadNum 32 --MaxThreadDeg 32

#### Output files (if you follow the above example)
result_Voc5000.model.Phi and result_Voc5000.model.U: model files with model parameters Phi and U, as described in the paper

result_Voc5000.perf and result_Voc5000.testscore: performance file and test score file

-------------------------------------------------------

### Classification demo using Multidomain Sentiment Classification data set, vocabulary size 1000
Raw data can be downloaded from https://www.cs.jhu.edu/~mdredze/datasets/sentiment/
#### Data: data_MultidomainSentiment/
.label: 0~1 binary class labels

.feature: each line is a document in bag-of-words representation. Colon is used to delimit word index and word counts. Tab is used to delimit different words.

#### Command
Supervised (alpha = 1.001):

    BP_sLDA.exe --nHid 5 --nHidLayer 10 --nInput 1000 --nOutput 2 --OutputType softmaxCE --alpha 1.001 --nEpoch 20 --BatchSize 100 --mu_Phi 0.01 --nSamplesPerDisplay 10000 --TrainLabelFile train.label --TestLabelFile test.label --TrainInputFile train.feature --TestInputFile test.feature --ResultFile result_Voc1000 --ThreadNum 32 --MaxThreadDeg 32

Supervised (alpha = 0.1):

    BP_sLDA.exe --nHid 5 --nHidLayer 10 --nInput 1000 --nOutput 2 --OutputType softmaxCE --alpha 0.1 --nEpoch 20 --BatchSize 100 --mu_Phi 0.001 --nSamplesPerDisplay 10000 --TrainLabelFile train.label --TestLabelFile test.label --TrainInputFile train.feature --TestInputFile test.feature --ResultFile result_Voc1000 --ThreadNum 32 --MaxThreadDeg 32

#### Output files
Same as the regression example.

-------------------------------------------------------

### Unsupervised demo using Amazon Movie Review data set, 1% data, vocabulary size 5000
#### Data: data_AmazonMovieReview_1percent/
#### Command
Unsupervised (alpha = 1.001):

    BP_LDA.exe --nHid 5 --nHidLayer 10 --nInput 5000 --alpha 1.001 --nEpoch 20 --BatchSize 1000 --flag_DumpFeature true --mu_Phi 0.01 --nSamplesPerDisplay 10000 --TrainInputFile train.feature --TestInputFile test.feature --ResultFile result_Voc5000 --ThreadNum 32 --MaxThreadDeg 32

Unsupervised (alpha = 0.1):

    BP_LDA.exe --nHid 5 --nHidLayer 10 --nInput 5000 --alpha 0.1 --nEpoch 20 --BatchSize 1000 --flag_DumpFeature true --mu_Phi 0.0001 --nSamplesPerDisplay 10000 --TrainInputFile train.feature --TestInputFile test.feature --ResultFile result_Voc5000 --ThreadNum 32 --MaxThreadDeg 32

#### Output files:
result_Voc5000.train.fea and result_Voc5000.test.fea: generated topic distribution vectors for each train/test documents.

-------------------------------------------------------

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

--BatchSizeSchedule: a scheduler for setting training batch size, e.g. 1:10,2:100,11:1000 means batch size = 10 in epoch 1, switching to batch size = 100 in epoch 2, and switching to 1000 in epoch 11.

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

--DebugLevel: high/low/medium

--flag_RunningAvg: true to smooth the training process (more stable model)

--flag_SaveAllModels: true means saving models at different epochs separately, and false means only saving the model at the last epoch

-------------------------------------------------------

#### References
1. Chen, Jianshu, et al. End-to-end Learning of LDA by Mirror-Descent Back Propagation over a Deep Architecture. Advances in Neural Information Processing Systems. 2015.

2. J. McAuley and J. Leskovec. From amateurs to connoisseurs: modeling the evolution of user expertise through online reviews. WWW, 2013.

3. John Blitzer, Mark Dredze, Fernando Pereira. Biographies, Bollywood, Boom-boxes and Blenders: Domain Adaptation for Sentiment Classification. Association of Computational Linguistics (ACL), 2007.
