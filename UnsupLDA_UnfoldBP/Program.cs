using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Diagnostics;
using LinearAlgebra;

/* 
 * Unfold the LDA model into a feedforward network using MDA and then use back-propagation
 * to train the model in an unsupervised manner. Use the tied-parameter case
 * only.
 */

namespace BP_LDA
{
    class Program
    {
        static void Main(string[] args)
        {
            // ======== Setup the default parameters ========
            paramModel_t paramModel = new paramModel_t();
            paramTrain_t paramTrain = new paramTrain_t();
            SetupDefaultParams(paramModel, paramTrain);
            // ---- Data Files ----
            string TrainInputFile = "";
            string TestInputFile = "";
            string ModelFile = "";
            string ResultFile = "";

            // ======== Parse the input parameters ========
            if (
                !ParseArgument(
                    args,
                    paramModel,
                    paramTrain,
                    ref TrainInputFile,
                    ref TestInputFile,
                    ref ModelFile,
                    ref ResultFile
                    )
                )
            {
                return;
            }
            paramModel.T = new float[paramModel.nHidLayer];
            for (int IdxLayer = 0; IdxLayer < paramModel.nHidLayer; IdxLayer++)
            {
                paramModel.T[IdxLayer] = paramModel.T_value;
            }

            // ======== Set the number of threads ========
            MatrixOperation.THREADNUM = paramTrain.ThreadNum;
            MatrixOperation.MaxMultiThreadDegree = paramTrain.MaxMultiThreadDegree;

            // ======== Load data from file ========
            SparseMatrix TrainData = DataLoader.InputDataLoader(TrainInputFile, paramModel.nInput);
            SparseMatrix TestData = DataLoader.InputDataLoader(TestInputFile, paramModel.nInput);
            paramTrain.nTrain = TrainData.nCols;
            paramTrain.nTest = TestData.nCols;

            // ======== Unsupervised learning of LDA model: unfolding and back-propagation
            // (i) Inference: Feedforward network via MDA unfolding
            // (ii) Learning: Projected (mini-batch) stochastic gradient descent (P-SGD) using back propagation
            LDA_Learn.TrainingBP_LDA(TrainData, TestData, paramModel, paramTrain, ModelFile, ResultFile);
        }        

        /*
         * Setup the default parameters
         */
        public static void SetupDefaultParams(paramModel_t paramModel, paramTrain_t paramTrain)
        {
            // ---- Model parameters ----
            paramModel.nHid = 16;
            paramModel.nHidLayer = 5;
            paramModel.T_value = 1;
            paramModel.To = 1;
            paramModel.eta = 0.5f;
            paramModel.alpha = 1.001f;
            paramModel.beta = 1.006f;
            paramModel.OutputType = "unsupLDA";
            paramModel.flag_AdaptivenHidLayer = false;
            paramModel.nInput = 784;
            // ---- Training parameters ----
            paramTrain.nEpoch = 100;
            paramTrain.BatchSize = 1000;
            paramTrain.BatchSize_Test = 100000;
            paramTrain.mu_Phi = 0.01f;
            paramTrain.mu_Phi_ReduceFactor = 10.0f;
            paramTrain.mu_U = 0.01f;
            paramTrain.LearnRateSchedule = "Constant";
            paramTrain.nSamplesPerDisplay = 10000;
            paramTrain.nEpochPerSave = 1;
            paramTrain.nEpochPerTest = 1;
            paramTrain.flag_DumpFeature = false;
            paramTrain.nEpochPerDump = 10;
            paramTrain.flag_BachSizeSchedule = false;
            paramTrain.ThreadNum = 100;
            paramTrain.MaxMultiThreadDegree = 32;
            paramTrain.DebugLevel = DebugLevel_t.low;
            paramTrain.flag_RunningAvg = false;
        }

        /*
         * Parse the input arguments
         */
        public static bool ParseArgument(
            string[] args,
            paramModel_t paramModel,
            paramTrain_t paramTrain,
            ref string TrainInputFile,
            ref string TestInputFile,
            ref string ModelFile,
            ref string ResultFile
            )
        {
            string ArgKey;
            string ArgValue;
            for (int IdxArg = 0; IdxArg < args.Length - 1; IdxArg += 2)
            {
                ArgKey = args[IdxArg];
                ArgValue = args[IdxArg + 1];
                switch (ArgKey)
                {
                    case "--nHid":
                        paramModel.nHid = int.Parse(ArgValue);
                        break;
                    case "--nHidLayer":
                        paramModel.nHidLayer = int.Parse(ArgValue);
                        break;
                    case "--alpha":
                        paramModel.alpha = float.Parse(ArgValue);
                        break;
                    case "--beta":
                        paramModel.beta = float.Parse(ArgValue);
                        break;
                    case "--nEpoch":
                        paramTrain.nEpoch = int.Parse(ArgValue);
                        break;
                    case "--BatchSize":
                        paramTrain.BatchSize = int.Parse(ArgValue);
                        break;
                    case "--BatchSize_Test":
                        paramTrain.BatchSize_Test = int.Parse(ArgValue);
                        break;
                    case "--mu_Phi":
                        paramTrain.mu_Phi = float.Parse(ArgValue);
                        break;
                    case "--mu_U":
                        paramTrain.mu_U = float.Parse(ArgValue);
                        break;
                    case "--nSamplesPerDisplay":
                        paramTrain.nSamplesPerDisplay = int.Parse(ArgValue);
                        break;
                    case "--nEpochPerSave":
                        paramTrain.nEpochPerSave = int.Parse(ArgValue);
                        break;
                    case "--nEpochPerTest":
                        paramTrain.nEpochPerTest = int.Parse(ArgValue);
                        break;
                    case "--TrainInputFile":
                        TrainInputFile = ArgValue;
                        paramTrain.TrainInputFile = TrainInputFile;
                        break;
                    case "--TestInputFile":
                        TestInputFile = ArgValue;
                        paramTrain.TestInputFile = TestInputFile;
                        break;
                    case "--ResultFile":
                        ResultFile = ArgValue;
                        break;
                    case "--nInput":
                        paramModel.nInput = int.Parse(ArgValue);
                        break;
                    case "--nOutput":
                        paramModel.nOutput = int.Parse(ArgValue);
                        break;
                    case "--LearnRateSchedule":
                        paramTrain.LearnRateSchedule = ArgValue;
                        break;
                    case "--flag_DumpFeature":
                        paramTrain.flag_DumpFeature = bool.Parse(ArgValue);
                        break;
                    case "--nEpochPerDump":
                        paramTrain.nEpochPerDump = int.Parse(ArgValue);
                        break;
                    case "--BatchSizeSchedule":
                        paramTrain.flag_BachSizeSchedule = true;
                        paramTrain.BachSizeSchedule = new Dictionary<int, int>();
                        string[] StrBatSched = ArgValue.Split(',');
                        for (int Idx = 0; Idx < StrBatSched.Length; Idx++)
                        {
                            string[] KeyValPair = StrBatSched[Idx].Split(':');
                            paramTrain.BachSizeSchedule.Add(int.Parse(KeyValPair[0]), int.Parse(KeyValPair[1]));
                        }
                        break;
                    case "--ThreadNum":
                        paramTrain.ThreadNum = int.Parse(ArgValue);
                        break;
                    case "--MaxThreadDeg":
                        paramTrain.MaxMultiThreadDegree = int.Parse(ArgValue);
                        break;
                    case "--T_value":
                        paramModel.T_value = float.Parse(ArgValue);
                        break;
                    case "--DebugLevel":
                        paramTrain.DebugLevel = (DebugLevel_t)Enum.Parse(typeof(DebugLevel_t), ArgValue, true);
                        break;
                    case "--flag_AdaptivenHidLayer":
                        paramModel.flag_AdaptivenHidLayer = bool.Parse(ArgValue);
                        break;
                    case "--flag_RunningAvg":
                        paramTrain.flag_RunningAvg = bool.Parse(ArgValue);
                        break;
                    default:
                        Console.WriteLine("Unknown ArgKey: {0}", ArgKey);
                        Program.DispHelp();
                        return false;
                }
            }
            if (String.IsNullOrEmpty(TrainInputFile) || String.IsNullOrEmpty(TestInputFile))
            {
                Console.WriteLine("Empty TrainInputFile or TestInputFile!");
                return false;
            }
            return true;
        }

        /*
         * Display help
         */
        public static void DispHelp()
        {
            Console.WriteLine("BP_LDA.exe");
        }
    }

    public class paramModel_t
    {
        public int nHid;
        public int nHidLayer;
        public float[] T = null;
        public float To;
        public int nOutput;
        public int nInput;
        public float eta;
        public float alpha;
        public float beta;
        public float T_value;
        public string OutputType;
        public bool flag_AdaptivenHidLayer;

        public DenseMatrix U;
        // For tied parameter case
        public DenseMatrix Phi;
        public DenseColumnVector b;
        
        public paramModel_t()
        {
        }

        public paramModel_t(paramModel_t SrcParamModel)
        {
            // Copy the model hyperparameters
            nInput = SrcParamModel.nInput;
            nOutput = SrcParamModel.nOutput;
            nHid = SrcParamModel.nHid;
            nHidLayer = SrcParamModel.nHidLayer;
            alpha = SrcParamModel.alpha;
            OutputType = SrcParamModel.OutputType;
            T = new float[SrcParamModel.T.Length];
            Array.Copy(SrcParamModel.T, T, SrcParamModel.T.Length);
            To = SrcParamModel.To;
            eta = SrcParamModel.eta;
            beta = SrcParamModel.beta;
            T_value = SrcParamModel.T_value;
            flag_AdaptivenHidLayer = SrcParamModel.flag_AdaptivenHidLayer;        
            // Copy the model parameters
            b = new DenseColumnVector(nHid, alpha - 1);
            Phi = new DenseMatrix(SrcParamModel.Phi);
            U = new DenseMatrix(SrcParamModel.U);
        }
    }


    public class paramTrain_t
    {
        public int nEpoch;
        public int BatchSize;
        public int BatchSize_Test;
        public float mu_Phi;
        public float mu_Phi_ReduceFactor;
        public int nTrain;
        public int nTest;
        public int nValid;
        public string LearnRateSchedule;
        public float mu_U;
        public int nSamplesPerDisplay;
        public int nEpochPerSave;
        public int nEpochPerTest;
        public bool flag_DumpFeature;
        public int nEpochPerDump;
        public bool flag_BachSizeSchedule;
        public Dictionary<int, int> BachSizeSchedule = null;
        public int ThreadNum;
        public int MaxMultiThreadDegree;
        public string ExternalEval;
        public bool flag_ExternalEval;
        public string TestLabelFile;
        public string TrainLabelFile;
        public string TestInputFile;
        public string TrainInputFile;
        public string ValidLabelFile;
        public string ValidInputFile;
        public bool flag_SaveAllModels;
        public bool flag_HasValidSet;
        public DebugLevel_t DebugLevel;
        public bool flag_RunningAvg;
    }

    public enum DebugLevel_t { low, medium, high};
    
}
