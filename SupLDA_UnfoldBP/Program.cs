using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using LinearAlgebra;
using BP_LDA;
using Common;

namespace BP_sLDA
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
            string ModelFile = "";
            string ResultFile = "";

            // ======== Parse the input parameters ========
            if (
                !ParseArgument(
                    args,
                    paramModel,
                    paramTrain,
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
            SparseMatrix TrainData = DataLoader.InputDataLoader(paramTrain.TrainInputFile, paramModel.nInput);
            SparseMatrix TrainLabel = DataLoader.LabelDataLoader(paramTrain.TrainLabelFile, paramModel.nOutput, paramModel.OutputType);
            SparseMatrix TestData = DataLoader.InputDataLoader(paramTrain.TestInputFile, paramModel.nInput);
            SparseMatrix TestLabel = DataLoader.LabelDataLoader(paramTrain.TestLabelFile, paramModel.nOutput, paramModel.OutputType);
            SparseMatrix ValidData = null;
            SparseMatrix ValidLabel = null;
            if (paramTrain.flag_HasValidSet)
            {
                ValidData = DataLoader.InputDataLoader(paramTrain.ValidInputFile, paramModel.nInput);
                ValidLabel = DataLoader.LabelDataLoader(paramTrain.ValidLabelFile, paramModel.nOutput, paramModel.OutputType);
            }
            paramTrain.nTrain = TrainData.nCols;
            paramTrain.nTest = TestData.nCols;
            if (paramTrain.flag_HasValidSet)
            {
                paramTrain.nValid = ValidData.nCols;
            }
            
            // ======== Supervised learning of BP-sLDA model: mirror-descent back-propagation
            // (i) Inference: Feedforward network via MDA unfolding
            // (ii) Learning: Projected (mini-batch) stochastic gradient descent (P-SGD) using back propagation
            LDA_Learn.TrainingBP_sLDA(TrainData, TrainLabel, TestData, TestLabel, ValidData, ValidLabel, paramModel, paramTrain, ModelFile, ResultFile);
            
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
            paramModel.OutputType = "softmaxCE";
            paramModel.nInput = 784;
            paramModel.flag_AdaptivenHidLayer = false;
            // ---- Training parameters ----
            paramTrain.nEpoch = 100;
            paramTrain.BatchSize = 100;
            paramTrain.BatchSize_Test = 100000;
            paramTrain.mu_Phi = 0.01f;
            paramTrain.mu_Phi_ReduceFactor = 10.0f;
            paramTrain.mu_U = 0.01f;
            paramTrain.LearnRateSchedule = "Constant";
            paramTrain.nSamplesPerDisplay = 1000;
            paramTrain.nEpochPerSave = 2;
            paramTrain.nEpochPerTest = 1;
            paramTrain.flag_DumpFeature = false;
            paramTrain.nEpochPerDump = 10;
            paramTrain.flag_BachSizeSchedule = false;
            paramTrain.ThreadNum = 100;
            paramTrain.MaxMultiThreadDegree = 32;
            paramTrain.flag_ExternalEval = false;
            paramTrain.flag_SaveAllModels = false;
            paramTrain.flag_HasValidSet = false;
            paramTrain.flag_RunningAvg = false;
        }

        /*
         * Parse the input arguments
         */
        public static bool ParseArgument(
            string[] args,
            paramModel_t paramModel,
            paramTrain_t paramTrain,
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
                    case "--To":
                        paramModel.To = float.Parse(ArgValue);
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
                        paramTrain.TrainInputFile = ArgValue;
                        break;
                    case "--TestInputFile":
                        paramTrain.TestInputFile = ArgValue;
                        break;
                    case "--TrainLabelFile":
                        paramTrain.TrainLabelFile = ArgValue;
                        break;
                    case "--TestLabelFile":
                        paramTrain.TestLabelFile = ArgValue;
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
                    case "--OutputType":
                        paramModel.OutputType = ArgValue;
                        if (paramModel.OutputType != "softmaxCE" && paramModel.OutputType != "linearQuad" && paramModel.OutputType != "linearCE")
                        {
                            throw new Exception("Unknown OutputType for supervised learning. Only softmaxCE/linearQuad/linearCE is supported.");
                        }
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
                    case "--ExternalEval":
                        paramTrain.flag_ExternalEval = true;
                        paramTrain.ExternalEval = ArgValue;
                        break;
                    case "--flag_SaveAllModels":
                        paramTrain.flag_SaveAllModels = bool.Parse(ArgValue);
                        break;
                    case "--ValidLabelFile":
                        paramTrain.ValidLabelFile = ArgValue;
                        paramTrain.flag_HasValidSet = true;
                        break;
                    case "--ValidInputFile":
                        paramTrain.ValidInputFile = ArgValue;
                        paramTrain.flag_HasValidSet = true;
                        break;
                    case "--T_value":
                        paramModel.T_value = float.Parse(ArgValue);
                        break;
                    case "--eta":
                        paramModel.eta = float.Parse(ArgValue);
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
            if (String.IsNullOrEmpty(paramTrain.TrainInputFile) || String.IsNullOrEmpty(paramTrain.TestInputFile)
                || String.IsNullOrEmpty(paramTrain.TrainLabelFile) || String.IsNullOrEmpty(paramTrain.TestLabelFile))
            {
                Console.WriteLine("Empty TrainInputFile, TestInputFile, TrainLabelFile, or TestLabelFile!");
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
}
