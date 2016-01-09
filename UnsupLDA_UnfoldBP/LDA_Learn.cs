using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using LinearAlgebra;
using System.Diagnostics;
using Common;
using System.IO;
using BP_LDA;

namespace BP_LDA
{
    public static class LDA_Learn
    {
        /*
         * Precompute learning rate schedule
         */
        public static double[] PrecomputeLearningRateSchedule(int nBatch, int nEpoch, double LearnRateStart, double LearnRateEnd, double Accuracy)
        {
            // Initialization
            double[] LearningRatePool = new double[nEpoch];
            LearningRatePool[nEpoch - 1] = LearnRateEnd;
            double b_min = 0;
            double b_max = 0;
            int iter = 0;
            double b;
            bool upper_flag;
            bool lower_flag;

            if (LearnRateEnd > LearnRateStart)
            {
                throw new System.ArgumentException("LearnRateEnd should be smaller than LearnRateStart");
            }

            // Precompute the optimal b by bi-section
            while (Math.Abs(LearningRatePool[0] - LearnRateStart) > Accuracy * LearnRateStart)
            {
                // Upper value of b
                b = b_max;
                for (int k = (nEpoch - 1); k >= 1; k--)
                {
                    LearningRatePool[k - 1] = 0.5 * (1 + 1 / (Math.Pow((1 - LearningRatePool[k] * b), 2 * nBatch))) * LearningRatePool[k];
                }
                upper_flag = ((LearningRatePool[0] > LearnRateStart) || (b * LearningRatePool.Max() >= 2)) ? true : false;

                // Lower value of b
                b = b_min;
                for (int k = (nEpoch - 1); k >= 1; k--)
                {
                    LearningRatePool[k - 1] = 0.5 * (1 + 1 / (Math.Pow((1 - LearningRatePool[k] * b), 2 * nBatch))) * LearningRatePool[k];
                }
                lower_flag = ((LearningRatePool[0] <= LearnRateStart) || (b * LearningRatePool.Max() < 2)) ? true : false;
                if (!lower_flag)
                {
                    throw new System.InvalidOperationException("lower_flag cannot be zero");
                }

                // Update
                if (!upper_flag)
                {
                    b_max = b_max + 1;
                }
                else
                {
                    b = (b_max + b_min) / 2;
                    for (int k = (nEpoch - 1); k >= 1; k--)
                    {
                        LearningRatePool[k - 1] = 0.5 * (1 + 1 / (Math.Pow((1 - LearningRatePool[k] * b), 2 * nBatch))) * LearningRatePool[k];
                    }
                    if ((LearningRatePool[0] > LearnRateStart) || (b * LearningRatePool.Max() > 2))
                    {
                        b_max = b;
                    }
                    else
                    {
                        b_min = b;
                    }
                }
                iter++;
                if (iter > 1e10)
                {
                    throw new System.InvalidOperationException("Maximum number of iterations has reached");
                }
            }

            return LearningRatePool;
        }

        /*
         * Training: unsupervised learning of feedforward (unfolding) LDA by back propagation
         */
        public static void TrainingBP_LDA(
            SparseMatrix TrainData,
            SparseMatrix TestData,
            paramModel_t paramModel,
            paramTrain_t paramTrain,
            string ModelFile,
            string ResultFile
            )
        {
            // ---- Extract the parameters ----
            // Model parameters
            int nInput = paramModel.nInput;
            int nHid = paramModel.nHid;
            int nHidLayer = paramModel.nHidLayer;
            int nOutput = paramModel.nOutput;
            float eta = paramModel.eta;
            float T_value = paramModel.T_value;
            string OutputType = paramModel.OutputType;
            float beta = paramModel.beta;
            // Training parameters
            int nEpoch = paramTrain.nEpoch;
            float mu_Phi = paramTrain.mu_Phi;
            float mu_U = paramTrain.mu_U;
            int nTrain = paramTrain.nTrain;
            float mu_Phi_ReduceFactor = paramTrain.mu_Phi_ReduceFactor;
            string LearnRateSchedule = paramTrain.LearnRateSchedule;
            int nSamplesPerDisplay = paramTrain.nSamplesPerDisplay;
            int nEpochPerSave = paramTrain.nEpochPerSave;
            int nEpochPerTest = paramTrain.nEpochPerTest;
            int nEpochPerDump = paramTrain.nEpochPerDump;

            // ---- Initialize the model ----
            ModelInit_LDA_Feedforward(paramModel);

            // ---- Initialize the training algorithm ----
            Console.WriteLine("=================================================================");
            Console.WriteLine("Unsupervised learning of BP-LDA: Mirror-Descent Back Propagation");
            Console.WriteLine("=================================================================");
            float TotLoss = 0.0f;
            float TotCE = 0.0f;
            double TotTime = 0.0f;
            double TotTimeThisEpoch = 0.0f;
            int TotSamples = 0;
            int TotSamplesThisEpoch = 0;
            double AvgnHidLayerEffective = 0.0;
            int CntRunningAvg = 0;
            int CntModelUpdate = 0;
            DenseRowVector mu_phi_search = new DenseRowVector(nHid, mu_Phi);
            DenseRowVector TestLoss_pool = new DenseRowVector(nEpoch / nEpochPerTest, 0.0f);
            DenseRowVector TestLoss_epoch = new DenseRowVector(nEpoch / nEpochPerTest, 0.0f);
            DenseRowVector TestLoss_time = new DenseRowVector(nEpoch / nEpochPerTest, 0.0f);
            int CountTest = 0;
            DenseRowVector G_Phi_pool = new DenseRowVector(paramModel.nHidLayer);
            DenseRowVector G_Phi_trunc_pool = new DenseRowVector(paramModel.nHidLayer, 0.0f);
            DenseRowVector AdaGradSum = new DenseRowVector(nHid, 0.0f);
            DenseRowVector TmpDenseRowVec = new DenseRowVector(nHid, 0.0f);
            int[] SparsePatternGradPhi = null;
            float nLearnLineSearch = 0.0f;
            int[] IdxPerm = null;
            int BatchSize_NormalBatch = paramTrain.BatchSize;
            int BatchSize_tmp = paramTrain.BatchSize;
            int nBatch = (int)Math.Ceiling(((float)nTrain) / ((float)BatchSize_NormalBatch));
            DNNRun_t DNNRun_NormalBatch = new DNNRun_t(nHid, BatchSize_NormalBatch, paramModel.nHidLayer, nOutput);
            DNNRun_t DNNRun_EndBatch = new DNNRun_t(nHid, nTrain - (nBatch - 1) * BatchSize_NormalBatch, paramModel.nHidLayer, nOutput);
            DNNRun_t DNNRun = null;
            Grad_t Grad = new Grad_t(nHid, nOutput, nInput, paramModel.nHidLayer, OutputType);
            DenseMatrix TmpGradDense = new DenseMatrix(nInput, nHid);
            DenseMatrix TmpMatDensePhi = new DenseMatrix(nInput, nHid);
            paramModel_t paramModel_avg = new paramModel_t(paramModel);
            Stopwatch stopWatch = new Stopwatch();
            // ---- Compute the schedule of the learning rate
            double[] stepsize_pool = null;
            switch (LearnRateSchedule)
            {
                case "PreCompute":
                    stepsize_pool = PrecomputeLearningRateSchedule(nBatch, nEpoch, mu_Phi, mu_Phi / mu_Phi_ReduceFactor, 1e-8f);
                    break;
                case "Constant":
                    stepsize_pool = new double[nEpoch];
                    for (int Idx = 0; Idx < nEpoch; Idx++)
                    {
                        stepsize_pool[Idx] = mu_Phi;
                    }
                    break;
                default:
                    throw new Exception("Unknown type of LearnRateSchedule");
            }
            // Now start training.........................
            for (int epoch = 0; epoch < nEpoch; epoch++)
            {
                TotSamplesThisEpoch = 0;
                TotTimeThisEpoch = 0.0;
                AvgnHidLayerEffective = 0.0;
                // -- Set the batch size if there is schedule --
                if (paramTrain.flag_BachSizeSchedule)
                {
                    if (paramTrain.BachSizeSchedule.TryGetValue(epoch + 1, out BatchSize_tmp))
                    {
                        BatchSize_NormalBatch = BatchSize_tmp;
                        nBatch = (int)Math.Ceiling(((float)nTrain) / ((float)BatchSize_NormalBatch));
                        DNNRun_NormalBatch = new DNNRun_t(nHid, BatchSize_NormalBatch, paramModel.nHidLayer, nOutput);
                        DNNRun_EndBatch = new DNNRun_t(nHid, nTrain - (nBatch - 1) * BatchSize_NormalBatch, paramModel.nHidLayer, nOutput);
                    }
                }

                // -- Shuffle the data (generating shuffled index) --
                IdxPerm = Statistics.RandPerm(nTrain);
                // -- Reset the (MDA) inference step-sizes --
                if (epoch > 0)
                {
                    for (int Idx = 0; Idx < paramModel.nHidLayer; Idx++)
                    {
                        paramModel.T[Idx] = T_value;
                    }
                }
                // -- Take the learning rate for the current epoch --
                mu_Phi = (float)stepsize_pool[epoch];
                // -- Start this epoch --
                Console.WriteLine("============== Epoch #{0}. BatchSize: {1} Learning Rate: {2} ==================", epoch + 1, BatchSize_NormalBatch, mu_Phi);
                for (int IdxBatch = 0; IdxBatch < nBatch; IdxBatch++)
                {
                    stopWatch.Start();
                    // Extract the batch
                    int BatchSize = 0;
                    if (IdxBatch < nBatch - 1)
                    {
                        BatchSize = BatchSize_NormalBatch;
                        DNNRun = DNNRun_NormalBatch;
                    }
                    else
                    {
                        BatchSize = nTrain - IdxBatch * BatchSize_NormalBatch;
                        DNNRun = DNNRun_EndBatch;
                    }
                    SparseMatrix Xt = new SparseMatrix(nInput, BatchSize);
                    SparseMatrix Dt = null;
                    int[] IdxSample = new int[BatchSize];
                    Array.Copy(IdxPerm, IdxBatch * BatchSize_NormalBatch, IdxSample, 0, BatchSize);
                    TrainData.GetColumns(Xt, IdxSample);

                    // Set the sparse pattern for the gradient
                    SparsePatternGradPhi = Xt.GetHorizontalUnionSparsePattern();
                    Grad.SetSparsePatternForAllGradPhi(SparsePatternGradPhi);

                    // Forward activation
                    LDA_Learn.ForwardActivation_LDA(Xt, DNNRun, paramModel, true);

                    // Back propagation
                    LDA_Learn.BackPropagation_LDA(Xt, Dt, DNNRun, paramModel, Grad);
                                        
                    // Compute the gradient and update the model (All gradients of Phi are accumulated into Grad.grad_Q_Phi)
                    MatrixOperation.ScalarDivideMatrix(Grad.grad_Q_Phi, (-1.0f) * ((beta - 1) / ((float)nTrain)), paramModel.Phi, true);
                    MatrixOperation.MatrixAddMatrix(Grad.grad_Q_Phi, Grad.grad_Q_TopPhi);                  
                    mu_phi_search.FillValue(mu_Phi);
                    // Different learning rate for different columns of Phi: Similar to AdaGrad but does not decay with time
                    ++CntModelUpdate;
                    MatrixOperation.ElementwiseMatrixMultiplyMatrix(TmpMatDensePhi, Grad.grad_Q_Phi, Grad.grad_Q_Phi);
                    MatrixOperation.VerticalSumMatrix(TmpDenseRowVec, TmpMatDensePhi);
                    MatrixOperation.ScalarMultiplyVector(TmpDenseRowVec, (1.0f / ((float)nInput)));
                    MatrixOperation.VectorAddVector(AdaGradSum, TmpDenseRowVec);
                    MatrixOperation.ScalarMultiplyVector(TmpDenseRowVec, AdaGradSum, 1.0f / ((float)CntModelUpdate));
                    MatrixOperation.ElementwiseSquareRoot(TmpDenseRowVec, TmpDenseRowVec);
                    MatrixOperation.ScalarAddVector(TmpDenseRowVec, mu_Phi);
                    MatrixOperation.ElementwiseVectorDivideVector(mu_phi_search, mu_phi_search, TmpDenseRowVec);
                    nLearnLineSearch = SMD_Update(paramModel.Phi, Grad.grad_Q_Phi, mu_phi_search, eta);
                    // Running average of the model
                    if (paramTrain.flag_RunningAvg && epoch >= (int)Math.Ceiling(((float)nEpoch) / 2.0f))
                    {
                        ++CntRunningAvg;
                        MatrixOperation.ScalarMultiplyMatrix(paramModel_avg.Phi, ((float)(CntRunningAvg - 1)));
                        MatrixOperation.MatrixAddMatrix(paramModel_avg.Phi, paramModel.Phi);
                        MatrixOperation.ScalarMultiplyMatrix(paramModel_avg.Phi, 1.0f / ((float)CntRunningAvg));
                    }


                    // Display the result
                    TotCE += ComputeCrossEntropy(Xt, paramModel.Phi,DNNRun.theta_pool, DNNRun.nHidLayerEffective);
                    //TotLoss += ComputeRegularizedCrossEntropy(Xt, paramModel.Phi, DNNRun.theta_pool[nHidLayer - 1], paramModel.b);
                    TotLoss = TotCE;
                    TotSamples += BatchSize;
                    TotSamplesThisEpoch += BatchSize;
                    AvgnHidLayerEffective = (((float)(TotSamplesThisEpoch-BatchSize))/((float)TotSamplesThisEpoch))*AvgnHidLayerEffective
                        + (1.0/((float)TotSamplesThisEpoch))*( DNNRun.nHidLayerEffective.Sum());
                    stopWatch.Stop();
                    TimeSpan ts = stopWatch.Elapsed;
                    TotTime += ts.TotalSeconds;
                    TotTimeThisEpoch += ts.TotalSeconds;
                    stopWatch.Reset();
                    if (TotSamplesThisEpoch % nSamplesPerDisplay == 0)
                    {
                        // Display results
                        Console.WriteLine(
                            "- Ep#{0}/{1} Bat#{2}/{3}. Loss={4:F3}. CE={5:F3}.  Speed={6} Samples/Sec.",
                            epoch + 1, nEpoch,
                            IdxBatch + 1, nBatch,
                            TotLoss / TotSamples, TotCE / TotSamples,
                            (int)((double)TotSamplesThisEpoch / TotTimeThisEpoch)
                            );
                        if (paramTrain.DebugLevel == DebugLevel_t.medium)
                        {
                            Console.WriteLine(
                                "  mu_phi_search_max={0} \n  mu_phi_search_min={1}",
                                mu_phi_search.VectorValue.Max(), mu_phi_search.VectorValue.Min()
                                );
                            Console.WriteLine("----------------------------------------------------");
                        }
                        if (paramTrain.DebugLevel == DebugLevel_t.high)
                        {
                            Console.WriteLine(
                                "  mu_phi_search_max={0} \n  mu_phi_search_min={1}",
                                mu_phi_search.VectorValue.Max(), mu_phi_search.VectorValue.Min()
                                );
                            Console.WriteLine(
                                "  AvgnHidLayerEff={0:F1}. G_Phi={1:F3}.",
                                AvgnHidLayerEffective,
                                Grad.grad_Q_Phi.MaxAbsValue()
                                );
                            Console.WriteLine("----------------------------------------------------");
                        }


                    }
                }
                // -- Test --
                if ((epoch + 1) % nEpochPerTest == 0)
                {
                    TestLoss_epoch.VectorValue[(epoch + 1) / nEpochPerTest - 1] = epoch + 1;
                    TestLoss_time.VectorValue[(epoch + 1) / nEpochPerTest - 1] = (float)TotTime;
                    if (paramTrain.flag_RunningAvg && epoch >= (int)Math.Ceiling(((float)nEpoch) / 2.0f))
                    {
                        TestLoss_pool.VectorValue[(epoch + 1) / nEpochPerTest - 1] = Testing_BP_LDA(TestData, paramModel_avg, paramTrain.BatchSize_Test);
                    }
                    else
                    {
                        TestLoss_pool.VectorValue[(epoch + 1) / nEpochPerTest - 1] = Testing_BP_LDA(TestData, paramModel, paramTrain.BatchSize_Test);
                    }
                    CountTest++;
                }

                // -- Save --
                if ((epoch + 1) % nEpochPerSave == 0)
                {
                    // Save model
                    if (paramTrain.flag_RunningAvg && epoch >= (int)Math.Ceiling(((float)nEpoch) / 2.0f))
                    {
                        string PhiCol = null;
                        (new FileInfo(ResultFile + ".model.Phi")).Directory.Create();
                        StreamWriter FileSaveModel = new StreamWriter(ResultFile + ".model.Phi", false);
                        for (int IdxCol = 0; IdxCol < paramModel_avg.Phi.nCols; IdxCol++)
                        {
                            PhiCol = String.Join("\t", paramModel_avg.Phi.DenseMatrixValue[IdxCol].VectorValue);
                            FileSaveModel.WriteLine(PhiCol);
                        }
                        FileSaveModel.Close();
                        // Save the final learning curves
                        StreamWriter FileSavePerf = new StreamWriter(ResultFile + ".perf", false);
                        FileSavePerf.WriteLine(String.Join("\t", TestLoss_epoch.VectorValue));
                        FileSavePerf.WriteLine(String.Join("\t", TestLoss_time.VectorValue));
                        FileSavePerf.WriteLine(String.Join("\t", TestLoss_pool.VectorValue));
                        FileSavePerf.Close();
                    }
                    {
                        string PhiCol = null;
                        (new FileInfo(ResultFile + ".model.Phi")).Directory.Create();
                        StreamWriter FileSaveModel = new StreamWriter(ResultFile + ".model.Phi", false);
                        for (int IdxCol = 0; IdxCol < paramModel.Phi.nCols; IdxCol++)
                        {
                            PhiCol = String.Join("\t", paramModel.Phi.DenseMatrixValue[IdxCol].VectorValue);
                            FileSaveModel.WriteLine(PhiCol);
                        }
                        FileSaveModel.Close();
                        // Save the final learning curves
                        StreamWriter FileSavePerf = new StreamWriter(ResultFile + ".perf", false);
                        FileSavePerf.WriteLine(String.Join("\t", TestLoss_epoch.VectorValue));
                        FileSavePerf.WriteLine(String.Join("\t", TestLoss_time.VectorValue));
                        FileSavePerf.WriteLine(String.Join("\t", TestLoss_pool.VectorValue));
                        FileSavePerf.Close();
                    }
                }

                // -- Dump feature --
                if (paramTrain.flag_DumpFeature && (epoch + 1) % nEpochPerDump == 0)
                {
                    if (paramTrain.flag_RunningAvg && epoch >= (int)Math.Ceiling(((float)nEpoch) / 2.0f))
                    {
                        DumpingFeature_BP_LDA(TrainData, paramModel_avg, paramTrain.BatchSize_Test, ResultFile + ".train.fea", "Train");
                        DumpingFeature_BP_LDA(TestData, paramModel_avg, paramTrain.BatchSize_Test, ResultFile + ".test.fea", "Test");
                    }
                    {
                        DumpingFeature_BP_LDA(TrainData, paramModel, paramTrain.BatchSize_Test, ResultFile + ".train.fea", "Train");
                        DumpingFeature_BP_LDA(TestData, paramModel, paramTrain.BatchSize_Test, ResultFile + ".test.fea", "Test");
                    }
                }


            }
        }



        /*
         * Training: supervised learning of feedforward (unfolding) LDA by back propagation
         */
        public static void TrainingBP_sLDA(
            SparseMatrix TrainData,
            SparseMatrix TrainLabel,
            SparseMatrix TestData,
            SparseMatrix TestLabel,
            SparseMatrix ValidData,
            SparseMatrix ValidLabel,
            paramModel_t paramModel,
            paramTrain_t paramTrain,
            string ModelFile,
            string ResultFile
            )
        {
            Console.WriteLine("=================================================================");
            Console.WriteLine("Supervised learning of BP-sLDA: Mirror-Descent Back Propagation");
            Console.WriteLine("=================================================================");
            // ---- Extract the parameters ----
            // Model parameters
            int nInput = paramModel.nInput;
            int nHid = paramModel.nHid;
            int nHidLayer = paramModel.nHidLayer;
            int nOutput = paramModel.nOutput;
            float eta = paramModel.eta;
            float T_value = paramModel.T_value;
            string OutputType = paramModel.OutputType;
            float beta = paramModel.beta;
            // Training parameters
            int nEpoch = paramTrain.nEpoch;
            float mu_Phi = paramTrain.mu_Phi;
            float mu_U = paramTrain.mu_U;
            int nTrain = paramTrain.nTrain;
            float mu_ReduceFactor = paramTrain.mu_Phi_ReduceFactor;
            string LearnRateSchedule = paramTrain.LearnRateSchedule;
            int nSamplesPerDisplay = paramTrain.nSamplesPerDisplay;
            int nEpochPerSave = paramTrain.nEpochPerSave;
            int nEpochPerTest = paramTrain.nEpochPerTest;
            int nEpochPerDump = paramTrain.nEpochPerDump;
            

            // ---- Initialize the model ----            
            ModelInit_LDA_Feedforward(paramModel);

            // ---- Initialize the training algorithm ----
            float TotLoss = 0.0f;
            float TotTrErr = 0.0f;
            double TotTime = 0.0f;
            double TotTimeThisEpoch = 0.0f;
            int TotSamples = 0;
            int TotSamplesThisEpoch = 0;
            float CntRunningAvg = 0.0f;
            float CntModelUpdate = 0.0f;
            double AvgnHidLayerEffective = 0.0f;
            DenseRowVector mu_phi_search = new DenseRowVector(nHid, mu_Phi);
            DenseRowVector mu_U_search = new DenseRowVector(nHid, mu_U);
            DenseRowVector AdaGradSum = new DenseRowVector(nHid, 0.0f);
            DenseRowVector TmpDenseRowVec = new DenseRowVector(nHid, 0.0f);
            DenseRowVector TestError_pool = new DenseRowVector(nEpoch / nEpochPerTest, 0.0f);
            DenseRowVector ValidError_pool = new DenseRowVector(nEpoch / nEpochPerTest, 0.0f);
            DenseRowVector TrainError_pool = new DenseRowVector(nEpoch / nEpochPerTest, 0.0f);
            DenseRowVector TrainLoss_pool = new DenseRowVector(nEpoch / nEpochPerTest, 0.0f);
            DenseRowVector TestError_epoch = new DenseRowVector(nEpoch / nEpochPerTest, 0.0f);
            DenseRowVector TestError_time = new DenseRowVector(nEpoch / nEpochPerTest, 0.0f);
            int CountTest = 0;
            float nLearnLineSearch = 0.0f;
            int[] IdxPerm = null;
            int BatchSize_NormalBatch = paramTrain.BatchSize;
            int BatchSize_tmp = paramTrain.BatchSize;
            int nBatch = (int)Math.Ceiling(((float)nTrain) / ((float)BatchSize_NormalBatch));
            DNNRun_t DNNRun_NormalBatch = new DNNRun_t(nHid, BatchSize_NormalBatch, paramModel.nHidLayer, nOutput);
            DNNRun_t DNNRun_EndBatch = new DNNRun_t(nHid, nTrain - (nBatch - 1) * BatchSize_NormalBatch, paramModel.nHidLayer, nOutput);
            DNNRun_t DNNRun = null;
            Grad_t Grad = new Grad_t(nHid, nOutput, nInput, paramModel.nHidLayer, OutputType);
            SparseMatrix TmpGrad = new SparseMatrix(nInput, nHid, true);
            DenseMatrix TmpMatDensePhi = new DenseMatrix(nInput, nHid);
            DenseMatrix TmpMatDenseU = new DenseMatrix(nOutput, nHid);
            paramModel_t paramModel_avg = new paramModel_t(paramModel);          
            Stopwatch stopWatch = new Stopwatch();            
            // ---- Compute the schedule of the learning rate
            double[] stepsize_pool_Phi = null;
            double[] stepsize_pool_U = null;
            switch (LearnRateSchedule)
            {
                case "PreCompute":
                    stepsize_pool_Phi = PrecomputeLearningRateSchedule(nBatch, nEpoch, mu_Phi, mu_Phi / mu_ReduceFactor, 1e-8f);
                    stepsize_pool_U = PrecomputeLearningRateSchedule(nBatch, nEpoch, mu_U, mu_U / mu_ReduceFactor, 1e-8f);
                    break;
                case "Constant":
                    stepsize_pool_Phi = new double[nEpoch];
                    stepsize_pool_U = new double[nEpoch];
                    for (int Idx = 0; Idx < nEpoch; Idx++)
                    {
                        stepsize_pool_Phi[Idx] = mu_Phi;
                        stepsize_pool_U[Idx] = mu_U;
                    }
                    break;
                default:
                    throw new Exception("Unknown type of LearnRateSchedule");
            }
            // Now start training.........................
            for (int epoch = 0; epoch < nEpoch; epoch++)
            {
                TotSamplesThisEpoch = 0;
                TotTimeThisEpoch = 0.0;
                AvgnHidLayerEffective = 0.0f;
                // -- Set the batch size if there is schedule --
                if (paramTrain.flag_BachSizeSchedule)
                {
                    if (paramTrain.BachSizeSchedule.TryGetValue(epoch + 1, out BatchSize_tmp))
                    {
                        BatchSize_NormalBatch = BatchSize_tmp;
                        nBatch = (int)Math.Ceiling(((float)nTrain) / ((float)BatchSize_NormalBatch));
                        DNNRun_NormalBatch = new DNNRun_t(nHid, BatchSize_NormalBatch, paramModel.nHidLayer, nOutput);
                        DNNRun_EndBatch = new DNNRun_t(nHid, nTrain - (nBatch - 1) * BatchSize_NormalBatch, paramModel.nHidLayer, nOutput);
                    }
                }

                // -- Shuffle the data (generating shuffled index) --
                IdxPerm = Statistics.RandPerm(nTrain);
                // -- Reset the (MDA) inference step-sizes --
                if (epoch > 0)
                {
                    for (int Idx = 0; Idx < paramModel.nHidLayer; Idx++)
                    {
                        paramModel.T[Idx] = T_value;
                    }
                }
                // -- Take the learning rate for the current epoch --
                mu_Phi = (float)stepsize_pool_Phi[epoch];
                mu_U = (float)stepsize_pool_U[epoch];
                // -- Start this epoch --
                Console.WriteLine("============== Epoch #{0}. BatchSize: {1} Learning Rate: Phi:{2}, U:{3} ==================",
                    epoch + 1, BatchSize_NormalBatch, mu_Phi, mu_U);
                for (int IdxBatch = 0; IdxBatch < nBatch; IdxBatch++)
                {
                    stopWatch.Start();
                    // Extract the batch
                    int BatchSize = 0;
                    if (IdxBatch < nBatch - 1)
                    {
                        BatchSize = BatchSize_NormalBatch;
                        DNNRun = DNNRun_NormalBatch;
                    }
                    else
                    {
                        BatchSize = nTrain - IdxBatch * BatchSize_NormalBatch;
                        DNNRun = DNNRun_EndBatch;
                    }
                    SparseMatrix Xt = new SparseMatrix(nInput, BatchSize);
                    SparseMatrix Dt = new SparseMatrix(nOutput, BatchSize);
                    int[] IdxSample = new int[BatchSize];
                    Array.Copy(IdxPerm, IdxBatch * BatchSize_NormalBatch, IdxSample, 0, BatchSize);
                    TrainData.GetColumns(Xt, IdxSample);
                    TrainLabel.GetColumns(Dt, IdxSample);
                    
                    // Forward activation
                    LDA_Learn.ForwardActivation_LDA(Xt, DNNRun, paramModel, true);

                    // Back propagation
                    LDA_Learn.BackPropagation_LDA(Xt, Dt, DNNRun, paramModel, Grad);
                    
                    // Compute the gradient and update the model (All gradients of Phi are accumulated into Grad.grad_Q_Phi)
                    // (i) Update Phi
                    MatrixOperation.ScalarDivideMatrix(Grad.grad_Q_Phi, (-1.0f) * ((beta - 1) / ((float)nTrain)), paramModel.Phi, true);       
                    mu_phi_search.FillValue(mu_Phi);
                    // Different learning rate for different columns of Phi: Similar to AdaGrad but does not decay with time
                    ++CntModelUpdate;
                    MatrixOperation.ElementwiseMatrixMultiplyMatrix(TmpMatDensePhi, Grad.grad_Q_Phi, Grad.grad_Q_Phi);
                    MatrixOperation.VerticalSumMatrix(TmpDenseRowVec, TmpMatDensePhi);
                    MatrixOperation.ScalarMultiplyVector(TmpDenseRowVec, 1.0f / ((float)nInput));
                    MatrixOperation.VectorSubtractVector(TmpDenseRowVec, AdaGradSum);
                    MatrixOperation.ScalarMultiplyVector(TmpDenseRowVec, 1.0f / CntModelUpdate);
                    MatrixOperation.VectorAddVector(AdaGradSum, TmpDenseRowVec);
                    MatrixOperation.ElementwiseSquareRoot(TmpDenseRowVec, AdaGradSum);
                    MatrixOperation.ScalarAddVector(TmpDenseRowVec, mu_Phi);
                    MatrixOperation.ElementwiseVectorDivideVector(mu_phi_search, mu_phi_search, TmpDenseRowVec);
                    nLearnLineSearch = SMD_Update(paramModel.Phi, Grad.grad_Q_Phi, mu_phi_search, eta);
                    // (ii) Update U                    
                    MatrixOperation.ScalarMultiplyMatrix(Grad.grad_Q_U, (-1.0f) * mu_U);
                    MatrixOperation.MatrixAddMatrix(paramModel.U, Grad.grad_Q_U);
                    // (iii) Running average of the model
                    if (paramTrain.flag_RunningAvg && epoch >= (int)Math.Ceiling(((float)nEpoch)/2.0f))
                    {
                        ++CntRunningAvg;
                        MatrixOperation.MatrixSubtractMatrix(TmpMatDensePhi, paramModel.Phi, paramModel_avg.Phi);
                        MatrixOperation.MatrixSubtractMatrix(TmpMatDenseU, paramModel.U, paramModel_avg.U);
                        MatrixOperation.ScalarMultiplyMatrix(TmpMatDensePhi, 1.0f / CntRunningAvg);
                        MatrixOperation.ScalarMultiplyMatrix(TmpMatDenseU, 1.0f / CntRunningAvg);
                        MatrixOperation.MatrixAddMatrix(paramModel_avg.Phi, TmpMatDensePhi);
                        MatrixOperation.MatrixAddMatrix(paramModel_avg.U, TmpMatDenseU);
                    }

                    // Display the result
                    TotTrErr += 100 * ComputeNumberOfErrors(Dt, DNNRun.y);
                    TotLoss += ComputeSupervisedLoss(Dt, DNNRun.y, paramModel.OutputType);
                    TotSamples += BatchSize;
                    TotSamplesThisEpoch += BatchSize;
                    AvgnHidLayerEffective =
                        (((double)(TotSamplesThisEpoch - BatchSize)) / ((double)TotSamplesThisEpoch)) * AvgnHidLayerEffective 
                        +
                        1.0 / ((double)TotSamplesThisEpoch) * DNNRun.nHidLayerEffective.Sum();
                    stopWatch.Stop();
                    TimeSpan ts = stopWatch.Elapsed;
                    TotTime += ts.TotalSeconds;
                    TotTimeThisEpoch += ts.TotalSeconds;
                    stopWatch.Reset();
                    if (TotSamplesThisEpoch % nSamplesPerDisplay == 0)
                    {
                        // Display results
                        Console.WriteLine(
                            "- Ep#{0}/{1} Bat#{2}/{3}. Loss={4:F3}. TrErr={5:F3}%. Speed={6} Samples/Sec.",
                            epoch + 1, nEpoch,
                            IdxBatch + 1, nBatch,
                            TotLoss / TotSamples, TotTrErr / TotSamples,
                            (int)((double)TotSamplesThisEpoch / TotTimeThisEpoch)
                            );
                        if (paramTrain.DebugLevel == DebugLevel_t.medium)
                        {
                            Console.WriteLine(
                                "  mu_phi_search_max={0} \n  mu_phi_search_min={1}",
                                mu_phi_search.VectorValue.Max(), mu_phi_search.VectorValue.Min()
                                );
                            Console.WriteLine("----------------------------------------------------");
                        }
                        if (paramTrain.DebugLevel == DebugLevel_t.high)
                        {
                            Console.WriteLine(
                                "  mu_phi_search_max={0} \n  mu_phi_search_min={1}",
                                mu_phi_search.VectorValue.Max(), mu_phi_search.VectorValue.Min()
                                );
                            float MaxAbsVal_Grad_Q_Phi = Grad.grad_Q_Phi.MaxAbsValue();
                            float MaxAbsVal_Grad_Q_U = Grad.grad_Q_U.MaxAbsValue();
                            Console.WriteLine(
                                "  AvgnHidLayerEff={0:F1}. G_Phi={1:F3}. G_U={2:F3}",
                                AvgnHidLayerEffective,
                                MaxAbsVal_Grad_Q_Phi,
                                MaxAbsVal_Grad_Q_U
                                );
                            // Save the screen into a log file
                            (new FileInfo(ResultFile + ".log")).Directory.Create();
                            using (StreamWriter LogFile = File.AppendText(ResultFile + ".log"))
                            {
                                LogFile.WriteLine(
                                    "- Ep#{0}/{1} Bat#{2}/{3}. Loss={4:F3}. TrErr={5:F3}%. Speed={6} Samples/Sec.",
                                    epoch + 1, nEpoch,
                                    IdxBatch + 1, nBatch,
                                    TotLoss / TotSamples, TotTrErr / TotSamples,
                                    (int)((double)TotSamplesThisEpoch / TotTimeThisEpoch)
                                    );
                                LogFile.WriteLine(
                                    "  mu_phi_search_max={0} \n  mu_phi_search_min={1}",
                                    mu_phi_search.VectorValue.Max(), mu_phi_search.VectorValue.Min()
                                    );
                                LogFile.WriteLine(
                                    "  AvgnHidLayerEff={0:F1}. G_Phi={1:F3}. G_U={2:F3}",
                                    AvgnHidLayerEffective,
                                    MaxAbsVal_Grad_Q_Phi,
                                    MaxAbsVal_Grad_Q_U
                                    );
                                LogFile.WriteLine("----------------------------------------------------");
                            }
                            Console.WriteLine("----------------------------------------------------");
                        }
                                                
                    }
                }
                // -- Test --
                if ((epoch + 1) % nEpochPerTest == 0)
                {
                    // Standard performance metric
                    TestError_epoch.VectorValue[(epoch + 1) / nEpochPerTest - 1] = epoch + 1;
                    TestError_time.VectorValue[(epoch + 1) / nEpochPerTest - 1] = (float)TotTime;
                    if (paramTrain.flag_RunningAvg && epoch >= (int)Math.Ceiling(((float)nEpoch) / 2.0f))
                    {
                        if (paramTrain.flag_HasValidSet)
                        {
                            ValidError_pool.VectorValue[(epoch + 1) / nEpochPerTest - 1]
                                = Testing_BP_sLDA(
                                    ValidData, 
                                    ValidLabel, 
                                    paramModel_avg, 
                                    paramTrain.BatchSize_Test, 
                                    ResultFile + ".validscore", 
                                    "Validation Set"
                                    );
                        }
                        TestError_pool.VectorValue[(epoch + 1) / nEpochPerTest - 1]
                            = Testing_BP_sLDA(
                                    TestData, 
                                    TestLabel, 
                                    paramModel_avg, 
                                    paramTrain.BatchSize_Test, 
                                    ResultFile + ".testscore", 
                                    "Test Set"
                                    );                        
                    }
                    else
                    {
                        if (paramTrain.flag_HasValidSet)
                        {
                            ValidError_pool.VectorValue[(epoch + 1) / nEpochPerTest - 1]
                                = Testing_BP_sLDA(
                                    ValidData, 
                                    ValidLabel, 
                                    paramModel, 
                                    paramTrain.BatchSize_Test, 
                                    ResultFile + ".validscore", 
                                    "Validation Set"
                                    );
                        }
                        TestError_pool.VectorValue[(epoch + 1) / nEpochPerTest - 1]
                            = Testing_BP_sLDA(
                                    TestData, 
                                    TestLabel, 
                                    paramModel, 
                                    paramTrain.BatchSize_Test, 
                                    ResultFile + ".testscore", 
                                    "Test Set"
                                    );
                    }
                    TrainError_pool.VectorValue[(epoch + 1) / nEpochPerTest - 1]
                            = TotTrErr / TotSamples;
                    TrainLoss_pool.VectorValue[(epoch + 1) / nEpochPerTest - 1]
                        = TotLoss / TotSamples;

                    // Performance metric evaluated using external evaluation tools, e.g., AUC, Top@K accuracy, etc.
                    if (paramTrain.flag_ExternalEval)
                    {
                        ExternalEvaluation(
                            paramTrain.ExternalEval, 
                            ResultFile, 
                            paramTrain.TestLabelFile, 
                            epoch, 
                            "Test Set"
                            );
                        if (paramTrain.flag_HasValidSet)
                        {
                            ExternalEvaluation(
                                paramTrain.ExternalEval, 
                                ResultFile, 
                                paramTrain.ValidLabelFile, 
                                epoch, 
                                "Validation Set"
                                );
                        }
                    }

                    CountTest++;
                }

                // -- Save --
                if ((epoch + 1) % nEpochPerSave == 0)
                {
                    // Save model
                    string PhiCol = null;
                    string UCol = null;
                    (new FileInfo(ResultFile + ".model.Phi")).Directory.Create();
                    string ModelName_Phi;
                    string ModelName_U;
                    if (paramTrain.flag_SaveAllModels)
                    {
                        ModelName_Phi = ResultFile + ".model.Phi" + ".iter" + (epoch + 1).ToString();
                        ModelName_U = ResultFile + ".model.U" + ".iter" + (epoch + 1).ToString();
                    }
                    else
                    {
                        ModelName_Phi = ResultFile + ".model.Phi";
                        ModelName_U = ResultFile + ".model.U";
                    }
                    if (paramTrain.flag_RunningAvg && epoch >= (int)Math.Ceiling(((float)nEpoch) / 2.0f))
                    {
                        using (StreamWriter FileSaveModel_Phi = new StreamWriter(ModelName_Phi, false))
                        {
                            for (int IdxCol = 0; IdxCol < paramModel_avg.Phi.nCols; IdxCol++)
                            {
                                PhiCol = String.Join("\t", paramModel_avg.Phi.DenseMatrixValue[IdxCol].VectorValue);
                                FileSaveModel_Phi.WriteLine(PhiCol);
                            }
                        }
                        using (StreamWriter FileSaveModel_U = new StreamWriter(ModelName_U, false))
                        {
                            for (int IdxCol = 0; IdxCol < paramModel_avg.U.nCols; IdxCol++)
                            {
                                UCol = String.Join("\t", paramModel_avg.U.DenseMatrixValue[IdxCol].VectorValue);
                                FileSaveModel_U.WriteLine(UCol);
                            }
                        }
                    }
                    else
                    {
                        using (StreamWriter FileSaveModel_Phi = new StreamWriter(ModelName_Phi, false))
                        {
                            for (int IdxCol = 0; IdxCol < paramModel.Phi.nCols; IdxCol++)
                            {
                                PhiCol = String.Join("\t", paramModel.Phi.DenseMatrixValue[IdxCol].VectorValue);
                                FileSaveModel_Phi.WriteLine(PhiCol);
                            }
                        }
                        using (StreamWriter FileSaveModel_U = new StreamWriter(ModelName_U, false))
                        {
                            for (int IdxCol = 0; IdxCol < paramModel.U.nCols; IdxCol++)
                            {
                                UCol = String.Join("\t", paramModel.U.DenseMatrixValue[IdxCol].VectorValue);
                                FileSaveModel_U.WriteLine(UCol);
                            }
                        }
                    }
                    // Save the final learning curves
                    using (StreamWriter FileSavePerf = new StreamWriter(ResultFile + ".perf", false))
                    {
                        FileSavePerf.Write("Epoch:\t");
                        FileSavePerf.WriteLine(String.Join("\t", TestError_epoch.VectorValue));
                        FileSavePerf.Write("TrainTime:\t");
                        FileSavePerf.WriteLine(String.Join("\t", TestError_time.VectorValue));
                        if (paramTrain.flag_HasValidSet)
                        {
                            FileSavePerf.Write("Validation:\t");
                            FileSavePerf.WriteLine(String.Join("\t", ValidError_pool.VectorValue));
                        }
                        FileSavePerf.Write("Test:\t");
                        FileSavePerf.WriteLine(String.Join("\t", TestError_pool.VectorValue));
                        FileSavePerf.Write("TrainError:\t");
                        FileSavePerf.WriteLine(String.Join("\t", TrainError_pool.VectorValue));
                        FileSavePerf.Write("TrainLoss:\t");
                        FileSavePerf.WriteLine(String.Join("\t", TrainLoss_pool.VectorValue));
                    }
                }

                // -- Dump feature --
                if (paramTrain.flag_DumpFeature && (epoch + 1) % nEpochPerDump == 0)
                {
                    if (paramTrain.flag_RunningAvg && epoch >= (int)Math.Ceiling(((float)nEpoch) / 2.0f))
                    {
                        DumpingFeature_BP_LDA(TrainData, paramModel_avg, paramTrain.BatchSize_Test, ResultFile + ".train.fea", "Train");
                        DumpingFeature_BP_LDA(TestData, paramModel_avg, paramTrain.BatchSize_Test, ResultFile + ".test.fea", "Test");
                        if (paramTrain.flag_HasValidSet)
                        {
                            DumpingFeature_BP_LDA(ValidData, paramModel_avg, paramTrain.BatchSize_Test, ResultFile + ".valid.fea", "Validation");
                        }
                    }
                    {
                        DumpingFeature_BP_LDA(TrainData, paramModel, paramTrain.BatchSize_Test, ResultFile + ".train.fea", "Train");
                        DumpingFeature_BP_LDA(TestData, paramModel, paramTrain.BatchSize_Test, ResultFile + ".test.fea", "Test");
                        if (paramTrain.flag_HasValidSet)
                        {
                            DumpingFeature_BP_LDA(ValidData, paramModel, paramTrain.BatchSize_Test, ResultFile + ".valid.fea", "Validation");
                        }
                    }
                }


            }
            

        }

        /*
         * Evaluating the model using external process, e.g., AUC, top@K accuracy, etc
         */
        public static void ExternalEvaluation(string ExternalEvalToolPath, string ResultFile, string TestLabelFile, int epoch, string EvalDataName)
        {
            string Extension;
            switch(EvalDataName)
            {
                case "Test Set":
                    Extension = ".testscore";
                    break;
                case "Validation Set":
                    Extension = ".validscore";
                    break;
                default:
                    throw new Exception("Unknown EvalDataName.");                    
            }
            using (Process callProcess = new Process()
            {
                StartInfo = new ProcessStartInfo()
                {
                    FileName = ExternalEvalToolPath,
                    Arguments = string.Format("\"{0}\" \"{1}\"", ResultFile + Extension, TestLabelFile),
                    CreateNoWindow = true,
                    UseShellExecute = false,
                }
            })
            {
                callProcess.Start();
                callProcess.WaitForExit();
            }

            StreamReader ExternalPerfFile = new StreamReader(ResultFile + Extension + ".extperf");
            using (StreamWriter ExternalPerfLogFile = File.AppendText(ResultFile + ".extperf"))
            {
                ExternalPerfLogFile.WriteLine("Epoch = {0}"+"\t[" + EvalDataName + "]", epoch + 1);
            }
            string StrLine;
            while ((StrLine = ExternalPerfFile.ReadLine()) != null)
            {
                Console.WriteLine("----------------------------------------------------");
                Console.Write(" " + "[" + EvalDataName + "] ");
                Console.WriteLine(StrLine);
                Console.WriteLine("----------------------------------------------------");
                using (StreamWriter ExternalPerfLogFile = File.AppendText(ResultFile + ".extperf"))
                {
                    ExternalPerfLogFile.WriteLine(StrLine);
                }
            }
            ExternalPerfFile.Close();
        }
        

        
        /*
         * Compute the training loss of the supervised learning at the current batch
         */
        public static float ComputeSupervisedLoss(SparseMatrix Dt, DenseMatrix y, string OutputType)
        {
            if (Dt.nCols != y.nCols || Dt.nRows != y.nRows)
            {
                throw new Exception("The numbers of samples from label and prediction do not match.");
            }
            DenseMatrix TmpDenseMat = new DenseMatrix(y);
            SparseMatrix TmpSparseMat = new SparseMatrix(Dt);
            DenseRowVector TmpDenseRowVec = new DenseRowVector(Dt.nCols);
            float TrainingLoss = 0.0f;
            switch (OutputType)
            {
                case "softmaxCE":
                    MatrixOperation.ScalarAddMatrix(TmpDenseMat, y, 1e-20f);
                    MatrixOperation.Log(TmpDenseMat);
                    MatrixOperation.ElementwiseMatrixMultiplyMatrix(TmpSparseMat, Dt, TmpDenseMat);
                    MatrixOperation.VerticalSumMatrix(TmpDenseRowVec, TmpSparseMat);
                    TrainingLoss = TmpDenseRowVec.Sum() * (-1.0f);
                    break;
                case "linearQuad":
                    MatrixOperation.MatrixSubtractMatrix(TmpDenseMat, Dt);
                    MatrixOperation.ElementwiseSquare(TmpDenseMat);
                    MatrixOperation.VerticalSumMatrix(TmpDenseRowVec, TmpDenseMat);
                    TrainingLoss = TmpDenseRowVec.Sum();
                    break;
                case "linearCE":
                    MatrixOperation.ScalarAddMatrix(TmpDenseMat, y, 1e-20f);
                    MatrixOperation.Log(TmpDenseMat);
                    MatrixOperation.ElementwiseMatrixMultiplyMatrix(TmpSparseMat, Dt, TmpDenseMat);
                    MatrixOperation.VerticalSumMatrix(TmpDenseRowVec, TmpSparseMat);
                    TrainingLoss = TmpDenseRowVec.Sum() * (-1.0f);
                    break;
                default:
                    throw new Exception("Unknown OutputType.");
            }

            return TrainingLoss;
        }
        public static float ComputeSupervisedLoss(SparseMatrix Dt, SparseMatrix y, string OutputType)
        {
            if (Dt.nCols != y.nCols || Dt.nRows != y.nRows)
            {
                throw new Exception("The numbers of samples from label and prediction do not match.");
            }
            SparseMatrix SparseMat = new SparseMatrix(y);
            SparseMatrix TmpSparseMat = new SparseMatrix(Dt);
            DenseRowVector TmpDenseRowVec = new DenseRowVector(Dt.nCols);
            float TrainingLoss = 0.0f;
            switch (OutputType)
            {
                case "softmaxCE":
                    MatrixOperation.ScalarAddMatrix(SparseMat, y, 1e-20f);
                    MatrixOperation.Log(SparseMat);
                    MatrixOperation.ElementwiseMatrixMultiplyMatrix(TmpSparseMat, Dt, SparseMat);
                    MatrixOperation.VerticalSumMatrix(TmpDenseRowVec, TmpSparseMat);
                    TrainingLoss = TmpDenseRowVec.Sum() * (-1.0f);
                    break;
                case "linearQuad":
                    MatrixOperation.MatrixSubtractMatrix(SparseMat, Dt);
                    MatrixOperation.ElementwiseSquare(SparseMat);
                    MatrixOperation.VerticalSumMatrix(TmpDenseRowVec, SparseMat);
                    TrainingLoss = TmpDenseRowVec.Sum();
                    break;
                case "linearCE":
                    MatrixOperation.ScalarAddMatrix(SparseMat, y, 1e-20f);
                    MatrixOperation.Log(SparseMat);
                    MatrixOperation.ElementwiseMatrixMultiplyMatrix(TmpSparseMat, Dt, SparseMat);
                    MatrixOperation.VerticalSumMatrix(TmpDenseRowVec, TmpSparseMat);
                    TrainingLoss = TmpDenseRowVec.Sum() * (-1.0f);
                    break;
                default:
                    throw new Exception("Unknown OutputType.");
            }

            return TrainingLoss;
        }


        /*
         * Compute the training error at the given batch
         */
        public static int ComputeNumberOfErrors(SparseMatrix Dt, DenseMatrix y)
        {
            if (Dt.nCols != y.nCols)
            {
                throw new Exception("The numbers of samples from label and prediction do not match.");
            }
            int nTotError = 0;
            int[] PredictedClass = y.IndexOfVerticalMax();
            for (int IdxCol = 0; IdxCol < Dt.nCols; IdxCol++)
            {
                if (Dt.SparseColumnVectors[IdxCol].Key[0] != PredictedClass[IdxCol])
                {
                    nTotError++;
                }
            }
            return nTotError;
        }
        public static int ComputeNumberOfErrors(SparseMatrix Dt, SparseMatrix y)
        {
            if (Dt.nCols != y.nCols)
            {
                throw new Exception("The numbers of samples from label and prediction do not match.");
            }
            int nTotError = 0;
            int[] PredictedClass = y.IndexOfVerticalMax();
            for (int IdxCol = 0; IdxCol < Dt.nCols; IdxCol++)
            {
                if (Dt.SparseColumnVectors[IdxCol].Key[0] != PredictedClass[IdxCol])
                {
                    nTotError++;
                }
            }
            return nTotError;
        }


        /*
         *  Initialize the Feedforward-LDA network (model). For the moment, we only implement the tied parameter case.
         */
        public static void ModelInit_LDA_Feedforward(paramModel_t paramModel)
        {
            // Extract the parameters
            int nInput = paramModel.nInput;
            int nOutput = paramModel.nOutput;
            int nHid = paramModel.nHid;
            int nHidLayer = paramModel.nHidLayer;
            float alpha = paramModel.alpha;
            string OutputType = paramModel.OutputType;

            // Initialzie the model
            paramModel.b = new DenseColumnVector(nHid, alpha - 1);
            paramModel.Phi = new DenseMatrix(nInput, nHid);
            paramModel.Phi.FillRandomValues();
            MatrixOperation.ScalarAddMatrix(paramModel.Phi, 1.0f);
            MatrixOperation.bsxfunMatrixRightDivideVector(paramModel.Phi, MatrixOperation.VerticalSumMatrix(paramModel.Phi));
            paramModel.U = new DenseMatrix(nOutput, nHid);
            paramModel.U.FillRandomValues();
            MatrixOperation.ScalarMultiplyMatrix(paramModel.U, 0.01f);
        }

        /*
         * Forward activation of Latent Dirichlet Allocation model (Mirror descent approach)
         */
        public static void ForwardActivation_LDA(SparseMatrix Xt, DNNRun_t DNNRun, paramModel_t paramModel, bool flag_IsTraining)
        {
            // -------- Extract parameters --------
            int nHid = paramModel.nHid;
            int nHidLayer = paramModel.nHidLayer;
            float eta = paramModel.eta;
            float T_value = paramModel.T_value;
            string OutputType = paramModel.OutputType;
            float To = paramModel.To;
            int BatchSize = Xt.nCols;

            // -------- Hidden activations --------
            // ---- Reset the effective number of hidden layers (mainly for alpha<1 case) ----
            Array.Clear(DNNRun.nHidLayerEffective,0,DNNRun.nHidLayerEffective.Length);
            // ---- T is different over layers (adaptive step-size MDA) ----
            DenseRowVector T = new DenseRowVector(BatchSize, T_value);
            SparseMatrix Phitheta = new SparseMatrix(Xt);
            DenseRowVector loss_pre = new DenseRowVector(BatchSize);
            DenseRowVector loss_post = new DenseRowVector(BatchSize);
            DenseRowVector loss_gap = new DenseRowVector(BatchSize);
            DenseRowVector loss_gap_thresh = new DenseRowVector(BatchSize);
            DenseRowVector gradproj = new DenseRowVector(BatchSize);
            SparseMatrix TmpSparseMat = new SparseMatrix(Xt);
            DenseMatrix TmpDenseMat = new DenseMatrix(nHid, BatchSize);
            DenseMatrix LogTheta = new DenseMatrix(nHid, BatchSize);
            DenseRowVector TmpDenseRowVec = new DenseRowVector(BatchSize);
            DenseMatrix NegGrad = new DenseMatrix(nHid, BatchSize);
            DenseMatrix LLR = new DenseMatrix(nHid, BatchSize);            
            //for (int IdxSample = 0; IdxSample < BatchSize; IdxSample++)
            Parallel.For(0, BatchSize, new ParallelOptions { MaxDegreeOfParallelism = MatrixOperation.MaxMultiThreadDegree }, IdxSample =>
            {
                float KLDivergence = 0.0f;
                // The forward activation for each data sample
                for (int IdxLayer = 0; IdxLayer < nHidLayer; IdxLayer++)
                {
                    // Compute the loss before unfolding the current layer
                    if (IdxLayer == 0)
                    {
                        MatrixOperation.MatrixMultiplyVector(
                            Phitheta.SparseColumnVectors[IdxSample], 
                            paramModel.Phi, 
                            DNNRun.theta0.DenseMatrixValue[IdxSample]
                            );
                    }
                    else
                    {
                        MatrixOperation.MatrixMultiplyVector(
                            Phitheta.SparseColumnVectors[IdxSample], 
                            paramModel.Phi, 
                            DNNRun.theta_pool[IdxLayer - 1].DenseMatrixValue[IdxSample]
                            );
                    }
                    if (IdxLayer > 1)
                    {
                        loss_pre.VectorValue[IdxSample] = loss_post.VectorValue[IdxSample];
                    }
                    else
                    {
                        MatrixOperation.ScalarAddVector(TmpSparseMat.SparseColumnVectors[IdxSample], Phitheta.SparseColumnVectors[IdxSample], 1e-12f);
                        MatrixOperation.Log(TmpSparseMat.SparseColumnVectors[IdxSample]);
                        MatrixOperation.ElementwiseVectorMultiplyVector(TmpSparseMat.SparseColumnVectors[IdxSample], Xt.SparseColumnVectors[IdxSample]);
                        loss_pre.VectorValue[IdxSample] = (-1.0f)*TmpSparseMat.SparseColumnVectors[IdxSample].Sum();
                        if (IdxLayer == 0)
                        {
                            MatrixOperation.ScalarAddVector(TmpDenseMat.DenseMatrixValue[IdxSample], DNNRun.theta0.DenseMatrixValue[IdxSample], 1e-12f);
                        }
                        else
                        {
                            MatrixOperation.ScalarAddVector(TmpDenseMat.DenseMatrixValue[IdxSample], DNNRun.theta_pool[IdxLayer - 1].DenseMatrixValue[IdxSample], 1e-12f);
                        }
                        MatrixOperation.Log(TmpDenseMat.DenseMatrixValue[IdxSample]);
                        MatrixOperation.ElementwiseVectorMultiplyVector(TmpDenseMat.DenseMatrixValue[IdxSample], paramModel.b);
                        TmpDenseRowVec.VectorValue[IdxSample] = TmpDenseMat.DenseMatrixValue[IdxSample].Sum();
                        loss_pre.VectorValue[IdxSample] -= TmpDenseRowVec.VectorValue[IdxSample];
                    }
                    // Compute the hidden activation of the current layer
                    MatrixOperation.ScalarAddVector(TmpSparseMat.SparseColumnVectors[IdxSample], Phitheta.SparseColumnVectors[IdxSample], 1e-12f);
                    MatrixOperation.ElementwiseVectorDivideVector(
                        TmpSparseMat.SparseColumnVectors[IdxSample], 
                        Xt.SparseColumnVectors[IdxSample], 
                        TmpSparseMat.SparseColumnVectors[IdxSample]
                        );
                    MatrixOperation.MatrixTransposeMultiplyVector(
                        TmpDenseMat.DenseMatrixValue[IdxSample], 
                        paramModel.Phi, 
                        TmpSparseMat.SparseColumnVectors[IdxSample]
                        );
                    if (IdxLayer == 0)
                    {
                        MatrixOperation.ScalarAddVector(
                            NegGrad.DenseMatrixValue[IdxSample], 
                            DNNRun.theta0.DenseMatrixValue[IdxSample], 
                            1e-12f
                            );
                    }
                    else
                    {
                        MatrixOperation.ScalarAddVector(
                            NegGrad.DenseMatrixValue[IdxSample], 
                            DNNRun.theta_pool[IdxLayer - 1].DenseMatrixValue[IdxSample], 
                            1e-12f
                            );
                    }
                    MatrixOperation.ElementwiseVectorDivideVector(NegGrad.DenseMatrixValue[IdxSample], paramModel.b, NegGrad.DenseMatrixValue[IdxSample]);
                    MatrixOperation.VectorAddVector(NegGrad.DenseMatrixValue[IdxSample], TmpDenseMat.DenseMatrixValue[IdxSample]);
                    // Line search for the parameter T
                    if (paramModel.alpha >= 1)
                    {
                        T.VectorValue[IdxSample] *= (1.0f / eta);
                    } // only perform line search for alpha>=1 case (convex)
                    loss_post.VectorValue[IdxSample] = loss_pre.VectorValue[IdxSample];
                    if (IdxLayer == 0)
                    {
                        MatrixOperation.Log(LogTheta.DenseMatrixValue[IdxSample], DNNRun.theta0.DenseMatrixValue[IdxSample]);
                    }
                    else
                    {
                        MatrixOperation.Log(LogTheta.DenseMatrixValue[IdxSample], DNNRun.theta_pool[IdxLayer - 1].DenseMatrixValue[IdxSample]);
                    }
                    while (true)
                    {
                        MatrixOperation.ScalarMultiplyVector(DNNRun.theta_pool[IdxLayer].DenseMatrixValue[IdxSample],
                            NegGrad.DenseMatrixValue[IdxSample], T.VectorValue[IdxSample]);
                        MatrixOperation.VectorAddVector(DNNRun.theta_pool[IdxLayer].DenseMatrixValue[IdxSample],
                            LogTheta.DenseMatrixValue[IdxSample]);
                        MatrixOperation.ScalarAddVector(DNNRun.theta_pool[IdxLayer].DenseMatrixValue[IdxSample],
                            (-1.0f) * DNNRun.theta_pool[IdxLayer].DenseMatrixValue[IdxSample].MaxValue());
                        MatrixOperation.Exp(DNNRun.theta_pool[IdxLayer].DenseMatrixValue[IdxSample]);
                        MatrixOperation.ScalarMultiplyVector(DNNRun.theta_pool[IdxLayer].DenseMatrixValue[IdxSample],
                            (1.0f / DNNRun.theta_pool[IdxLayer].DenseMatrixValue[IdxSample].Sum()));
                        // Compute the loss after undfolding the current layer
                        MatrixOperation.MatrixMultiplyVector(Phitheta.SparseColumnVectors[IdxSample],
                            paramModel.Phi, DNNRun.theta_pool[IdxLayer].DenseMatrixValue[IdxSample]);
                        MatrixOperation.Log(Phitheta.SparseColumnVectors[IdxSample]);
                        loss_post.VectorValue[IdxSample]
                            = (-1.0f) * MatrixOperation.InnerProduct(Xt.SparseColumnVectors[IdxSample], Phitheta.SparseColumnVectors[IdxSample]);
                        MatrixOperation.ScalarAddVector(TmpDenseMat.DenseMatrixValue[IdxSample], DNNRun.theta_pool[IdxLayer].DenseMatrixValue[IdxSample], 1e-12f);
                        MatrixOperation.Log(TmpDenseMat.DenseMatrixValue[IdxSample]);
                        loss_post.VectorValue[IdxSample] -= MatrixOperation.InnerProduct(TmpDenseMat.DenseMatrixValue[IdxSample], paramModel.b);
                        if (IdxLayer == 0)
                        {
                            MatrixOperation.VectorSubtractVector(TmpDenseMat.DenseMatrixValue[IdxSample],
                                DNNRun.theta_pool[IdxLayer].DenseMatrixValue[IdxSample],
                                DNNRun.theta0.DenseMatrixValue[IdxSample]);
                        }
                        else
                        {
                            MatrixOperation.VectorSubtractVector(TmpDenseMat.DenseMatrixValue[IdxSample],
                                DNNRun.theta_pool[IdxLayer].DenseMatrixValue[IdxSample],
                                DNNRun.theta_pool[IdxLayer - 1].DenseMatrixValue[IdxSample]);
                        }
                        loss_gap.VectorValue[IdxSample] = loss_post.VectorValue[IdxSample] - loss_pre.VectorValue[IdxSample];
                        gradproj.VectorValue[IdxSample]
                            = (-1.0f) * MatrixOperation.InnerProduct(NegGrad.DenseMatrixValue[IdxSample],
                                TmpDenseMat.DenseMatrixValue[IdxSample]);
                        loss_gap_thresh.VectorValue[IdxSample] = gradproj.VectorValue[IdxSample]
                            + (0.5f / T.VectorValue[IdxSample]) * (float)Math.Pow((double)TmpDenseMat.DenseMatrixValue[IdxSample].L1Norm(), 2.0);
                        if (loss_gap.VectorValue[IdxSample] > loss_gap_thresh.VectorValue[IdxSample] + 1e-12 && paramModel.alpha>=1)
                        {
                            T.VectorValue[IdxSample] *= eta;
                        } // Only perform line search for alpha>=1 case (convex)
                        else
                        {
                            DNNRun.T_pool.DenseMatrixValuePerRow[IdxLayer].VectorValue[IdxSample] = T.VectorValue[IdxSample];
                            break;
                        }
                    }
                    // Count the effective number of hidden layers
                    ++DNNRun.nHidLayerEffective[IdxSample];
                    // stop MDA if termination condition holds
                    if (paramModel.flag_AdaptivenHidLayer)
                    {
                        if (IdxLayer == 0)
                        {
                            MatrixOperation.ElementwiseVectorDivideVector(
                                LLR.DenseMatrixValue[IdxSample],
                                DNNRun.theta_pool[IdxLayer].DenseMatrixValue[IdxSample],
                                DNNRun.theta0.DenseMatrixValue[IdxSample]
                                );
                            MatrixOperation.Log(LLR.DenseMatrixValue[IdxSample]);
                        }
                        else
                        {
                            MatrixOperation.ElementwiseVectorDivideVector(
                                LLR.DenseMatrixValue[IdxSample],
                                DNNRun.theta_pool[IdxLayer].DenseMatrixValue[IdxSample],
                                DNNRun.theta_pool[IdxLayer - 1].DenseMatrixValue[IdxSample]
                                );
                            MatrixOperation.Log(LLR.DenseMatrixValue[IdxSample]);
                            MatrixOperation.ResetVectorSparsePattern(
                                LLR.DenseMatrixValue[IdxSample], 
                                DNNRun.theta_pool[IdxLayer].DenseMatrixValue[IdxSample]
                                );
                        }
                        KLDivergence = MatrixOperation.InnerProduct(
                            LLR.DenseMatrixValue[IdxSample], 
                            DNNRun.theta_pool[IdxLayer].DenseMatrixValue[IdxSample]
                            );
                        if (KLDivergence < 1e-12f)
                        {
                            break;
                        }
                    }
                }
                // ---- Generate output ----
                switch (OutputType)
                {
                    case "softmaxCE":
                        MatrixOperation.MatrixMultiplyVector(
                            DNNRun.y.DenseMatrixValue[IdxSample],
                            paramModel.U,
                            DNNRun.theta_pool[DNNRun.nHidLayerEffective[IdxSample] - 1].DenseMatrixValue[IdxSample]
                            );
                        MatrixOperation.ScalarAddVector(DNNRun.y.DenseMatrixValue[IdxSample], To);
                        TmpDenseRowVec.VectorValue[IdxSample] = DNNRun.y.DenseMatrixValue[IdxSample].MaxValue();
                        MatrixOperation.ScalarAddVector(DNNRun.y.DenseMatrixValue[IdxSample], (-1.0f) * TmpDenseRowVec.VectorValue[IdxSample]);
                        MatrixOperation.Exp(DNNRun.y.DenseMatrixValue[IdxSample]);
                        TmpDenseRowVec.VectorValue[IdxSample] = DNNRun.y.DenseMatrixValue[IdxSample].Sum();
                        MatrixOperation.ScalarMultiplyVector(DNNRun.y.DenseMatrixValue[IdxSample], (1.0f) / TmpDenseRowVec.VectorValue[IdxSample]);
                        break;
                    case "unsupLDA":
                        // Will not compute the reconstructed input at forward activation to save time during training.
                        break;
                    case "linearQuad":
                        MatrixOperation.MatrixMultiplyVector(
                            DNNRun.y.DenseMatrixValue[IdxSample],
                            paramModel.U,
                            DNNRun.theta_pool[DNNRun.nHidLayerEffective[IdxSample] - 1].DenseMatrixValue[IdxSample]
                            );
                        break;
                    case "linearCE":
                        throw new Exception("linearCE not implemented.");
                    default:
                        throw new Exception("Unknown OutputType.");
                }
            });            
        }

        /*
         * Back propagation of the unfolded LDA model (Mirror descent approach)
         */
        // Implemented without atomic operation
        public static void BackPropagation_LDA(SparseMatrix Xt, SparseMatrix Dt, DNNRun_t DNNRun, paramModel_t paramModel, Grad_t Grad)
        {
            // -------- Extract parameters --------
            int nHid = paramModel.nHid;
            int nHidLayer = paramModel.nHidLayer;
            int nOutput = paramModel.nOutput;
            float To = paramModel.To;
            string OutputType = paramModel.OutputType;
            int BatchSize = Xt.nCols;
            int nInput = paramModel.nInput;



            // -------- Back propagation --------
            DenseMatrix grad_Q_po = new DenseMatrix(DNNRun.y);
            SparseMatrix TmpSparseMat = new SparseMatrix(Xt);
            SparseMatrix grad_Q_po_Sparse = new SparseMatrix(Xt);
            DenseMatrix xi = new DenseMatrix(nHid, BatchSize);
            DenseMatrix TmpDenseMat = new DenseMatrix(nHid, BatchSize);
            DenseMatrix ThetaRatio = new DenseMatrix(nHid, BatchSize);
            DenseRowVector TmpDenseRowVec = new DenseRowVector(BatchSize);            
            DenseMatrix tmp_theta_xi_b_T_OVER_theta_lm1_2 = new DenseMatrix(nHid, BatchSize);
            SparseMatrix tmp_Xt_OVER_Phitheta = new SparseMatrix(Xt);
            SparseMatrix tmp_Phi_theta_xi = new SparseMatrix(Xt);
            Grad.grad_Q_Phi.ClearValue();
            // ---- Offset of effective number of layers ----
            int[] OffsetEffNumLayer = new int[BatchSize];
            OffsetEffNumLayer[0] = 0;
            int NumTotalLayer = DNNRun.nHidLayerEffective[0];
            for (int IdxSample = 1; IdxSample < BatchSize; ++IdxSample)
            {
                OffsetEffNumLayer[IdxSample] = OffsetEffNumLayer[IdxSample - 1] + DNNRun.nHidLayerEffective[IdxSample-1];
                NumTotalLayer += DNNRun.nHidLayerEffective[IdxSample];
            }
            // ---- Temporary variables that stores the intermediate results for computing the gradients ----
            DenseMatrix tmp_theta_xi_pool = new DenseMatrix(nHid, NumTotalLayer, 0.0f);
            DenseMatrix tmp_theta_xi = new DenseMatrix(nHid, BatchSize, 0.0f);
            DenseMatrix theta_l_minus_one = new DenseMatrix(nHid, NumTotalLayer, 0.0f);
            SparseMatrix tmp_Xt_OVER_Phitheta_pool = new SparseMatrix(nInput, NumTotalLayer);
            SparseMatrix TmpSparseMat_pool = new SparseMatrix(nInput, NumTotalLayer);
            int NumTotalNz = 0;
            for (int IdxSample = 0; IdxSample < BatchSize; ++IdxSample)
            {
                int Layer_begin = OffsetEffNumLayer[IdxSample];
                int Layer_end = Layer_begin + DNNRun.nHidLayerEffective[IdxSample];
                SparseColumnVector[] tmp1 = tmp_Xt_OVER_Phitheta_pool.SparseColumnVectors;
                SparseColumnVector[] tmp2 = TmpSparseMat_pool.SparseColumnVectors;
                SparseColumnVector xt = Xt.SparseColumnVectors[IdxSample];
                NumTotalNz += xt.nNonzero;
                for (int IdxLayer = Layer_begin; IdxLayer < Layer_end; ++IdxLayer)
                {
                    tmp1[IdxLayer] = new SparseColumnVector(xt);
                    tmp2[IdxLayer] = new SparseColumnVector(xt);
                }
            }
            int[] SparsePatternGradPhi = Xt.GetHorizontalUnionSparsePattern();
            SparseMatrix TmpGrad = new SparseMatrix(nInput, nHid, true);
            TmpGrad.SetSparsePatternForAllColumn(SparsePatternGradPhi);
            // ---- Compute grad Q wrt po if possible ----
            switch (OutputType)
            {
                case "softmaxCE":
                    MatrixOperation.MatrixSubtractMatrix(grad_Q_po, Dt);
                    MatrixOperation.ScalarMultiplyMatrix(grad_Q_po, To);
                    Grad.grad_Q_U.ClearValue();
                    break;
                case "linearQuad":
                    MatrixOperation.MatrixSubtractMatrix(grad_Q_po, Dt);
                    MatrixOperation.ScalarMultiplyMatrix(grad_Q_po, 2.0f);
                    Grad.grad_Q_U.ClearValue();
                    break;
                case "unsupLDA":
                    Grad.grad_Q_TopPhi.SetAllValuesToZero();
                    break;
                case "linearCE":
                    throw new Exception("linearCE is not implemented.");
                default:
                    throw new Exception("Unknown OutputType");
            }
            Parallel.For(0, BatchSize, new ParallelOptions { MaxDegreeOfParallelism = MatrixOperation.MaxMultiThreadDegree }, IdxSample =>
            {
                // ***************************************************************************

                // -------- Back propagation: top layer --------                    
                switch (OutputType)
                {
                    case "softmaxCE":
                        // ---- grad Q wrt pL (x_L) ----
                        MatrixOperation.MatrixTransposeMultiplyVector(
                            xi.DenseMatrixValue[IdxSample],
                            paramModel.U,
                            grad_Q_po.DenseMatrixValue[IdxSample]
                            );
                        MatrixOperation.ElementwiseVectorMultiplyVector(
                            TmpDenseMat.DenseMatrixValue[IdxSample],
                            DNNRun.theta_pool[DNNRun.nHidLayerEffective[IdxSample] - 1].DenseMatrixValue[IdxSample],
                            xi.DenseMatrixValue[IdxSample]
                            );
                        TmpDenseRowVec.VectorValue[IdxSample] = TmpDenseMat.DenseMatrixValue[IdxSample].Sum();
                        MatrixOperation.ScalarAddVector(
                            xi.DenseMatrixValue[IdxSample],
                            xi.DenseMatrixValue[IdxSample],
                            TmpDenseRowVec.VectorValue[IdxSample] * (-1.0f)
                            );
                        break;
                    case "linearQuad":
                        // ---- grad Q wrt pL (x_L) ----
                        MatrixOperation.MatrixTransposeMultiplyVector(
                            xi.DenseMatrixValue[IdxSample],
                            paramModel.U,
                            grad_Q_po.DenseMatrixValue[IdxSample]
                            );
                        MatrixOperation.ElementwiseVectorMultiplyVector(
                            TmpDenseMat.DenseMatrixValue[IdxSample],
                            DNNRun.theta_pool[DNNRun.nHidLayerEffective[IdxSample] - 1].DenseMatrixValue[IdxSample],
                            xi.DenseMatrixValue[IdxSample]
                            );
                        TmpDenseRowVec.VectorValue[IdxSample] = TmpDenseMat.DenseMatrixValue[IdxSample].Sum();
                        MatrixOperation.ScalarAddVector(
                            xi.DenseMatrixValue[IdxSample],
                            xi.DenseMatrixValue[IdxSample],
                            (-1.0f) * TmpDenseRowVec.VectorValue[IdxSample]
                            );
                        break;
                    case "unsupLDA":
                        // ---- grad Q wrt po ----
                        MatrixOperation.MatrixMultiplyVector(
                            grad_Q_po_Sparse.SparseColumnVectors[IdxSample],
                            paramModel.Phi,
                            DNNRun.theta_pool[DNNRun.nHidLayerEffective[IdxSample] - 1].DenseMatrixValue[IdxSample]
                            );
                        MatrixOperation.ElementwiseVectorDivideVector(
                            grad_Q_po_Sparse.SparseColumnVectors[IdxSample],
                            Xt.SparseColumnVectors[IdxSample],
                            grad_Q_po_Sparse.SparseColumnVectors[IdxSample]
                            );
                        // ---- grad Q wrt pL (x_L) ----
                        MatrixOperation.MatrixTransposeMultiplyVector(
                            xi.DenseMatrixValue[IdxSample],
                            paramModel.Phi,
                            grad_Q_po_Sparse.SparseColumnVectors[IdxSample]
                            );
                        MatrixOperation.ScalarMultiplyVector(
                            xi.DenseMatrixValue[IdxSample],
                            -1.0f
                            );
                        MatrixOperation.ElementwiseVectorMultiplyVector(
                            TmpDenseMat.DenseMatrixValue[IdxSample],
                            xi.DenseMatrixValue[IdxSample],
                            DNNRun.theta_pool[DNNRun.nHidLayerEffective[IdxSample] - 1].DenseMatrixValue[IdxSample]
                            );
                        TmpDenseRowVec.VectorValue[IdxSample] = TmpDenseMat.DenseMatrixValue[IdxSample].Sum();
                        MatrixOperation.ScalarAddVector(
                            xi.DenseMatrixValue[IdxSample],
                            xi.DenseMatrixValue[IdxSample],
                            (-1.0f) * TmpDenseRowVec.VectorValue[IdxSample]
                            );
                        break;
                    case "linearCE":
                        throw new Exception("linearCE is not implemented.");
                    //break;
                    default:
                        throw new Exception("Unknown OutputType");
                }


                // ***************************************************************************

                // -------- Back propagation: hidden layers --------
                for (int IdxLayer = DNNRun.nHidLayerEffective[IdxSample] - 1; IdxLayer >= 0; IdxLayer--)
                {
                    // ---- Compute the position in the temporary variable for the current layer at the current sample ----
                    int IdxTmpVar = OffsetEffNumLayer[IdxSample] + IdxLayer;
                    // ---- grad wrt b ---
                    // Not implemented at the moment. (Can be used to update the Dirichlet parameter automatically.)
                    // ---- Compute the intermediate variables ----
                    MatrixOperation.ElementwiseVectorMultiplyVector(
                        tmp_theta_xi_pool.DenseMatrixValue[IdxTmpVar],
                        DNNRun.theta_pool[IdxLayer].DenseMatrixValue[IdxSample],
                        xi.DenseMatrixValue[IdxSample]
                        );
                    if (IdxLayer == 0)
                    {
                        MatrixOperation.ElementwiseVectorDivideVector(
                            tmp_theta_xi_b_T_OVER_theta_lm1_2.DenseMatrixValue[IdxSample],
                            tmp_theta_xi_pool.DenseMatrixValue[IdxTmpVar],
                            DNNRun.theta0.DenseMatrixValue[IdxSample]
                            );
                    }
                    else
                    {
                        MatrixOperation.ElementwiseVectorDivideVector(
                            tmp_theta_xi_b_T_OVER_theta_lm1_2.DenseMatrixValue[IdxSample],
                            tmp_theta_xi_pool.DenseMatrixValue[IdxTmpVar],
                            DNNRun.theta_pool[IdxLayer - 1].DenseMatrixValue[IdxSample]
                            );
                    }
                    if (IdxLayer == 0)
                    {
                        MatrixOperation.ElementwiseVectorDivideVector(
                            tmp_theta_xi_b_T_OVER_theta_lm1_2.DenseMatrixValue[IdxSample],
                            tmp_theta_xi_b_T_OVER_theta_lm1_2.DenseMatrixValue[IdxSample],
                            DNNRun.theta0.DenseMatrixValue[IdxSample]
                            );
                    }
                    else
                    {
                        MatrixOperation.ElementwiseVectorDivideVector(
                            tmp_theta_xi_b_T_OVER_theta_lm1_2.DenseMatrixValue[IdxSample],
                            tmp_theta_xi_b_T_OVER_theta_lm1_2.DenseMatrixValue[IdxSample],
                            DNNRun.theta_pool[IdxLayer - 1].DenseMatrixValue[IdxSample]
                            );
                    }
                    MatrixOperation.ElementwiseVectorMultiplyVector(
                        tmp_theta_xi_b_T_OVER_theta_lm1_2.DenseMatrixValue[IdxSample],
                        paramModel.b
                        );
                    MatrixOperation.ScalarMultiplyVector(
                        tmp_theta_xi_b_T_OVER_theta_lm1_2.DenseMatrixValue[IdxSample],
                        DNNRun.T_pool.DenseMatrixValuePerRow[IdxLayer].VectorValue[IdxSample]
                        );
                    // Reset the elements to zero if theta_{l-1} is zero at these positions (mainly for alpha<1 case)
                    if (IdxLayer > 0)
                    {
                        MatrixOperation.ResetVectorSparsePattern(
                            tmp_theta_xi_b_T_OVER_theta_lm1_2.DenseMatrixValue[IdxSample],
                            DNNRun.theta_pool[IdxLayer - 1].DenseMatrixValue[IdxSample]
                            );
                    }
                    // Continue to intermediate variable computation
                    if (IdxLayer == 0) // TmpSparseMat is Phitheta_lm1
                    {
                        MatrixOperation.MatrixMultiplyVector(
                            TmpSparseMat.SparseColumnVectors[IdxSample],
                            paramModel.Phi,
                            DNNRun.theta0.DenseMatrixValue[IdxSample]
                            );
                    }
                    else
                    {
                        MatrixOperation.MatrixMultiplyVector(
                            TmpSparseMat.SparseColumnVectors[IdxSample],
                            paramModel.Phi,
                            DNNRun.theta_pool[IdxLayer - 1].DenseMatrixValue[IdxSample]
                            );
                    }
                    MatrixOperation.ElementwiseVectorDivideVector(
                        tmp_Xt_OVER_Phitheta_pool.SparseColumnVectors[IdxTmpVar],
                        Xt.SparseColumnVectors[IdxSample],
                        TmpSparseMat.SparseColumnVectors[IdxSample]
                        );
                    MatrixOperation.ElementwiseVectorDivideVector(
                        TmpSparseMat.SparseColumnVectors[IdxSample],
                        tmp_Xt_OVER_Phitheta_pool.SparseColumnVectors[IdxTmpVar],
                        TmpSparseMat.SparseColumnVectors[IdxSample]
                        ); // TmpSparseMat is tmp_Xt_OVER_Phitheta2
                    MatrixOperation.MatrixMultiplyVector(
                        tmp_Phi_theta_xi.SparseColumnVectors[IdxSample],
                        paramModel.Phi,
                        tmp_theta_xi_pool.DenseMatrixValue[IdxTmpVar]
                        );
                    MatrixOperation.ElementwiseVectorMultiplyVector(
                        TmpSparseMat.SparseColumnVectors[IdxSample],
                        tmp_Phi_theta_xi.SparseColumnVectors[IdxSample]
                        ); // TmpSparseMat is ( tmp_Phi_theta_xi.*tmp_Xt_OVER_Phitheta2 )
                    MatrixOperation.MatrixTransposeMultiplyVector(
                        TmpDenseMat.DenseMatrixValue[IdxSample],
                        paramModel.Phi,
                        TmpSparseMat.SparseColumnVectors[IdxSample]
                        );
                    MatrixOperation.ScalarMultiplyVector(
                        TmpDenseMat.DenseMatrixValue[IdxSample],
                        DNNRun.T_pool.DenseMatrixValuePerRow[IdxLayer].VectorValue[IdxSample]
                        ); // TmpDenseMat is tmp_Tl_Phit_xtPhiTheta2_Phi_theta_xi
                    // ---- Compute the gradient wrt Phi ----     
                    MatrixOperation.ScalarMultiplyVector(
                        tmp_Xt_OVER_Phitheta_pool.SparseColumnVectors[IdxTmpVar],
                        DNNRun.T_pool.DenseMatrixValuePerRow[IdxLayer].VectorValue[IdxSample]
                        );
                    MatrixOperation.ScalarMultiplyVector(
                        TmpSparseMat_pool.SparseColumnVectors[IdxTmpVar],
                        TmpSparseMat.SparseColumnVectors[IdxSample],
                        DNNRun.T_pool.DenseMatrixValuePerRow[IdxLayer].VectorValue[IdxSample]*(-1.0f)
                        );                      
                    if (IdxLayer == 0)
                    {
                        theta_l_minus_one.DenseMatrixValue[IdxTmpVar] = DNNRun.theta0.DenseMatrixValue[IdxSample];
                    }
                    else
                    {
                        theta_l_minus_one.DenseMatrixValue[IdxTmpVar] = DNNRun.theta_pool[IdxLayer - 1].DenseMatrixValue[IdxSample];
                    }                    
                    // ---- Compute xi_{l-1} via back propagation ----
                    if (IdxLayer > 0)
                    {
                        // Reset the elements to zero if theta_{l-1} is zero at these positions (mainly for alpha<1 case)
                        MatrixOperation.ElementwiseVectorDivideVector(
                            ThetaRatio.DenseMatrixValue[IdxSample],
                            DNNRun.theta_pool[IdxLayer].DenseMatrixValue[IdxSample],
                            DNNRun.theta_pool[IdxLayer - 1].DenseMatrixValue[IdxSample]
                            );
                        MatrixOperation.ResetVectorSparsePattern(
                            ThetaRatio.DenseMatrixValue[IdxSample],
                            DNNRun.theta_pool[IdxLayer - 1].DenseMatrixValue[IdxSample]
                            );
                        MatrixOperation.ElementwiseVectorMultiplyVector(
                            xi.DenseMatrixValue[IdxSample],
                            xi.DenseMatrixValue[IdxSample],
                            ThetaRatio.DenseMatrixValue[IdxSample]
                            );
                        // Compute xi_{l-1} now
                        MatrixOperation.VectorSubtractVector(
                            TmpDenseMat.DenseMatrixValue[IdxSample],
                            xi.DenseMatrixValue[IdxSample],
                            TmpDenseMat.DenseMatrixValue[IdxSample]
                            );
                        MatrixOperation.VectorSubtractVector(
                            TmpDenseMat.DenseMatrixValue[IdxSample],
                            TmpDenseMat.DenseMatrixValue[IdxSample],
                            tmp_theta_xi_b_T_OVER_theta_lm1_2.DenseMatrixValue[IdxSample]
                            );
                        MatrixOperation.ElementwiseVectorMultiplyVector(
                            tmp_theta_xi.DenseMatrixValue[IdxSample],
                            DNNRun.theta_pool[IdxLayer - 1].DenseMatrixValue[IdxSample],
                            TmpDenseMat.DenseMatrixValue[IdxSample]
                            ); // tmp_theta_xi is tmp1 in matlab code
                        TmpDenseRowVec.VectorValue[IdxSample] = tmp_theta_xi.DenseMatrixValue[IdxSample].Sum();
                        MatrixOperation.ScalarAddVector(
                            xi.DenseMatrixValue[IdxSample],
                            TmpDenseMat.DenseMatrixValue[IdxSample],
                            TmpDenseRowVec.VectorValue[IdxSample] * (-1.0f)
                            );
                    }

                }
            });


            // -------- Compute the gradients --------
            // ---- Gradient with respect to U ----
            DenseMatrix Theta_Top = new DenseMatrix(nHid, BatchSize);
            for (int IdxSample = 0; IdxSample < BatchSize; ++IdxSample )
            {
                Theta_Top.DenseMatrixValue[IdxSample] = DNNRun.theta_pool[DNNRun.nHidLayerEffective[IdxSample] - 1].DenseMatrixValue[IdxSample];
            }
            switch (OutputType)
            {
                case "softmaxCE":
                    // ---- grad Q wrt U ----
                    MatrixOperation.MatrixMultiplyMatrixTranspose(Grad.grad_Q_U, grad_Q_po, Theta_Top);
                    MatrixOperation.ScalarMultiplyMatrix(Grad.grad_Q_U, (1.0f / (float)BatchSize));
                    break;
                case "linearQuad":
                    // ---- grad Q wrt U ----
                    MatrixOperation.MatrixMultiplyMatrixTranspose(Grad.grad_Q_U, grad_Q_po, Theta_Top);
                    MatrixOperation.ScalarMultiplyMatrix(Grad.grad_Q_U, (1.0f / (float)BatchSize));
                    break;
                case "unsupLDA":
                    // ---- grad Q wrt Phi on top ----
                    MatrixOperation.MatrixMultiplyMatrixTranspose(Grad.grad_Q_TopPhi, grad_Q_po_Sparse, Theta_Top, false);
                    MatrixOperation.ScalarMultiplyMatrix(Grad.grad_Q_TopPhi, Grad.grad_Q_TopPhi, (-1.0f / (float)BatchSize));
                    break;
                case "linearCE":
                    throw new Exception("linearCE is not implemented.");
                //break;
                default:
                    throw new Exception("Unknown OutputType");
            }
            // ---- Gradient with respect to Phi ----
            TmpGrad.SetAllValuesToZero();
            MatrixOperation.MatrixMultiplyMatrixTranspose(TmpGrad, tmp_Xt_OVER_Phitheta_pool, tmp_theta_xi_pool, true);
            MatrixOperation.MatrixMultiplyMatrixTranspose(TmpGrad, TmpSparseMat_pool, theta_l_minus_one, true);
            MatrixOperation.ScalarMultiplyMatrix(TmpGrad, TmpGrad, (1.0f / (float)BatchSize));
            MatrixOperation.MatrixAddMatrix(Grad.grad_Q_Phi, TmpGrad);

        }

        
        public static float SMD_Update(DenseMatrix X, DenseMatrix Grad, DenseRowVector LearningRatePerCol, float eta)
        {
            if (X.nCols != Grad.nCols || X.nRows != Grad.nRows)
            {
                throw new Exception("Dimension mismatch.");
            }
            DenseRowVector nLearnLineSearchPerCol = new DenseRowVector(X.nCols, 0.0f);
            DenseMatrix Update = new DenseMatrix(Grad.nRows, Grad.nCols);
            DenseRowVector TmpRowVec = new DenseRowVector(LearningRatePerCol);
            MatrixOperation.ScalarMultiplyVector(TmpRowVec, -1.0f);
            MatrixOperation.bsxfunVectorMultiplyMatrix(Update, Grad, TmpRowVec);
            MatrixOperation.VerticalMaxMatrix(TmpRowVec, Update);
            MatrixOperation.bsxfunMatrixSubtractVector(Update, Update, TmpRowVec);
            MatrixOperation.Exp(Update);
            MatrixOperation.ElementwiseMatrixMultiplyMatrix(X, X, Update);
            MatrixOperation.VerticalSumMatrix(TmpRowVec, X);
            MatrixOperation.bsxfunMatrixRightDivideVector(X, TmpRowVec);

            return 0.0f;
        }
        


        /*
         * Compute inverse document frequency (IDF) of the data
         */
        public static DenseColumnVector ComputeInverseDocumentFrequency(SparseMatrix InputData)
        {
            Console.WriteLine("=================================================="); 
            DenseColumnVector IDF = new DenseColumnVector(InputData.nRows);
            int[] DocFreq = new int[InputData.nRows];
            int Cnt = 0;
            for (int IdxCol = 0; IdxCol < InputData.nCols; ++IdxCol)
            {
                int nNonzero = InputData.SparseColumnVectors[IdxCol].nNonzero;
                int[] ColKey = InputData.SparseColumnVectors[IdxCol].Key;
                for (int IdxRow = 0; IdxRow < nNonzero; ++IdxRow)
                {
                    ++DocFreq[ColKey[IdxRow]];
                }
                ++Cnt;
                if (Cnt % 10000 == 0)
                {
                    Console.Write("Generating document frequency: {0}/{1}\r", Cnt, InputData.nCols);
                }
            }
            Console.WriteLine("Generating document frequency: {0}/{1}", Cnt, InputData.nCols);
            Cnt = 0;
            for (int IdxRow = 0; IdxRow < InputData.nRows; ++IdxRow )
            {
                if (DocFreq[IdxRow] > 0)
                {
                    IDF.VectorValue[IdxRow] = 1.0f / ((float)DocFreq[IdxRow]);
                }
                else
                {
                    IDF.VectorValue[IdxRow] = 1.0f / ((float)InputData.nCols);
                }
                ++Cnt;
                if (Cnt % 10000 == 0)
                {
                    Console.Write("Generating inverse document frquency: {0}/{1}\r", Cnt, InputData.nRows);
                }
            }
            Console.WriteLine("Generating inverse document frquency: {0}/{1}", Cnt, InputData.nRows);

            Console.WriteLine("=================================================="); 
            return IDF;
        }

        /*
         * Compute Cross Entropy between the reconstructed input and the actual input. (Unsupervised learning case)
         */
        public static float ComputeCrossEntropy(SparseMatrix Xt, DenseMatrix Phi, DenseMatrix theta_top)
        {
            SparseMatrix TmpSparseMat = new SparseMatrix(Xt);
            DenseRowVector TmpDenseRowVec = new DenseRowVector(Xt.nCols);
            MatrixOperation.MatrixMultiplyMatrix(TmpSparseMat, Phi, theta_top);
            MatrixOperation.Log(TmpSparseMat);
            MatrixOperation.ElementwiseMatrixMultiplyMatrix(TmpSparseMat, Xt);
            MatrixOperation.VerticalSumMatrix(TmpDenseRowVec, TmpSparseMat);
            return (-1.0f) * TmpDenseRowVec.VectorValue.Sum();
        }
        public static float ComputeCrossEntropy(SparseMatrix Xt, DenseMatrix Phi, DenseMatrix[] theta_pool, int[] nHidLayerEffective)
        {
            SparseMatrix TmpSparseMat = new SparseMatrix(Xt);
            DenseRowVector TmpDenseRowVec = new DenseRowVector(Xt.nCols);
            Parallel.For(0, Xt.nCols, IdxSample =>
            {
                MatrixOperation.MatrixMultiplyVector(
                    TmpSparseMat.SparseColumnVectors[IdxSample], 
                    Phi, 
                    theta_pool[nHidLayerEffective[IdxSample] - 1].DenseMatrixValue[IdxSample]
                    );
            });
            MatrixOperation.Log(TmpSparseMat);
            MatrixOperation.ElementwiseMatrixMultiplyMatrix(TmpSparseMat, Xt);
            MatrixOperation.VerticalSumMatrix(TmpDenseRowVec, TmpSparseMat);
            return (-1.0f) * TmpDenseRowVec.VectorValue.Sum();
        }

        /*
         * Compute Regularized Cross Entropy between the reconstructed input and the actual input. (Loss funtion for the unsupervised learning case)
         */
        public static float ComputeRegularizedCrossEntropy(SparseMatrix Xt, DenseMatrix Phi, DenseMatrix theta_top, DenseColumnVector b)
        {
            SparseMatrix TmpSparseMat = new SparseMatrix(Xt);
            DenseRowVector TmpDenseRowVec = new DenseRowVector(Xt.nCols);
            MatrixOperation.MatrixMultiplyMatrix(TmpSparseMat, Phi, theta_top);
            MatrixOperation.Log(TmpSparseMat);
            MatrixOperation.ElementwiseMatrixMultiplyMatrix(TmpSparseMat, Xt);
            MatrixOperation.VerticalSumMatrix(TmpDenseRowVec, TmpSparseMat);
            float CE = (-1.0f) * TmpDenseRowVec.VectorValue.Sum();
            DenseMatrix TmpDenseMat = new DenseMatrix(theta_top.nRows, theta_top.nCols);
            MatrixOperation.Log(TmpDenseMat, theta_top);
            MatrixOperation.bsxfunVectorMultiplyMatrix(TmpDenseMat, b);
            MatrixOperation.VerticalSumMatrix(TmpDenseRowVec, TmpDenseMat);
            CE = CE - TmpDenseRowVec.VectorValue.Sum();
            return CE;
        }

        /*
         * Testing the cross entropy on the test data (Unsupervised learning)
         */
        public static float Testing_BP_LDA(SparseMatrix TestData, paramModel_t paramModel)
        {
            Console.WriteLine("----------------------------------------------------");
            Console.Write(" Testing: ");
            DNNRun_t DNNRun = new DNNRun_t(paramModel.nHid, TestData.nCols, paramModel.nHidLayer, paramModel.nOutput);
            ForwardActivation_LDA(TestData, DNNRun, paramModel, false);
            float NegLogLoss = ComputeCrossEntropy(TestData, paramModel.Phi, DNNRun.theta_pool, DNNRun.nHidLayerEffective);
            NegLogLoss /= (float)TestData.nCols;
            Console.WriteLine(" -LogLoss = {0}", NegLogLoss);
            Console.WriteLine("----------------------------------------------------");
            return NegLogLoss;
        }
        public static float Testing_BP_LDA(SparseMatrix TestData, paramModel_t paramModel, int BatchSize_normal)
        {
            Console.WriteLine("----------------------------------------------------");
            int nTest = TestData.nCols;
            int nBatch = (int)Math.Ceiling(((float)nTest) / ((float)BatchSize_normal));
            float NegLogLoss = 0.0f;
            DNNRun_t DNNRun_NormalBatch = new DNNRun_t(paramModel.nHid, BatchSize_normal, paramModel.nHidLayer, paramModel.nOutput);
            DNNRun_t DNNRun_EndBatch = new DNNRun_t(paramModel.nHid, nTest - (nBatch - 1) * BatchSize_normal, paramModel.nHidLayer, paramModel.nOutput);
            DNNRun_t DNNRun = null;
            int[] IdxSample_Tot = new int[nTest];
            for (int Idx = 0; Idx < nTest; Idx++)
            {
                IdxSample_Tot[Idx] = Idx;
            }
            for (int IdxBatch = 0; IdxBatch < nBatch; IdxBatch++)
            {
                // Extract the batch
                int BatchSize = 0;
                if (IdxBatch < nBatch - 1)
                {
                    BatchSize = BatchSize_normal;
                    DNNRun = DNNRun_NormalBatch;
                }
                else
                {
                    BatchSize = nTest - IdxBatch * BatchSize_normal;
                    DNNRun = DNNRun_EndBatch;
                }
                SparseMatrix Xt = new SparseMatrix(paramModel.nInput, BatchSize);
                int[] IdxSample = new int[BatchSize];
                Array.Copy(IdxSample_Tot, IdxBatch * BatchSize_normal, IdxSample, 0, BatchSize);
                TestData.GetColumns(Xt, IdxSample);

                // Forward activation
                LDA_Learn.ForwardActivation_LDA(Xt, DNNRun, paramModel, false);

                // Compute loss
                NegLogLoss += ComputeCrossEntropy(Xt, paramModel.Phi, DNNRun.theta_pool, DNNRun.nHidLayerEffective);

                Console.Write(" Testing: Bat#{0}/{1}\r", (IdxBatch + 1), nBatch);
            }
            NegLogLoss = NegLogLoss / ((float)nTest);
            Console.WriteLine(" Testing: -LogLoss = {0}", NegLogLoss);
            Console.WriteLine("----------------------------------------------------");
            return NegLogLoss;
        }

        /*
         * Testing the model on the test data (Supervised learning). Compute the classification error (softmaxCE) or MSE (linearQuad).
         */
        public static float Testing_BP_sLDA(SparseMatrix TestData, SparseMatrix TestLabel, paramModel_t paramModel)
        {
            Console.WriteLine("----------------------------------------------------");
            Console.Write(" Testing: ");
            DNNRun_t DNNRun = new DNNRun_t(paramModel.nHid, TestData.nCols, paramModel.nHidLayer, paramModel.nOutput);
            ForwardActivation_LDA(TestData, DNNRun, paramModel, false);
            int nTotError;
            float TestError;
            switch (paramModel.OutputType)
            {
                case "softmaxCE":
                    nTotError = ComputeNumberOfErrors(TestLabel, DNNRun.y);
                    TestError = 100 * ((float)nTotError) / ((float)TestLabel.nCols);
                    Console.WriteLine(" TestError = {0}%", TestError);
                    break;
                case "linearQuad":
                    TestError = ComputeSupervisedLoss(TestLabel, DNNRun.y, paramModel.OutputType);
                    Console.WriteLine(" MSE = {0}", TestError);
                    break;
                default:
                    throw new Exception("Unknown OutputType.");
            }
            Console.WriteLine("----------------------------------------------------");
            return TestError;
        }
        public static float Testing_BP_sLDA(SparseMatrix TestData, SparseMatrix TestLabel, paramModel_t paramModel, int BatchSize_normal, string ScoreFileName, string EvalDataName)
        {
            Console.WriteLine("----------------------------------------------------");
            int nTest = TestData.nCols;
            int nBatch = (int)Math.Ceiling(((float)nTest) / ((float)BatchSize_normal));
            float TestError = 0.0f;
            DNNRun_t DNNRun_NormalBatch = new DNNRun_t(paramModel.nHid, BatchSize_normal, paramModel.nHidLayer, paramModel.nOutput);
            DNNRun_t DNNRun_EndBatch = new DNNRun_t(paramModel.nHid, nTest - (nBatch - 1) * BatchSize_normal, paramModel.nHidLayer, paramModel.nOutput);
            DNNRun_t DNNRun = null;
            int[] IdxSample_Tot = new int[nTest];
            (new FileInfo(ScoreFileName)).Directory.Create();
            StreamWriter ScoreFile = new StreamWriter(ScoreFileName);
            for (int Idx = 0; Idx < nTest; Idx++)
            {
                IdxSample_Tot[Idx] = Idx;
            }
            // ---- Test in a batch-wise manner over the test data ----
            for (int IdxBatch = 0; IdxBatch < nBatch; IdxBatch++)
            {
                // Extract the batch
                int BatchSize;
                if (IdxBatch < nBatch - 1)
                {
                    BatchSize = BatchSize_normal;
                    DNNRun = DNNRun_NormalBatch;
                }
                else
                {
                    BatchSize = nTest - IdxBatch * BatchSize_normal;
                    DNNRun = DNNRun_EndBatch;
                }
                SparseMatrix Xt = new SparseMatrix(paramModel.nInput, BatchSize);
                SparseMatrix Dt = new SparseMatrix(paramModel.nOutput, BatchSize);
                int[] IdxSample = new int[BatchSize];
                Array.Copy(IdxSample_Tot, IdxBatch * BatchSize_normal, IdxSample, 0, BatchSize);
                TestData.GetColumns(Xt, IdxSample);
                TestLabel.GetColumns(Dt, IdxSample);

                // Forward activation
                LDA_Learn.ForwardActivation_LDA(Xt, DNNRun, paramModel, false);

                // Compute loss
                switch (paramModel.OutputType)
                {
                    case "softmaxCE":
                        TestError += ComputeNumberOfErrors(Dt, DNNRun.y);
                        break;
                    case "linearQuad":
                        TestError += ComputeSupervisedLoss(Dt, DNNRun.y, paramModel.OutputType);
                        break;
                    case "linearCE":
                        TestError += ComputeNumberOfErrors(Dt, DNNRun.y);
                        break;
                    default:
                        throw new Exception("Unknown OutputType.");
                }

                // Write the score into file
                for (int IdxCol = 0; IdxCol < DNNRun.y.nCols; IdxCol++)
                {
                    ScoreFile.WriteLine(String.Join("\t", DNNRun.y.DenseMatrixValue[IdxCol].VectorValue));
                }

                Console.Write(" Testing on " + EvalDataName + ": Bat#{0}/{1}\r", (IdxBatch + 1), nBatch);
            }
            switch (paramModel.OutputType)
            {
                case "softmaxCE":
                    TestError = 100 * TestError / nTest;
                    Console.WriteLine(" [" + EvalDataName + "]" + " TestError = {0}%          ", TestError);
                    break;
                case "linearQuad":
                    TestError = TestError / nTest;
                    Console.WriteLine(" [" + EvalDataName + "]" + " MSE = {0}                 ", TestError);
                    break;
                case "linearCE":
                    TestError = 100 * TestError / nTest;
                    Console.WriteLine(" [" + EvalDataName + "]" + " TestError = {0}%          ", TestError);
                    break;
                default:
                    throw new Exception("Unknown OutputType.");
            }
            Console.WriteLine("----------------------------------------------------");
            ScoreFile.Close();
            return TestError;
        }

        /*
         * Dumping features
         */
        public static void DumpingFeature_BP_LDA(SparseMatrix InputData, paramModel_t paramModel, int BatchSize_normal, string FeatureFileName, string DataName)
        {
            Console.WriteLine("----------------------------------------------------");
            int nTest = InputData.nCols;
            int nBatch = (int)Math.Ceiling(((float)nTest) / ((float)BatchSize_normal));
            DNNRun_t DNNRun_NormalBatch = new DNNRun_t(paramModel.nHid, BatchSize_normal, paramModel.nHidLayer, paramModel.nOutput);
            DNNRun_t DNNRun_EndBatch = new DNNRun_t(paramModel.nHid, nTest - (nBatch - 1) * BatchSize_normal, paramModel.nHidLayer, paramModel.nOutput);
            DNNRun_t DNNRun = null;
            Console.Write(" Dumping feature ({0}): Bat#{1}/{2}\r", DataName, 0, nBatch);
            int[] IdxSample_Tot = new int[nTest];
            for (int Idx = 0; Idx < nTest; Idx++)
            {
                IdxSample_Tot[Idx] = Idx;
            }
            StreamWriter FeatureFile = new StreamWriter(FeatureFileName);
            for (int IdxBatch = 0; IdxBatch < nBatch; IdxBatch++)
            {
                // Extract the batch
                int BatchSize = 0;
                if (IdxBatch < nBatch - 1)
                {
                    BatchSize = BatchSize_normal;
                    DNNRun = DNNRun_NormalBatch;
                }
                else
                {
                    BatchSize = nTest - IdxBatch * BatchSize_normal;
                    DNNRun = DNNRun_EndBatch;
                }
                SparseMatrix Xt = new SparseMatrix(paramModel.nInput, BatchSize);
                int[] IdxSample = new int[BatchSize];
                Array.Copy(IdxSample_Tot, IdxBatch * BatchSize_normal, IdxSample, 0, BatchSize);
                InputData.GetColumns(Xt, IdxSample);

                // Forward activation
                LDA_Learn.ForwardActivation_LDA(Xt, DNNRun, paramModel, false);

                // Dump the feature into file
                for (int Idx = 0; Idx < BatchSize; Idx++)
                {
                    FeatureFile.WriteLine(String.Join("\t", DNNRun.theta_pool[DNNRun.nHidLayerEffective[Idx] - 1].DenseMatrixValue[Idx].VectorValue));
                }

                Console.Write(" Dumping feature ({0}): Bat#{1}/{2}\r", DataName, (IdxBatch + 1), nBatch);
            }
            Console.Write("\n");
            Console.WriteLine("----------------------------------------------------");
            FeatureFile.Close();
        }

        /*
         * Prediction: generating the output score
         */
        public static void PredictingOutput_BP_sLDA(SparseMatrix TestData, paramModel_t paramModel, int BatchSize_normal, string ScoreFileName)
        {
            Console.WriteLine("----------------------------------------------------");
            int nTest = TestData.nCols;
            int nBatch = (int)Math.Ceiling(((float)nTest) / ((float)BatchSize_normal));
            DNNRun_t DNNRun_NormalBatch = new DNNRun_t(paramModel.nHid, BatchSize_normal, paramModel.nHidLayer, paramModel.nOutput);
            DNNRun_t DNNRun_EndBatch = new DNNRun_t(paramModel.nHid, nTest - (nBatch - 1) * BatchSize_normal, paramModel.nHidLayer, paramModel.nOutput);
            DNNRun_t DNNRun = null;
            int[] IdxSample_Tot = new int[nTest];
            (new FileInfo(ScoreFileName)).Directory.Create();
            StreamWriter ScoreFile = new StreamWriter(ScoreFileName);
            for (int Idx = 0; Idx < nTest; Idx++)
            {
                IdxSample_Tot[Idx] = Idx;
            }
            // ---- Test in a batch-wise manner over the test data ----
            for (int IdxBatch = 0; IdxBatch < nBatch; IdxBatch++)
            {
                // Extract the batch
                int BatchSize;
                if (IdxBatch < nBatch - 1)
                {
                    BatchSize = BatchSize_normal;
                    DNNRun = DNNRun_NormalBatch;
                }
                else
                {
                    BatchSize = nTest - IdxBatch * BatchSize_normal;
                    DNNRun = DNNRun_EndBatch;
                }
                SparseMatrix Xt = new SparseMatrix(paramModel.nInput, BatchSize);
                SparseMatrix Dt = new SparseMatrix(paramModel.nOutput, BatchSize);
                int[] IdxSample = new int[BatchSize];
                Array.Copy(IdxSample_Tot, IdxBatch * BatchSize_normal, IdxSample, 0, BatchSize);
                TestData.GetColumns(Xt, IdxSample);

                // Forward activation
                LDA_Learn.ForwardActivation_LDA(Xt, DNNRun, paramModel, false);

                // Write the score into file
                for (int IdxCol = 0; IdxCol < DNNRun.y.nCols; IdxCol++)
                {
                    ScoreFile.WriteLine(String.Join("\t", DNNRun.y.DenseMatrixValue[IdxCol].VectorValue));
                }

                Console.Write(" Testing: Bat#{0}/{1}\r", (IdxBatch + 1), nBatch);
            }
            Console.WriteLine("----------------------------------------------------");
            ScoreFile.Close();
        }
    }

    public class DNNRun_t
    {
        public DenseMatrix[] theta_pool = null;
        public DenseMatrix theta0 = null;
        public DenseMatrix T_pool = null;
        public DenseMatrix y = null;
        public SparseMatrix y_sparse = null;
        public int nHid = 0;
        public int BatchSize = 0;
        public int nHidLayer = 0;
        public int nOutput = 0;
        public int[] nHidLayerEffective = null;

        public DNNRun_t()
        {
        }

        public DNNRun_t(int NumHiddenNode, int InputBatchSize, int NumHiddenLayer, int NumOutput)
        {
            nHid = NumHiddenNode;
            BatchSize = InputBatchSize;
            nHidLayer = NumHiddenLayer;
            nOutput = NumOutput;

            theta0 = new DenseMatrix(nHid, BatchSize, 1.0f / ((float)nHid));
            theta_pool = new DenseMatrix[nHidLayer];
            for (int IdxLayer = 0; IdxLayer < nHidLayer; IdxLayer++)
            {
                theta_pool[IdxLayer] = new DenseMatrix(nHid, BatchSize);
            }

            T_pool = new DenseMatrix(nHidLayer, BatchSize, false);
            y = new DenseMatrix(nOutput, BatchSize);

            nHidLayerEffective = new int[InputBatchSize];
        }

        public void SetSparsePatternForY(SparseMatrix SourceMatrix)
        {
            y_sparse = new SparseMatrix(SourceMatrix);
        }



    }
    
    /*
     * Gradient for the model (Dense gradient)
     */
    public class Grad_t
    {
        public DenseMatrix grad_Q_U = null;
        public SparseMatrix grad_Q_U_Sparse = null;
        public DenseColumnVector[] grad_Q_b_pool = null;
        public SparseMatrix[] grad_Q_Phi_pool = null;
        public SparseMatrix grad_Q_TopPhi = null;
        public DenseMatrix grad_Q_Phi = null;

        public Grad_t()
        {
        }

        public Grad_t(int NumHiddenNode, int NumOutput, int NumInput, int nHidLayer, string OutputType)
        {
            grad_Q_b_pool = new DenseColumnVector[nHidLayer];
            for (int IdxLayer = 0; IdxLayer < nHidLayer; IdxLayer++)
            {
                grad_Q_b_pool[IdxLayer] = new DenseColumnVector(NumHiddenNode);
            }
            grad_Q_Phi_pool = new SparseMatrix[nHidLayer];
            for (int IdxLayer = 0; IdxLayer < nHidLayer; IdxLayer++)
            {
                grad_Q_Phi_pool[IdxLayer] = new SparseMatrix(NumInput, NumHiddenNode, true);
            }
            if (OutputType == "linearQuad" || OutputType == "softmaxCE" || OutputType == "linearCE")
            {
                grad_Q_U = new DenseMatrix(NumOutput, NumHiddenNode);
                if (OutputType == "linearCE")
                {
                    grad_Q_U_Sparse = new SparseMatrix(NumOutput, NumHiddenNode, true);
                }
            }
            else if (OutputType == "unsupLDA")
            {
                grad_Q_TopPhi = new SparseMatrix(NumInput, NumHiddenNode, true);
            }
            grad_Q_Phi = new DenseMatrix(NumInput, NumHiddenNode);
        }

        public void SetSparsePatternForAllGradPhi(int[] SourceKey)
        {
            if (grad_Q_TopPhi != null)
            {
                grad_Q_TopPhi.SetSparsePatternForAllColumn(SourceKey);
            }
            for (int IdxLayer = 0; IdxLayer < grad_Q_Phi_pool.Length; IdxLayer++)
            {
                grad_Q_Phi_pool[IdxLayer].SetSparsePatternForAllColumn(SourceKey);
            }
        }
    }
    



}
