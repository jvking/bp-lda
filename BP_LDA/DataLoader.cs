using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using LinearAlgebra;
using System.Threading;

namespace BP_LDA
{
	public class DataLoader
	{
		/*
         * Load input data (tsv format) and return it in a sparse matrix
         */
		public static SparseMatrix InputDataLoader(string InputDataFile, int nInput)
		{
			Console.WriteLine("==================================================");


			// Scan through the entire file to get the number of lines (number of samples)
			Console.WriteLine("Scanning the file: {0}", InputDataFile);
			StreamReader InputDataStream = new StreamReader(InputDataFile);
			string StrLine;
			int nLine = 0;
			int nSamples = 0;
			List<string> AllRawInput = new List<string>();
			while ((StrLine = InputDataStream.ReadLine()) != null)
			{
				AllRawInput.Add(StrLine);
				nLine++;
				if (nLine % 10000 == 0)
				{
					Console.Write("Number of lines (samples): {0}\r", nLine);
				}
			}
			nSamples = nLine;
			Console.Write("Number of lines (samples): {0}\n", nLine);
			Console.WriteLine("Finished scanning the input data file");
			InputDataStream.Close();            

			// Parse each line and store it into each column of the sparse matrix
			Console.WriteLine("Loading input data...");
			SparseMatrix InputDataMatrix = new SparseMatrix(nInput, nLine);            
			nLine = 0;            
			int nTotNonzero = 0;
			int nEmptyLine = 0;
			Parallel.For(0, InputDataMatrix.nCols, new ParallelOptions { MaxDegreeOfParallelism = MatrixOperation.MaxMultiThreadDegree }, IdxCol =>
				{
					string[] StrLineSplit = AllRawInput[IdxCol].Split('\t');
					int nNonzero = StrLineSplit.Length;
					Interlocked.Add(ref nTotNonzero, nNonzero);
					int[] Key = null;
					float[] Val = null;
					Key = new int[nNonzero];
					Val = new float[nNonzero];
					string[] StrKeyValPair = null;
					if (StrLineSplit.Length == 1 && StrLineSplit[0] == "")
					{
						Key[0] = 0;
						Val[0] = 0.0f;
						Interlocked.Increment(ref nEmptyLine);
					}
					else
					{
						for (int Idx = 0; Idx < nNonzero; Idx++)
						{
							StrKeyValPair = StrLineSplit[Idx].Split(':');
							Key[Idx] = int.Parse(StrKeyValPair[0]);
							Val[Idx] = float.Parse(StrKeyValPair[1]);
						}
					}
					InputDataMatrix.FillColumn(Key, Val, IdxCol);

					Interlocked.Increment(ref nLine);
					if (nLine % 10000 == 0)
					{
						Console.Write("Number of lines (samples): {0}, with {1} empty lines.\r", nLine, nEmptyLine);
					}
				});

			Console.Write("Number of lines (samples): {0}, with {1} empty lines.\n", nLine, nEmptyLine);
			Console.WriteLine("Finished loading the input data file");
			Console.WriteLine("# Samples = {0}, # Inputs = {1}", InputDataMatrix.nCols, InputDataMatrix.nRows);
			Console.WriteLine("# Nonzeros = {0}/{1} ({2}%)", nTotNonzero, (long)InputDataMatrix.nRows * (long)InputDataMatrix.nCols, (((float)nTotNonzero) / ((float)InputDataMatrix.nRows * InputDataMatrix.nCols)) * 100);
			Console.WriteLine("==================================================");



			return InputDataMatrix;
		}

		/*
         * Load the label data (single column) and return it in a sparse matrix
         */
		public static SparseMatrix LabelDataLoader(string LabelDataFile, int nOutput, string OutputType)
		{            
			Console.WriteLine("==================================================");
			Console.WriteLine("Scanning the file: {0}", LabelDataFile);
			StreamReader LabelDataStream = new StreamReader(LabelDataFile);
			string StrLine;
			int nLine = 0;
			int nSamples = 0;
			List<string> AllRawLabel = new List<string>();

			// Load all the data
			nLine = 0;
			while ((StrLine = LabelDataStream.ReadLine())!=null)
			{
				AllRawLabel.Add(StrLine);

				nLine++;
				if (nLine % 10000 == 0)
				{
					Console.Write("Number of lines (samples): {0}\r", nLine);
				}
			}
			Console.WriteLine("Number of lines (samples): {0}", nLine);
			Console.WriteLine("Finished scanning the input data file");
			LabelDataStream.Close();


			// Parse the raw text into actual labels to be used in the learning algorithm
			Console.WriteLine("Loading input data...");
			nSamples = nLine;
			SparseMatrix LabelData = new SparseMatrix(nOutput, nSamples);
			nLine = 0;
			switch (OutputType)
			{
			case "linearCE":
				Parallel.For(0, LabelData.nCols, new ParallelOptions { MaxDegreeOfParallelism = MatrixOperation.MaxMultiThreadDegree }, IdxCol =>
					{
						int[] Key = new int[1];
						float[] Val = new float[1];
						if (!int.TryParse(AllRawLabel[IdxCol], out Key[0]))
						{
							Key[0] = (int)float.Parse(AllRawLabel[IdxCol]);
						}
						Val[0] = 1.0f;
						LabelData.FillColumn(Key, Val, IdxCol);
						Interlocked.Increment(ref nLine);
						if (nLine % 10000 == 0)
						{
							Console.Write("Number of lines (samples): {0}\r", nLine);
						}
					});
				break;
			case "softmaxCE":
				Parallel.For(0, LabelData.nCols, new ParallelOptions { MaxDegreeOfParallelism = MatrixOperation.MaxMultiThreadDegree }, IdxCol =>
					{
						int[] Key = new int[1];
						float[] Val = new float[1];
						if (!int.TryParse(AllRawLabel[IdxCol], out Key[0]))
						{
							Key[0] = (int)float.Parse(AllRawLabel[IdxCol]);
						}
						Val[0] = 1.0f;
						LabelData.FillColumn(Key, Val, IdxCol);
						Interlocked.Increment(ref nLine);
						if (nLine % 10000 == 0)
						{
							Console.Write("Number of lines (samples): {0}\r", nLine);
						}
					});
				break;
			case "linearQuad":
				Parallel.For(0, LabelData.nCols, new ParallelOptions { MaxDegreeOfParallelism = MatrixOperation.MaxMultiThreadDegree }, IdxCol =>
					{
						string[] StrLineSplit = AllRawLabel[IdxCol].Split('\t');
						int[] Key = new int[StrLineSplit.Length];
						float[] Val = new float[StrLineSplit.Length];
						for (int IdxOutput = 0; IdxOutput < StrLineSplit.Length; IdxOutput++)
						{
							Key[IdxOutput] = IdxOutput;
							Val[IdxOutput] = float.Parse(StrLineSplit[IdxOutput]);
						}
						LabelData.FillColumn(Key, Val, IdxCol);
						Interlocked.Increment(ref nLine);
						if (nLine % 10000 == 0)
						{
							Console.Write("Number of lines (samples): {0}\r", nLine);
						}
					});
				break;
			default:
				throw new Exception("Unknown OutputType are supported.");
			}
			Console.Write("Number of lines (samples): {0}\n", nLine);
			Console.WriteLine("==================================================");

			return LabelData;
		}


		/*
         * Load dense matrix and transpose it.
         */
		public static DenseMatrix DenseMatrixTransposeLoader(string DenseMatrixFileName)
		{
			StreamReader DenseMatrixFile = new StreamReader(DenseMatrixFileName);

			// Scan the data file to determine the dimension of the matrix
			int nCol = 0;
			int nRow = 0;
			string StrLine;
			while((StrLine = DenseMatrixFile.ReadLine()) != null)
			{
				if (nCol == 0)
				{
					string[] StrLineSplit = StrLine.Split('\t');
					nRow = StrLineSplit.Length;
				}
				++nCol;
			}
			DenseMatrixFile.Close();

			// Load the data and store it into the matrix
			DenseMatrix Matrix = new DenseMatrix(nRow,nCol);
			DenseMatrixFile = new StreamReader(DenseMatrixFileName);            
			int IdxCol = 0;
			while ((StrLine = DenseMatrixFile.ReadLine()) != null)
			{
				float[] LoadedColumn = new float[nRow];
				string[] StrLineSplit = StrLine.Split('\t');
				for (int IdxRow = 0; IdxRow < nRow; ++IdxRow)
				{
					LoadedColumn[IdxRow] = float.Parse(StrLineSplit[IdxRow]);
				}

				Matrix.FillColumn(LoadedColumn, IdxCol);

				++IdxCol;
			}
			DenseMatrixFile.Close();
			return Matrix;
		}

		/*
         * Load ParamModel from file
         */
		public static paramModel_t ParamModelHyperParamLoader(string ParamModelHyperParamFileName)
		{
			paramModel_t ParamModel = new paramModel_t();
			using(StreamReader ParamModelFile = new StreamReader(ParamModelHyperParamFileName))
			{
				string StrLine;
				while((StrLine = ParamModelFile.ReadLine())!=null)
				{
					string[] StrLineSplit = StrLine.Split('\t');
					string Key = StrLineSplit[0];
					string Val = StrLineSplit[1];
					switch (Key)
					{
					case "nHid":
						ParamModel.nHid = int.Parse(Val);
						break;
					case "nHidLayer":
						ParamModel.nHidLayer = int.Parse(Val);
						break;
					case "To":
						ParamModel.To = float.Parse(Val);
						break;
					case "nOutput":
						ParamModel.nOutput = int.Parse(Val);
						break;
					case "nInput":
						ParamModel.nInput = int.Parse(Val);
						break;
					case "eta":
						ParamModel.eta = float.Parse(Val);
						break;
					case "alpha":
						ParamModel.alpha = float.Parse(Val);
						break;
					case "beta":
						ParamModel.beta = float.Parse(Val);
						break;
					case "T_value":
						ParamModel.T_value = float.Parse(Val);
						break;
					case "OutputType":
						ParamModel.OutputType = Val;
						break;
					default:
						throw new Exception("Unknown type of Key in hyperparameter file.");
					}
				}
				ParamModel.T = new float[ParamModel.nHidLayer];
				for (int Idx = 0; Idx<ParamModel.T.Length; ++Idx)
				{
					ParamModel.T[Idx] = ParamModel.T_value;
				}
				ParamModel.b = new DenseColumnVector(ParamModel.nHid, ParamModel.alpha - 1.0f);
			}

			return ParamModel;
		}
	}
}
