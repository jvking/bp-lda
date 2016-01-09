using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace LinearAlgebra
{
    public static class MatrixOperation
    {
        public static int THREADNUM = 100;
        public static int MaxMultiThreadDegree = 32;
        

        /*
         * Set the rows of a matrix to zero if the corresponding rows of a column vector are negative
         */
        public static void SetRowsToZeroGivenColVector(DenseMatrix Z, DenseColumnVector x)
        {
            int nCols = Z.nCols;
            List<int> IdxNegX = new List<int>();
            var xVal = x.VectorValue;
            int xDim = x.Dim;
            for (int IdxRow = 0; IdxRow < xDim; ++IdxRow )
            {
                if (xVal[IdxRow]<0.0f)
                {
                    IdxNegX.Add(IdxRow);
                }
            }
            Parallel.For(0, nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                var ZVal = Z.DenseMatrixValue[IdxCol].VectorValue; 
                int nCnt = IdxNegX.Count;
                for (int IdxRow = 0; IdxRow < nCnt; ++IdxRow)
                {
                    ZVal[IdxNegX[IdxRow]] = 0.0f;
                }
            });
        }

        /*
         * Z = log(X). or z = log(x) elementwise log
         */
        public static void Log(SparseMatrix Z, SparseMatrix X)
        {
            // Dimension check
            if (Z.nCols != X.nCols || Z.nRows != X.nRows)
            {
                throw new Exception("Dimension mismatch.");
            }
            // Computation
            for (int IdxCol = 0; IdxCol < Z.nCols; IdxCol++)
            {
                var nNonzero = Z.SparseColumnVectors[IdxCol].nNonzero;
                var zKey = Z.SparseColumnVectors[IdxCol].Key;
                var xKey = X.SparseColumnVectors[IdxCol].Key;
                var zVal = Z.SparseColumnVectors[IdxCol].Val;
                var xVal = X.SparseColumnVectors[IdxCol].Val;
                for (int IdxRow = 0; IdxRow < nNonzero; ++IdxRow)
                {
                    if (zKey[IdxRow] != xKey[IdxRow])
                    {
                        throw new Exception("Sparse patterns do not match in elementwise matrix multiplication.");
                    }
                    zVal[IdxRow] = (float) Math.Log( (double) (xVal[IdxRow]));
                }
            }
        }
        public static void Log(DenseMatrix Z, DenseMatrix X)
        {
            // Dimension check
            if (Z.nCols != X.nCols || Z.nCols != X.nCols)
            {
                throw new Exception("Dimension mismatch.");
            }
            // Computation
            var nRows = Z.nRows;
            var nCols = Z.nCols;
            Parallel.For(0, nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                var zVal = Z.DenseMatrixValue[IdxCol].VectorValue;
                var xVal = X.DenseMatrixValue[IdxCol].VectorValue;
                for (int IdxRow = 0; IdxRow < nRows; ++IdxRow)
                {
                    zVal[IdxRow] = (float)Math.Log((double)xVal[IdxRow]);
                }
            });
        }
        public static void Log(DenseColumnVector z, DenseColumnVector x)
        {
            // Dimension check
            if (z.Dim != x.Dim)
            {
                throw new Exception("Dimension mismatch.");
            }
            // Computation
            for (int IdxCol = 0; IdxCol < z.Dim; IdxCol++)
            {
                z.VectorValue[IdxCol] = (float)Math.Log((double)x.VectorValue[IdxCol]);
            }
        }
        
        /*
         * Z = log(Z), or z = log(z)
         */
        public static void Log(SparseMatrix Z)
        {
            int nCols = Z.nCols;
            Parallel.For(0, nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                var nNonzero = Z.SparseColumnVectors[IdxCol].nNonzero;
                var zVal = Z.SparseColumnVectors[IdxCol].Val;
                for (int IdxRow = 0; IdxRow < nNonzero; ++IdxRow)
                {
                    zVal[IdxRow] = (float)Math.Log((double)zVal[IdxRow]);
                }
            });
        }
        public static void Log(DenseMatrix Z)
        {
            int nCols = Z.nCols;
            int nRows = Z.nRows;
            Parallel.For(0, nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                var zVal = Z.DenseMatrixValue[IdxCol].VectorValue;
                for (int IdxRow = 0; IdxRow < nRows; ++IdxRow)
                {
                    zVal[IdxRow] = (float)Math.Log((double)zVal[IdxRow]);
                }
            });
        }
        public static void Log(DenseColumnVector z)
        {
            var zVal = z.VectorValue;
            for (int IdxRow = 0; IdxRow < z.Dim; ++IdxRow)
            {
                zVal[IdxRow] = (float) Math.Log((double) zVal[IdxRow]);
            }
        }
        public static void Log(SparseColumnVector z)
        {
            var zVal = z.Val;
            for (int IdxRow = 0; IdxRow < z.nNonzero; ++IdxRow)
            {
                zVal[IdxRow] = (float)Math.Log((double)zVal[IdxRow]);
            }
        }

        /*
         * Z = exp(Z), or z = exp(z)
         */
        public static void Exp(DenseColumnVector z)
        {
            var zVal = z.VectorValue;
            for (int IdxCol = 0; IdxCol < z.Dim; ++IdxCol)
            {
                zVal[IdxCol] = (float)Math.Exp((double)zVal[IdxCol]);
            }
        }
        public static void Exp(DenseMatrix Z)
        {
            int nCols = Z.nCols;
            int nRows = Z.nRows;

            Parallel.For(0, nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                var zVal = Z.DenseMatrixValue[IdxCol].VectorValue;
                for (int IdxRow = 0; IdxRow < nRows; ++IdxRow)
                {
                    zVal[IdxRow] = (float)Math.Exp((double)zVal[IdxRow]);
                }
            });
            
        }
        public static void Exp(SparseMatrix Z)
        {
            int nCols = Z.nCols;
            int nRows = Z.nRows;
            Parallel.For(0, nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                var zVal = Z.SparseColumnVectors[IdxCol].Val;
                int nNz = Z.SparseColumnVectors[IdxCol].nNonzero;
                for (int IdxRow = 0; IdxRow < nNz; ++IdxRow)
                {
                    zVal[IdxRow] = (float)Math.Exp(zVal[IdxRow]);
                }
            });
        }
        

        /*
         * z = x^T y
         * Inner product
         */
        public static float InnerProduct(SparseColumnVector x, SparseColumnVector y)
        {
            if (x.Dim != y.Dim || x.nNonzero != y.nNonzero)
            {
                throw new Exception("Dimension mismatch.");
            }
            float z = 0.0f;
            int nNonzero = x.nNonzero;
            var xVal = x.Val;
            var yVal = y.Val;
            for (int Idx = 0; Idx < nNonzero; ++Idx)
            {
                z += xVal[Idx] * yVal[Idx];
            }
            return z;
        }
        public static float InnerProduct(DenseColumnVector x, DenseColumnVector y)
        {
            if (x.Dim != y.Dim)
            {
                throw new Exception("Dimension mismatch.");
            }
            float z = 0.0f;
            var xVal = x.VectorValue;
            var yVal = y.VectorValue;
            int Dim = x.Dim;
            for(int Idx = 0; Idx < Dim; ++Idx)
            {
                z += xVal[Idx] * yVal[Idx];
            }
            return z;
        }

        /*
         * Z = Z.^2 
         */
        public static void ElementwiseSquare(DenseMatrix Z)
        {
            int nCols = Z.nCols;
            int nRows = Z.nRows;
            Parallel.For(0, nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                var zVal = Z.DenseMatrixValue[IdxCol].VectorValue;
                for (int IdxRow = 0; IdxRow < nRows; ++IdxRow)
                {
                    zVal[IdxRow] *= zVal[IdxRow];
                }
            });
        }
        public static void ElementwiseSquare(SparseMatrix Z)
        {
            int nCols = Z.nCols;
            int nRows = Z.nRows;
            Parallel.For(0, nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                var zVal = Z.SparseColumnVectors[IdxCol].Val;
                int nNonzero = Z.SparseColumnVectors[IdxCol].nNonzero;
                for (int IdxRow = 0; IdxRow < nNonzero; ++IdxRow)
                {
                    zVal[IdxRow] *= zVal[IdxRow];
                }
            });
        }

        /*
         * Z = Z.^{1/2} 
         */
        public static void ElementwiseSquareRoot(DenseMatrix Z)
        {
            int nCols = Z.nCols;
            int nRows = Z.nRows;
            Parallel.For(0, nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                var zVal = Z.DenseMatrixValue[IdxCol].VectorValue;
                for (int IdxRow = 0; IdxRow < nRows; ++IdxRow)
                {
                    zVal[IdxRow] = (float)Math.Sqrt(zVal[IdxRow]);
                }
            });
        }
        /*
         * z = x.^{1/2}
         */
        public static void ElementwiseSquareRoot(DenseRowVector z, DenseRowVector x)
        {
            if (z.Dim != x.Dim)
            {
                throw new Exception("Dimension mismatch.");
            }
            int zDim = z.Dim;
            var zVal = z.VectorValue;
            var xVal = x.VectorValue;
            for (int Idx = 0; Idx < zDim; ++Idx )
            {
                zVal[Idx] = (float)Math.Sqrt(xVal[Idx]);
            }
        }

        /*
         * Z = X.*Y, where X and Y are arrays
         */
        public static float[] ElementwiseArrayProduct(float[] X, float[] Y)
        {
            if (X.Length != Y.Length)
            {
                throw new Exception("Lengths of X and Y are not equal!");
            }
            int Dim = X.Length;
            float[] Z = new float[Dim];
            for (int Idx = 0; Idx < Dim; Idx++)
            {
                Z[Idx] = X[Idx] * Y[Idx];
            }
            return Z;
        }
        public static double[] ElementwiseArrayProduct(double[] X, double[] Y)
        {
            if (X.Length != Y.Length)
            {
                throw new Exception("Lengths of X and Y are not equal!");
            }
            int Dim = X.Length;
            double[] Z = new double[Dim];
            for (int Idx = 0; Idx < Dim; Idx++)
            {
                Z[Idx] = X[Idx] * Y[Idx];
            }
            return Z;
        }
        public static int[] ElementwiseArrayProduct(int[] X, int[] Y)
        {
            if (X.Length != Y.Length)
            {
                throw new Exception("Lengths of X and Y are not equal!");
            }
            int Dim = X.Length;
            int[] Z = new int[Dim];
            for (int Idx = 0; Idx < Dim; Idx++)
            {
                Z[Idx] = X[Idx] * Y[Idx];
            }
            return Z;
        }

        /*
         * Z = X.*Y
         */
        public static void ElementwiseMatrixMultiplyMatrix(SparseMatrix Z, SparseMatrix X, SparseMatrix Y)
        {
            // Dimension check
            if (X.nCols != Y.nCols || X.nRows != Y.nRows || Z.nCols != X.nCols || Z.nRows != X.nRows)
            {
                throw new Exception("Dimension mismatch during elementwise matrix multiplication.");
            }
            // Elementwise matrix multiplication
            for (int IdxCol = 0; IdxCol < Z.nCols; IdxCol++)
            {
                for (int IdxRow = 0; IdxRow < Z.SparseColumnVectors[IdxCol].nNonzero; IdxRow++)
                {
                    if (Z.SparseColumnVectors[IdxCol].Key[IdxRow] != X.SparseColumnVectors[IdxCol].Key[IdxRow] 
                        || Z.SparseColumnVectors[IdxCol].Key[IdxRow] != Y.SparseColumnVectors[IdxCol].Key[IdxRow])
                    {
                        throw new Exception("Sparse patterns do not match in elementwise matrix multiplication.");
                    }
                    Z.SparseColumnVectors[IdxCol].Val[IdxRow] = X.SparseColumnVectors[IdxCol].Val[IdxRow] * Y.SparseColumnVectors[IdxCol].Val[IdxRow];
                }
            }
        }
        public static void ElementwiseMatrixMultiplyMatrix(SparseMatrix Z, SparseMatrix X, DenseMatrix Y)
        {
            // Dimension check
            if (X.nCols != Y.nCols || X.nRows != Y.nRows || Z.nCols != X.nCols || Z.nRows != X.nRows)
            {
                throw new Exception("Dimension mismatch during elementwise matrix multiplication.");
            }
            // Elementwise matrix multiplication
            Parallel.For(0, Z.nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                var zVal = Z.SparseColumnVectors[IdxCol].Val;
                var xVal = X.SparseColumnVectors[IdxCol].Val;
                var yVal = Y.DenseMatrixValue[IdxCol].VectorValue;
                var zKey = Z.SparseColumnVectors[IdxCol].Key;
                int nNonzero = Z.SparseColumnVectors[IdxCol].nNonzero;
                for (int IdxRow = 0; IdxRow < nNonzero; ++IdxRow)
                {
                    zVal[IdxRow]
                        = xVal[IdxRow] * yVal[zKey[IdxRow]];
                }
            });
        }
        public static void ElementwiseMatrixMultiplyMatrix(DenseMatrix Z, DenseMatrix X, DenseMatrix Y)
        {
            if (Z.nCols != X.nCols || Z.nRows != X.nRows || Z.nCols != Y.nCols || Z.nRows != Y.nRows)
            {
                throw new Exception("Dimension mismatch.");
            }
            int nCols = Z.nCols;
            int nRows = Z.nRows;
            Parallel.For(0, nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                var zVal = Z.DenseMatrixValue[IdxCol].VectorValue;
                var xVal = X.DenseMatrixValue[IdxCol].VectorValue;
                var yVal = Y.DenseMatrixValue[IdxCol].VectorValue;
                for (int IdxRow = 0; IdxRow < nRows; ++IdxRow)
                {
                    zVal[IdxRow] = xVal[IdxRow] * yVal[IdxRow];
                }
            });
        }        
        
        /*
         * Z = Z.*X
         */
        public static void ElementwiseMatrixMultiplyMatrix(SparseMatrix Z, SparseMatrix X)
        { 
            // Check dimension
            if (Z.nCols != X.nCols || Z.nRows != X.nRows)
            {
                throw new Exception("Dimension mismatch.");
            }
            // Computation
            Parallel.For(0, Z.nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                //if (Z.SparseColumnVectors[IdxCol].nNonzero != X.SparseColumnVectors[IdxCol].nNonzero)
                //{
                //    throw new Exception("Sparse pattern mismatch.");
                //}
                var zVal = Z.SparseColumnVectors[IdxCol].Val;
                var xVal = X.SparseColumnVectors[IdxCol].Val;
                int nNonzero = Z.SparseColumnVectors[IdxCol].nNonzero;
                for (int IdxRow = 0; IdxRow < nNonzero; ++IdxRow)
                {
                    //if (Z.SparseColumnVectors[IdxCol].Key[IdxRow] != X.SparseColumnVectors[IdxCol].Key[IdxRow])
                    //{
                    //    throw new Exception("Sparse pattern mismatch.");
                    //}
                    zVal[IdxRow] *= xVal[IdxRow];
                }
            });            
        }
        public static void ElementwiseMatrixMultiplyMatrix(DenseMatrix Z, SparseMatrix X)
        {
            if (Z.nCols != X.nCols || Z.nRows != X.nRows)
            {
                throw new Exception("Dimension mismatch.");
            }
            Parallel.For(0, Z.nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
                {
                    int nNz = X.SparseColumnVectors[IdxCol].nNonzero;
                    var xKey = X.SparseColumnVectors[IdxCol].Key;
                    var xVal = X.SparseColumnVectors[IdxCol].Val;
                    var zVal = Z.DenseMatrixValue[IdxCol].VectorValue;
                    for (int IdxRow = 0; IdxRow < nNz; ++IdxRow)
                    {
                        zVal[xKey[IdxRow]] *= xVal[IdxRow];
                    }
                }
            );
        }

        /*
         * z = x.*y
         */
        public static void ElementwiseVectorMultiplyVector(DenseColumnVector z, DenseColumnVector x, DenseColumnVector y)
        {
            if (z.Dim != x.Dim || z.Dim != y.Dim)
            {
                throw new Exception("Dimension mismatch.");
            }
            int zDim = z.Dim;
            var zVal = z.VectorValue;
            var xVal = x.VectorValue;
            var yVal = y.VectorValue;
            for (int IdxRow = 0; IdxRow < zDim; ++IdxRow)
            {
                zVal[IdxRow] = xVal[IdxRow] * yVal[IdxRow];
            }
        }

        /*
         * z = z.*x
         */
        public static void ElementwiseVectorMultiplyVector(DenseColumnVector z, DenseColumnVector x)
        {
            // Dimension check
            if (z.Dim != x.Dim)
            {
                throw new Exception("Dimension mismatch.");
            }
            // Computation
            int Dim = z.Dim;
            var zVal = z.VectorValue;
            var xVal = x.VectorValue;
            for (int IdxCol = 0; IdxCol < Dim; ++IdxCol)
            {
                zVal[IdxCol] *= xVal[IdxCol];
            }
        }
        public static DenseRowVector ElementwiseVectorMultiplyVector(DenseRowVector x, DenseRowVector y)
        {
            if (x.Dim != y.Dim)
            {
                throw new Exception("Dimension mismatch.");
            }
            DenseRowVector z = new DenseRowVector(x.Dim);
            for (int IdxCol = 0; IdxCol < z.Dim; IdxCol++ )
            {
                z.VectorValue[IdxCol] = x.VectorValue[IdxCol] * y.VectorValue[IdxCol];
            }
                return z;
        }
        public static void ElementwiseVectorMultiplyVector(SparseColumnVector x, SparseColumnVector y)
        {
            if (x.Dim != y.Dim || x.nNonzero != y.nNonzero)
            {
                throw new Exception("Dimension mismatch.");
            }
            int nz = x.nNonzero;
            var xVal = x.Val;
            var yVal = y.Val;
            for (int IdxRow = 0; IdxRow < nz; ++IdxRow)
            {
                xVal[IdxRow] *= yVal[IdxRow];
            }
        }

        /*
         * Z = X ./ Y
         * Elementwise division
         */
        public static void ElementwiseMatrixDivideMatrix(SparseMatrix Z, SparseMatrix X, SparseMatrix Y)
        {
            // Dimension check
            if (Z.nCols != X.nCols || Z.nRows != X.nRows || Z.nCols != Y.nCols || Z.nRows != Y.nRows)
            {
                throw new Exception("Dimension mismatch.");
            }
            // Computation
            int nCols = Z.nCols;
            Parallel.For(0, nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                var zVal = Z.SparseColumnVectors[IdxCol].Val;
                var xVal = X.SparseColumnVectors[IdxCol].Val;
                var yVal = Y.SparseColumnVectors[IdxCol].Val;
                int nNonzero = Z.SparseColumnVectors[IdxCol].nNonzero;
                for (int IdxRow = 0; IdxRow < nNonzero; ++IdxRow)
                {
                    zVal[IdxRow] = xVal[IdxRow] / (yVal[IdxRow]+1e-12f);
                }
            });
        }
        public static void ElementwiseMatrixDivideMatrix(DenseMatrix Z, DenseMatrix X, DenseMatrix Y)
        {
            if (Z.nRows != X.nRows || Z.nCols != X.nCols || Z.nRows != Y.nRows || Z.nCols != Y.nCols)
            {
                throw new Exception("Dimension mismatch.");
            }
            int nRows = Z.nRows;
            Parallel.For(0, Z.nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                var zVal = Z.DenseMatrixValue[IdxCol].VectorValue;
                var xVal = X.DenseMatrixValue[IdxCol].VectorValue;
                var yVal = Y.DenseMatrixValue[IdxCol].VectorValue;
                for (int IdxRow = 0; IdxRow < nRows; ++IdxRow)
                {
                    zVal[IdxRow] = xVal[IdxRow] / (yVal[IdxRow]+1e-12f);
                }
            });
        }

        /*
         * z = x./y, where x and y are vectors
         */
        public static void ElementwiseVectorDivideVector(SparseColumnVector z, SparseColumnVector x, SparseColumnVector y)
        {
            if (z.Dim != x.Dim || z.Dim != y.Dim || z.nNonzero != x.nNonzero || z.nNonzero != y.nNonzero)
            {
                throw new Exception("Dimension mismatch.");
            }
            int nz = z.nNonzero;
            var zVal = z.Val;
            var xVal = x.Val;
            var yVal = y.Val;
            for (int IdxRow = 0; IdxRow < nz; ++IdxRow)
            {
                zVal[IdxRow] = xVal[IdxRow] / (yVal[IdxRow]+1e-12f);
            }
        }
        
        /*
         * z = x./y
         */
        public static void ElementwiseVectorDivideVector(DenseColumnVector z, DenseColumnVector x, DenseColumnVector y)
        {
            if (z.Dim != x.Dim || z.Dim != y.Dim)
            {
                throw new Exception("Dimension mismatch.");
            }
            var zVal = z.VectorValue;
            var xVal = x.VectorValue;
            var yVal = y.VectorValue;
            int zDim = z.Dim;
            for (int IdxRow = 0; IdxRow < zDim; ++IdxRow)
            {
                zVal[IdxRow] = xVal[IdxRow] / (yVal[IdxRow]+1e-12f);
            }
        }
        public static void ElementwiseVectorDivideVector(DenseRowVector z, DenseRowVector x, DenseRowVector y)
        {
            if (z.Dim != x.Dim || z.Dim != y.Dim)
            {
                throw new Exception("Dimension mismatch.");
            }
            var zVal = z.VectorValue;
            var xVal = x.VectorValue;
            var yVal = y.VectorValue;
            int zDim = z.Dim;
            for (int IdxRow = 0; IdxRow < zDim; ++IdxRow)
            {
                zVal[IdxRow] = xVal[IdxRow] / (yVal[IdxRow]+1e-12f);
            }
        }
        
        /*
         * Z = x ./ Y, where x is a scalar
         */
        public static void ScalarDivideMatrix(DenseMatrix Z, float x, DenseMatrix Y)
        {
            if (Z.nCols != Y.nCols || Z.nRows != Y.nRows)
            {
                throw new Exception("Dimension mismatch.");
            }
            int nRows = Z.nRows;
            Parallel.For(0, Z.nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                var zVal = Z.DenseMatrixValue[IdxCol].VectorValue;
                var yVal = Y.DenseMatrixValue[IdxCol].VectorValue;
                for (int IdxRow = 0; IdxRow < nRows; ++IdxRow)
                {
                    zVal[IdxRow] = x / (yVal[IdxRow]+1e-12f);
                }
            });
        }
        public static void ScalarDivideMatrix(DenseMatrix Z, float x, DenseMatrix Y, bool IsCumSum)
        {
            if (Z.nCols != Y.nCols || Z.nRows != Y.nRows)
            {
                throw new Exception("Dimension mismatch.");
            }
            int nRows = Z.nRows;
            if (IsCumSum)
            {
                Parallel.For(0, Z.nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
                {
                    var zVal = Z.DenseMatrixValue[IdxCol].VectorValue;
                    var yVal = Y.DenseMatrixValue[IdxCol].VectorValue;
                    for (int IdxRow = 0; IdxRow < nRows; ++IdxRow)
                    {
                        zVal[IdxRow] += x / (yVal[IdxRow]+1e-12f);
                    }
                });
            }
            else
            {
                Parallel.For(0, Z.nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
                {
                    var zVal = Z.DenseMatrixValue[IdxCol].VectorValue;
                    var yVal = Y.DenseMatrixValue[IdxCol].VectorValue;
                    for (int IdxRow = 0; IdxRow < nRows; ++IdxRow)
                    {
                        zVal[IdxRow] = x / (yVal[IdxRow]+1e-12f);
                    }
                });
            }
        }

        /*
         * Z = x ./ Z or b = bsxfun(@rdivide, x, Z)
         * Vector divide by matrix
         */
        public static void bsxfunVectorDivideMatrix(DenseMatrix Z, DenseColumnVector x)
        {
            // Dimension check
            if (x.Dim != Z.nRows)
            {
                throw new Exception("Dimension mismatch.");
            }
            // Computation
            int nRows = Z.nRows;
            var xVal = x.VectorValue;
            Parallel.For(0, Z.nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                var zVal = Z.DenseMatrixValue[IdxCol].VectorValue;
                for (int IdxRow = 0; IdxRow < nRows; ++IdxRow)
                {
                    zVal[IdxRow] = xVal[IdxRow] / (zVal[IdxRow]+1e-12f);
                }
            });
        }

        /*
         * Z = x ./ Y or b = bsxfun(@rdivide, x, Y)
         * Vector divide by matrix
         */
        public static void bsxfunVectorDivideMatrix(DenseMatrix Z, DenseColumnVector x, DenseMatrix Y)
        {
            // Dimension check
            if (x.Dim != Z.nRows || Z.nCols != Y.nCols || Z.nRows != Y.nRows)
            {
                throw new Exception("Dimension mismatch.");
            }
            // Computation
            for (int IdxCol = 0; IdxCol < Z.nCols; IdxCol++)
            {
                for (int IdxRow = 0; IdxRow < Z.nRows; IdxRow++)
                {
                    Z.DenseMatrixValue[IdxCol].VectorValue[IdxRow] = x.VectorValue[IdxRow] / (Y.DenseMatrixValue[IdxCol].VectorValue[IdxRow]+1e-12f);
                }
            }
        }

        /*
         * X = X * y (scalar)
         */
        public static void ScalarMultiplyMatrix(DenseMatrix X, float y)
        {
            int nRows = X.nRows;
            Parallel.For(0, X.nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                var xVal = X.DenseMatrixValue[IdxCol].VectorValue;
                for (int IdxRow = 0; IdxRow < nRows; ++IdxRow)
                {
                    xVal[IdxRow] *= y;
                }
            });
        }

        /*
         * Z = X * y (scalar)
         */
        public static void ScalarMultiplyMatrix(SparseMatrix Z, SparseMatrix X, float y)
        {
            // Dimension check
            if (Z.nCols != X.nCols || Z.nRows != X.nRows)
            {
                throw new Exception("Dimension mismatch.");
            }
            // Computation
            Parallel.For(0, Z.nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                var zVal = Z.SparseColumnVectors[IdxCol].Val;
                var xVal = X.SparseColumnVectors[IdxCol].Val;
                int nNonzero = Z.SparseColumnVectors[IdxCol].nNonzero;
                for (int IdxRow = 0; IdxRow < nNonzero; ++IdxRow)
                {
                    //if (Z.SparseColumnVectors[IdxCol].Key[IdxRow] != X.SparseColumnVectors[IdxCol].Key[IdxRow])
                    //{
                    //    throw new Exception("Sparse patterns do not match");
                    //}
                    zVal[IdxRow] = xVal[IdxRow] * y;
                }
            });
        }

        /*
         * z = z * y (scalar): Scalar multiplies vector
         */
        public static void ScalarMultiplyVector(DenseRowVector z, float y)
        {
            var zVal = z.VectorValue;
            int Dim = z.Dim;
            for (int IdxCol = 0; IdxCol < Dim; ++IdxCol)
            {
                zVal[IdxCol] *= y;
            }
        }
        public static void ScalarMultiplyVector(DenseColumnVector z, float y)
        {
            int Dim = z.Dim;
            var zVal = z.VectorValue;
            for (int IdxCol = 0; IdxCol < Dim; ++IdxCol)
            {
                zVal[IdxCol] *= y;
            }
        }
        public static void ScalarMultiplyVector(SparseColumnVector z, float y)
        {
            int zNz = z.nNonzero;
            var zVal = z.Val;
            for (int IdxRow = 0; IdxRow < zNz; ++IdxRow)
            {
                zVal[IdxRow] *= y;
            }
        }
        public static void ScalarMultiplyVector(SparseColumnVector z, SparseColumnVector x, float y)
        {
            int zNz = z.nNonzero;
            var zVal = z.Val;
            var xVal = x.Val;
            for (int IdxRow = 0; IdxRow < zNz; ++IdxRow)
            {
                zVal[IdxRow] = xVal[IdxRow] * y;
            }
        }
        /*
         * z = x * y, where y is a scalar
         */
        public static void ScalarMultiplyVector(DenseRowVector z, DenseRowVector x, float y)
        {
            var zVal = z.VectorValue;
            var xVal = x.VectorValue;
            int Dim = z.Dim;
            for (int IdxCol = 0; IdxCol < Dim; ++IdxCol)
            {
                zVal[IdxCol] = xVal[IdxCol] * y;
            }
        }
        
        /*
         * z = x * y, where x is a vector and y is a scalar
         */
        public static void ScalarMultiplyVector(DenseColumnVector z, DenseColumnVector x, float y)
        {
            // Dimension check
            if (z.Dim != x.Dim)
            {
                throw new Exception("Dimension mismatch.");
            }
            // Computation
            int Dim = z.Dim;
            var zVal = z.VectorValue;
            var xVal = x.VectorValue;
            for (int IdxCol = 0; IdxCol < Dim; ++IdxCol)
            {
                zVal[IdxCol] = xVal[IdxCol] * y;
            }
        }

        /*
         * z = z - x: vector subtracts vector
         */
        public static void VectorSubtractVector(DenseRowVector z, DenseRowVector x)
        {
            // Dimension check
            if (z.Dim != x.Dim)
            {
                throw new Exception("Dimension mismatch.");
            }
            // Computation
            var zVal = z.VectorValue;
            var xVal = x.VectorValue;
            int Dim = z.Dim;
            for (int IdxCol=0; IdxCol<Dim; ++IdxCol)
            {
                zVal[IdxCol] -= xVal[IdxCol];
            }
        }
        public static void VectorSubtractVector(DenseColumnVector z, DenseColumnVector x, DenseColumnVector y)
        {
            if (z.Dim != x.Dim || z.Dim != y.Dim)
            {
                throw new Exception("Dimension mismatch.");
            }
            var zVal = z.VectorValue;
            var xVal = x.VectorValue;
            var yVal = y.VectorValue;
            int Dim = z.Dim;
            for (int IdxRow = 0; IdxRow < Dim; ++IdxRow)
            {
                zVal[IdxRow] = xVal[IdxRow] - yVal[IdxRow];
            }
        }

        /*
         * Z = Z - Y
         */
        public static void MatrixSubtractMatrix(DenseMatrix Z, SparseMatrix Y)
        {
            if (Z.nRows != Y.nRows || Z.nCols != Y.nCols)
            {
                throw new Exception("Dimension mismatch.");
            }
            Parallel.For(0, Z.nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                int nNonzero = Y.SparseColumnVectors[IdxCol].nNonzero;
                var zVal = Z.DenseMatrixValue[IdxCol].VectorValue;
                var yKey = Y.SparseColumnVectors[IdxCol].Key;
                var yVal = Y.SparseColumnVectors[IdxCol].Val;
                for (int IdxRow = 0; IdxRow < nNonzero; ++IdxRow)
                {
                    zVal[yKey[IdxRow]] -= yVal[IdxRow];
                }
            });
        }
        public static void MatrixSubtractMatrix(SparseMatrix Z, SparseMatrix Y)
        {
            if (Z.nRows != Y.nRows || Z.nCols != Y.nCols)
            {
                throw new Exception("Dimension mismatch.");
            }
            Parallel.For(0, Z.nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                int nNonzero = Y.SparseColumnVectors[IdxCol].nNonzero;
                var zVal = Z.SparseColumnVectors[IdxCol].Val;
                var yVal = Y.SparseColumnVectors[IdxCol].Val;
                for (int IdxRow = 0; IdxRow < nNonzero; ++IdxRow)
                {
                    zVal[IdxRow] -= yVal[IdxRow];
                }
            });
        }
        public static void MatrixSubtractMatrix(DenseMatrix Z, DenseMatrix Y)
        {
            if (Z.nCols != Y.nCols || Z.nRows != Y.nRows)
            {
                throw new Exception("Dimension mismatch.");
            }
            Parallel.For(0, Z.nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                var zVal = Z.DenseMatrixValue[IdxCol].VectorValue;
                var yVal = Y.DenseMatrixValue[IdxCol].VectorValue;
                int nRows = Z.nRows;
                for (int IdxRow = 0; IdxRow < nRows; ++IdxRow)
                {
                    zVal[IdxRow] -= yVal[IdxRow];
                }
            });
        }

        /*
         * Z = X - Y
         */
        public static void MatrixSubtractMatrix(DenseMatrix Z, DenseMatrix X, DenseMatrix Y)
        {
            if (Z.nCols != X.nCols || Z.nRows != X.nRows || Z.nCols != Y.nCols || Z.nRows != Y.nRows)
            {
                throw new Exception("Dimension mismatch.");
            }
            Parallel.For(0, Z.nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                var zVal = Z.DenseMatrixValue[IdxCol].VectorValue;
                var xVal = X.DenseMatrixValue[IdxCol].VectorValue;
                var yVal = Y.DenseMatrixValue[IdxCol].VectorValue;
                int nRows = Z.nRows;
                for (int IdxRow = 0; IdxRow < nRows; IdxRow++)
                {
                    zVal[IdxRow] = xVal[IdxRow] - yVal[IdxRow];
                }
            });
        }

        /*
         * X = X + y (scalar)
         */
        public static void ScalarAddMatrix(DenseMatrix X, float y)
        {
            Parallel.For(0, X.nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                var xVal = X.DenseMatrixValue[IdxCol].VectorValue;
                for (int IdxRow = 0; IdxRow < X.nRows; IdxRow++)
                {
                    xVal[IdxRow] += y;
                }
            });
        }

        /*
         * z = z + y, where y is a scalar
         */
        public static void ScalarAddVector(DenseColumnVector z, float y)
        {
            var zVal = z.VectorValue;
            for (int IdxCol = 0; IdxCol < z.Dim; IdxCol++)
            {
                zVal[IdxCol] += y;
            }
        }
        public static void ScalarAddVector(DenseRowVector z, float y)
        {
            var zVal = z.VectorValue;
            for (int IdxCol = 0; IdxCol < z.Dim; IdxCol++)
            {
                zVal[IdxCol] += y;
            }
        }
        public static void ScalarAddVector(SparseColumnVector z, float y)
        {
            var zVal = z.Val;
            int nNonzero = z.nNonzero;
            for (int IdxRow = 0; IdxRow < nNonzero; ++IdxRow)
            {
                zVal[IdxRow] += y;
            }
        }
        
        
        /*
         * z = x + y, where x is a vector and y is a scalar
         */
        public static void ScalarAddVector(DenseColumnVector z, DenseColumnVector x, float y)
        {
            if (z.Dim != x.Dim)
            {
                throw new Exception("Dimension mismatch.");
            }
            var zVal = z.VectorValue;
            var xVal = x.VectorValue;
            int Dim = z.Dim;
            for (int IdxCol = 0; IdxCol < Dim; ++IdxCol)
            {
                zVal[IdxCol] = xVal[IdxCol] + y;
            }
        }
        public static void ScalarAddVector(SparseColumnVector z, SparseColumnVector x, float y)
        {
            if (z.Dim != x.Dim || z.nNonzero != x.nNonzero)
            {
                throw new Exception("Dimension mismatch");
            }
            var zVal = z.Val;
            var xVal = x.Val;
            int nz = z.nNonzero;
            for (int IdxRow = 0; IdxRow < nz; ++IdxRow)
            {
                zVal[IdxRow] = xVal[IdxRow] + y;
            }
        }

        /*
         * Z = X + y (scalar)
         */
        public static void ScalarAddMatrix(SparseMatrix Z, SparseMatrix X, float y)
        {
            // Dimension check
            if (Z.nCols != X.nCols || Z.nRows != X.nRows)
            {
                throw new Exception("Matrix dimension mismatch.");
            }
            // Computation
            Parallel.For(0, X.nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                var zVal = Z.SparseColumnVectors[IdxCol].Val;
                var xVal = X.SparseColumnVectors[IdxCol].Val;
                int nNonzero = X.SparseColumnVectors[IdxCol].nNonzero;
                for (int IdxRow = 0; IdxRow < nNonzero; ++IdxRow)
                {
                    //if (Z.SparseColumnVectors[IdxCol].Key[IdxRow] != X.SparseColumnVectors[IdxCol].Key[IdxRow])
                    //{
                    //    throw new Exception("Sparse patterns do not match.");
                    //}
                    zVal[IdxRow] = xVal[IdxRow] + y;
                }
            });
        }
        public static void ScalarAddMatrix(DenseMatrix Z, DenseMatrix X, float y)
        {
            // Dimension check
            if (Z.nCols != X.nCols || Z.nRows != X.nRows)
            {
                throw new Exception("Matrix dimension mismatch.");
            }
            // Computation
            Parallel.For(0, X.nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                var zVal = Z.DenseMatrixValue[IdxCol].VectorValue;
                var xVal = X.DenseMatrixValue[IdxCol].VectorValue;
                int nRows = X.nRows;
                for (int IdxRow = 0; IdxRow < nRows; ++IdxRow)
                {
                    zVal[IdxRow] = xVal[IdxRow] + y;
                }
            });
        }
                
        /*
         * Z = X + Z
         */
        public static void MatrixAddMatrix(DenseMatrix Z, DenseMatrix X)
        {
            // Check dimension
            if (Z.nRows != X.nRows || Z.nCols != X.nCols)
            {
                throw new Exception("Dimension mismatch.");
            }
            // Computation
            Parallel.For(0, Z.nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                var zVal = Z.DenseMatrixValue[IdxCol].VectorValue;
                var xVal = X.DenseMatrixValue[IdxCol].VectorValue;
                int nRows = Z.nRows;
                for (int IdxRow = 0; IdxRow < nRows; ++IdxRow)
                {
                    zVal[IdxRow] += xVal[IdxRow];
                }
            });
        }
        public static void MatrixAddMatrix(DenseMatrix Z, SparseMatrix X)
        {
            // Check dimension
            if (Z.nRows != X.nRows || Z.nCols != X.nCols)
            {
                throw new Exception("Dimension mismatch.");
            }
            // Computation
            Parallel.For(0, Z.nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                var zVal = Z.DenseMatrixValue[IdxCol].VectorValue;
                var xKey = X.SparseColumnVectors[IdxCol].Key;
                var xVal = X.SparseColumnVectors[IdxCol].Val;
                int nNonzero = X.SparseColumnVectors[IdxCol].nNonzero;
                for (int IdxRow = 0; IdxRow < nNonzero; ++IdxRow)
                {
                    zVal[xKey[IdxRow]] += xVal[IdxRow];
                }
            });
        }
        public static void MatrixAddMatrix(SparseMatrix Z, DenseMatrix X)
        {
            // Check dimension
            if (Z.nRows != X.nRows || Z.nCols != X.nCols)
            {
                throw new Exception("Dimension mismatch.");
            }
            // Computation
            Parallel.For(0, Z.nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                var nNonzero = Z.SparseColumnVectors[IdxCol].nNonzero;
                var zVal = Z.SparseColumnVectors[IdxCol].Val;
                var zKey = Z.SparseColumnVectors[IdxCol].Key;
                var xVal = X.DenseMatrixValue[IdxCol].VectorValue;
                for (int IdxRow = 0; IdxRow < nNonzero; ++IdxRow)
                {
                    zVal[IdxRow] += xVal[zKey[IdxRow]];
                }
            });
        }
        public static void MatrixAddMatrix(SparseMatrix Z, SparseMatrix X)
        {
            // Check dimension
            if (Z.nRows != X.nRows || Z.nCols != X.nCols)
            {
                throw new Exception("Dimension mismatch.");
            }
            // Computation
            Parallel.For(0, Z.nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                var nNonzero = Z.SparseColumnVectors[IdxCol].nNonzero;
                var zVal = Z.SparseColumnVectors[IdxCol].Val;
                var xVal = X.SparseColumnVectors[IdxCol].Val;
                for (int IdxRow = 0; IdxRow < nNonzero; ++IdxRow)
                {
                    zVal[IdxRow] += xVal[IdxRow];
                }
            });
        }
        /*
         * Z = X + Y
         */
        public static void MatrixAddMatrix(DenseMatrix Z, DenseMatrix X, SparseMatrix Y)
        {
            // Check dimension
            if (Z.nRows != X.nRows || Z.nCols != X.nCols || Z.nRows != Y.nRows || Z.nCols != Y.nCols)
            {
                throw new Exception("Dimension mismatch.");
            }
            // Computation
            Parallel.For(0, Z.nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                Array.Copy(X.DenseMatrixValue[IdxCol].VectorValue, Z.DenseMatrixValue[IdxCol].VectorValue, Z.DenseMatrixValue[IdxCol].VectorValue.Length);
            });
            MatrixAddMatrix(Z, Y);
        }

        /*
         * z = z + x, where z and x are vectors
         */
        public static void VectorAddVector(DenseColumnVector z, DenseColumnVector x)
        {
            // Dimension check
            if (z.Dim != x.Dim)
            {
                throw new Exception("Dimension mismatch.");
            }
            // Computation
            var zVal = z.VectorValue;
            var xVal = x.VectorValue;
            int Dim = z.Dim;
            for (int IdxCol = 0; IdxCol < Dim; ++IdxCol)
            {
                zVal[IdxCol] += xVal[IdxCol];
            }
        }
        public static void VectorAddVector(DenseRowVector z, DenseRowVector x)
        {
            if (z.Dim != x.Dim)
            {
                throw new Exception("Dimension mismatch.");
            }
            // Computation
            var zVal = z.VectorValue;
            var xVal = x.VectorValue;
            int Dim = z.Dim;
            for (int IdxCol = 0; IdxCol < Dim; ++IdxCol)
            {
                zVal[IdxCol] += xVal[IdxCol];
            }
        }

        /*
         * z = sum(X,1): vertical sum
         */
        public static DenseRowVector VerticalSumMatrix(DenseMatrix X)
        {
            DenseRowVector z = new DenseRowVector(X.nCols,0.0f);
            var zVal = z.VectorValue;
            Parallel.For(0, X.nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                var xVal = X.DenseMatrixValue[IdxCol].VectorValue;
                int nRows = X.nRows;
                for (int IdxRow = 0; IdxRow < nRows; ++IdxRow)
                {
                    zVal[IdxCol] += xVal[IdxRow];
                }
            });
            return z;
        }
        public static void VerticalSumMatrix(DenseRowVector z, DenseMatrix X)
        {
            z.FillValue(0.0f);
            int nRows = X.nRows;
            Parallel.For(0, X.nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                var zVal = z.VectorValue;
                var xVal = X.DenseMatrixValue[IdxCol].VectorValue;
                for (int IdxRow = 0; IdxRow < nRows; ++IdxRow)
                {
                    zVal[IdxCol] += xVal[IdxRow];
                }
            });
        }
        public static void VerticalSumMatrix(DenseRowVector z, SparseMatrix X)
        {
            Array.Clear(z.VectorValue, 0, z.VectorValue.Length);
            Parallel.For(0, X.nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                var zVal = z.VectorValue;
                var xVal = X.SparseColumnVectors[IdxCol].Val;
                int nNonzero = X.SparseColumnVectors[IdxCol].nNonzero;
                for (int IdxRow = 0; IdxRow < nNonzero; ++IdxRow)
                {
                    zVal[IdxCol] += xVal[IdxRow];
                }
            });
        }

        /*
         * z = sum(X,2): horizontal sum
         */
        public static DenseColumnVector HorizontalSumMatrix(DenseMatrix X)
        {
            DenseColumnVector z = new DenseColumnVector(X.nRows, 0.0f);
            for (int IdxRow = 0; IdxRow < X.nRows; IdxRow++)
            {
                for (int IdxCol = 0; IdxCol < X.nCols; IdxCol++)
                {
                    z.VectorValue[IdxRow] += X.DenseMatrixValue[IdxCol].VectorValue[IdxRow];
                }
            }
            return z;
        }
        public static void HorizontalSumMatrix(DenseColumnVector z, DenseMatrix X)
        {
            Array.Clear(z.VectorValue, 0, z.VectorValue.Length);
            Parallel.For(0, X.nRows, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxRow =>
            {
                float sum = 0.0f;
                var xCol = X.DenseMatrixValue;
                int nCols = X.nCols;
                for (int IdxCol = 0; IdxCol < nCols; ++IdxCol)
                {
                    sum += xCol[IdxCol].VectorValue[IdxRow];
                }
                z.VectorValue[IdxRow] = sum;
            });
        }
        public static DenseColumnVector HorizontalSumMatrix(SparseMatrix X)
        {
            DenseColumnVector z = new DenseColumnVector(X.nRows);
            Array.Clear(z.VectorValue, 0, z.VectorValue.Length);

            for (int IdxCol = 0; IdxCol < X.nCols; IdxCol++ )
            {
                int nNonzero = X.SparseColumnVectors[IdxCol].nNonzero;
                var xKey = X.SparseColumnVectors[IdxCol].Key;
                var xVal = X.SparseColumnVectors[IdxCol].Val;
                var zVal = z.VectorValue;
                for (int IdxRow = 0; IdxRow < nNonzero; ++IdxRow)
                {
                    zVal[xKey[IdxRow]] += xVal[IdxRow];
                }
            }

            return z;
        }
        public static void HorizontalSumMatrix(DenseColumnVector z, SparseMatrix X)
        {
            if (z.Dim != X.nRows)
            {
                throw new Exception("Dimension mismatch.");
            }
            Array.Clear(z.VectorValue, 0, z.VectorValue.Length);

            for (int IdxCol = 0; IdxCol < X.nCols; IdxCol++)
            {
                int nNonzero = X.SparseColumnVectors[IdxCol].nNonzero;
                var xKey = X.SparseColumnVectors[IdxCol].Key;
                var xVal = X.SparseColumnVectors[IdxCol].Val;
                var zVal = z.VectorValue;
                for (int IdxRow = 0; IdxRow < nNonzero; ++IdxRow)
                {
                    zVal[xKey[IdxRow]] += xVal[IdxRow];
                }
            }
        }

        /*
         * z = mean(X,2): horizontal mean
         */
        public static DenseColumnVector HorizontalMeanMatrix(SparseMatrix X)
        {
            DenseColumnVector z = HorizontalSumMatrix(X);
            return z;
        }

        /*
         * X = bsxfun(@rdivide, X, y) or X = X / y, where y is a dense row or column vector
         */
        public static void bsxfunMatrixRightDivideVector(DenseMatrix X, DenseRowVector y)
        {
            if (X.nCols != y.Dim)
            {
                throw new Exception("The Number of columns in the two inputs does not match!");
            }
            Parallel.For(0, X.nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                var xVal = X.DenseMatrixValue[IdxCol].VectorValue;
                var yVal = y.VectorValue[IdxCol];
                int nRows = X.nRows;
                for (int IdxRow = 0; IdxRow < nRows; ++IdxRow)
                {
                    xVal[IdxRow] /= (yVal+1e-12f);
                }
            });
        }
        public static void bsxfunMatrixRightDivideVector(DenseMatrix X, DenseColumnVector y)
        {
            if (X.nRows != y.Dim)
            {
                throw new Exception("The Number of rows in the two inputs does not match!");
            }
            for (int IdxRow = 0; IdxRow < X.nRows; IdxRow++)
            {
                for (int IdxCol = 0; IdxCol < X.nCols; IdxCol++)
                {
                    X.DenseMatrixValue[IdxCol].VectorValue[IdxRow] /= (y.VectorValue[IdxRow]+1e-12f);
                }
            }
        }


        /*
         * X = bsxfun(@times, X, y) or X = X * y, where y is a dense row or column vector
         */
        public static void bsxfunVectorMultiplyMatrix(DenseMatrix X, DenseRowVector y)
        {
            if (X.nCols != y.Dim)
            {
                throw new Exception("The Number of columns in the two inputs does not match!");
            }
            Parallel.For(0, X.nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                var xVal = X.DenseMatrixValue[IdxCol].VectorValue;
                var yVal = y.VectorValue[IdxCol];
                int nRows = X.nRows;
                for (int IdxRow = 0; IdxRow < nRows; ++IdxRow)
                {
                    xVal[IdxRow] *= yVal;
                }
            });
        }
        public static void bsxfunVectorMultiplyMatrix(DenseMatrix X, DenseColumnVector y)
        {
            if (X.nRows != y.Dim)
            {
                throw new Exception("The Number of rows in the two inputs does not match!");
            }
            int nRows = X.nRows;
            Parallel.For(0, X.nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                var xVal = X.DenseMatrixValue[IdxCol].VectorValue;
                var yVal = y.VectorValue;
                for (int IdxRow = 0; IdxRow < nRows; ++IdxRow)
                {
                    xVal[IdxRow] *= yVal[IdxRow];
                }
            });
        }
        public static void bsxfunVectorMultiplyMatrix(SparseMatrix X, DenseRowVector y)
        {
            if (X.nCols != y.Dim)
            {
                throw new Exception("Dimension mismatch.");
            }
            Parallel.For(0, X.nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                var xVal = X.SparseColumnVectors[IdxCol].Val;
                var yVal = y.VectorValue[IdxCol];
                var nNonzero = X.SparseColumnVectors[IdxCol].nNonzero;
                for (int IdxRow = 0; IdxRow < nNonzero; ++IdxRow)
                {
                    xVal[IdxRow] *= yVal;
                }
            });
        }

        /*
         * Z = bsxfun(@times, X, y) or Z = X * y, where y is a dense row or column vector
         */
        public static void bsxfunVectorMultiplyMatrix(SparseMatrix Z, SparseMatrix X, DenseRowVector y)
        {
            if (Z.nCols != X.nCols || Z.nRows != X.nRows || Z.nCols != y.Dim)
            {
                throw new Exception("Dimension mismatch!");
            }
            int ZnCols = Z.nCols;
            Parallel.For(0, ZnCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                int nNz = Z.SparseColumnVectors[IdxCol].nNonzero;
                var ZVal = Z.SparseColumnVectors[IdxCol].Val;
                var XVal = X.SparseColumnVectors[IdxCol].Val;
                var yVal = y.VectorValue;
                for (int IdxRow = 0; IdxRow < nNz; ++IdxRow)
                {
                    ZVal[IdxRow] = XVal[IdxRow] * yVal[IdxCol];
                }
            });
        }
        public static void bsxfunVectorMultiplyMatrix(DenseMatrix Z, DenseMatrix X, DenseRowVector y)
        {
            if (X.nCols != y.Dim || Z.nCols != X.nCols || Z.nRows != X.nRows)
            {
                throw new Exception("The Number of columns in the two inputs does not match!");
            }

            int total = Z.nCols * Z.nRows;
            int process_len = (total + THREADNUM - 1) / THREADNUM;
            Parallel.For(0, THREADNUM, new ParallelOptions{ MaxDegreeOfParallelism = MaxMultiThreadDegree}, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int id = thread_idx * process_len + t;
                    if (id < total)
                    {
                        int IdxCol = id / Z.nRows;
                        int IdxRow = id % Z.nRows;
                        Z.DenseMatrixValue[IdxCol].VectorValue[IdxRow] = X.DenseMatrixValue[IdxCol].VectorValue[IdxRow] * y.VectorValue[IdxCol];
                    }
                    else
                    {
                        break;
                    }
                }
            });
        }

        /*
         * Z = bsxfun(@minus, X, y)
         */
        public static void bsxfunMatrixSubtractVector(DenseMatrix Z, DenseMatrix X, DenseRowVector y)
        {
            if (Z.nCols != X.nCols || Z.nRows != X.nRows || Z.nCols != y.Dim)
            {
                throw new Exception("Dimension mismatch.");
            }

            int total = Z.nCols * Z.nRows;
            int process_len = (total + THREADNUM - 1) / THREADNUM;
            Parallel.For(0, THREADNUM, new ParallelOptions{ MaxDegreeOfParallelism = MaxMultiThreadDegree}, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int id = thread_idx * process_len + t;
                    if (id < total)
                    {
                        int IdxCol = id / Z.nRows;
                        int IdxRow = id % Z.nRows;
                        Z.DenseMatrixValue[IdxCol].VectorValue[IdxRow] = X.DenseMatrixValue[IdxCol].VectorValue[IdxRow] - y.VectorValue[IdxCol];
                    }
                    else
                        break;
                }
            });
        }
        public static void bsxfunMatrixSubtractVector(SparseMatrix Z, SparseMatrix X, DenseRowVector y)
        {
            if (Z.nCols != X.nCols || Z.nRows != X.nRows || Z.nCols != y.Dim)
            {
                throw new Exception("Dimension mismatch.");
            }

            int total = Z.nCols;
            int process_len = (total + THREADNUM - 1) / THREADNUM;
            Parallel.For(0, THREADNUM, new ParallelOptions{ MaxDegreeOfParallelism = MaxMultiThreadDegree}, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int IdxCol = thread_idx * process_len + t;
                    if (IdxCol < total)
                    {
                        var zVal = Z.SparseColumnVectors[IdxCol].Val;
                        var xVal = X.SparseColumnVectors[IdxCol].Val;
                        var yVal = y.VectorValue[IdxCol];
                        int nNonzero = Z.SparseColumnVectors[IdxCol].nNonzero;
                        for (int IdxRow = 0; IdxRow < nNonzero; ++IdxRow)
                        {
                            zVal[IdxRow] = xVal[IdxRow] - yVal;
                        }
                    }
                }
            });
        }

        /*
         * Z = X*Y, where X and Y are dense. 
         * Only nonzero positions of Z will be computed if Z is sparse.
         */
        public static void MatrixMultiplyMatrix(SparseMatrix Z, DenseMatrix X, DenseMatrix Y)
        {
            // Dimension check
            if (Z.nRows != X.nRows || Z.nCols != Y.nCols || X.nCols != Y.nRows)
            {
                throw new Exception("Matrix dimension mismatch in multiplication.");
            }

            int total = Z.nCols ;
            int process_len = (total + THREADNUM - 1) / THREADNUM;
            Parallel.For(0, THREADNUM, new ParallelOptions{ MaxDegreeOfParallelism = MaxMultiThreadDegree}, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int IdxCol = thread_idx * process_len + t;
                    if (IdxCol < total)
                    {
                        // V1
                        //SparseColumnVector z = Z.SparseColumnVectors[IdxCol];
                        //int nNonzero = z.nNonzero;
                        //for (int IdxRow = 0; IdxRow < nNonzero; IdxRow++)
                        //{
                        //    // Get the actual row index of Z
                        //    int IdxTrueRow = z.Key[IdxRow];
                        //    // Compute the (IdxTrueRow, IdxCol)-th entry of Z, stored at (IdxRow, IdxCol)
                        //    float sum = 0;
                        //    for (int Idx = 0; Idx < X.nCols; Idx++)
                        //    {
                        //        sum += X.DenseMatrixValue[Idx].VectorValue[IdxTrueRow]
                        //               * Y.DenseMatrixValue[IdxCol].VectorValue[Idx];
                        //    }
                        //    z.Val[IdxRow] = sum;

                        //}

                        // V2
                        var yVector = Y.DenseMatrixValue[IdxCol].VectorValue;
                        SparseColumnVector z = Z.SparseColumnVectors[IdxCol];
                        var zVal = z.Val;
                        var zKey = z.Key;
                        int nNonzero = z.nNonzero;
                        for (int IdxRow = 0; IdxRow < nNonzero; ++IdxRow)
                        {
                            zVal[IdxRow] = 0;
                        }
                        for (int Idx = 0; Idx < X.nCols; ++ Idx)
                        {
                            // Compute the (IdxTrueRow, IdxCol)-th entry of Z, stored at (IdxRow, IdxCol)
                            var xVector = X.DenseMatrixValue[Idx].VectorValue;
                            var y = yVector[Idx];
                            for (int IdxRow = 0; IdxRow < nNonzero; ++ IdxRow)
                            {
                                // Get the actual row index of Z
                                zVal[IdxRow] += xVector[zKey[IdxRow]] * y;
                            }
                        }

                    }
                    else
                        break;
                }
            });
        }
        public static void MatrixMultiplyMatrix(DenseMatrix Z, DenseMatrix X, DenseMatrix Y)
        {
            // Dimension check
            if (Z.nRows != X.nRows || Z.nCols != Y.nCols || X.nCols != Y.nRows)
            {
                throw new Exception("Matrix dimension mismatch in multiplication.");
            }

            int total = Z.nCols * Z.nRows;
            int process_len = (total + THREADNUM - 1) / THREADNUM;
            Parallel.For(0, THREADNUM, new ParallelOptions{ MaxDegreeOfParallelism = MaxMultiThreadDegree}, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int id = thread_idx * process_len + t;
                    if (id < total)
                    {
                        int IdxCol = id / Z.nRows;
                        int IdxRow = id % Z.nRows;
                        float sum = 0;
                        var yVal = Y.DenseMatrixValue[IdxCol].VectorValue;
                        for(int Idx = 0; Idx<X.nCols;Idx++)
                        {
                            sum += X.DenseMatrixValue[Idx].VectorValue[IdxRow]
                                * yVal[Idx];
                        }
                        Z.DenseMatrixValue[IdxCol].VectorValue[IdxRow] = sum;
                    }
                    else
                        break;
                }
            });
        }


        /*
         * Z = X*Y^T
         */
        public static void MatrixMultiplyMatrixTranspose(DenseMatrix Z, DenseMatrix X, DenseMatrix Y)
        {
            if (Z.nRows != X.nRows || Z.nCols != Y.nRows || X.nCols != Y.nCols)
            {
                throw new Exception("Dimension mismatch.");
            }

            int total = Z.nCols * Z.nRows;
            int process_len = (total + THREADNUM - 1) / THREADNUM;
            Parallel.For(0, THREADNUM, new ParallelOptions{ MaxDegreeOfParallelism = MaxMultiThreadDegree}, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int id = thread_idx * process_len + t;
                    if (id < total)
                    {
                        int IdxCol = id / Z.nRows;
                        int IdxRow = id % Z.nRows;
                        float sum = 0;
                        for (int Idx = 0; Idx < X.nCols; Idx++)
                        {
                            sum += X.DenseMatrixValue[Idx].VectorValue[IdxRow] * Y.DenseMatrixValue[Idx].VectorValue[IdxCol];
                        }
                        Z.DenseMatrixValue[IdxCol].VectorValue[IdxRow] = sum;
                    }
                    else
                        break;
                }
            });
        }

        public static void MatrixMultiplyMatrixTranspose(DenseMatrix Z, SparseMatrix X, DenseMatrix Y)
        {
            if (Z.nRows != X.nRows || Z.nCols != Y.nRows || X.nCols != Y.nCols)
            {
                throw new Exception("Dimension mismatch.");
            }
            
            int total = Z.nCols;
            int process_len = (total + THREADNUM - 1) / THREADNUM;
            Parallel.For(0, THREADNUM, new ParallelOptions{ MaxDegreeOfParallelism = MaxMultiThreadDegree}, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int IdxCol = thread_idx * process_len + t;
                    if (IdxCol < total)
                    {
                        DenseColumnVector z = Z.DenseMatrixValue[IdxCol];
                        Array.Clear(z.VectorValue, 0, Z.nRows);
                        for (int Idx = 0; Idx < X.nCols; Idx++)
                        {
                            SparseColumnVector x = X.SparseColumnVectors[Idx];
                            float yvalue = Y.DenseMatrixValue[Idx].VectorValue[IdxCol];
                            for (int IdxRow = 0; IdxRow < x.nNonzero; IdxRow++)
                            {
                                z.VectorValue[x.Key[IdxRow]]
                                    += x.Val[IdxRow] * yvalue;
                            }
                        }
                    }
                    else
                        break;
                }
            });
        }

        /*
         * Z  = X * Y^T if IsCumSum == false
         * Z += X * Y^T is IsCumSum == true
         */
        public static void MatrixMultiplyMatrixTranspose(DenseMatrix Z, SparseMatrix X, DenseMatrix Y, bool IsCumSum)
        {
            if (Z.nRows != X.nRows || Z.nCols != Y.nRows || X.nCols != Y.nCols)
            {
                throw new Exception("Dimension mismatch.");
            }
            int total = Z.nCols;
            int process_len = (total + THREADNUM - 1) / THREADNUM;
            Parallel.For(0, THREADNUM, new ParallelOptions{ MaxDegreeOfParallelism = MaxMultiThreadDegree}, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int IdxCol = thread_idx * process_len + t;
                    if (IdxCol < total)
                    {
                        DenseColumnVector z = Z.DenseMatrixValue[IdxCol];
                        if (!IsCumSum) Array.Clear(z.VectorValue, 0, Z.nRows);
                        for (int Idx = 0; Idx < X.nCols; Idx++)
                        {
                            SparseColumnVector x = X.SparseColumnVectors[Idx];
                            float yvalue = Y.DenseMatrixValue[Idx].VectorValue[IdxCol];
                            for (int IdxRow = 0; IdxRow < x.nNonzero; IdxRow++)
                            {
                                z.VectorValue[x.Key[IdxRow]] += x.Val[IdxRow] * yvalue;
                            }
                        }
                    }
                    else
                        break;
                }
            });
        }

        public static void MatrixMultiplyMatrixTranspose(SparseMatrix Z, SparseMatrix X, DenseMatrix Y, bool IsCumSum)
        {
            if (Z.nRows != X.nRows || Z.nCols != Y.nRows || X.nCols != Y.nCols)
            {
                throw new Exception("Dimension mismatch.");
            }
            
            int total = Z.nCols;
            int process_len = (total + THREADNUM - 1) / THREADNUM;
            Parallel.For(0, THREADNUM, new ParallelOptions{ MaxDegreeOfParallelism = MaxMultiThreadDegree}, thread_idx =>
            {
                var ZSparsePatternOfEachColumn = Z.SparsePatternOfEachColumn;
                var xnCols = X.nCols;
                for (int t = 0; t < process_len; ++t)
                {
                    int IdxCol = thread_idx * process_len + t;
                    if (IdxCol < total)
                    {
                        SparseColumnVector z = Z.SparseColumnVectors[IdxCol];
                        var zVal = z.Val;
                        if (!IsCumSum) Array.Clear(zVal, 0, z.nNonzero);
                        for (int Idx = 0; Idx < xnCols; ++Idx)
                        {
                            float yvalue = Y.DenseMatrixValue[Idx].VectorValue[IdxCol];
                            var xKey = X.SparseColumnVectors[Idx].Key;
                            var xVal = X.SparseColumnVectors[Idx].Val;
                            var xnNonzero = X.SparseColumnVectors[Idx].nNonzero;
                            for (int IdxRow = 0; IdxRow < xnNonzero; ++IdxRow)
                            {
                                int xkey = xKey[IdxRow];
                                float xvalue = xVal[IdxRow];
                                zVal[ZSparsePatternOfEachColumn[xkey]]
                                    += xvalue * yvalue;
                            }
                        }
                    }
                    else
                        break;
                }
            });

        }

        /*
         * Z += a * x * y^T : Atomic add & thread safe, where x and y are column vectors, and a is a scalar.
         */
        public static void AtomicAddVectorMultiplyVectorTranspose(SparseMatrix Z, SparseColumnVector x, DenseColumnVector y, float a)
        {
            if (Z.nRows != x.Dim || Z.nCols != y.Dim)
            {
                throw new Exception("Dimension mismatch.");
            }
            float product = 0.0f;
            float InitVal = 0.0f;
            float ComputedVal = 0.0f;
            int ZnCols = Z.nCols;
            int xNz = x.nNonzero;
            var xVal = x.Val;
            var xKey = x.Key;
            var yVal = y.VectorValue;
            var ZSparseColumnPattern = Z.SparsePatternOfEachColumn;
            for (int IdxCol = 0; IdxCol < ZnCols; ++IdxCol)
            {
                var ZVal = Z.SparseColumnVectors[IdxCol].Val;
                for (int IdxRow = 0; IdxRow < xNz; ++IdxRow)
                {
                    product = xVal[IdxRow] * yVal[IdxCol] * a;
                    int ZIdx = ZSparseColumnPattern[xKey[IdxRow]];
                    do
                    {
                        InitVal = ZVal[ZIdx];
                        ComputedVal = InitVal + product;
                    } while (InitVal != Interlocked.CompareExchange(ref ZVal[ZIdx], ComputedVal, InitVal));
                }
            }
        }
        public static void AtomicAddVectorMultiplyVectorTranspose(DenseMatrix Z, DenseColumnVector x, DenseColumnVector y, float a)
        {
            if (Z.nRows != x.Dim || Z.nCols != y.Dim)
            {
                throw new Exception("Dimension mismatch.");
            }
            float product = 0.0f;
            float InitVal = 0.0f;
            float ComputedVal = 0.0f;
            int ZnCols = Z.nCols;
            int ZnRows = Z.nRows;
            var xVal = x.VectorValue;
            var yVal = y.VectorValue;
            for (int IdxCol = 0; IdxCol < ZnCols; ++IdxCol)
            {
                var ZVal = Z.DenseMatrixValue[IdxCol].VectorValue;
                for (int IdxRow = 0; IdxRow < ZnRows; ++IdxRow)
                {
                    product = xVal[IdxRow] * yVal[IdxCol] * a;
                    do
                    {
                        InitVal = ZVal[IdxRow];
                        ComputedVal = InitVal + product;
                    } while (InitVal != Interlocked.CompareExchange(ref ZVal[IdxRow], ComputedVal, InitVal));
                }
            }
        }
        public static void AtomicAddVectorMultiplyVectorTranspose(DenseMatrix Z, SparseColumnVector x, DenseColumnVector y, float a)
        {
            if (Z.nRows != x.Dim || Z.nCols != y.Dim)
            {
                throw new Exception("Dimension mismatch.");
            }
            float product = 0.0f;
            float InitVal = 0.0f;
            float ComputedVal = 0.0f;
            int ZnCols = Z.nCols;
            int xNz = x.nNonzero;
            var xVal = x.Val;
            var xKey = x.Key;
            var yVal = y.VectorValue;
            for (int IdxCol = 0; IdxCol < ZnCols; ++IdxCol)            
            {
                var ZVal = Z.DenseMatrixValue[IdxCol].VectorValue;
                for (int IdxRow = 0; IdxRow < xNz; ++IdxRow)
                {
                    product = xVal[IdxRow] * yVal[IdxCol] * a;
                    do
                    {
                        InitVal = ZVal[xKey[IdxRow]];
                        ComputedVal = InitVal + product;
                    } while (InitVal != Interlocked.CompareExchange(ref ZVal[xKey[IdxRow]], ComputedVal, InitVal));
                }
            }
        }




        /* 
         * z = X * y
         * Matrix multiply vector
         */
        public static void MatrixMultiplyVector(DenseColumnVector z, DenseMatrix X, DenseColumnVector y)
        {
            if (z.Dim != X.nRows || X.nCols != y.Dim)
            {
                throw new Exception("Dimension mismatch.");
            }
            Array.Clear(z.VectorValue, 0, z.VectorValue.Length);
            int XnCols = X.nCols;
            int zDim = z.Dim;

            for (int IdxCol = 0; IdxCol < XnCols; ++IdxCol)
            {
                var zVal = z.VectorValue;
                var XVal = X.DenseMatrixValue[IdxCol].VectorValue;
                var yVal = y.VectorValue;
                for (int IdxRow = 0; IdxRow < zDim; ++IdxRow)
                {
                    zVal[IdxRow] += XVal[IdxRow] * yVal[IdxCol];
                }
            }
        }
        
        public static void MatrixMultiplyVector(SparseColumnVector z, DenseMatrix X, DenseColumnVector y)
        {
            // Dimension check
            if (X.nCols != y.Dim || z.Dim != X.nRows)
            {
                throw new Exception("Dimension mismatch.");
            }
            // Computation
            //for (int IdxRow = 0; IdxRow < z.nNonzero; IdxRow++)
            //{
            //    z.Val[IdxRow] = 0.0f;
            //    for (int Idx = 0; Idx < X.nCols; Idx++)
            //    {
            //        z.Val[IdxRow] += X.DenseMatrixValue[Idx].VectorValue[z.Key[IdxRow]] * y.VectorValue[Idx];
            //    }
            //}

            var zVal = z.Val;
            var zKey = z.Key;
            Array.Clear(zVal, 0, zVal.Length);            
            var zNNonzero = z.nNonzero;
            var xnCols = X.nCols;
            float[] xColumn = null;
            float yValue;
            for (int Idx = 0; Idx < xnCols; ++ Idx)
            {
                xColumn = X.DenseMatrixValue[Idx].VectorValue;
                yValue = y.VectorValue[Idx];
                for (int IdxRow = 0; IdxRow < zNNonzero; ++ IdxRow)
                {
                    zVal[IdxRow] += xColumn[zKey[IdxRow]] * yValue;
                }
            }
        }

        /*
         * z = X^T * y
         */
        public static void MatrixTransposeMultiplyVector(DenseColumnVector z, DenseMatrix X, DenseColumnVector y)
        {
            if (z.Dim != X.nCols || X.nRows != y.Dim)
            {
                throw new Exception("Dimension mismatch.");
            }
            int zDim = z.Dim;
            int yDim = y.Dim;
            var XMat = X.DenseMatrixValue;
            var zVal = z.VectorValue;
            var yVal = y.VectorValue;
            Parallel.For(0, zDim, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxRow =>
            {
                float sum = 0.0f;
                var XCol = XMat[IdxRow].VectorValue;
                for (int Idx = 0; Idx < yDim; ++Idx)
                {
                    sum += XCol[Idx] * yVal[Idx];
                }
                zVal[IdxRow] = sum;
            });
        }
        public static void MatrixTransposeMultiplyVector(DenseColumnVector z, DenseMatrix X, SparseColumnVector y)
        {
            if (z.Dim != X.nCols || X.nRows != y.Dim)
            {
                throw new Exception("Dimension mismatch.");
            }
            int zDim = z.Dim;
            float sum = 0.0f;
            int yNz = y.nNonzero;
            var XMat = X.DenseMatrixValue;
            var zVal = z.VectorValue;
            for (int IdxRow = 0; IdxRow < zDim; ++IdxRow)
            {
                sum = 0.0f;
                var XCol = XMat[IdxRow].VectorValue;
                var yKey = y.Key;
                var yVal = y.Val;
                for (int Idx = 0; Idx < yNz; ++Idx)
                {
                    sum += XCol[yKey[Idx]] * yVal[Idx];
                }
                zVal[IdxRow] = sum;
            }
            
        }

        /* 
         * Z = X^T * Y
         * Matrix transpose mulplication
         */
        public static void MatrixTransposeMultiplyMatrix(DenseMatrix Z,DenseMatrix X, SparseMatrix Y)
        {
            // Dimension check
            if (X.nRows != Y.nRows || Z.nRows != X.nCols || Z.nCols != Y.nCols)
            {
                throw new Exception("Dimension mismatch.");
            }
            // Computation
            int total = Z.nCols * Z.nRows;
            int process_len = (total + THREADNUM - 1) / THREADNUM;
            Parallel.For(0, THREADNUM, new ParallelOptions{ MaxDegreeOfParallelism = MaxMultiThreadDegree}, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int id = thread_idx * process_len + t;
                    if (id < total)
                    {
                        int IdxCol = id / Z.nRows;
                        int IdxRow = id % Z.nRows;
                        DenseColumnVector z = Z.DenseMatrixValue[IdxCol];
                        var x = X.DenseMatrixValue[IdxRow].VectorValue;
                        SparseColumnVector y = Y.SparseColumnVectors[IdxCol];
                        var nNonzero = y.nNonzero;
                        float sum = 0;
                        var yKey = y.Key;
                        var yVal = y.Val;
                        for (int Idx = 0; Idx < nNonzero; ++ Idx)
                        {
                            sum += x[yKey[Idx]]
                                   * yVal[Idx];
                        }
                        z.VectorValue[IdxRow] = sum;
                    }
                    else
                        break;
                }
            });
        }
        public static void MatrixTransposeMultiplyMatrix(DenseMatrix Z, DenseMatrix X, DenseMatrix Y)
        {
            if (Z.nRows != X.nCols || Z.nCols != Y.nCols || X.nRows != Y.nRows)
            {
                throw new Exception("Dimension mismatch.");
            }

            int total = Z.nCols * Z.nRows;
            int process_len = (total + THREADNUM - 1) / THREADNUM;
            Parallel.For(0, THREADNUM, new ParallelOptions{ MaxDegreeOfParallelism = MaxMultiThreadDegree}, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int id = thread_idx * process_len + t;
                    if (id < total)
                    {
                        int IdxCol = id / Z.nRows;
                        int IdxRow = id % Z.nRows;
                        Z.DenseMatrixValue[IdxCol].VectorValue[IdxRow] = 0;
                        DenseColumnVector z = Z.DenseMatrixValue[IdxCol];
                        DenseColumnVector x = X.DenseMatrixValue[IdxRow];
                        DenseColumnVector y = Y.DenseMatrixValue[IdxCol];
                        for (int Idx = 0; Idx < X.nRows; Idx++)
                        {
                            z.VectorValue[IdxRow]
                                += x.VectorValue[Idx] * y.VectorValue[Idx];
                        }
                    }
                    else
                        break;
                }
            });
        }

        /*
         * z = max(X, [], 1)
         * Vertial max: generate a row vector contains the maximum of each column
         */
        public static void VerticalMaxMatrix(DenseRowVector z, DenseMatrix X)
        {
            //for (int IdxCol = 0; IdxCol < z.Dim; IdxCol++)
            //{
            //    z.VectorValue[IdxCol] = X.DenseMatrixValue[IdxCol].MaxValue();
            //}

            int zDim = z.Dim;
            var zVal = z.VectorValue;
            var XMat = X.DenseMatrixValue;
            Parallel.For(0, zDim, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                zVal[IdxCol] = XMat[IdxCol].MaxValue();
            });
        }
        public static void VerticalMaxMatrix(DenseRowVector z, SparseMatrix X)
        {
            //int zDim = z.Dim;
            //var zVal = z.VectorValue;
            //var XMat = X.SparseColumnVectors;
            //for (int IdxCol = 0; IdxCol < zDim; ++IdxCol)
            //{
            //    zVal[IdxCol] = XMat[IdxCol].Val.Max();
            //}

            int zDim = z.Dim;
            var zVal = z.VectorValue;
            var XMat = X.SparseColumnVectors;
            Parallel.For(0, zDim, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                zVal[IdxCol] = XMat[IdxCol].Val.Max();
            });
        }

        /*
         * Count the values less than a certain value at each column of a matrix
         */
        public static void CountValuesLessThanThreshold(DenseRowVector NumSpecialElementPerCol, DenseMatrix X, float Threshold)
        {
            if (NumSpecialElementPerCol.Dim != X.nCols)
            {
                throw new Exception("Dimension mismatch.");
            }

            for (int IdxCol = 0; IdxCol < X.nCols; IdxCol++ )
            {
                NumSpecialElementPerCol.VectorValue[IdxCol] = 0.0f;
                for (int IdxRow = 0; IdxRow < X.nRows; IdxRow++)
                {                    
                    if (X.DenseMatrixValue[IdxCol].VectorValue[IdxRow]<Threshold)
                    {
                        NumSpecialElementPerCol.VectorValue[IdxCol]++;
                    }
                }
            }
        }
        public static int CountValuesLessThanThreshold(DenseColumnVector x, float Threshold)
        {
            int NumSpecialElement = 0;
            var xVal = x.VectorValue;
            for (int IdxRow = 0; IdxRow < x.Dim; IdxRow++)
            {     
                if(xVal[IdxRow]<Threshold)
                {
                    NumSpecialElement++;
                }
            }
            return NumSpecialElement;
        }
        /*
         * Set elements of Z to zero if the corresponding elements of X is zero.
         */
        public static void ResetMatrixSparsePattern(DenseMatrix Z, DenseMatrix X)
        { 
            if (Z.nCols != X.nCols || Z.nRows != X.nRows)
            {
                throw new Exception("Dimension mismatch.");
            }

            Parallel.For(0, Z.nCols, new ParallelOptions { MaxDegreeOfParallelism = MaxMultiThreadDegree }, IdxCol =>
            {
                int nRows = Z.nRows;
                var XVal = X.DenseMatrixValue[IdxCol].VectorValue;
                var ZVal = Z.DenseMatrixValue[IdxCol].VectorValue;
                for (int IdxRow = 0; IdxRow < nRows; ++IdxRow)
                {
                    if (Math.Abs(XVal[IdxRow]) < 1e-30f)
                    {
                        ZVal[IdxRow] = 0.0f;
                    }
                }
            });
        }

        /*
         * Set elements of z to zero if the corresponding elements of x is zero.
         */
        public static void ResetVectorSparsePattern(DenseColumnVector z, DenseColumnVector x)
        {
            if (z.Dim != x.Dim)
            {
                throw new Exception("Dimension mismatch.");
            }
            var xVal = x.VectorValue;
            var zVal = z.VectorValue;
            int zDim = z.Dim;
            for (int IdxRow = 0; IdxRow < zDim; ++IdxRow)
            {
                if (Math.Abs(xVal[IdxRow])<1e-30f)
                {
                    zVal[IdxRow] = 0.0f;
                }
            }
        }
    }

    public static class Projection
    {
        /*
         * Project each column of the input matrix X onto the orthogonal space of the plane: 1^T x = 1
         * X = Proj(X) or Z = Proj(Z)
         */
        public static void ProjCols2OrthogonalSimplexPlane(DenseMatrix X)
        {
            DenseRowVector TmpDenseRowVec = new DenseRowVector(X.nCols);
            MatrixOperation.VerticalSumMatrix(TmpDenseRowVec, X);
            MatrixOperation.ScalarMultiplyVector(TmpDenseRowVec, 1.0f / ((float)X.nRows));
            MatrixOperation.bsxfunMatrixSubtractVector(X, X, TmpDenseRowVec);
        }
        public static void ProjCols2OrthogonalSimplexPlane(DenseMatrix Z, DenseMatrix X)
        {
            if (Z.nCols != X.nCols || Z.nRows != X.nRows)
            {
                throw new Exception("Dimension mismatch.");
            }
            DenseRowVector TmpDenseRowVec = new DenseRowVector(X.nCols);
            MatrixOperation.VerticalSumMatrix(TmpDenseRowVec, X);
            MatrixOperation.ScalarMultiplyVector(TmpDenseRowVec, 1.0f / ((float)X.nRows));
            MatrixOperation.bsxfunMatrixSubtractVector(Z, X, TmpDenseRowVec);
        }
        public static void ProjCols2OrthogonalSimplexPlane(SparseMatrix Z, SparseMatrix X)
        {
            if (Z.nCols != X.nCols || Z.nRows != X.nRows)
            {
                throw new Exception("Dimension mismatch.");
            }
            DenseRowVector TmpDenseRowVec = new DenseRowVector(X.nCols);
            MatrixOperation.VerticalSumMatrix(TmpDenseRowVec, X);
            MatrixOperation.ScalarMultiplyVector(TmpDenseRowVec, 1.0f / ((float)X.nRows));
            MatrixOperation.bsxfunMatrixSubtractVector(Z, X, TmpDenseRowVec);
        }

        /*
         * Project each column of the input matrix X onto the affine space defined by 1^T x = 1
         */
        public static void ProjCols2SimplexPlane(DenseMatrix X)
        {
            DenseRowVector TmpDenseRowVec = new DenseRowVector(X.nCols);
            MatrixOperation.VerticalSumMatrix(TmpDenseRowVec, X);
            MatrixOperation.ScalarMultiplyVector(TmpDenseRowVec, 1.0f / ((float)X.nRows));
            MatrixOperation.bsxfunMatrixSubtractVector(X, X, TmpDenseRowVec);
            MatrixOperation.ScalarAddMatrix(X, 1.0f / ((float)X.nRows));
        }
        public static void ProjCols2SimplexPlane(DenseColumnVector x)
        {
            float Mean = x.VectorValue.Average();
            MatrixOperation.ScalarAddVector(x, ((-1.0f) * Mean + 1.0f / ((float)x.Dim)));
        }
        public static void ProjCols2SimplexPlane(DenseMatrix Z, DenseMatrix X)
        {
            if (Z.nCols != X.nCols || Z.nRows != X.nRows)
            {
                throw new Exception("Dimension mismatch.");
            }
            DenseRowVector TmpDenseRowVec = new DenseRowVector(X.nCols);
            MatrixOperation.VerticalSumMatrix(TmpDenseRowVec, X);
            MatrixOperation.ScalarMultiplyVector(TmpDenseRowVec, 1.0f / ((float)X.nRows));
            MatrixOperation.bsxfunMatrixSubtractVector(Z, X, TmpDenseRowVec);
            MatrixOperation.ScalarAddMatrix(Z, 1.0f / ((float)X.nRows));
        }
    }

    public class SparseColumnVector
    {
        public int[] Key = null;
        public float[] Val = null;
        public int Dim = 0;
        public int nNonzero = 0;

        public SparseColumnVector()
        {
        }

        public SparseColumnVector(int Dimension)
        {
            Dim = Dimension;
        }

        public SparseColumnVector(int NumNonzero, int Dimension)
        {
            nNonzero = NumNonzero;
            Dim = Dimension;
            Key = new int[nNonzero];
            Val = new float[nNonzero];
        }

        // This constructor initialize the sparse column vector by deep copy
        public SparseColumnVector(SparseColumnVector SourceSparseColumnVector)
        {
            Dim = SourceSparseColumnVector.Dim;
            nNonzero = SourceSparseColumnVector.nNonzero;
            Key = new int[nNonzero];
            Val = new float[nNonzero];
            DeepCopySparseColumnVectorFrom(SourceSparseColumnVector);
        }

        public void DeepCopySparseColumnVectorFrom(SparseColumnVector SourceSparseColumnVector)
        {
            if (Dim != SourceSparseColumnVector.Dim || nNonzero != SourceSparseColumnVector.nNonzero)
            {
                throw new Exception("Dimension or nNonzero mismatch between the source and target SparseColumnVectors");
            }

            Array.Copy(SourceSparseColumnVector.Key, Key, nNonzero);
            Array.Copy(SourceSparseColumnVector.Val, Val, nNonzero);
        }

        // Set the sparse pattern of the sparse column vector
        public void SetSparsePattern(int[] SourceKey)
        {
            if (SourceKey.Max()>Dim)
            {
                throw new Exception("The dimension of the SourceKey exceeds the original dimension of the sparse column vector.");
            }
            nNonzero = SourceKey.Length;
            Key = SourceKey;
            Val = new float[nNonzero];
        }

        // Get the L_inf norm of the column
        public float LInfNorm()
        {
            float Max = Val.Max();
            float Min = Val.Min();
            return Math.Max(Math.Abs(Max), Math.Abs(Min));
        }

        // Sum of all the values
        public float Sum()
        {
            float Sum = 0.0f;
            for (int IdxRow = 0; IdxRow < nNonzero; ++IdxRow)
            {
                Sum += Val[IdxRow];
            }
            return Sum;
        }
    }

    public class SparseMatrix
    {
        public SparseColumnVector[] SparseColumnVectors = null;
        public int nRows = 0;
        public int nCols = 0;
        public bool flag_SameSparsePatterForAllColumns = false;
        public int[] SparsePatternOfEachColumn = null;

        public SparseMatrix(int NumRows, int NumCols)
        {
            nRows = NumRows;
            nCols = NumCols;
            SparseColumnVectors = new SparseColumnVector[nCols];
            for (int IdxCol = 0; IdxCol < nCols; IdxCol++)
            {
                SparseColumnVectors[IdxCol] = new SparseColumnVector(nRows);
            }            
        }

        public SparseMatrix(int NumRows, int NumCols, bool SameSparsePatternForAllColumn)
        {
            nRows = NumRows;
            nCols = NumCols;
            SparseColumnVectors = new SparseColumnVector[nCols];
            for (int IdxCol = 0; IdxCol < nCols; IdxCol++)
            {
                SparseColumnVectors[IdxCol] = new SparseColumnVector(nRows);
            }
            flag_SameSparsePatterForAllColumns = SameSparsePatternForAllColumn;
            if (flag_SameSparsePatterForAllColumns)
            {
                SparsePatternOfEachColumn = new int[nRows];
            }

        }

        // This constructor performs deep copy from the source sparse matrix
        public SparseMatrix(SparseMatrix SourceSparseMatrix)
        {
            nRows = SourceSparseMatrix.nRows;
            nCols = SourceSparseMatrix.nCols;
            SparseColumnVectors = new SparseColumnVector[nCols];
            for (int IdxCol = 0; IdxCol < nCols; IdxCol++)
            {
                SparseColumnVectors[IdxCol] = new SparseColumnVector(SourceSparseMatrix.SparseColumnVectors[IdxCol]);                
            }
            flag_SameSparsePatterForAllColumns = SourceSparseMatrix.flag_SameSparsePatterForAllColumns;
            if (flag_SameSparsePatterForAllColumns)
            {
                SparsePatternOfEachColumn = new int[nRows];
                Array.Copy(SourceSparseMatrix.SparsePatternOfEachColumn, SparsePatternOfEachColumn, nRows);
            }
        }

        public void FillColumn(int[] InputKey, float[] InputVal, int IdxCol)
        {
            if (InputKey.Length != InputVal.Length)
            {
                throw new Exception("Number of Keys != Number of Values");
            }
            int NumNonzero = InputKey.Length;
            SparseColumnVectors[IdxCol].nNonzero = NumNonzero;
            SparseColumnVectors[IdxCol].Key = InputKey;
            SparseColumnVectors[IdxCol].Val = InputVal;
        }


        // Get the desired columns from this sparse matrix to form a new (sub-)sparse matrix
        public void GetColumns(SparseMatrix SubMatrix, int[] IdxColumns)
        {
            if (SubMatrix.nCols != IdxColumns.Length)
            {
                throw new Exception("Number of desired columns is not equal to the target SubMatrix!");
            }
            for (int IdxCol = 0; IdxCol < IdxColumns.Length; IdxCol++)
            {
                Array.Copy(SparseColumnVectors, IdxColumns[IdxCol], SubMatrix.SparseColumnVectors, IdxCol, 1);
            }
        }
        public SparseMatrix GetColumns(int[] IdxColumns)
        {
            SparseMatrix SubMatrix = new SparseMatrix(nRows, IdxColumns.Length);
            if (SubMatrix.nCols != IdxColumns.Length)
            {
                throw new Exception("Number of desired columns is not equal to the target SubMatrix!");
            }
            for (int IdxCol = 0; IdxCol < IdxColumns.Length; IdxCol++)
            {
                Array.Copy(SparseColumnVectors, IdxColumns[IdxCol], SubMatrix.SparseColumnVectors, IdxCol, 1);
            }
            return SubMatrix;
        }
        public void SetSparsePatternForAllColumn(int[] SourceKey)
        {
            for (int IdxCol = 0; IdxCol < nCols; IdxCol++)
            {
                SparseColumnVectors[IdxCol].SetSparsePattern(SourceKey);
            }
            // An auxiliary array stores the index for each key
            flag_SameSparsePatterForAllColumns = true;
            for(int Idx = 0; Idx < SourceKey.Length; Idx++)
            {
                SparsePatternOfEachColumn[SourceKey[Idx]] = Idx;
            }
        }

        public void SetAllValuesToZero()
        {
            Parallel.For(0, nCols, IdxCol =>
            {
                Array.Clear(SparseColumnVectors[IdxCol].Val, 0, SparseColumnVectors[IdxCol].nNonzero);
            });
        }

        public float MaxAbsValue()
        {
            float[] MaxAbsColValue = new float[nCols];
            for (int IdxCol = 0; IdxCol < nCols; IdxCol++)
            {
                MaxAbsColValue[IdxCol] = SparseColumnVectors[IdxCol].LInfNorm();
            }
            return MaxAbsColValue.Max();
        }

        public int[] GetHorizontalUnionSparsePattern()
        {
            //int[] SparsePattern = null;
            //if (nCols > 1)
            //{
            //    for (int Idx = 0; Idx < nCols; Idx++)
            //    {
            //        if (Idx == 0)
            //        {
            //            SparsePattern = new int[SparseColumnVectors[0].nNonzero];
            //            Array.Copy(SparseColumnVectors[0].Key, SparsePattern, SparseColumnVectors[0].nNonzero);
            //        }
            //        else
            //        {
            //            SparsePattern = SparsePattern.Union(SparseColumnVectors[Idx].Key).ToArray<int>();
            //        }
            //    }
            //}
            //else
            //{
            //    SparsePattern = new int[SparseColumnVectors[0].nNonzero];
            //    Array.Copy(SparseColumnVectors[0].Key, SparsePattern, SparseColumnVectors[0].nNonzero);
            //}

            // Newer version
            bool[] ElemMap = new bool[nRows];
            int[] KeyPool = new int[nRows];
            int IdxUniqueKey = 0;
            for (int IdxCol = 0; IdxCol < nCols; ++IdxCol )
            {
                int nNz = SparseColumnVectors[IdxCol].nNonzero;
                var ColKey = SparseColumnVectors[IdxCol].Key;
                for(int IdxRow = 0; IdxRow < nNz; ++IdxRow)
                {
                    if (!ElemMap[ColKey[IdxRow]])
                    {
                        KeyPool[IdxUniqueKey] = ColKey[IdxRow];
                        ElemMap[ColKey[IdxRow]] = true;
                        ++IdxUniqueKey;
                    }
                }
            }
            int[] SparsePattern = new int[IdxUniqueKey];
            Array.Copy(KeyPool, SparsePattern, IdxUniqueKey);
            return SparsePattern;
        }

        public int[] IndexOfVerticalMax()
        {
            int[] IdxVerticalMax = new int[nCols];
            float[] ValVerticalMax = new float[nCols];

            for (int IdxCol = 0; IdxCol < nCols; IdxCol++)
            {
                ValVerticalMax[IdxCol] = SparseColumnVectors[IdxCol].Val[0];
                IdxVerticalMax[IdxCol] = SparseColumnVectors[IdxCol].Key[0];
                for (int IdxRow = 0; IdxRow < SparseColumnVectors[IdxCol].nNonzero; IdxRow++)
                {
                    if (SparseColumnVectors[IdxCol].Val[IdxRow] > ValVerticalMax[IdxCol])
                    {
                        ValVerticalMax[IdxCol] = SparseColumnVectors[IdxCol].Val[IdxRow];
                        IdxVerticalMax[IdxCol] = SparseColumnVectors[IdxCol].Key[IdxRow];
                    }
                }
            }

            return IdxVerticalMax;
        }

    }

    public class DenseMatrix
    {
        public int nRows = 0;
        public int nCols = 0;

        public DenseColumnVector[] DenseMatrixValue = null;
        public DenseRowVector[] DenseMatrixValuePerRow = null;

        public bool isPerColumn = true;

        public DenseMatrix()
        {
        }

        public DenseMatrix(int NumRows, int NumCols)
        {
            nRows = NumRows;
            nCols = NumCols;
            DenseMatrixValue = new DenseColumnVector[nCols];
            for (int IdxCol=0;IdxCol<nCols;IdxCol++)
            {
                DenseMatrixValue[IdxCol] = new DenseColumnVector(nRows);
            }
        }

        public DenseMatrix(int NumRows, int NumCols, bool IsPerColumn)
        {
            if (IsPerColumn)
            {
                nRows = NumRows;
                nCols = NumCols;
                isPerColumn = true;
                DenseMatrixValue = new DenseColumnVector[nCols];
                for (int IdxCol = 0; IdxCol < nCols; IdxCol++)
                {
                    DenseMatrixValue[IdxCol] = new DenseColumnVector(nRows);
                }
            }
            else
            {
                nRows = NumRows;
                nCols = NumCols;
                isPerColumn = false;
                DenseMatrixValuePerRow = new DenseRowVector[nRows];
                for (int IdxRow = 0; IdxRow < nRows; IdxRow++)
                {
                    DenseMatrixValuePerRow[IdxRow] = new DenseRowVector(nCols);
                }
            }
        }

        public DenseMatrix(int NumRows, int NumCols, float Value)
        {
            nRows = NumRows;
            nCols = NumCols;
            DenseMatrixValue = new DenseColumnVector[nCols];
            for (int IdxCol = 0; IdxCol < nCols; IdxCol++)
            {
                DenseMatrixValue[IdxCol] = new DenseColumnVector(nRows,Value);
            }
        }

        public DenseMatrix(DenseMatrix SourceMatrix)
        {
            nRows = SourceMatrix.nRows;
            nCols = SourceMatrix.nCols;
            DenseMatrixValue = new DenseColumnVector[nCols];
            for (int IdxCol = 0; IdxCol < nCols; IdxCol++)
            {
                DenseMatrixValue[IdxCol] = new DenseColumnVector(nRows);
                DenseMatrixValue[IdxCol].DeepCopyFrom(SourceMatrix.DenseMatrixValue[IdxCol]);
            }
        }        
        
        public void FillRandomValues()
        {
            Random random = new Random();
            for (int IdxCol = 0; IdxCol < nCols; ++IdxCol) 
            {
                var ColVal = DenseMatrixValue[IdxCol].VectorValue;
                for (int IdxRow = 0; IdxRow < nRows; ++IdxRow)
                {
                    ColVal[IdxRow] = (float)random.NextDouble();
                }
            }
        }

        public void FillRandomValues(int RandomSeed)
        {
            Random random = new Random(RandomSeed);
            for (int IdxRow = 0; IdxRow < nRows; IdxRow++)
            {
                for (int IdxCol = 0; IdxCol < nCols; IdxCol++)
                {
                    DenseMatrixValue[IdxCol].VectorValue[IdxRow] = (float)random.NextDouble();
                }
            }
        }

        public void FillValue(float Value)
        {
            for (int IdxRow = 0; IdxRow < nRows; IdxRow++)
            {
                for (int IdxCol = 0; IdxCol < nCols; IdxCol++)
                {
                    DenseMatrixValue[IdxCol].VectorValue[IdxRow] = Value;
                }
            }
        }
        public void ClearValue()
        {
            Parallel.For(0, nCols, IdxCol =>
                {
                    Array.Clear(DenseMatrixValue[IdxCol].VectorValue, 0, DenseMatrixValue[IdxCol].VectorValue.Length);
                }
                );
        }

        public void FillColumn(float[] InputColumn, int IdxCol)
        {
            if (nRows!=InputColumn.Length)
            {
                throw new Exception("Dimension mismatch.");
            }
            if (IdxCol < 0 || IdxCol >= nCols)
            {
                throw new Exception("IdxCol out of the range of the DenseMatrix.");
            }
            DenseMatrixValue[IdxCol].VectorValue = InputColumn;            
        }

        public float MaxAbsValue()
        {
            float[] MaxAbsColValue = new float[nCols];
            for (int IdxCol = 0; IdxCol < nCols; IdxCol++)
            {
                MaxAbsColValue[IdxCol] = DenseMatrixValue[IdxCol].LInfNorm();
            }
            return MaxAbsColValue.Max();
        }

        public float Min()
        {
            float[] MinColValue = new float[nCols];
            for (int IdxCol = 0; IdxCol < nCols; IdxCol++)
            {
                MinColValue[IdxCol] = DenseMatrixValue[IdxCol].VectorValue.Min();
            }
            return MinColValue.Min();
        }
        public float Max()
        {
            float[] MaxColValue = new float[nCols];
            for (int IdxCol = 0; IdxCol < nCols; IdxCol++)
            {
                MaxColValue[IdxCol] = DenseMatrixValue[IdxCol].VectorValue.Max();
            }
            return MaxColValue.Max();
        }

        public int[] IndexOfVerticalMax()
        {
            int[] IdxVerticalMax = new int[nCols];
            float[] ValVerticalMax = new float[nCols];

            for (int IdxCol = 0; IdxCol < nCols; IdxCol++ )
            {
                ValVerticalMax[IdxCol] = DenseMatrixValue[IdxCol].VectorValue[0];
                IdxVerticalMax[IdxCol] = 0;
                for (int IdxRow = 1; IdxRow < nRows; IdxRow++)
                {
                    if (DenseMatrixValue[IdxCol].VectorValue[IdxRow] > ValVerticalMax[IdxCol])
                    {
                        ValVerticalMax[IdxCol] = DenseMatrixValue[IdxCol].VectorValue[IdxRow];
                        IdxVerticalMax[IdxCol] = IdxRow;
                    }
                }
            }
            
            return IdxVerticalMax;
        }

        
        
    }

    public class DenseColumnVector
    {
        public int Dim = 0;
        public float[] VectorValue = null;

        public DenseColumnVector()
        {
        }

        // This constructor allocate the memory of the specified size
        public DenseColumnVector(int Dimension)
        {
            Dim = Dimension;
            VectorValue = new float[Dim];
        }

        // This constructor initialize the vector with the same value
        public DenseColumnVector(int Dimension, float Value)
        {
            Dim = Dimension;
            VectorValue = new float[Dim];
            FillValue(Value);
        }

        public void FillValue(float Value)
        {
            for (int Idx=0;Idx<Dim;++Idx)
            {
                VectorValue[Idx] = Value;
            }
        }

        public void DeepCopyFrom(DenseColumnVector SourceVector)
        {
            // Check dimension
            if (Dim != SourceVector.Dim)
            {
                throw new Exception("Dimension mismatch during deep copy of DenseColumnVector.");
            }
            // Deep copy of the float array
            Array.Copy(SourceVector.VectorValue,VectorValue,Dim);
        }

        public float MaxValue()
        {
            return VectorValue.Max();
        }

        public float Sum()
        {
            return VectorValue.Sum();
        }

        public float L1Norm()
        {
            float z = 0;
            for (int Idx = 0; Idx < Dim; Idx++)
            {
                z += Math.Abs(VectorValue[Idx]);
            }
            return z;
        }

        public float LInfNorm()
        {
            float Max = VectorValue.Max();
            float Min = VectorValue.Min();
            return Math.Max(Math.Abs(Max), Math.Abs(Min));
        }
    }

    public class DenseRowVector
    {
        public int Dim = 0;
        public float[] VectorValue = null;

        public DenseRowVector()
        {
        }

        // This constructor allocate the memory of the specified size
        public DenseRowVector(int Dimension)
        {
            Dim = Dimension;
            VectorValue = new float[Dim];
        }

        // This constructor initialize the vector with the same value
        public DenseRowVector(int Dimension, float Value)
        {
            Dim = Dimension;
            VectorValue = new float[Dim];
            FillValue(Value);
        }

        public DenseRowVector(DenseRowVector SourceVector)
        {
            Dim = SourceVector.Dim;
            VectorValue = new float[Dim];
            DeepCopyFrom(SourceVector);
        }

        public void FillValue(float Value)
        {
            for (int Idx = 0; Idx < Dim; Idx++)
            {
                VectorValue[Idx] = Value;
            }
        }

        public void DeepCopyFrom(DenseRowVector SourceVector)
        {
            // Check dimension
            if (Dim != SourceVector.Dim)
            {
                throw new Exception("Dimension mismatch during deep copy of DenseRowVector.");
            }
            // Deep copy of the float array
            Array.Copy(SourceVector.VectorValue,VectorValue,Dim);
        }

        public float Sum()
        {
            return VectorValue.Sum();
        }

    }
}
