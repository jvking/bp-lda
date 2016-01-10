using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using LinearAlgebra;

namespace Common
{
	public static class Statistics
	{
		/*
        * Random permulation of 1 to n
        */
		public static int[] RandPerm(int N)
		{
			int[] IdxPerm = new int[N];
			for (int Idx = 0; Idx < N; Idx++)
			{
				IdxPerm[Idx] = Idx;
			}

			Random random = new Random();

			int n = N;
			while (n > 1)
			{
				n--;
				int i = random.Next(n + 1);
				int tmp = IdxPerm[i];
				IdxPerm[i] = IdxPerm[n];
				IdxPerm[n] = tmp;
			}

			return IdxPerm;
		}


	}
}
