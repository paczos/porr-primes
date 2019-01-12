import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.tools
# import pycuda.cumath as maths
from pycuda.cumath import fmod
from pycuda.compiler import SourceModule

import time

import numpy as np
whole_program = time.time()


lower_bound = 1
upper_bound = 2050
diff = upper_bound - lower_bound
d = diff
d = np.int32(d)

nums = list(range (lower_bound, upper_bound))
numki = np.asarray(nums)
numki = numki.astype(np.int32)
res = np.zeros(diff)
res = res.astype(np.int64)


nums_gpu = cuda.mem_alloc(numki.nbytes)
res_gpu = cuda.mem_alloc(res.nbytes)

# arr_gpu = cuda.mem_alloc(arr.nbytes)

cuda.memcpy_htod(nums_gpu, numki)
cuda.memcpy_htod(res_gpu, res)


mod = SourceModule("""
  __global__ void primify (long long int *a, long long int *res, int *diff)
  {
    long long int idx = threadIdx.x;
    if(idx == 0) {
            printf(" ");

    }
    long long int j = 0;
    long long int i = 0;
    long long int sq = sqrtf(a[idx]);
    for (i = 2; i<=sq; ++i){
        long long int rem = remainder(a[idx], i);

        if(rem == 0){
            j++;
        }

    }
    if(j == 0){
        int l = 0;
        for(l; l<sq; ++l){
            if(res[l] == 0){
                res[l] = a[idx];
                break;
            }
        }
    }


  }
  __global__ void test (int *a)
  {
    int idx = threadIdx.x+ blockIdx.x * blockDim.x;

    printf("  %d   ", a[idx]);
   }
  """)

func = mod.get_function("test")
start = time.time()
#
# bdim = (16, 16, 1)
# dx, mx = divmod(cols, bdim[0])
# dy, my = divmod(rows, bdim[1])
#
# gdim = ( (dx + (mx>0)) * bdim[0], (dy + (my>0)) * bdim[1]) )


func( nums_gpu, block=(1024,1,1), grid=(2,1))
gpu_time = time.time() - start
print ("GPU operations took  % seconds" % gpu_time)
res_fromgpu = np.empty_like(res)
#
cuda.memcpy_dtoh(res_fromgpu, res_gpu)
# for x in range(0,diff):
#     if res_fromgpu[x] == 0:
#         res_fromgpu = res_fromgpu[0:x]
#         break
#
# print (res_fromgpu)
program_time = time.time() - whole_program

print ("total time:  % seconds" % program_time)
