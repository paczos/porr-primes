import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.tools
# import pycuda.cumath as maths
from pycuda.cumath import fmod
from pycuda.compiler import SourceModule

import time

import numpy
whole_program = time.time()


lower_bound = 1739834324567
upper_bound = 1739834324667
diff = (upper_bound - lower_bound)-1
d = int((upper_bound - lower_bound)/20)

d = numpy.int32(d)

nums = list(range (lower_bound, upper_bound))
nums = numpy.asarray(nums)
nums = nums.astype(numpy.int64)
res = numpy.zeros(diff)
res = res.astype(numpy.int64)

nums_gpu = cuda.mem_alloc(nums.nbytes)
res_gpu = cuda.mem_alloc(res.nbytes)
# d_gpu = cuda.mem_alloc(d.nbytes)

# arr_gpu = cuda.mem_alloc(arr.nbytes)

cuda.memcpy_htod(nums_gpu, nums)
cuda.memcpy_htod(res_gpu, res)


mod = SourceModule("""
  __global__ void primify (long long int *a, long long int *res, int diff)
  {
    int idx = threadIdx.x;
    int d = diff;
    int k = 0;
    int j = 0;
    long long int i = 0;
    long long int sq = 0;
    int m = idx*5;

    for (k = 0; k<=d; ++k){
        m = k+idx*5;
        i = 0;
        j = 0;
        sq = sqrtf(a[m]);
        for (i = 2; i<=sq; ++i){
            int rem = remainder(a[m], i);
            if(rem == 0){
                j++;
            }

        }
        if(j == 0){
            res[m] = a[m];
        }

    }

  }
  """)

func = mod.get_function("primify")

func( nums_gpu,res_gpu, d, block=(20,1,1))
res_fromgpu = numpy.empty_like(res)
#
cuda.memcpy_dtoh(res_fromgpu, res_gpu)

res = res_fromgpu[res_fromgpu != 0]
print (res)
program_time = time.time() - whole_program


print ("total time:  % s seconds" % program_time)
