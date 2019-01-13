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
diff = upper_bound - lower_bound
d = diff
d = numpy.int32(d)

nums = list(range (lower_bound, upper_bound))
nums = numpy.asarray(nums)
nums = nums.astype(numpy.int64)
res = numpy.zeros(diff)
res = res.astype(numpy.int64)

nums_gpu = cuda.mem_alloc(nums.nbytes)
res_gpu = cuda.mem_alloc(res.nbytes)

# arr_gpu = cuda.mem_alloc(arr.nbytes)

cuda.memcpy_htod(nums_gpu, nums)
cuda.memcpy_htod(res_gpu, res)


mod = SourceModule("""
  __global__ void primify (long long int *a, long long int *res, int *diff)
  {
    long long int idx = threadIdx.x;
    if(idx == 0) {
            printf(" " );
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
  """)

func = mod.get_function("primify")

start = time.time()
func( nums_gpu,res_gpu, d, block=(diff,1,1))
gpu_time = time.time() - start
print ("GPU operations took  % s seconds" % gpu_time)
res_fromgpu = numpy.empty_like(res)
#
cuda.memcpy_dtoh(res_fromgpu, res_gpu)
for x in range(0,diff):
    if res_fromgpu[x] == 0:
        res_fromgpu = res_fromgpu[0:x]
        break

print (res_fromgpu)
program_time = time.time() - whole_program

print ("total time:  % s seconds" % program_time)
