f="result.csv"
echo "gpu_or_cpu,num_particles,block_size,num_iteration,execution_time" > ${f}
for num_particles in 10 50 100 500 1000 5000 10000 50000 100000 500000 1000000 5000000 10000000
do
  for block_size in 16 32 64 128 256
  do
    ./exercise_3.out ${num_particles} ${block_size} ${f}
  done
done
