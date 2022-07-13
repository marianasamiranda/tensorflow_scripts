#!/bin/bash


WORKSPACE=$(dirname $(dirname $(realpath $0)))
DATA_DIR="/scratch1/08486/mmiranda/mysharedirectory"

HH=$(hostname)
echo $HH 

echo "cp ${DATA_DIR}/objects/44g.zip /dev/shm"
cp "${DATA_DIR}/objects/44g.zip" /dev/shm

echo "cp ${DATA_DIR}/objects/7g.zip /dev/shm"
cp "${DATA_DIR}/objects/7g.zip" /dev/shm

echo "cp ${DATA_DIR}/objects/6g.zip /dev/shm"
cp "${DATA_DIR}/objects/6g.zip" /dev/shm

echo "cp ${DATA_DIR}/objects/3g /dev/shm"
cp "${DATA_DIR}/objects/3g" /dev/shm

echo "module load cuda/10.1 cudnn/7.6.5 nccl/2.5.6"
#module load intel/19.1.1 
#module load impi/19.0.9
#module load python3/3.7.0
module load cuda/10.1
module load cudnn/7.6.5
module load nccl/2.5.6

echo "module load remora"
module load remora

module list

cd "${WORKSPACE}/scripts"

EPOCHS=1
BATCH_SIZE=256
DATE="$(date +%Y_%m_%d-%H_%M)"
TARGET_DIR="/tmp"

# 100g

DATASET="${DATA_DIR}/imagenet_processed/100g_tfrecords"
 
for i in {1..1}; do
  RUN_DIR="${TARGET_DIR}/lenet-100g-bs${BATCH_SIZE}-ep${EPOCHS}-${DATE}"
  #./train_global.sh -j 0 -o -m lenet -b $BATCH_SIZE -e $EPOCHS -g 4 -i autotune -l -d "$DATASET" -r $RUN_DIR > aaaa1.txt 
  #./train_global.sh -o -m lenet -b $BATCH_SIZE -e $EPOCHS -g 4 -i autotune -l -v -d "$DATASET" -r $RUN_DIR -j 0 
  #strace -cf -o strace_tensor_dist_e3_c5_b256.5_n2.txt ./train_global.sh -j 2 -o -m lenet -b $BATCH_SIZE -e $EPOCHS -g 4 -i autotune -l -v -d "$DATASET" -r $RUN_DIR 
  ./train_global.sh -j 2 -o -m lenet -b $BATCH_SIZE -e $EPOCHS -g 4 -i autotune -l -v -d "$DATASET" -r $RUN_DIR 
  sleep 10
  mv "remora_${SLURM_JOB_ID}"  $RUN_DIR

#  RUN_DIR="${TARGET_DIR}/alexnet-100g-bs${BATCH_SIZE}-ep${EPOCHS}-${DATE}"
#  remora ./train.sh -o -m alexnet -b $BATCH_SIZE -e $EPOCHS -g 4 -i autotune -l -v -d "$DATASET" -r $RUN_DIR
#  sleep 10
#  mv "remora_${SLURM_JOB_ID}" $RUN_DIR
#
#  RUN_DIR="${TARGET_DIR}/resnet-100g-bs${BATCH_SIZE}-ep${EPOCHS}-${DATE}"
#  remora ./train.sh -o -m resnet -b $BATCH_SIZE -e $EPOCHS -g 4 -i autotune -l -v -d "$DATASET" -r $RUN_DIR
#  sleep 10  
#  mv "remora_${SLURM_JOB_ID}" $RUN_DIR
done

# 200g

DATASET="${DATA_DIR}/imagenet_processed/200g_2048_tfrecords"

#for i in {1..1}; do
#  RUN_DIR="${TARGET_DIR}/lenet-200g-bs${BATCH_SIZE}-ep${EPOCHS}-${DATE}"
#  remora ./train.sh -o -m lenet -b $BATCH_SIZE -e $EPOCHS -g 4 -i autotune -v -d "$DATASET" -r $RUN_DIR -s 2048
#  sleep 10
#  mv "remora_${SLURM_JOB_ID}" $RUN_DIR
# 
#  RUN_DIR="${TARGET_DIR}/alexnet-200g-bs${BATCH_SIZE}-ep${EPOCHS}-${DATE}"
#  remora ./train.sh -o -m alexnet -b $BATCH_SIZE -e $EPOCHS -g 4 -i autotune -v -d "$DATASET" -r $RUN_DIR -s 2048
#  sleep 10
#  mv "remora_${SLURM_JOB_ID}" $RUN_DIR
# 
#  RUN_DIR="${TARGET_DIR}/resnet-200g-bs${BATCH_SIZE}-ep${EPOCHS}-${DATE}"
#  remora ./train.sh -o -m resnet -b $BATCH_SIZE -e $EPOCHS -g 4 -i autotune -v -d "$DATASET" -r $RUN_DIR -s 2048
#  sleep 10
#  mv "remora_${SLURM_JOB_ID}" $RUN_DIR
#done
