#!/bin/sh

# You can control the resources and scheduling with '#SBATCH' settings
# (see 'man sbatch' for more information on setting these parameters)

# The default partition is the 'general' partition
#SBATCH --partition=general

# The default Quality of Service is the 'short' QoS (maximum run time: 4 hours)
#SBATCH --qos=medium

# The default run (wall-clock) time is 1 minute
#SBATCH --time=3:00:00

# The default number of parallel tasks per job is 1
#SBATCH --ntasks=1

# The default number of CPUs per task is 1 (note: CPUs are always allocated to jobs per 2)
# Request 1 CPU per active thread of your program (assume 1 unless you specifically set this)
#SBATCH --cpus-per-task=2

# The default memory per node is 1024 megabytes (1GB) (for multiple tasks, specify --mem-per-cpu instead)16384
#SBATCH --mem=20480

# Request a GPU
#SBATCH --gres=gpu:p100:1

# Set mail type to 'END' to receive a mail when the job finishes
# Do not enable mails when submitting large numbers (>20) of jobs at once
#SBATCH --mail-type=BEGIN

# Measure GPU usage of your job (initialization)
#previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

# Use this simple command to check that your sbatch settings are working (it should show the GPU that you requested)
/usr/bin/nvidia-smi -L

# Use this simple command to check that your sbatch settings are working
/usr/bin/scontrol show job -d "$SLURM_JOB_ID"

# Your job commands go below here

# Uncomment these lines and adapt them to load the software that your job requires
module use /opt/insy/modulefiles
module load cuda/11.5 cudnn/11.5-8.3.0.98

source /home/nfs/arkhan/venv0/bin/activate
# pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# pip install comet_ml --upgrade --quiet
# pip install transformers sklearn pandas

# pip install -r /tudelft.net/staff-umbrella/tcr/requirements.txt
# Computations should be started with 'srun'. For example: -m torch.distributed.launch --nproc_per_node 2 
srun python eletcr_ep2.py