#!/usr/bin/env bash
set -ux

USE_PGLE=${1:-0}
MODEL_CONFIG=${2:-"src/maxdiffusion/configs/base_flux_schnell.yml"}

export HF_TOKEN=${HF_TOKEN:-""}
export CODE_DIR=${CODE_DIR:-"."}
export LOG_DIR=${LOG_DIR:-"."}

AG_COMBINE_THRESHOLD=134217728
AR_COMBINE_THRESHOLD=134217728
RS_COMBINE_THRESHOLD=67108864

export BASE_XLA_FLAGS="
		--xla_gpu_all_reduce_combine_threshold_bytes=$AR_COMBINE_THRESHOLD --xla_gpu_all_gather_combine_threshold_bytes=$AG_COMBINE_THRESHOLD
                --xla_gpu_reduce_scatter_combine_threshold_bytes=$RS_COMBINE_THRESHOLD
                --xla_gpu_enable_pipelined_all_gather=true
                --xla_gpu_enable_pipelined_reduce_scatter=true
                --xla_gpu_enable_pipelined_all_reduce=true
                --xla_gpu_enable_while_loop_double_buffering=true
                --xla_gpu_enable_all_gather_combine_by_dim=false
                --xla_gpu_enable_reduce_scatter_combine_by_dim=false
                --xla_disable_hlo_passes=rematerialization
                --xla_gpu_enable_command_buffer= 
                --xla_gpu_enable_triton_gemm=false
                --xla_dump_to=${LOG_DIR}/xla_dump
                --xla_gpu_exhaustive_tiling_search=true
                --xla_gpu_graph_level=0"
                # --xla_gpu_experimental_parallel_collective_overlap_limit=32
                # --xla_gpu_enable_nccl_user_buffers=true
                # --xla_dump_hlo_pass_re=.*
                # --xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL,CUDNN,COLLECTIVES,CONDITIONAL,DYNAMIC_SLICE_FUSION"

# # no overlap
# export BASE_XLA_FLAGS="
# 		--xla_gpu_all_reduce_combine_threshold_bytes=$AR_COMBINE_THRESHOLD --xla_gpu_all_gather_combine_threshold_bytes=$AG_COMBINE_THRESHOLD
#                 --xla_gpu_reduce_scatter_combine_threshold_bytes=$RS_COMBINE_THRESHOLD
#                 --xla_gpu_enable_pipelined_all_gather=false
#                 --xla_gpu_enable_pipelined_reduce_scatter=false
#                 --xla_gpu_enable_pipelined_all_reduce=false
#                 --xla_gpu_enable_while_loop_double_buffering=true
#                 --xla_gpu_enable_all_gather_combine_by_dim=false
#                 --xla_gpu_enable_reduce_scatter_combine_by_dim=false
#                 --xla_disable_hlo_passes=rematerialization
#                 --xla_gpu_enable_command_buffer= 
#                 --xla_gpu_graph_level=0"

# export KERAS_BACKEND="jax"
# #export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump"
# # export JAX_DEFAULT_PRNG_IMPL="rbg"
# export JAX_SPMD_MODE="allow_all"
# export JAX_PLATFORMS="cuda"
# #export JAX_LOG_COMPILES="True"
# #export TPU_STDERR_LOG_LEVEL="0"
# #export TPU_MIN_LOG_LEVEL="0"
# #export TF_CPP_MIN_LOG_LEVEL="0"
# export HUGGINGFACE_HUB_VERBOSITY="warning"
# export TRANSFORMERS_NO_ADVISORY_WARNINGS="1"
# export TOKENIZERS_PARALLELISM="1"
# #export JAX_TRACEBACK_FILTERING=off
# #export GCS_RESOLVE_REFRESH_SECS=60
# #export GCS_REQUEST_CONNECTION_TIMEOUT_SECS=300
# #export GCS_METADATA_REQUEST_TIMEOUT_SECS=300
# #export GCS_READ_REQUEST_TIMEOUT_SECS=300
# #export GCS_WRITE_REQUEST_TIMEOUT_SECS=600

# export NCCL_OPTIMAL_CTAS=1
# export NCCL_ALGO=NVLS
# export NCCL_NVLS_ENABLE=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=NVLS,COLL,P2P,TUNING

# export TF_CPP_VMODULE=gemm_rewriter=10
# export TF_CPP_MIN_LOG_LEVEL=0
# export TF_CPP_MAX_LOG_LEVEL=100

export NVTE_FUSED_ATTN=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

# For multi process run
if [ -n "${SLURM_NTASKS_PER_NODE:-}" ] && [ $SLURM_NTASKS_PER_NODE -gt 1 ]; then
    echo "SLURM Configuration:"
    echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
    echo "Total tasks: $SLURM_NTASKS"
    echo "Number of nodes: $SLURM_NNODES"
    echo "Node ID: $SLURM_NODEID"
    export NNODES=8
    export NODE_RANK=$SLURM_PROCID
    export JAX_COORDINATOR_IP=${SLURM_LAUNCH_NODE_IPADDR}
    export JAX_COORDINATOR_PORT=12345
    export GPUS_PER_NODE=8
fi

if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token $HF_TOKEN
fi

PROJECT_NAME=benchmark
RUN_NAME=test
#uv run
if [ $USE_PGLE -eq 1 ]; then
    echo "PGLE is enabled"
    # export TF_CPP_VMODULE=profile_guided_latency_estimator=10
    # export TF_CPP_MIN_LOG_LEVEL=0
    # export TF_CPP_MAX_LOG_LEVEL=100

    NSYS_OUTPUT_FILE="${LOG_DIR}/profile-run"
    export XLA_FLAGS=$BASE_XLA_FLAGS" --xla_gpu_enable_latency_hiding_scheduler=false --xla_gpu_disable_async_collectives=allreduce,allgather,reducescatter,collectivebroadcast,alltoall,collectivepermute"
    echo $XLA_FLAGS

    nsys profile -t cuda,nvtx -o ${NSYS_OUTPUT_FILE} --cuda-graph-trace=node --force-overwrite=true --capture-range=cudaProfilerApi --capture-range-end=stop python3 src/maxdiffusion/train_flux.py $MODEL_CONFIG hardware=gpu run_name=flux attention=cudnn_flash_te max_train_steps=10 enable_profiler=True profiler_steps=2 profiler=nsys 

    echo "generate pbtxt"
    export PGLE_PROFILE_PATH=${NSYS_OUTPUT_FILE}.pbtxt
    python pgo_nsys_converter.py --profile_path $NSYS_OUTPUT_FILE.nsys-rep --post_process --pgle_output_path $PGLE_PROFILE_PATH 

    NSYS_OUTPUT_FILE="${LOG_DIR}/perf-run"
    export XLA_FLAGS=$BASE_XLA_FLAGS" --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_pgle_profile_file_or_directory_path=$PGLE_PROFILE_PATH"
    echo $XLA_FLAGS

    nsys profile -t cuda,nvtx -o ${NSYS_OUTPUT_FILE} --cuda-graph-trace=node --force-overwrite=true --capture-range=cudaProfilerApi --capture-range-end=stop python3 src/maxdiffusion/train_flux.py $MODEL_CONFIG hardware=gpu run_name=flux attention=cudnn_flash_te max_train_steps=10 enable_profiler=True profiler_steps=2 profiler=nsys 
else
    echo "PGLE is disabled"
    NSYS_OUTPUT_FILE="${LOG_DIR}/normal-run"
    export XLA_FLAGS=$BASE_XLA_FLAGS" --xla_gpu_enable_latency_hiding_scheduler=true"
    # export XLA_FLAGS=$BASE_XLA_FLAGS" --xla_gpu_enable_latency_hiding_scheduler=false"
    echo $XLA_FLAGS

    # export PYTHONPATH=$CODE_DIR/benchmark:$PYTHONPATH

    # nsys profile -t cuda,nvtx -o ${NSYS_OUTPUT_FILE} --cuda-graph-trace=node --force-overwrite=true --capture-range=cudaProfilerApi --capture-range-end=stop python3 src/maxdiffusion/train_flux.py src/maxdiffusion/configs/base_flux_dev.yml hardware=gpu save_final_checkpoint=False run_name=flux attention=cudnn_flash_te max_train_steps=10 enable_profiler=True profiler_steps=2 profiler=nsys
    # nsys profile -t cuda,nvtx -o ${NSYS_OUTPUT_FILE} --cuda-graph-trace=node --force-overwrite=true --capture-range=cudaProfilerApi --capture-range-end=stop python3 src/maxdiffusion/train_flux.py src/maxdiffusion/configs/base_flux_schnell.yml hardware=gpu run_name=flux attention=cudnn_flash_te max_train_steps=10 enable_profiler=True profiler_steps=2 profiler=nsys 

    nsys profile -t cuda,nvtx -o ${NSYS_OUTPUT_FILE} --cuda-graph-trace=node --force-overwrite=true --capture-range=cudaProfilerApi --capture-range-end=stop python3 src/maxdiffusion/train_flux.py $MODEL_CONFIG hardware=gpu run_name=flux attention=cudnn_flash_te max_train_steps=20 enable_profiler=True profiler_steps=5 skip_first_n_steps_for_profiler=10 profiler=nsys 
    # python3 src/maxdiffusion/train_flux.py $MODEL_CONFIG hardware=gpu run_name=flux attention=cudnn_flash_te max_train_steps=10 enable_profiler=True profiler_steps=2 profiler=xplane 
fi
set +x

