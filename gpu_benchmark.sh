#!/usr/bin/env bash
set -ux

MODEL_CONFIG=${1:-"train/benchmark/configs/tpu.yaml"}
USE_PGLE=${2:-0}

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
                --xla_dump_to=/tmp/xla_dump
                --xla_gpu_graph_level=0"
                # --xla_dump_hlo_pass_re=.*
                # --xla_gpu_enable_nccl_user_buffers=true
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

    nsys profile -t cuda,nvtx -o ${NSYS_OUTPUT_FILE} --cuda-graph-trace=node --force-overwrite=true --capture-range=cudaProfilerApi --capture-range-end=stop python3 -m train.$PROJECT_NAME \
        --checkpoint.run_name $RUN_NAME \
        --config $MODEL_CONFIG

    echo "generate pbtxt"
    export PGLE_PROFILE_PATH=${NSYS_OUTPUT_FILE}.pbtxt
    python pgo_nsys_converter.py --profile_path $NSYS_OUTPUT_FILE.nsys-rep --post_process --pgle_output_path $PGLE_PROFILE_PATH 

    NSYS_OUTPUT_FILE="${LOG_DIR}/perf-run"
    export XLA_FLAGS=$BASE_XLA_FLAGS" --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_pgle_profile_file_or_directory_path=$PGLE_PROFILE_PATH"
    echo $XLA_FLAGS

    nsys profile -t cuda,nvtx -o ${NSYS_OUTPUT_FILE} --cuda-graph-trace=node --force-overwrite=true --capture-range=cudaProfilerApi --capture-range-end=stop python3 -m train.$PROJECT_NAME \
        --checkpoint.run_name $RUN_NAME \
        --config $MODEL_CONFIG
else
    echo "PGLE is disabled"
    NSYS_OUTPUT_FILE="${LOG_DIR}/normal-run"
    export XLA_FLAGS=$BASE_XLA_FLAGS" --xla_gpu_enable_latency_hiding_scheduler=true"
    # export XLA_FLAGS=$BASE_XLA_FLAGS" --xla_gpu_enable_latency_hiding_scheduler=false"
    echo $XLA_FLAGS

    # export PYTHONPATH=$CODE_DIR/benchmark:$PYTHONPATH

    nsys profile -t cuda,nvtx -o ${NSYS_OUTPUT_FILE} --cuda-graph-trace=node --force-overwrite=true --capture-range=cudaProfilerApi --capture-range-end=stop python3 src/maxdiffusion/train_flux.py src/maxdiffusion/configs/base_flux_dev.yml hardware=gpu save_final_checkpoint=False run_name=flux attention=cudnn_flash_te max_train_steps=10 enable_profiler=True profiler_steps=2 profiler=nsys
fi
set +x