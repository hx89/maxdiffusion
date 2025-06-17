# syntax=docker/dockerfile:1-labs
ARG BASE_IMAGE=ghcr.io/nvidia/jax-mealkit:jax-2025-06-16
# ARG URLREF_MAXDIFFUSION=https://github.com/AI-Hypercomputer/maxdiffusion.git#main
ARG URLREF_MAXDIFFUSION=https://github.com/hx89/maxdiffusion.git#haixinl/gpu_opt
ARG SRC_PATH_MAXDIFFUSION=/opt/maxdiffusion
# ARG REQUIREMENTS_FILE=requirements.txt
# ARG REQUIREMENTS_FILE=requirements_with_jax_stable_stack.txt
ARG REQUIREMENTS_FILE=requirements_with_jax_ai_image.txt

###############################################################################
## Download source and add auxiliary scripts
###############################################################################
FROM ${BASE_IMAGE} AS mealkit
ARG URLREF_MAXDIFFUSION
ARG SRC_PATH_MAXDIFFUSION
ARG REQUIREMENTS_FILE
RUN <<"EOF" bash -ex
git-clone.sh ${URLREF_MAXDIFFUSION} ${SRC_PATH_MAXDIFFUSION}
# grain/grain-nightly distinction unclear; grain==0.2.4 seems broken
sed -i 's|^grain-nightly==.*|grain==0.2.3|' "${SRC_PATH_MAXDIFFUSION}/${REQUIREMENTS_FILE}"
# Install CPU-only TensorFlow
sed -i 's|^tensorflow>=|tensorflow-cpu>=|' "${SRC_PATH_MAXDIFFUSION}/${REQUIREMENTS_FILE}"
# Install CPU-only PyTorch
sed -i '/^torch==/ s/$/+cpu/' "${SRC_PATH_MAXDIFFUSION}/${REQUIREMENTS_FILE}"
sed -i 's|^huggingface-hub==.*|huggingface-hub==0.24.7|' "${SRC_PATH_MAXDIFFUSION}/${REQUIREMENTS_FILE}"
sed -i 's|^huggingface_hub==.*|huggingface_hub==0.30.0|' "${SRC_PATH_MAXDIFFUSION}/${REQUIREMENTS_FILE}"
echo "--extra-index-url https://download.pytorch.org/whl/cpu" >> /opt/pip-tools.d/requirements-maxdiffusion.in
# Skip torchvision
sed -i 's|^torchvision|#torchvision|' "${SRC_PATH_MAXDIFFUSION}/${REQUIREMENTS_FILE}"
(cd ${SRC_PATH_MAXDIFFUSION} && git diff)
echo "-e ${SRC_PATH_MAXDIFFUSION}" >> /opt/pip-tools.d/requirements-maxdiffusion.in
echo "-r ${SRC_PATH_MAXDIFFUSION}/${REQUIREMENTS_FILE}" >> /opt/pip-tools.d/requirements-maxdiffusion.in
EOF

###############################################################################
## Add test script to the path
###############################################################################
# ADD test-maxdiffusion.sh /usr/local/bin

###############################################################################
## Install accumulated packages from the base image and the previous stage
###############################################################################
FROM mealkit AS final
RUN pip-finalize.sh
WORKDIR ${SRC_PATH_MAXDIFFUSION}