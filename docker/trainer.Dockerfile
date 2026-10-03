# Trainer image: the pinned vLLM image (same torch / transformers as serving) with its
# `vllm serve` entrypoint cleared. Extra training deps go here when they are needed.
ARG VLLM_IMAGE
FROM ${VLLM_IMAGE}
ENTRYPOINT []
WORKDIR /work
