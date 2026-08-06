IMAGE=$1

#docker run -it \
#        --ulimit memlock=-1:-1 \
#        --ulimit stack=67108864:67108864 \
#        --device /dev/dri \
#        --device /dev/kfd \
#        --network host \
#        --ipc host \
#        --group-add video \
#        --cap-add SYS_PTRACE \
#        --security-opt seccomp=unconfined \
#        --privileged \
#        --env HUGGINGFACE_HUB_CACHE=/models\
#        --env MODELSCOPE_CACHE=/models\
#        -v /apps/data/models/:/models \
#        -v /apps:/apps \
#        --shm-size 32G \
#        --name ${CONTAINER_NAME} \
#        ${IMAGE_NAME} /bin/bash


docker run -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --network=host \
  --pid=host \
  --ipc=host \
  --ulimit memlock=-1 \
  --cap-add=IPC_LOCK \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --privileged \
  --shm-size=64g \
  --group-add video \
  --group-add rdma \
  -v /:/dockerx \
  -v /dev/infiniband:/dev/infiniband \
  ${IMAGE}
