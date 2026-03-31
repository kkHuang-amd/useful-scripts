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


docker run -it --privileged --network=host --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ipc=host --shm-size 64G -v /:/dockerx $IMAGE
