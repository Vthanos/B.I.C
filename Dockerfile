ADD file:7009ad0ee0bbe5ed7f381792e07347e260e6896aeee0d80597808065120fa96b in /
CMD ["bash"]
ENV NVARCH=x86_64
ENV NVIDIA_REQUIRE_CUDA=cuda>=11.2 brand=tesla,driver>=418,driver<419
ENV NV_CUDA_CUDART_VERSION=11.2.146-1
ENV NV_CUDA_COMPAT_PACKAGE=cuda-compat-11-2
ARG TARGETARCH
RUN |1 TARGETARCH=amd64 /bin/sh -c apt-get update && apt-get install -y --no-install-recommends     gnupg2 curl ca-certificates &&     curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/${NVARCH}/3bf863cc.pub | apt-key add - &&     echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/${NVARCH} /" > /etc/apt/sources.list.d/cuda.list &&     apt-get purge --autoremove -y curl     && rm -rf /var/lib/apt/lists/* # buildkit
ENV CUDA_VERSION=11.2.1
RUN |1 TARGETARCH=amd64 /bin/sh -c apt-get update && apt-get install -y --no-install-recommends     cuda-cudart-11-2=${NV_CUDA_CUDART_VERSION}     ${NV_CUDA_COMPAT_PACKAGE}     && ln -s cuda-11.2 /usr/local/cuda &&     rm -rf /var/lib/apt/lists/* # buildkit
RUN |1 TARGETARCH=amd64 /bin/sh -c echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf     && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf # buildkit
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
COPY NGC-DL-CONTAINER-LICENSE / # buildkit
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ARG CUDA
ARG ARCH
ARG CUDNN=8.1.0.77-1
ARG CUDNN_MAJOR_VERSION=8
ARG LIB_DIR_PREFIX=x86_64
ARG LIBNVINFER=7.2.2-1
ARG LIBNVINFER_MAJOR_VERSION=7
ENV DEBIAN_FRONTEND=noninteractive
/bin/bash -c #(nop)  SHELL [/bin/bash -c]
|7 ARCH= CUDA=11.2 CUDNN=8.1.0.77-1 CUDNN_MAJOR_VERSION=8 LIBNVINFER=7.2.2-1 LIBNVINFER_MAJOR_VERSION=7 LIB_DIR_PREFIX=x86_64 /bin/bash -c apt-get update && apt-get install -y --no-install-recommends         build-essential         cuda-command-line-tools-${CUDA/./-}         libcublas-${CUDA/./-}         cuda-nvrtc-${CUDA/./-}         libcufft-${CUDA/./-}         libcurand-${CUDA/./-}         libcusolver-${CUDA/./-}         libcusparse-${CUDA/./-}         curl         libcudnn8=${CUDNN}+cuda${CUDA}         libfreetype6-dev         libhdf5-serial-dev         libzmq3-dev         pkg-config         software-properties-common         unzip
|7 ARCH= CUDA=11.2 CUDNN=8.1.0.77-1 CUDNN_MAJOR_VERSION=8 LIBNVINFER=7.2.2-1 LIBNVINFER_MAJOR_VERSION=7 LIB_DIR_PREFIX=x86_64 /bin/bash -c [[ "${ARCH}" = "ppc64le" ]] || { apt-get update &&         apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub &&         echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /"  > /etc/apt/sources.list.d/tensorRT.list &&         apt-get update &&         apt-get install -y --no-install-recommends libnvinfer${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda11.0         libnvinfer-plugin${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda11.0         && apt-get clean         && rm -rf /var/lib/apt/lists/*; }
/bin/bash -c #(nop)  ENV LD_LIBRARY_PATH=/usr/local/cuda-11.0/targets/x86_64-linux/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
|7 ARCH= CUDA=11.2 CUDNN=8.1.0.77-1 CUDNN_MAJOR_VERSION=8 LIBNVINFER=7.2.2-1 LIBNVINFER_MAJOR_VERSION=7 LIB_DIR_PREFIX=x86_64 /bin/bash -c ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1     && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf     && ldconfig
/bin/bash -c #(nop)  ENV LANG=C.UTF-8
|7 ARCH= CUDA=11.2 CUDNN=8.1.0.77-1 CUDNN_MAJOR_VERSION=8 LIBNVINFER=7.2.2-1 LIBNVINFER_MAJOR_VERSION=7 LIB_DIR_PREFIX=x86_64 /bin/bash -c apt-get update && apt-get install -y     python3     python3-pip
|7 ARCH= CUDA=11.2 CUDNN=8.1.0.77-1 CUDNN_MAJOR_VERSION=8 LIBNVINFER=7.2.2-1 LIBNVINFER_MAJOR_VERSION=7 LIB_DIR_PREFIX=x86_64 /bin/bash -c python3 -m pip --no-cache-dir install --upgrade     "pip<20.3"     setuptools
|7 ARCH= CUDA=11.2 CUDNN=8.1.0.77-1 CUDNN_MAJOR_VERSION=8 LIBNVINFER=7.2.2-1 LIBNVINFER_MAJOR_VERSION=7 LIB_DIR_PREFIX=x86_64 /bin/bash -c ln -s $(which python3) /usr/local/bin/python
/bin/bash -c #(nop)  ARG TF_PACKAGE=tensorflow
/bin/bash -c #(nop)  ARG TF_PACKAGE_VERSION=
|9 ARCH= CUDA=11.2 CUDNN=8.1.0.77-1 CUDNN_MAJOR_VERSION=8 LIBNVINFER=7.2.2-1 LIBNVINFER_MAJOR_VERSION=7 LIB_DIR_PREFIX=x86_64 TF_PACKAGE=tensorflow TF_PACKAGE_VERSION=2.9.1 /bin/bash -c python3 -m pip install --no-cache-dir ${TF_PACKAGE}${TF_PACKAGE_VERSION:+==${TF_PACKAGE_VERSION}}
/bin/bash -c #(nop) COPY file:8d4bb213681b66936e970b7417387fa77847a8c72277464035eae2510a23cbf6 in /etc/bash.bashrc
|9 ARCH= CUDA=11.2 CUDNN=8.1.0.77-1 CUDNN_MAJOR_VERSION=8 LIBNVINFER=7.2.2-1 LIBNVINFER_MAJOR_VERSION=7 LIB_DIR_PREFIX=x86_64 TF_PACKAGE=tensorflow TF_PACKAGE_VERSION=2.9.1 /bin/bash -c chmod a+rwx /etc/bash.bashrc
|9 ARCH= CUDA=11.2 CUDNN=8.1.0.77-1 CUDNN_MAJOR_VERSION=8 LIBNVINFER=7.2.2-1 LIBNVINFER_MAJOR_VERSION=7 LIB_DIR_PREFIX=x86_64 TF_PACKAGE=tensorflow TF_PACKAGE_VERSION=2.9.1 /bin/bash -c python3 -m pip install --no-cache-dir jupyter matplotlib
|9 ARCH= CUDA=11.2 CUDNN=8.1.0.77-1 CUDNN_MAJOR_VERSION=8 LIBNVINFER=7.2.2-1 LIBNVINFER_MAJOR_VERSION=7 LIB_DIR_PREFIX=x86_64 TF_PACKAGE=tensorflow TF_PACKAGE_VERSION=2.9.1 /bin/bash -c python3 -m pip install --no-cache-dir jupyter_http_over_ws ipykernel==5.1.1 nbformat==4.4.0 jedi==0.17.2
|9 ARCH= CUDA=11.2 CUDNN=8.1.0.77-1 CUDNN_MAJOR_VERSION=8 LIBNVINFER=7.2.2-1 LIBNVINFER_MAJOR_VERSION=7 LIB_DIR_PREFIX=x86_64 TF_PACKAGE=tensorflow TF_PACKAGE_VERSION=2.9.1 /bin/bash -c jupyter serverextension enable --py jupyter_http_over_ws
|9 ARCH= CUDA=11.2 CUDNN=8.1.0.77-1 CUDNN_MAJOR_VERSION=8 LIBNVINFER=7.2.2-1 LIBNVINFER_MAJOR_VERSION=7 LIB_DIR_PREFIX=x86_64 TF_PACKAGE=tensorflow TF_PACKAGE_VERSION=2.9.1 /bin/bash -c mkdir -p /tf/tensorflow-tutorials && chmod -R a+rwx /tf/
|9 ARCH= CUDA=11.2 CUDNN=8.1.0.77-1 CUDNN_MAJOR_VERSION=8 LIBNVINFER=7.2.2-1 LIBNVINFER_MAJOR_VERSION=7 LIB_DIR_PREFIX=x86_64 TF_PACKAGE=tensorflow TF_PACKAGE_VERSION=2.9.1 /bin/bash -c mkdir /.local && chmod a+rwx /.local
|9 ARCH= CUDA=11.2 CUDNN=8.1.0.77-1 CUDNN_MAJOR_VERSION=8 LIBNVINFER=7.2.2-1 LIBNVINFER_MAJOR_VERSION=7 LIB_DIR_PREFIX=x86_64 TF_PACKAGE=tensorflow TF_PACKAGE_VERSION=2.9.1 /bin/bash -c apt-get update && apt-get install -y --no-install-recommends wget git
/bin/bash -c #(nop) WORKDIR /tf/tensorflow-tutorials
|9 ARCH= CUDA=11.2 CUDNN=8.1.0.77-1 CUDNN_MAJOR_VERSION=8 LIBNVINFER=7.2.2-1 LIBNVINFER_MAJOR_VERSION=7 LIB_DIR_PREFIX=x86_64 TF_PACKAGE=tensorflow TF_PACKAGE_VERSION=2.9.1 /bin/bash -c wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/classification.ipynb
|9 ARCH= CUDA=11.2 CUDNN=8.1.0.77-1 CUDNN_MAJOR_VERSION=8 LIBNVINFER=7.2.2-1 LIBNVINFER_MAJOR_VERSION=7 LIB_DIR_PREFIX=x86_64 TF_PACKAGE=tensorflow TF_PACKAGE_VERSION=2.9.1 /bin/bash -c wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/overfit_and_underfit.ipynb
|9 ARCH= CUDA=11.2 CUDNN=8.1.0.77-1 CUDNN_MAJOR_VERSION=8 LIBNVINFER=7.2.2-1 LIBNVINFER_MAJOR_VERSION=7 LIB_DIR_PREFIX=x86_64 TF_PACKAGE=tensorflow TF_PACKAGE_VERSION=2.9.1 /bin/bash -c wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/regression.ipynb
|9 ARCH= CUDA=11.2 CUDNN=8.1.0.77-1 CUDNN_MAJOR_VERSION=8 LIBNVINFER=7.2.2-1 LIBNVINFER_MAJOR_VERSION=7 LIB_DIR_PREFIX=x86_64 TF_PACKAGE=tensorflow TF_PACKAGE_VERSION=2.9.1 /bin/bash -c wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/save_and_load.ipynb
|9 ARCH= CUDA=11.2 CUDNN=8.1.0.77-1 CUDNN_MAJOR_VERSION=8 LIBNVINFER=7.2.2-1 LIBNVINFER_MAJOR_VERSION=7 LIB_DIR_PREFIX=x86_64 TF_PACKAGE=tensorflow TF_PACKAGE_VERSION=2.9.1 /bin/bash -c wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/text_classification.ipynb
|9 ARCH= CUDA=11.2 CUDNN=8.1.0.77-1 CUDNN_MAJOR_VERSION=8 LIBNVINFER=7.2.2-1 LIBNVINFER_MAJOR_VERSION=7 LIB_DIR_PREFIX=x86_64 TF_PACKAGE=tensorflow TF_PACKAGE_VERSION=2.9.1 /bin/bash -c wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/text_classification_with_hub.ipynb
/bin/bash -c #(nop) COPY file:771a5d1feaa479d6972f4d612cc687e2e487b16c4ce6359830a25bcdd1a16151 in README.md
|9 ARCH= CUDA=11.2 CUDNN=8.1.0.77-1 CUDNN_MAJOR_VERSION=8 LIBNVINFER=7.2.2-1 LIBNVINFER_MAJOR_VERSION=7 LIB_DIR_PREFIX=x86_64 TF_PACKAGE=tensorflow TF_PACKAGE_VERSION=2.9.1 /bin/bash -c apt-get autoremove -y && apt-get remove -y wget
/bin/bash -c #(nop) WORKDIR /tf
/bin/bash -c #(nop)  EXPOSE 8888
|9 ARCH= CUDA=11.2 CUDNN=8.1.0.77-1 CUDNN_MAJOR_VERSION=8 LIBNVINFER=7.2.2-1 LIBNVINFER_MAJOR_VERSION=7 LIB_DIR_PREFIX=x86_64 TF_PACKAGE=tensorflow TF_PACKAGE_VERSION=2.9.1 /bin/bash -c python3 -m ipykernel.kernelspec
/bin/bash -c #(nop)  CMD ["bash" "-c" "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root"]


