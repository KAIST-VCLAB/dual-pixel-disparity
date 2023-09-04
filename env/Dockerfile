FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
RUN DEBIAN_FRONTEND=noninteractive

# Basic apps. You may not need libgl1-* and x11-apps if you don't need GUI
RUN apt update && apt install -y git nano wget unzip libgl1-mesa-glx x11-apps
RUN conda init bash
RUN conda install -y \
    wget unzip \
    scipy==1.7.3
RUN pip install --no-input \
    easydict \
    tqdm \
    opencv-python-headless==4.6.0.66 \
    opencv-contrib-python-headless==4.6.0.66 \
    scikit-image==0.19.3 \
    opt_einsum==3.3.0
