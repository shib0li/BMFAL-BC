FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install git -y 

RUN /opt/conda/bin/pip install jupyter \
  && /opt/conda/bin/pip install scipy fire tqdm scikit-learn pandas matplotlib  \
  && /opt/conda/bin/pip install pyDOE sobol_seq hdf5storage \
  && /opt/conda/bin/pip install torchnet 