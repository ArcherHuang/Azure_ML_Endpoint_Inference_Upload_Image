FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu20.04:20220902.v1
ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/tensorflow-2.6
# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.8 pip=20.2.4
# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH
# Install pip dependencies
RUN HOROVOD_WITH_TENSORFLOW=1 pip install 'azureml-core==1.45.0' \
                                          'azureml-defaults==1.45.0' \
                                          'azureml-mlflow==1.45.0' \
                                          'tensorflow-gpu~=2.6.0' \
                                          'opencv-python==4.1.2.30' \
                                          'azure-storage-blob==12.18.3' \
                                          'typing_extensions==4.4.0' \
                                          'numpy~=1.19.0' \
                                          'Pillow'
                          
# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH