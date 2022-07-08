FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN pip install tensorflow-gpu
RUN pip install Pillow
RUN pip install numpy
RUN pip install matplotlib
RUN pip install pandas
RUN pip install scipy
RUN pip install plotly-express

