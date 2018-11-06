FROM travisci/ci-garnet:packer-1512502276-986baf0

ARG PYTHON_VERSION=3.6

# install
RUN sudo apt-get update
RUN wget -q https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p /opt/miniconda
ENV PATH="/opt/miniconda/bin:$PATH"
RUN conda config --set always_yes yes --set changeps1 no
RUN conda update -q conda

RUN conda create -q -n test-environment python=$PYTHON_VERSION
ENV PATH="/opt/miniconda/envs/test-environment/bin:$PATH"
RUN pip install future
RUN pip install chainer -q
RUN pip install torchvision -q
RUN pip uninstall torch -y
RUN pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
RUN pip install moviepy==0.2.3.2 -q
RUN pip install matplotlib -q
RUN pip install requests -q
RUN python -c "import imageio; imageio.plugins.ffmpeg.download()"
RUN pip install --upgrade pytest flake8

ADD . /src
WORKDIR /src
RUN python setup.py develop

# script
CMD bash -c "conda info -a; \
    conda list; \
    which python; \
    flake8 tensorboardX; \
    pytest; \
    pip uninstall -y tensorboardX; \
    pip install tensorboardX; \
    pytest"
