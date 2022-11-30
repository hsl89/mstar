FROM 747303060528.dkr.ecr.us-east-2.amazonaws.com/mstar-gitlab:base1.6 as base

# Copy mstar
RUN mkdir -p /usr/local/src/mstar
COPY mstar /usr/local/src/mstar

# Install mstar or p3 and p4
RUN cd /usr/local/src/mstar && \
    TORCH_CUDA_ARCH_LIST="8.0 7.5 7.0" FORCE_CUDA=1 pip3 install /usr/local/src/mstar --no-build-isolation
RUN python3 -c "from mstar import AutoModel"

# Install lm_eval
RUN cd /usr/local/src/ && \
    git clone https://github.com/bigscience-workshop/lm-evaluation-harness.git && \
    cd lm-evaluation-harness && \
    git checkout bdd1d3f9e4480b5fe89850b98ac1cc8b8dfab024 && \
    cp /usr/local/src/mstar/tools/robustness-testing/harness-setup.py setup.py && \
    pip3 install -e .

# Install promptsource
RUN cd  /usr/local/src/lm-evaluation-harness && \
    git clone https://github.com/bigscience-workshop/promptsource.git && \
    cd promptsource && \
    git checkout eval-hackathon && \
    pip3 install -e .

# Install requirements
RUN pip3 install -r /usr/local/src/mstar/tools/robustness-testing/requirements.txt

#Change workdirectory
WORKDIR /usr/local/src/mstar/tools/robustness-testing


