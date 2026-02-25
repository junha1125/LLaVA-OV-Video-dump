FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /workspace

RUN apt update && apt install -y tmux ffmpeg htop && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/junha1125/LLaVA-OV-Video-dump.git

WORKDIR /workspace/LLaVA-OV-Video-dump

RUN pip install --upgrade pip && \
    pip install -e ".[train]" && \
    pip install --upgrade "Pillow[webp]>=10.0.0" && \
    pip install huggingface-hub==0.31.1 && \
    pip install -r requirements_wan.txt && \
    pip install flash-attn==2.3.3 --no-build-isolation

RUN curl -fsSL https://claude.ai/install.sh | bash

ENV PATH="/root/.local/bin:${PATH}"

CMD ["/bin/bash"]