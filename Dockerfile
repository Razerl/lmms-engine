FROM fatbao55/lmms-engine:v1.0
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
WORKDIR /app/lmms-engine
COPY . /app/lmms-engine/
RUN bash uv_sync_linux.sh
ENV PATH="/app/lmms-engine/.venv/bin:$PATH"
ENV PYTHONPATH="/app/lmms-engine:${PYTHONPATH:-}"
CMD ["/bin/bash"]
