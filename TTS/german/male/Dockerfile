FROM python:3.8.0 as build

RUN apt update
RUN apt install --yes --no-install-recommends espeak libsndfile1 git

RUN mkdir -p /app
RUN cd /app && \
    git clone https://github.com/mozilla/TTS && \
    cd TTS && \
    git checkout 540d811

RUN cd /app/TTS && \
    python3 -m venv .venv

RUN cd /app/TTS && \
    .venv/bin/pip3 install --upgrade pip && \
    .venv/bin/pip3 install torch==1.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    .venv/bin/pip3 install -r requirements.txt && \
    .venv/bin/python3 setup.py install

RUN cd /app/TTS && \
    .venv/bin/pip3 install inflect 'numba==0.48'

RUN cd /app/TTS && \
    .venv/bin/pip3 install 'flask' 'flask-cors'

#-----------------------

FROM python:3.8-slim

RUN uname -a

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
        espeak libsndfile1

COPY --from=build /app/TTS/.venv/ /app/
COPY vocoder/ /app/vocoder/
COPY model/ /app/model/
COPY templates/ /app/templates/
COPY tts.py scale_stats.npy /app/

WORKDIR /app

EXPOSE 5002

ENTRYPOINT ["/app/bin/python3", "/app/tts.py"]