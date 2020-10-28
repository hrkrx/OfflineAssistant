FROM tensorflow/tensorflow:1.15.4

RUN apt update
RUN apt install --yes --no-install-recommends sox git ffmpeg

RUN mkdir -p /app
RUN mkdir -p /app/model
RUN cd /app && \
    git clone https://github.com/mozilla/DeepSpeech.git

RUN pip install --upgrade pip

RUN cd /app/DeepSpeech && pip install -r requirements_transcribe.txt && \
    python setup.py install
    
COPY model /app/model

RUN cd app && \ 
    python3 /app/DeepSpeech/util/taskcluster.py --source tensorflow --artifact convert_graphdef_memmapped_format --branch r1.15 --target model/

RUN cd app/model && \
    chmod +x convert_graphdef_memmapped_format && \
    ./convert_graphdef_memmapped_format --in_graph=output_graph.pb --out_graph=output_graph.pbmm

RUN pip install flask flask-cors werkzeug

RUN cd /app && \
    git clone https://github.com/mozilla/DeepSpeech-examples.git && \
    cd DeepSpeech-examples/vad_transcriber && \
    pip install -r requirements.txt

RUN cp /app/DeepSpeech-examples/vad_transcriber/wavTranscriber.py /app/
RUN cp /app/DeepSpeech-examples/vad_transcriber/wavSplit.py /app/

COPY templates/ /app/
COPY stt.py /app/

ENTRYPOINT ["python3", "/app/stt.py"]