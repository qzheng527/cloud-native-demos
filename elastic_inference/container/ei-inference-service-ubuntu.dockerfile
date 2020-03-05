FROM openvino/ubuntu18_runtime:2020.1

ARG pip_mirror

USER root

RUN apt update && apt install -y libpython3.6-dev libsm6 libxext6 libxrender-dev

COPY ./container/docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

USER openvino

COPY ./apps /apps
COPY ./models /models

RUN pip3 install wheel ${pip_mirror} --user
RUN pip3 install ${pip_mirror} -r /apps/requirements.ois.txt --user

ENV INFER_MODEL_PATH="/models"
ENV INFER_MODEL_NAME="SqueezeNetSSD-5Class"
ENV INPUT_QUEUE_HOST="127.0.0.1"
ENV OUTPUT_BROKER_HOST="127.0.0.1"
ENV INFER_TYPE="people"

# for prometheums metrics
EXPOSE 8000

HEALTHCHECK CMD curl --fail http://localhost:8000/ || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]

CMD ["/apps/infer_service.py"]
