FROM continuumio/miniconda3:23.10.0-1

RUN apt update &&\
    apt upgrade -y &&\
    apt install -y libgl1

WORKDIR /gec

COPY environment.yml /gec/environment.yml
RUN conda env create -f environment.yml

ENV PYTHONPATH="${PYTHONPATH}:/gec/src"
ENV TFHUB_CACHE_DIR="./tfhub_cache/"

COPY . /gec

CMD ["conda", "run", "--no-capture-output", "-n", "gec", "python", "app.py"]
