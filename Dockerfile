FROM nvidia/cuda:12.2.0-base-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# ENV BUNDLER_VERSION="2.4.22"

WORKDIR /app

RUN apt-get update -y && \
      apt-get upgrade -y && \
      apt-get install -y python3.11 pip

# RUN gem install bundler:$BUNDLER_VERSION && rm -rf /var/cache/apk/*

COPY requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

# ADD Gemfile* /app/

# RUN  bundle config --local without "development test" && \
#      bundle install -j8 --no-cache && \
#      bundle clean --force


COPY . /app

CMD python3 app.py
