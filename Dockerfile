FROM tensorflow/tensorflow:2.4.2-gpu
WORKDIR /home

CMD bash

COPY [\
     "Main.py",\
     "Config.py",\
     "CreateQRCode.py",\
     "Network.py",\
     "Utils.py",\
     "requirement.txt",\
     "/home/"\
]

RUN \
alias sudo="";\
set -e;\
    pip install -r requirement.txt