# QRCodeDLReader Quick Start

## Environment
- OS: Linux (Can use Docker)
- GPU: Nvidia (No device is OK)

## Install Docker
If you haven't installed Docker, look [this page](https://docs.docker.com/engine/install/) and install it.

3. Install Nvidia Docker (When only use GPU)
If you haven't installed Nvidia Docker, look [this page](https://github.com/NVIDIA/nvidia-docker) and install it.

## Clone Repository
```
git clone https://github.com/playre0001/QRCodeDLReader
```

## Build Docker Image
Build dockerfile.

```
cd QRCodeDLReader
docker build . -t qrcodedlreader
```

## Start Container
If use GPU:
```
docker run -it --gpus all qrcodedlreader
```

If use CPU:
```
docker run -it qrcodedlreader
```

## Learning
```
python Main.py
```

## Evaluate
```
sh CalcWordAccuracy.sh
```

## Predict
When predict, you have to set word length what is for generating test word.
```
python Main.py -m 2 -l [WORDLENGTH]
```

## Generating QRcode on Number Only
If you want to generate only number, replace 
```
TARGET_WORDS=string.digits+string.ascii_letters
```
as
```
TARGET_WORDS=string.digits#+string.ascii_letters
```
in "Config.py".

If you want to replace it by command line, use
```
sed -i s/TARGET_WORDS=string.digits+string.ascii_letters/TARGET_WORDS=string.digits#+string.ascii_letters/ Config.py
```
