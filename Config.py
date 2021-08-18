import string
import os

TARGET_WORDS=string.digits+string.ascii_letters

TEMPLATE_IMAGE_DIR_PATH="TEMP"

TEMPLATE_IMAGE_PATH=os.path.join(TEMPLATE_IMAGE_DIR_PATH,"temp.png")

#Show: https://www.keyence.co.jp/ss/products/autoid/codereader/basic2d-qr-types.jsp#sect_03
if TARGET_WORDS==string.digits:
    QRCODE_SIZE_LIST=[
        17,
        34,
        58,
        82,
        106,
        139,
        154,
        202,
        235
    ]
else:
    QRCODE_SIZE_LIST=[
        7,
        14,
        24,
        34,
        44,
        58,
        64,
        84,
        98
    ]

EVALUATE_SAMPLE_NUM=100