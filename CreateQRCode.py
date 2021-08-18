import qrcode
import os
import re

from Config import TARGET_WORDS

def CreateQRCode(text,image_path):
    """
    Create QRcode function.

    Args:
        text (str): test fro generating QRcode
        image_path (str): save directory path for QRcode image
    Return:
        QRcode image
    """
    assert text is not None
    assert image_path is not None

    for c in text:
        if not c in TARGET_WORDS:
            raise ValueError(c,": Argument can only use",TARGET_WORDS)

    qr=qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=5,
        border=2
    )
    qr.add_data(text.encode("ascii"))
    qr.make(fit=True)
    image=qr.make_image(fill_color="black",back_color="white")

    os.makedirs(os.path.dirname(image_path),exist_ok=True)

    image.save(image_path)

    return image