import tensorflow as tf
import random
import os
import PIL

from CreateQRCode import CreateQRCode
from Config import TARGET_WORDS

def SetGPUConfig(using_gpu_number,memory_allocate_flag):
    """
    Setting gpu config.

    Args:
        using_gpu_number (int): using gpu number. -1 is use all.
        memory_allocate_flag (bool): The flag allowing auto allocate memory.
    """
    #GPU confi
    #Get all gpu list. Then, setting using all gpu
    physical_devices = tf.config.list_physical_devices("GPU")

    if len(physical_devices)<=0:
        print("Not enough GPU hardware devices available")

    #Check is exist number what is using_gpu_number.
    if using_gpu_number>=len(physical_devices):
        print("Using GPU number;",using_gpu_number,"is not available.")
        print("GPU list:",physical_devices)

    #Setting using GPU
    if using_gpu_number!=-1:
        tf.config.set_visible_devices(physical_devices[using_gpu_number],"GPU")

    #Setting allowing auto allocate memory.
    if memory_allocate_flag:
        if using_gpu_number==-1:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device,True)
        else:
            tf.config.experimental.set_memory_growth(physical_devices[using_gpu_number],True)

    using_gpu_list=tf.config.get_visible_devices("GPU")
    print("[GPU Config]")
    for i,device in enumerate(physical_devices):
        if device in using_gpu_list:
            print("\tGPU"+str(i)+": USE")
        else:
            print("\tGPU"+str(i)+": DISUSE")

def GenerateRandomString(max_string_length):
    """
    Create random string.
    Using characters was defined by Config.py.
    Args:
        max_string_length (int): max length of string
    Return:
        string
    """
    return "".join(random.choices(TARGET_WORDS,k=max_string_length))

def QRCodeRandomGenerator(generate_num,image_save_dir,max_string_length):
    """
    Create QRCodeImage by random string.
    Args:
        generate_num (int): Generate QRcode number
        image_save_dir (string): Save path of saveing QRcode
        max_string_length (int): Max string lengh for generateing QRcode
    Return:
        return 2 lists.
            0. QRcode image paths
            1. String for generating QRcode
            2. QRcode size
    """
    dst=[[],[]]

    os.makedirs(image_save_dir,exist_ok=True)

    counter=0
    for filename in os.listdir(image_save_dir):
        if os.path.splitext(filename)[1] == ".jpg":
            counter+=1
            img=PIL.Image.open(os.path.join(image_save_dir,filename))
            dst[0].append(os.path.join(image_save_dir,filename))
            dst[1].append(os.path.splitext(filename)[0])

            if counter>generate_num:
                break

    generate_num=generate_num-counter

    if generate_num>0:
        print("Start generate QRCode\n")
        for i in range(generate_num):
            print("%d / %d"%(i+1,generate_num),end="\r")
            
            #Create random string langh
            random_string=GenerateRandomString(max_string_length)

            #Create QRcodes
            file_path=os.path.join(image_save_dir,random_string+".jpg")

            img=CreateQRCode(random_string,file_path)

            dst[0].append(file_path)
            dst[1].append(random_string)

    dst.append((*img.size,1))

    return dst

def NormalizeString(string):
    """
    Convert string to [0,1] float
    Args:
        string (string): string
    Return:
        list of normalized string
    """
    dst=[]
    tmp=float(len(TARGET_WORDS))

    for c in string:
        dst.append(float(TARGET_WORDS.find(c))/tmp)

    return dst

def UnNormalizeString(numbers):
    """
    Convert [0,1] to string
    Args:
        numbers (float in list): numbers
    Return:
        string
    """
    dst=""
    tmp=float(len(TARGET_WORDS))

    for n in numbers:
        dst+=TARGET_WORDS[int(n*tmp)]

    return dst

def LoadImage(image_path):
    image=tf.io.read_file(image_path)
    image=tf.io.decode_image(image)
    image=tf.cast(image,tf.float32)
    image/=225.
    image=tf.expand_dims(image,axis=0)

    return image