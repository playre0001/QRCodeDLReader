import os
#delete tensorflow log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import sys
import argparse
import shutil

from Utils import QRCodeRandomGenerator, SetGPUConfig
from Config import QRCODE_SIZE_LIST, TEMPLATE_IMAGE_PATH, TEMPLATE_IMAGE_DIR_PATH, EVALUATE_SAMPLE_NUM, TARGET_WORDS
from CreateQRCode import CreateQRCode
import Network
import Utils

def Learning(model,model_save_path,max_string_length,image_size,epoch,batch_size,version,train,validation):
    """
    For learning model function

    Args:
        model (tf.keras.Model): Deep learning model
        model_save_path (string): Save directory path for model
        max_string_length (int): Max string lengh for generateing QRcode 
        epoch (int): Epoch
        batch_size (int): Batch size
        version (int): Learning QRcode version
        train (list): Train dataset
        validation (list): Validation dataset
    Retrun:
        Model fit history
    """
    os.makedirs(model_save_path,exist_ok=True)

    mcp=tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_save_path,"Version%d.hdf5"%(version)),
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        monitor="val_loss"
    )

    def Generator(image_paths,answers):
        for i in range(len(image_paths)):
            image_path=image_paths[i]
            answer=answers[i]

            image=tf.io.read_file(image_path)

            image=tf.io.decode_image(image)

            image=tf.cast(image,tf.float32)

            image/=225.

            yield image,answer

    train_ds=tf.data.Dataset.from_generator(
        generator=Generator,
        args=train,
        output_types=(tf.float32,tf.float32),
        output_shapes=(tf.TensorShape(image_size),tf.TensorShape((max_string_length,)))
    )

    valid_ds=tf.data.Dataset.from_generator(
        generator=Generator,
        args=validation,
        output_types=(tf.float32,tf.float32),
        output_shapes=(tf.TensorShape(image_size),tf.TensorShape((max_string_length,)))
    )

    history=model.fit(
        train_ds.batch(batch_size),
        validation_data=valid_ds.batch(batch_size),
        epochs=epoch,
        callbacks=[mcp]
    )

    return history

def Prediction(model,image_path):
    #Load Image
    image=Utils.LoadImage(image_path)

    #Predict
    answer=model.predict(image)

    #Show Answer
    print("AI's answer!: ",Utils.UnNormalizeString(answer[0]))

def Evaluate(model,words_length):
    """
    Evaluate a learned model.
    
    Args:
        model (tf.keras.Model)  : Learned model
        words_length (int) : max length of string what will be used in Evaluate.
    Return: 
        None
    """

    x_path,y,image_size=QRCodeRandomGenerator(EVALUATE_SAMPLE_NUM,TEMPLATE_IMAGE_DIR_PATH,words_length)

    test=[]

    for path in x_path:
        test.append(Utils.LoadImage(path))

    test=tf.concat(test,axis=0)

    answers=model.predict(test,batch_size=None)

    #Check Answer Accuracy
    zero_accuracy=0.0
    one_accuracy=0.0
    five_accuracy=0.0

    for i,answer in enumerate(answers):
        ans_string=Utils.UnNormalizeString(answer)

        zero_correct_counter=0
        one_correct_counter=0
        five_correct_counter=0
        for k in range(len(y[i])):
            char_index=TARGET_WORDS.index(y[i][k])

            zero_list  = [TARGET_WORDS[char_index+error_capacity] for error_capacity in range(0,1)]
            one_list   = [TARGET_WORDS[char_index+error_capacity] for error_capacity in range(-1,2) if char_index+error_capacity > 0 and char_index+error_capacity < len(TARGET_WORDS)]
            five_list  = [TARGET_WORDS[char_index+error_capacity] for error_capacity in range(-5,6) if char_index+error_capacity > 0 and char_index+error_capacity < len(TARGET_WORDS)]

            if ans_string[k] in zero_list:
                zero_correct_counter+=1
                one_correct_counter+=1
                five_correct_counter+=1

            elif ans_string[k] in one_list:
                one_correct_counter+=1
                five_correct_counter+=1

            elif ans_string[k] in five_list:
                five_correct_counter+=1

        zero_accuracy+=float(zero_correct_counter/len(y[i]))
        one_accuracy+=float(one_correct_counter/len(y[i]))
        five_accuracy+=float(five_correct_counter/len(y[i]))

    zero_accuracy   =   zero_accuracy / float(len(answers))
    one_accuracy    =   one_accuracy  / float(len(answers))
    five_accuracy   =   five_accuracy / float(len(answers))

    print("<Zero error capacity> Average Accuracy: %3.3f%%"%(zero_accuracy*100.))
    print("<One  error capacity> Average Accuracy: %3.3f%%"%(one_accuracy*100.))
    print("<Five error capacity> Average Accuracy: %3.3f%%"%(five_accuracy*100.))
        
def Main(args):
    """
    Entory point
    Args:
        args (argparser): arguments parser
    Return: 
        None
    """
    mode=args.mode
    epoch=args.epoch
    batch_size=args.batch_size
    split_size=args.split_size
    model_save_path=args.model_save_path
    load_model_path=args.load_model_path
    save_dir_image_path=args.save_dir_image_path
    words=args.words
    words_length=args.words_length
    turn_off_memory_allocate=args.turn_off_memory_allocate
    using_gpu_number=args.using_gpu_number

    Utils.SetGPUConfig(using_gpu_number,turn_off_memory_allocate)

    if mode==0:
        for version,max_string_length in enumerate(QRCODE_SIZE_LIST):
            save_path=os.path.join(save_dir_image_path,str(version+1))

            x,y,image_size=Utils.QRCodeRandomGenerator(10000,save_path,max_string_length)
            y=[tf.constant(Utils.NormalizeString(k),dtype=tf.float32) for k in y]

            model=Network.CreateNetwork(max_string_length,image_size,"mse","adagrad",[])

            sep=int(len(x)*split_size)
            Learning(model,model_save_path,max_string_length,image_size,epoch,batch_size,version+1,(x[sep:],y[sep:]),(x[:sep],y[:sep]))

    elif mode==1:
        if load_model_path is None:
            raise ValueError("In evaluate, load_model_path = None is invalid.")

        model=Network.LoadNetwork(load_model_path)
        Evaluate(model,words_length)

        shutil.rmtree(TEMPLATE_IMAGE_DIR_PATH)

    elif mode==2:
        if words is None:
            words=Utils.GenerateRandomString(words_length)
            print("Generate String:",words)

        if load_model_path is None:
            for i in range(len(QRCODE_SIZE_LIST)):
                if QRCODE_SIZE_LIST[i]>=len(words):
                    print("Auto select: Version=%d"%(i+1))
                    load_model_path=os.path.join("Models","Version%d.hdf5"%(i+1))
                    break

        model=Network.LoadNetwork(load_model_path)

        CreateQRCode(words,TEMPLATE_IMAGE_PATH)

        Prediction(model,TEMPLATE_IMAGE_PATH)

        shutil.rmtree(TEMPLATE_IMAGE_DIR_PATH)
        
def ParseArguments(args):
    """
    Argument Parser.

    Args:
        args (sys.args[1:]): Argment by this program.

    Return:
        paser
    """

    #parserの定義
    parser=argparse.ArgumentParser(description="サンプル画像に写った物体の位置検出を行うプログラム")

    #引数の設定
    parser.add_argument("-m","--mode",default=0,type=int,choices=[0,1,2],help="Starting mode (0: Learning, 1: Evaluate, 2: Predict)")
    parser.add_argument("-e","--epoch",default=20,type=int,help="Epoch")
    parser.add_argument("-b","--batch_size",default=16,type=int,help="batch size")
    parser.add_argument("-v","--split_size",default=0.2,type=float,help="validation percentage")
    parser.add_argument("-msp","--model_save_path",default="Models",type=str,help="path of save model")
    parser.add_argument("-lmp","--load_model_path",default=None,type=str,help="path of learned model. If this value is None, use optimal model in Models/")
    parser.add_argument("-s","--save_dir_image_path",default="QRImages",type=str,help="The dir path to save for QRCode Image")
    parser.add_argument("-w","--words",default=None,type=str,help="The words for generating QRCode")
    parser.add_argument("-l","--words_length",default=7,type=int,help="This argument is to define input words length for prediction.If already define words(w) option, this argument will be ignored.")
    parser.add_argument("--turn_off_memory_allocate",action="store_true",help="Turn off auto gpu memory allocate function")
    parser.add_argument("-g","--using_gpu_number",default=-1,type=int,help="Useing GPU Number. -1 is use all gpu")

    
    #引数の解析
    return parser.parse_args()

#エントリポイント
if __name__=="__main__":
    Main(ParseArguments(sys.argv[1:]))
