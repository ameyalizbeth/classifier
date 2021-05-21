# importing necessary libraries and functions

import random
from PIL import Image
from flask import (Flask, flash, render_template, redirect, request, session,
                   send_file, url_for,  jsonify)
from werkzeug.utils import secure_filename
import torch
import urllib.request
import torch.nn as nn
from torchvision.transforms import transforms
import numpy as np
from torch.autograd import Variable
from torchvision.models import squeezenet1_1
import torch.functional as F
from io import open
import os
from PIL import Image
import pathlib
import glob
import cv2
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
classes=['mammooty', 'mohanlal']
LETTER_SET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'


class ConvNet(nn.Module):
    def __init__(self,num_classes=6):
        super(ConvNet,self).__init__()
        
        #Output size after convolution filter
        #((w-f+2P)/s) +1
        
        #Input shape= (256,3,150,150)
        
        self.conv1=nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
        #Shape= (256,12,150,150)
        self.bn1=nn.BatchNorm2d(num_features=12)
        #Shape= (256,12,150,150)
        self.relu1=nn.ReLU()
        #Shape= (256,12,150,150)
        
        self.pool=nn.MaxPool2d(kernel_size=2)
        #Reduce the image size be factor 2
        #Shape= (256,12,75,75)
        
        
        self.conv2=nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
        #Shape= (256,20,75,75)
        self.relu2=nn.ReLU()
        #Shape= (256,20,75,75)
        
        
        
        self.conv3=nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
        #Shape= (256,32,75,75)
        self.bn3=nn.BatchNorm2d(num_features=32)
        #Shape= (256,32,75,75)
        self.relu3=nn.ReLU()
        #Shape= (256,32,75,75)
        
        
        self.fc=nn.Linear(in_features=75 * 75 * 32,out_features=num_classes)
        
        
        
        #Feed forwad function
        
    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)
            
        output=self.pool(output)
            
        output=self.conv2(output)
        output=self.relu2(output)
            
        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)
            
            
            #Above output will be in matrix form, with shape (256,32,75,75)
            
        output=output.view(-1,32*75*75)
            
            
        output=self.fc(output)
            
        return output

checkpoint=torch.load('best_checkpoint.model', map_location=torch.device('cpu'))
model=ConvNet(num_classes=2)
model.load_state_dict(checkpoint)
model.eval()

def generate_random_name(filename):
    """ Generate a random name for an uploaded file. """
    ext = filename.split('.')[-1]
    rns = [random.randint(0, len(LETTER_SET) - 1) for _ in range(3)]
    name = ''.join([LETTER_SET[rn] for rn in rns])
    return "{new_fn}.{ext}".format(new_fn=name, ext=ext)

def is_allowed_file(filename):
    """ Checks if a filename's extension is acceptable """
    allowed_ext = filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    return '.' in filename and allowed_ext


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def prediction(img_path,transformer):
    
    image=pil_loader(img_path)
    
    
    image_tensor=transformer(image).float()
    
    
    image_tensor=image_tensor.unsqueeze_(0)
    
    if torch.cuda.is_available():
        image_tensor.cuda()
        
    input=Variable(image_tensor)
    
    
    output=model(input)
    if(output.data.numpy().max() < 0.035):
         pred="defintely not mamooty or mohanlal"
         return pred
    index=output.data.numpy().argmax()
    print(output)
    pred=classes[index]
    
    return pred




app = Flask(__name__) #Initialize the flask App
app.secret_key = '1234'
app.config['UPLOAD_FOLDER'] = '.\\predicting_images'
pred_path = '.\\predicting_images'

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        # show the upload form
        return render_template('index.html')

    if request.method == 'POST':
        if request.form.to_dict()['url'] != '' :
            url =  request.form.to_dict()
            imgurl = url['url']
            imgurl2 = imgurl
            filename = imgurl.split('/')[-1]
            print(filename)
            if is_allowed_file(filename):
                
                opener = urllib.request.URLopener()
                opener.addheader('User-Agent', 'whatever')
                filename2, headers = opener.retrieve(imgurl2, './predicting_images/'+filename )
                #urllib.request.urlretrieve(imgurl2, './predicting_images/'+filename)
                
                return redirect(url_for('predict', filename=filename))
            else:
                return render_template('error.html',text="oops try another url")
                
                

            return redirect(request.url)
        # check if a file was passed into the POST request
        if 'image' not in request.files:
            return render_template('error.html',text="no file was uploaded")

        image_file = request.files['image']

        # if filename is empty, then assume no upload
        if image_file.filename == '':
            return render_template('error.html',text="no file was uploaded")

        # check if the file is "legit"
        if image_file and is_allowed_file(image_file.filename):
            filename = secure_filename(generate_random_name(image_file.filename))
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(filepath)
            # HACK: Defer this to celery, might take time
            
            
            return redirect(url_for('predict', filename=filename))
        else:
            return render_template('error.html',text="hey upload files of format png,jpg and jpeg. ")

@app.route('/predict/<filename>')
def predict(filename):
    
    #image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    images_path1=glob.glob(pred_path+'/*.jpg')
    images_path2=glob.glob(pred_path+'/*.jpeg')
    images_path3=glob.glob(pred_path+'/*.png')
    image_path = images_path1 + images_path2 + images_path3
     
    print(image_path)
    transformer=transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5,0.5,0.5], # 0-1 to [-1,1] , formula (x-mean)/std
                        [0.5,0.5,0.5])
    ])
 	
    pred_dict=""

    for i in image_path:
        pred_dict=prediction(i,transformer)
    os.remove(image_path[0])

    return render_template(
        'predict.html',prediction_text=pred_dict
      )
    
if __name__ == "__main__":
    app.run(debug=True)
