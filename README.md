# SimpleSelfAttention (Created 5/14/2019)

## (x * x^T) * (W * x)

Python 3.7, Pytorch 1.0.0, fastai 1.0.52


The purpose of this repository is two-fold:
- demonstrate improvements brought by the use of a self-attention layer in an image classification model.
- introduce a new layer which I call SimpleSelfAttention, which is a modified version of the SelfAttention described in [4]

## Updates

v0.3 (6/21/2019)
- Changed the order of operations in SimpleSelfAttention (in xresnet.py), it should run much faster (see Self Attention Time Complexity.ipynb)
- added fast.ai's csv logging in train.py

v0.2 (5/31/2019)
- Original standalone notebook is now in folder "v0.1"
- model is now in xresnet.py, training is done via train.py (both adapted from fastai repository)
- Added option for symmetrical self-attention (thanks @mgrankin for the implementation)
- Added support for multiple GPU (thanks to fastai)
- Added option to run fastai's learning rate finder
- Added option to use xresnet18 to xresnet152 baseline architectures

Note: we recommend starting with a single GPU, as running multiple GPU will require additional hyperparameter tuning.

## How to run (see 'examples' notebook):

%run train.py --woof 1 --size 256 --bs 64 --mixup 0.2 --sa 1 --epoch 5  --lr 3e-3

- woof: 0 for Imagenette, 1 for Imagewoof (dataset will download automatically)
- size: image size
- bs: batch size
- mixup: 0 for no mixup data augmentation
- sa: 1 if we use SimpleSelfAttention, otherwise 0
- sym: 1 if we add symmetry to SimpleSelfAttention (need to have sa=1)
- epoch: number of epochs
- lr: learning rate
- lrfinder: 1 to run learning rate finder, don't train
- dump: 1 to print model, don't train
- arch: default is 'xresnet50'
- gpu: gpu to train on (by default uses all available GPUs??)
- log: name of csv file to save training log to (folder path is displayed when running)


For faster training on multiple GPUs, you can try running: python -m fastai.launch train.py (not tested much)


## Image classification results (work in progress)

We compare a baseline resnet model to the same model with an extra self-attention layer (SimpleSelfAttention, which I will describe further down).

### Same run time ~50 epochs test (xresnet18, 128px, Imagewoof dataset[1])

#### 1) We first run the original xresnet18 model for 50 epochs with a range of learning rates and pick the best one:

| Model | Dataset | Image Size | Epochs | Learning Rate | # of runs | Avg (Max Accuracy) |
|---|---|---|---|---|---|---|
| xresnet18 | Imagewoof | 128 | 50 | 1e-3  | 10 | 0.821 |
| xresnet18 | Imagewoof | 128 | 50 | 3e-3  | 30  | 0.845 |
| xresnet18 | Imagewoof | 128 | 50 | 5e-3  | 10  | 0.846 |
| xresnet18 | Imagewoof | 128 | 50 | **8e-3**  | 20  | **0.850** |
| xresnet18 | Imagewoof | 128 | 50 | 1e-2 | 20 | 0.846 |
| xresnet18 | Imagewoof | 128 | 50 | 12e-3  | 20 | 0.844 |
| xresnet18 | Imagewoof | 128 | 50 | 14e-3 | 20 | 0.847 |

Note: we are not using mixup.

#### 2) We pick a number of epochs for our xresnet18+SimpleSelfAttention model that gives the same runtime or less as the baseline model and use the learning rate from step 1

Results using the original self-attention layer are added as a reference.



| Model | Dataset | Image Size | Epochs | Learning Rate | # of runs | Avg (Max Accuracy) | Stdev (Max Accuracy) | Avg Wall Time (# of obs) |
|---|---|---|---|---|---|---|---|---|
| xresnet18 | Imagewoof | 128 | 50 | 8e-3  | 20 | 0.8498 | 0.00782 | 9:37 (4)|
| xresnet18 + simple sa | Imagewoof | 128 | 47 | 8e-3  | 20  | **0.8567** | 0.00937 | 9:28 (4) |
| xresnet18 + original sa | Imagewoof | 128 | 47 | 8e-3  | 20  | 0.8547 | 0.00652 | 11:20 (1) |

This is using a single RTX 2080 Ti GPU. We use the %%time function on Jupyter notebooks. 


Parameters:

%run train.py --woof 1 --size 128 --bs 64 --mixup 0 --sa 0 --epoch 50  --lr 8e-3 --arch 'xresnet18'

%run train.py --woof 1 --size 128 --bs 64 --mixup 0 --sa 1 --epoch 47  --lr 8e-3 --arch 'xresnet18'




We can compare the results using an independent samples t-test (https://www.medcalc.org/calc/comparison_of_means.php):

- Difference: 0.007
- 95% confidence interval: 0.0014 to 0.0124
- Significance level: P = 0.0157


Adding a SimpleSelfAttention layer seems to provide a statistically significant boost in accuracy after training for ~50 epochs, without additional run time, and while using a learning rate optimized for the original model.

SimpleSelfAttention provides similar results as the original SelfAttention, while decreasing run time.


### Same run time ~100 epochs test (xresnet18, 128px, Imagewoof dataset[1])

We use the same parameters as for 50 epochs and double the number of epochs:



| Model | Dataset | Image Size | Epochs | Learning Rate | # of runs | Avg (Max Accuracy) | Stdev (Max Accuracy) | Avg Wall Time(# of obs) |
|---|---|---|---|---|---|---|---|---|
| xresnet18 | Imagewoof | 128 | 100 | 8e-3  | 23 | 0.8576 | 0.00817 | 20:05 (4) |
| xresnet18 + simple sa | Imagewoof | 128 | 94 | 8e-3  | 23  | **0.8634** | 0.00740 | 19:27 (4) |

- Difference: 0.006
- 95% CI	0.0012 to 0.0104
- Significance level	P = 0.0153


### ~100 epochs test with Mixup=0.2 (xresnet18, 128px, Imagewoof dataset[1])

| Model | Dataset | Image Size | Epochs | Learning Rate | # of runs | Avg (Max Accuracy) | Stdev (Max Accuracy) | Avg Wall Time(# of obs) |
|---|---|---|---|---|---|---|---|---|
| xresnet18 | Imagewoof | 128 | 100 | 8e-3  | 15 | 0.8636 | 0.00585 | ? |
| xresnet18 + simple sa | Imagewoof | 128 | 94 | 8e-3  | 15  | 0.87106 | 0.00726 | ? |
| xresnet18 + original sa | Imagewoof | 128 | 94 | 8e-3  | 15  | 0.8697 | 0.00726 | ? |

Again here, SimpleSelfAttention performs as well as the original self-attention layer and beats the baseline model.

### ~50 epochs , 256px images, Mixup = 0.2

| Model | Dataset | Image Size | Epochs | Learning Rate | # of runs | Avg (Max Accuracy) | Stdev (Max Accuracy) | Avg Wall Time(# of obs) |
|---|---|---|---|---|---|---|---|---|
| xresnet18 | Imagewoof | 256 | 50 | 8e-3  | 15 | 0.9005 | 0.00595 | _ |
| xresnet18 + simple sa | Imagewoof | 256 | 47 | 8e-3  | 15  | 0.9002 | 0.00478 | _ |

So far, no detected improvement when using 256px wide images.


## Simple Self Attention layer

The only difference between baseline and proposed model is the addition of a self-attention layer at a specific position in the architecture. 

The new layer, which I call SimpleSelfAttention, is a modified and simplified version of the fastai implementation ([3]) of the self attention layer described in the SAGAN paper ([4]).


#### Original layer:

          
     class SelfAttention(nn.Module):
    
      "Self attention layer for nd."

      def __init__(self, n_channels:int):
          super().__init__()
          self.query = conv1d(n_channels, n_channels//8)
          self.key   = conv1d(n_channels, n_channels//8)
          self.value = conv1d(n_channels, n_channels)
          self.gamma = nn.Parameter(tensor([0.]))

      def forward(self, x):
          #Notation from https://arxiv.org/pdf/1805.08318.pdf
          size = x.size()
          x = x.view(*size[:2],-1)
          f,g,h = self.query(x),self.key(x),self.value(x)
          beta = F.softmax(torch.bmm(f.permute(0,2,1).contiguous(), g), dim=1)
          o = self.gamma * torch.bmm(h, beta) + x
          return o.view(*size).contiguous()

#### Proposed layer:
      
   Edit (6/21/2019): order of operations matters to reduce complexity! Changed from x * (x^T * (conv(x))) to (x * x^T) * conv(x)
        
    class SimpleSelfAttention(nn.Module):
    
    def __init__(self, n_in:int, ks=1):#, n_out:int):
        super().__init__()           
        self.conv = conv1d(n_in, n_in, ks, padding=ks//2, bias=False)    
        self.gamma = nn.Parameter(tensor([0.]))       
        self.sym = sym
        self.n_in = n_in
        
    def forward(self,x):               
                  
        size = x.size()  
        x = x.view(*size[:2],-1)   # (C,N)             
        
        convx = self.conv(x)   # (C,C) * (C,N) = (C,N)   => O(NC^2)
        xxT = torch.bmm(x,x.permute(0,2,1).contiguous())   # (C,N) * (N,C) = (C,C)   => O(NC^2)
        
        o = torch.bmm(xxT, convx)   # (C,C) * (C,N) = (C,N)   => O(NC^2)
          
        o = self.gamma * o + x
        
          
        return o.view(*size).contiguous()      


As described in the SAGAN paper ([4]), the original layer takes the image features x of shape (C,N) (where N = H * W), and transforms them into f(x) = Wf * x and g(x) = Wg * x, where Wf and Wg have shape (C,C'), and C' is chosen to be C/8. Those matrix multiplications can be expressed as (1 * 1) convolution layers. Then, we compute S = (f(x))^T * g(x).

Therefore, S = (Wf * x)^T * (Wg * x) = x^T * (Wf ^T * Wg) * x. My first proposed simplification is to combine (Wf ^T * Wg) into a single (C * C) matrix W. So S = x^T * W * x.  S = S(x,x) (bilinear form) is of shape (N * N) and will represent the influence of each pixel on other pixels ("the extent to which the model attends to the ith location when synthesizing the jth region" [4]). Note that S(x,x) depends on the input, whereas W does not. (I suspect that having the same bilinear form for every input might be the reason we do better on Imagewoof = 10 dog breeds than Imagenette = 10 very different classes)

Thus, we only learn weights W for one convolution layer instead of weights Wf and Wg for two convolution layers. Advantages are: simplicity, removal of one design choice (C' = C/8), and a matrix W that offers more possibilities than Wf ^T * Wg. One possible drawback is that we have more parameters to learn (C^2 vs C^2/4). One option we haven't tried here is to force W to be a symmetrical matrix. This would reduce the number of parameters and force the influence of "pixel" j on pixel i to be the same as pixel i on pixel j.

Edit: @mgrankin tested symmetry and got a small improvement [5]

The next step in the original version of the layer is to compute the softmax of matrix S. I decided to remove this step completely and work with unrestricted weights instead of normalized probability-like weights.

The final step in the original version is to compute h(x) = Wh * x (Wh of shape (C * C)), which is also implemented as a 1 * 1 convolution layer. Then our final output is o = gamma * h(x) * S + x.  We propose to remove this final convolution layer and have the output be o = gamma * x * S + x. This final convolution could be re-added as a separate layer if desired, although this implies a different position for the skip connection.





# References

[1] https://github.com/fastai/imagenette

[2] https://github.com/fastai/fastai/blob/master/examples/train_imagenette.py

[3] https://github.com/fastai/fastai/blob/5c51f9eabf76853a89a9bc5741804d2ed4407e49/fastai/layers.py

[4] https://arxiv.org/abs/1805.08318

[5] https://github.com/mgrankin/SimpleSelfAttention/blob/master/Imagenette%20Simple%20Symmetric%20Self%20Attention.ipynb
