# Sobre o Projeto
1. CycleGans são ferramentas de transformação de estilos em imagens, ou seja, a partir de uma imagem existente podemos gerar outras imagens
2. Esse tipo de transformação é comumente chamada de tradução de imagem, nesse projeto vamos realizar a tradução de imagens utilizando cyclegans
3. Para mais informações, consulte: Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros. "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks", in IEEE International Conference on Computer Vision (ICCV), 2017. (* indicates equal contributions) Bibtex
4. ![image](https://github.com/Antonio-Borges-Rufino/Traducao_Imagem/assets/86124443/c58784fe-ac92-46ff-9733-9845b91f9688)


# Arquitetura de CycleGan
1. Uma rede cycle gan possui 2 discriminadores e 2 geradores
2. Nesse projeto, os geradores serão redes resnet
3. Basicamente, existe um ciclo entre um dominio A e um dominio B onde a imagem fica sendo gerada e discriminada para realizar a transferencia de estilo
4. O funcionamento é exemplificado abaixo
![image](https://github.com/Antonio-Borges-Rufino/Traducao_Imagem/assets/86124443/3aedc1d2-e5fc-47a7-92c0-1a62bfcb32bf)

# Cycle Loss
1. O primeiro tipo de função de perda que vamos empregar é a função de perda ciclica.
2. A função de perda ciclica é baseada na imagem acima representada entre os 2 dominios A e B
3. Imagine uma imagem que queremos traduzir em outra, a grande dificuldade é que, muitas vezes não temos a imagem alvo que queremos produzir, precisando assim realizar uma função de perda correspondente.
4. Para a função de perda ciclica, primeiro dividimos em 2 funções de perda diferentes, a relacionada ao gerador AB e o seu discriminador e a relacionada ao gerador BA e seu discriminador.
5. Gerador AB: As imagens do dominio A passam pelo tradutor (gerador AB) para o dominio B, após isso, as imagens passam pelo disciminador B e depois passam pelo gerador BA, com a intenção de reescrever a imagem novamente no dominio A, com isso, temos o caminho A->B->A', então realizamos a diferença entre A e A' por alguma função de perda convecional, como MAE = |A-A'|
6. Gerador BA: O gerador BA é o inverso do gerador AB, com isso, o caminho de geração fica B->A->B' e a função de perda é MAE = |B-B'|
7. Por fim, a função de perda final é a soma das 2 funções de perda anteriores, ficando = Final_Loss = MAE|A-A'| + MAE|B-B'|

# Identity Loss
1. A perda de identidade serve para preservar as cores das imagens que pertecem a um dominio
2. Vamos usar o mesmo esquema ciclico com A->GeradorAB->A' e B->GeradorBA->B' para produzir as perdas ciclicas A = MAE(|A-A'|) e B = MAE(|B-B'|)
3. Dessa vez, a função de perda final fica total_loss = (cycleA+cycleB)/2

# Patch-GAN Loss
1. Essa é a perda do discriminador.
2. O discrimador nas redes GAN's convencionais geralmente tem uma saída bolena indicando se algo é ou não é.
3. Para a rede ciclica, a saida é uma matriz relacionada a redução do espaço matricial da matriz da imagem verdadeira.
4. O custo então e computado através da redução da imagem verdadeira com a da rede neural.
5. Como isso acontece: Suponha-se que a imagem de um gerador seja uma matriz de 256x256, então essa matriz é reduzida de forma que ela se transforma em uma matriz de 16x16.
6. Agora a rede neural precisa só ajustar para a matrix 16x16, que é a matriz alvo

# Implementação do Residual Block
```
import torch.nn as nn
class ResidualBlock(nn.Module):
  def __init__(self, in_features):
    super(ResidualBlock,self).__init__()
    self.block = nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_features,in_features,3),
        nn.InstanceNorm2d(in_features),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_features,in_features,3),
        nn.InstanceNorm2d(in_features)
    )
  def forward(self,x):
    return x + self.block(x)
```

# Implementação do gerador
```
import torch.nn as nn

class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
```

# Implementação do discriminador
```
import torch.nn as nn
class Discriminator(nn.Module):
  def __init__(self, input_shape):
    super(Discriminator,self).__init__()
    channels, height, widht = input_shape
    self.output_shape = (1,height//2**4,widht//2**4)

    def discriminator_block(in_filters,out_filters,normalize = True):
      layer = [nn.Conv2d(in_filters,out_filters,4,stride=2,padding=1)]
      if normalize:
        layer.append(nn.InstanceNorm2d(out_filters))
      layer.append(nn.LeakyReLU(0.2,inplace=True))
      return layer

    self.model = nn.Sequential(
        *discriminator_block(channels,64,normalize=False),
        *discriminator_block(64,128),
        *discriminator_block(128,256),
        *discriminator_block(256,512),
        nn.ZeroPad2d((1,0,1,0)),
        nn.Conv2d(512,1,4,padding=1)
    )

  def forward(self, img):
        return self.model(img)
```

# Implementação de métodos auxiliares
1. Importação das bibliotecas
```
import glob
import random
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import random
import time
import datetime
import sys
from torch.autograd import Variable
import torch
import numpy as np
```
2. Classe ReplayBuffer
* Recebe como parâmetro um tamanho máximo fixo chamada self.max_size
* Possui um método chamado push_and_pop, esse método recebe como parâmetro a imagem gerada por um gerador
* Um for percorre essa imagem indo de vetor em vetor dentro dela, a imagem portanto é limitada pelo tamanho máximo fixo passado como parâmetro
* O trecho de código torch.unsqueeze(element, 0) tem como objetivo transformar o vetor do pixel da imagem em uma matriz, por exemplo, um vetor [56,56,56] é transformado em [[56,56,56]]
* Então verifica-se se a variavel global self.data é menor que a variavel global self.max_size, caso seja menor, self.data recebe o resultado de torch.unsqueeze(element, 0) caso não ocorre uma verificação a partir de aleatoriedade, caso um numero aleatorio seja maior que o pré-definido, de forma aleatoria pega-se uma matriz de self.data e insere na variavel final chamada to_return e essa mesma matriz de self.data é substituida pela nova matriz de torch.unsqueeze(element, 0). Caso o numero aleatório seja menor não é alterado nada em self.data e a variavel final to_return recebe a matriz de torch.unsqueeze(element, 0)
* Por fim, o método retorna a concatenação de todas as matrizes de to_return em uma matriz imagem nova atraveés do método torch.cat em formato de tensor com Variable do pytorch
* Abaixo está o código
```
class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))
```
3. Class LambdaLR  
* Classe que serve para descrever uma taxa de aprendizagem dinâmica a partir de uma taxa inicial
* Com o tempo no treinamento, a taxa de aprendizado vai mudando com base no método step da classe instanciada
* recebe os parâmetros n_epochs (é a quantidade de épocas de treinamento), offset (é a época inicial), decay_start_epoch (época em que começa a decair a taxa de aprendizado)
```
class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
```
4. Função que gera uma nova imagem
* Funciona para transformar em RGB uma imagem que não é RGB
```
def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image
```
5. Classe ImageDataset
* O método __init__ recebe como parâmetro o caminho dos arquivos de imagem (root), parâmetros de transformação de imagem do pytorch vision (transforms_), um parâmetro unaligned que serve para pegar aleatoriamente imagens caso seja True. Ainda em __init__ as variaveis globais files_A e files_B servem para buscar as imagens, no caso, indicada por mode, que indica se ela busca nos arquivos de treinamento ou de teste ou algum outro arquivo. A variavel self.transform recebe parâmetros de transformação de imagem do pytorch vision chamado torchvision.transform.Compose.
* O método __getiten__ recupera as imagens carregadas, transforma usando pytorch torchvision e retorna apenas as imagens transformadas
```
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*"))

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
```
# Implementação do looping de treinamento
* Créditos:  PyTorch-GAN Git repo
1. Importação das bibliotecas
```
import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
```
2. Conjunto de hiperparâmetro
```
epoch = 0
n_epochs = 30
dataset_name = "horse2zebra"
batch_size = 2
lr = 0.0002
b1 = 0.5
b2 = 0.0002
decay_epoch = 10
n_cpu = 4
img_height=256
img_width=256
channels=3
sample_interval = 100
n_residual_blocks = 9
lambda_cyc = 10
lambda_id = 5.0
checkpoint_interval = 1
```
3. Inicializando diretorios de imagens de amostras e pesos do modelo
```
os.makedirs("images/%s" % dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % dataset_name, exist_ok=True)
```
4. Inicializando variaveis de perdas (Loss_Function), geradores e discriminadores
```
#Funções de perda
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
#Tamanho das imagens
input_shape = [channels, img_height, img_width]
#Discrimadores e Geradores
G_AB = GeneratorResNet(input_shape, n_residual_blocks)
G_BA = GeneratorResNet(input_shape, n_residual_blocks)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)
```
5. Caso exista placas de video, utilizar as implemetações cuda
```
torch.cuda.empty_cache()
cuda = torch.cuda.is_available()
if cuda:
  G_AB = G_AB.cuda()
  G_BA = G_BA.cuda()
  D_A = D_A.cuda()
  D_B = D_B.cuda()
  criterion_GAN.cuda()
  criterion_cycle.cuda()
  criterion_identity.cuda()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
```
6. Inicializando os pesos do modelo, caso já exista pesos ele carrega, caso não inicia com os pesos de uma distribuição normal
```
def weights_init_normal(m):
  classname = m.__class__.__name__
  if classname.find('Linear') != -1:
    y = m.in_features
    m.weight.data.normal_(0.0,1/np.sqrt(y))
    m.bias.data.fill_(0)

if epoch != 0:
  G_AB.load_state_dict(torch.load("saved_models/%s/G_AB_%d.pth" % (dataset_name, epoch)))
  G_BA.load_state_dict(torch.load("saved_models/%s/G_BA_%d.pth" % (dataset_name, epoch)))
  D_A.load_state_dict(torch.load("saved_models/%s/D_A_%d.pth" % (dataset_name, epoch)))
  D_B.load_state_dict(torch.load("saved_models/%s/D_B_%d.pth" % (dataset_name, epoch)))
else:
  G_AB.apply(weights_init_normal)
  G_BA.apply(weights_init_normal)
  D_A.apply(weights_init_normal)
  D_B.apply(weights_init_normal)
```
7. Inicializando otimizadores
* O otimizador das redes geradores é compartilhado
```
optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(b1, b2)
    )
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(b1, b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(b1, b2))
```
8. Inicializa as otimizações dinâmicas das taxas de aprendizado
* É aqui onde é implementada a função LambdaLR diretamente na função do pytorch
* Segue o mesmo esquema de compartilhamento entre as redes geradoras
```
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step
    )

lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step
    )

lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step
    )
```
9. Inicialização da classe ReplayBuffer para guardar imagens já geradas
```
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()
```
10. Inicialização dos parâmetros de processamento de imagem do pytorch com torchvision
```
transforms_ = [
  transforms.Resize(int(img_height * 1.12), Image.BICUBIC),
  transforms.RandomCrop((img_height, img_width)),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ]
```
11. Loading das imagens para treinamento e teste
```
!unzip /content/data.zip
```
```
# Training data loader
dataloader = DataLoader(
  ImageDataset("data/%s" % dataset_name, transforms_=transforms_, unaligned=True),
  batch_size=batch_size,
  shuffle=True,
  num_workers=n_cpu,
    )
    # Test data loader
val_dataloader = DataLoader(
  ImageDataset("data/%s" % dataset_name, transforms_=transforms_, unaligned=True, mode="test"),
  batch_size=5,
  shuffle=True,
  num_workers=1,
    )
```
12. Função auxiliar para geração de imagens e salvamento das imagens de teste
```
def sample_images(batches_done):
  imgs = next(iter(val_dataloader))
  G_AB.eval()
  G_BA.eval()
  real_A = Variable(imgs["A"].type(Tensor))
  fake_B = G_AB(real_A)
  real_B = Variable(imgs["B"].type(Tensor))
  fake_A = G_BA(real_B)
  # Arange images along x-axis
  real_A = make_grid(real_A, nrow=5, normalize=True)
  real_B = make_grid(real_B, nrow=5, normalize=True)
  fake_A = make_grid(fake_A, nrow=5, normalize=True)
  fake_B = make_grid(fake_B, nrow=5, normalize=True)
  # Arange images along y-axis
  image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
  save_image(image_grid, "images/%s/%s.png" % (dataset_name, batches_done), normalize=False)
```
13. Looping de treinamento
```
prev_time = time.time()
for epoch in range(epoch, n_epochs):
  for i, batch in enumerate(dataloader):
    # Set model input
    real_A = Variable(batch["A"].type(Tensor))
    real_B = Variable(batch["B"].type(Tensor))

    # Adversarial ground truths
    valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
    fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)

    # ------------------
    #  Train Generators
    # ------------------

    G_AB.train()
    G_BA.train()

    optimizer_G.zero_grad()

    # Identity loss
    loss_id_A = criterion_identity(G_BA(real_A), real_A)
    loss_id_B = criterion_identity(G_AB(real_B), real_B)

    loss_identity = (loss_id_A + loss_id_B) / 2

    # GAN loss
    fake_B = G_AB(real_A)
    loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
    fake_A = G_BA(real_B)
    loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

    loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

    # Cycle loss
    recov_A = G_BA(fake_B)
    loss_cycle_A = criterion_cycle(recov_A, real_A)
    recov_B = G_AB(fake_A)
    loss_cycle_B = criterion_cycle(recov_B, real_B)

    loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

    # Total loss
    loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity

    loss_G.backward()
    optimizer_G.step()

    # -----------------------
    #  Train Discriminator A
    # -----------------------

    optimizer_D_A.zero_grad()

    # Real loss
    loss_real = criterion_GAN(D_A(real_A), valid)
    # Fake loss (on batch of previously generated samples)
    fake_A_ = fake_A_buffer.push_and_pop(fake_A)
    loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
    # Total loss
    loss_D_A = (loss_real + loss_fake) / 2

    loss_D_A.backward()
    optimizer_D_A.step()

    # -----------------------
    #  Train Discriminator B
    # -----------------------

    optimizer_D_B.zero_grad()

    # Real loss
    loss_real = criterion_GAN(D_B(real_B), valid)
    # Fake loss (on batch of previously generated samples)
    fake_B_ = fake_B_buffer.push_and_pop(fake_B)
    loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
    # Total loss
    loss_D_B = (loss_real + loss_fake) / 2

    loss_D_B.backward()
    optimizer_D_B.step()

    loss_D = (loss_D_A + loss_D_B) / 2

    # --------------
    #  Log Progress
    # --------------

    # Determine approximate time left
    batches_done = epoch * len(dataloader) + i
    batches_left = n_epochs * len(dataloader) - batches_done
    time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
    prev_time = time.time()

    print("================================")
    print("Epoch: {}".format(epoch/n_epochs))
    print("Batch: {}".format(i/len(dataloader)))
    print("D loss: {}".format(loss_D.item()))
    print("G loss: {}".format(loss_G.item()))
    print("adv: {}".format(loss_GAN.item()))
    print("cycle: {}".format(loss_cycle.item()))
    print("identity: {}".format(loss_identity.item()))
    print("ETA: {}".format(time_left))

    # If at sample interval save image
    if batches_done % sample_interval == 0:
      sample_images(batches_done)
    #sys.stdout.write(torch.cuda.memory_summary(device=None, abbreviated=False))

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
    # Save model checkpoints
      torch.save(G_AB.state_dict(), "savedModels/G_AB_%d.pth" % (epoch))
      torch.save(G_BA.state_dict(), "savedModels/G_BA_%d.pth" % (epoch))
      torch.save(D_A.state_dict(), "savedModels/D_A_%d.pth" % (epoch))
      torch.save(D_B.state_dict(), "savedModels/D_B_%d.pth" % (epoch))
```



















