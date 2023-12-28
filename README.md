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
