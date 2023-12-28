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
5. ![image](https://github.com/Antonio-Borges-Rufino/Traducao_Imagem/assets/86124443/3aedc1d2-e5fc-47a7-92c0-1a62bfcb32bf)
