
---
displayName: CIFAR-100
labelTypes:
- Classification
license:
- Unknown
mediaTypes:
- Image
paperUrl: https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
publishDate: "2009-04-08"
publishUrl: http://www.cs.toronto.edu/~kriz/cifar.html
publisher:
- Massachusetts Institute of Technology
- New York University
tags: []
taskTypes:
- Image Classification

---
# 数据集介绍
  ## 简介
  这个数据集就像 CIFAR-10，除了它有 100 个类，每个类包含 600 张图像。每个类别有 500 个训练图像和 100 个测试图像。 CIFAR-100 中的 100 个类被分为 20 个超类。每个图像都带有一个“精细”标签（它所属的类）和一个“粗糙”标签（它所属的超类）。
  ## 类定义
  ```
aquatic mammals: beaver, dolphin, otter, seal, whale
fish: aquarium fish, flatfish, ray, shark, trout
flowers: orchids, poppies, roses, sunflowers, tulips
food containers: bottles, bowls, cans, cups, plates
fruit and vegetables: apples, mushrooms, oranges, pears, sweet peppers
household electrical devices: clock, computer keyboard, lamp, telephone, television
household furniture: bed, chair, couch, table, wardrobe
insects: bee, beetle, butterfly, caterpillar, cockroach
large carnivores: bear, leopard, lion, tiger, wolf
large man-made outdoor things: bridge, castle, house, road, skyscraper
large natural outdoor scenes: cloud, forest, mountain, plain, sea
large omnivores and herbivores: camel, cattle, chimpanzee, elephant, kangaroo
medium-sized mammals: fox, porcupine, possum, raccoon, skunk
non-insect invertebrates: crab, lobster, snail, spider, worm
people: baby, boy, girl, man, woman
reptiles: crocodile, dinosaur, lizard, snake, turtle
small mammals: hamster, mouse, rabbit, shrew, squirrel
trees: maple, oak, palm, pine, willow
vehicles 1: bicycle, bus, motorcycle, pickup truck, train
vehicles 2: lawn-mower, rocket, streetcar, tank, tractor
```
  ## 引文
  ```
@article{krizhevsky2009learning,
  title={Learning multiple layers of features from tiny images},
  author={Krizhevsky, Alex and Hinton, Geoffrey and others},
  year={2009},
  publisher={Citeseer}
}
```
  
## Download dataset
:modelscope-code[]{type="git"}