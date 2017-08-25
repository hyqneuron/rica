
# Reconstruction ICA

Reproducing [Reconstruction ICA](http://ai.stanford.edu/~quocle/LeKarpenkoNgiamNg.pdf), without whitening.

You'll need [PyTorch](http://pytorch.org). A GPU helps but instructruction is given in `main.py` about how to run it
without a GPU. The tunable parameters are also explained in `main.py`.

Just run `python main.py`. Will generate a set of weight images with different lambda values.

# sample weights

`lambda=0.4`
![weight_images_0.4.jpg](./rica_weight_images_0.4.jpg)

`lambda=0.8`
![weight_images_0.4.jpg](./rica_weight_images_0.8.jpg)

`lambda=1.2`
![weight_images_0.4.jpg](./rica_weight_images_1.2.jpg)

`lambda=1.6`
![weight_images_0.4.jpg](./rica_weight_images_1.6.jpg)

`lambda=2.0`
![weight_images_0.4.jpg](./rica_weight_images_2.0.jpg)

`lambda=2.4`
![weight_images_0.4.jpg](./rica_weight_images_2.4.jpg)

`lambda=2.8`
![weight_images_0.4.jpg](./rica_weight_images_2.8.jpg)

`lambda=3.2`
![weight_images_0.4.jpg](./rica_weight_images_3.2.jpg)

`lambda=3.6`
![weight_images_0.4.jpg](./rica_weight_images_3.6.jpg)

