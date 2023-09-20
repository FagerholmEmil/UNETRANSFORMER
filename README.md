# UNETRANSFORMER
Summary: Fusion of UNET and Transformer

This project leverages deep learning to identify the tumor locations in MRI images. Recently, encoder-decoder architectures (e.g. U-net) have become the de facto standard for image segmentation. However, the locality of convolutional layers limits the global context and long-range spatial dependencies. These limitations are resolved through the fusion between Transformers, which extract long-range dependencies with self-attention mechanisms, with U-Net which captures low-level spatial details efficiently. The problems with data variability between clinics and scarcity of data are solved by fine-tuning the network and augmenting new training data. 

<img width="713" alt="19c387ec-df70-4d60-847f-331910aa384f" src="https://github.com/FagerholmEmil/UNETRANSFORMER/assets/114832342/fbe8c483-31cb-4b53-a59e-a5a60369361b">

