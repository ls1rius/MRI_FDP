from .autoencoder import AutoencoderKL


class MRIVAE(AutoencoderKL):
    def __init__(self, ddconfig, lossconfig, embed_dim, **kwargs):
        super().__init__(ddconfig=ddconfig, lossconfig=lossconfig, embed_dim=embed_dim, **kwargs)

    def validation_step(self, batch, batch_idx):
        data = batch[self.image_key]
        ind = data.shape[-1] // 2
        batch[self.image_key] = data[..., ind:ind + 1]
        super().validation_step(batch, batch_idx)
