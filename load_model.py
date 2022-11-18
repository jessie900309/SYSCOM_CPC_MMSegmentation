from mmseg.apis import init_segmentor


def load_MMSmodel():
    config_file = 'mmsegmentation/configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py'
    checkpoint_file = 'mmsegmentation/checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
    return model


if __name__ == '__main__':
    load_MMSmodel()
