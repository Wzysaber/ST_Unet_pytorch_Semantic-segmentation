import ml_collections


def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    # config.resnet.att_type = 'CBAM'
    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None
    # config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
    config.patch_size = 16

    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return config


def get_testing():
    """Returns a minimal configuration for testing."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


def get_r50_b16_config():
    """Returns the Resnet50 + ViT-B/16 configuration.-------------------------wo yong de """
    config = get_b16_config()

    # 构建config.data容器，将不同的类型给放入进去
    config.data = ml_collections.ConfigDict()
    config.data.img_size = 256  # 6144
    config.data.in_chans = 3

    # 放入种类数目和相应的patch，就是256*256的图片划分成为4*4的patch结构，共256/4的数量
    config.n_classes = 6
    config.patches.grid = (4, 4)

    # 构建config.resnet容器，将不同的类型给放入进去
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 6, 3)  # resnet的层数结构
    config.resnet.width_factor = 0.5

    config.classifier = 'seg'  # 种类名称

    # 构建 config.trans容器，也就是辅助encoder（swin transformer）中的各个必要参数
    config.trans = ml_collections.ConfigDict()
    config.trans.num_heads = [3, 6, 12, 24]  # 注意力的头的数目
    config.trans.depths = [2, 2, 6, 2]  # swin transformer的网络结构深度
    config.trans.embed_dim = 96
    config.trans.window_size = 8

    # config.pretrained_path = '../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz' #yuxunlian

    # (256,128,64,16)#
    # #1024,512,256,128,64)
    # #(2048,1024,512,256,128)
    # #(256, 128, 64, 16)
    # 解码的通道数
    config.decoder_channels = (512, 256, 128, 64)

    # 链接的通道数量
    # [256,128,64,16]#[512,256,128,64,16]#[512,256,128,64,32]#[1024,512,256,128,64]#[512, 256, 64, 16]
    config.skip_channels = [512, 256, 128, 64]

    config.n_classes = 6  # 分类的个数
    config.n_skip = 4  # 链接的次数，或者直接理解成阶段数
    config.activation = 'softmax'

    return config


def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.patches.size = (32, 32)
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_32.npz'
    return config


def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.representation_size = None

    # custom
    config.classifier = 'seg'
    config.resnet_pretrained_path = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-L_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return config


def get_r50_l16_config():
    """Returns the Resnet50 + ViT-L/16 configuration. customized """
    config = get_l16_config()
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.resnet_pretrained_path = '../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.activation = 'softmax'
    return config


def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.patches.size = (32, 32)
    return config


def get_h14_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (14, 14)})
    config.hidden_size = 1280
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 5120
    config.transformer.num_heads = 16
    config.transformer.num_layers = 32
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None

    return config
