import argparse
import time
import os


# 函数参数定义
def parse_args():
    parser = argparse.ArgumentParser(description="RemoteSensingSegmentation by PyTorch")

    # dataset
    parser.add_argument('--dataset-name', type=str, default='Vaihingen')
    parser.add_argument('--train-data-root', type=str,
                        default='/home/students/master/2022/wangzy/PyCharm-Remote/ST_Unet_test/Vaihingen_Img/Train/')
    parser.add_argument('--val-data-root', type=str,
                        default='/home/students/master/2022/wangzy/PyCharm-Remote/ST_Unet_test/Vaihingen_Img/Test/')
    parser.add_argument('--train-batch-size', type=int, default=8, metavar='N',
                        help='batch size for training (default:16)')
    parser.add_argument('--val-batch-size', type=int, default=8, metavar='N',
                        help='batch size for testing (default:16)')

    # output_save_path
    # strftime格式化时间，显示当前的时间
    parser.add_argument('--experiment-start-time', type=str,
                        default=time.strftime('%m-%d-%H:%M:%S', time.localtime(time.time())))
    parser.add_argument('--save-pseudo-data-path', type=str,
                        default='/home/students/master/2022/wangzy/PyCharm-Remote/ST_Unet_test/pseudo_data')
    parser.add_argument('--save-file', default=False)

    # augmentation
    parser.add_argument('--base-size', type=int, default=256, help='base image size')
    parser.add_argument('--crop-size', type=int, default=256, help='crop image size')
    parser.add_argument('--flip-ratio', type=float, default=0.5)
    parser.add_argument('--resize-scale-range', type=str, default='0.5, 2.0')

    # model
    parser.add_argument('--model', type=str, default='Swin_Transformer', help='model name')
    parser.add_argument('--pretrained', action='store_true', default=True)

    # criterion
    # 损失的权重值
    parser.add_argument('--class-loss-weight', type=list, default=
    # [0.007814952234152803, 0.055862295151291756, 0.029094606950899726, 0.03104357983254851, 0.22757710412943985, 0.19666243636646102, 0.6088052968747066, 0.15683966777104494, 0.5288489922602664, 0.21668940382940433, 0.04310240828376457, 0.18284053575941367, 0.571096349549462, 0.32601488184885147, 0.45384359272537766, 1.0])
    # [0.007956167959807792, 0.05664417300631733, 0.029857031694750392, 0.03198534634969046, 0.2309102255169529,
    #  0.19627322641039702, 0.6074939752850792, 0.16196525436190998, 0.5396602408824741, 0.22346488456565283,
    #  0.04453628275090391, 0.18672995330033487, 0.5990724459491834, 0.33183887346397484, 0.47737597643193597, 1.0]
    [0.008728536232175135, 0.05870821984204281, 0.030766985878693004, 0.03295408432939304, 0.2399409412190348,
     0.20305583055639448, 0.6344888568739531, 0.16440413437125656, 0.5372260524694122, 0.22310945250778813,
     0.04659596810284655, 0.19246378709444723, 0.6087430986295436, 0.34431415558778183, 0.4718853977371564, 1.0])

    # loss
    parser.add_argument('--loss-names', type=str, default='cross_entropy')
    parser.add_argument('--classes-weight', type=str, default=None)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default:0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0001, metavar='M', help='weight-decay (default:1e-4)')

    # optimizer
    parser.add_argument('--optimizer-name', type=str, default='SGD')

    # learning_rate
    parser.add_argument('--base-lr', type=float, default=0.01, metavar='M', help='')

    # environment
    parser.add_argument('--use-cuda', action='store_true', default=True, help='using CUDA training')
    parser.add_argument('--num-GPUs', type=int, default=1, help='numbers of GPUs')
    parser.add_argument('--num_workers', type=int, default=32)

    # validation
    parser.add_argument('--eval', action='store_true', default=False, help='evaluation only')
    parser.add_argument('--no-val', action='store_true', default=False)

    parser.add_argument('--best-miou', type=float, default=0)

    parser.add_argument('--total-epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 120)')
    parser.add_argument('--start-epoch', type=int, default=0, metavar='N', help='start epoch (default:0)')

    parser.add_argument('--resume-path', type=str, default=None)

    args = parser.parse_args()

    directory = "weight/%s/%s/%s/" % (args.dataset_name, args.model, args.experiment_start_time)
    args.directory = directory

    if args.save_file:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print("Creat and Save model.pth!")

    return args
