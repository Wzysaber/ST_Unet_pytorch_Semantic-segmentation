from PIL import Image
import numpy as np

# 染色板将图片进行染色
palette = [
    255, 255, 255,  # 0  #surface
    0, 0, 255,      # 1  #building
    0, 255, 255,    # 2  #low vegetation
    0, 255, 0,      # 3  #tree
    255, 255, 0,    # 4  #car
    255, 0, 0,      # 5  #clutter/background red
]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


# 将grey mask转化为彩色mask

# putpalette
# 为“P”或者“L”图像增加一个调色板。对于“L”图像，它的模式将变化为“P”。
# 调色板序列需要包含768项整数，每组三个值表示对应像素的红，绿和蓝。用户可以使用768个byte的字符串代替这个整数序列。

def colorize_mask(mask):
    mask_color = Image.fromarray(mask.astype(np.uint8)).convert('P')
    mask_color.putpalette(palette)
    return mask_color
