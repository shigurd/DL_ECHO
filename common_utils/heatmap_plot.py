import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def show_preds_heatmap(preds):
    ''' creates heatmaps and orders them for saving in summary_writer '''
    preds_detached = preds.detach().cpu().numpy()

    stack_np = []
    for layer in preds_detached:
        for channel in layer:
            maxval = np.max(channel)
            minval = np.min(channel)
            temp = (channel - minval) / (maxval - minval)

            cmap = plt.get_cmap('jet')
            rgba_img = cmap(temp.squeeze())
            rgb_img = rgba_img[:, :, :-1] # deletes alphachannel
            plot_tb = (rgb_img * 255).astype(np.uint8).transpose((2, 0, 1))
            stack_np.append(plot_tb)

            # plot = Image.fromarray((rgb_img * 255).astype(np.uint8))
            # plot.show()

    plot_np = np.stack(stack_np)

    return plot_np