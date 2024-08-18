
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import display
from pandas import option_context
import torch
import numpy as np
from matplotlib.colors import LightSource
from matplotlib import cm
import matplotlib.pyplot as plt
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import pandas as pd
import dataframe_image as dfi

from . import data_utils

# ccmap = getattr(cm, 'hot')
# ccmap = getattr(cm, 'CMRmap')
# ccmap = getattr(cm, 'turbo')
# ccmap = getattr(cm, 'gnuplot')
ccmap = getattr(cm, 'jet')
# ccmap = getattr(cm, 'gray')


def set_ccmap(k):
    global ccmap
    ccmap = getattr(cm, k)


# ANCHOR: Plotting


def ori_sal_his(ori, sal, title=None):
    n = 4
    fig, axs = plt.subplots(1, n, figsize=(n * 4, 4), dpi=100, gridspec_kw={
                            'width_ratios': [1 for i in range(n)], 'height_ratios': [1]})
    if title is not None:
        print(title)
    plt.sca(axs[0])
    plt.axis(False)
    plt.imshow(ori)
    plt.sca(axs[1])
    axs1 = plt.imshow(sal, cmap=ccmap)
    fig.colorbar(axs1, shrink=0.7)
    plt.sca(axs[2])
    plt.axis(False)
    overlap = show_cam_on_image(ori, sal, use_rgb=True)
    plt.imshow(overlap)
    plt.sca(axs[3])
    plt.axis(False)
    plt.imshow(data_utils.mask(ori, sal))
    plt.show()


def_su_arg = dict(
    # rstride=1,
    # cstride=1,
    linewidth=1,
    antialiased=False,
    shade=False
)


def plot_hor(img_arrs, cmap=ccmap,
             solo=False, rows=None, columns=None,
             subplot_titles=None, idx=[], cb=False, dpi=200):
    if isinstance(cmap, str) and hasattr(cm, cmap):
        cmap = getattr(cm, cmap)
    size = 4
    # elif rows is not None and rows > 1 and columns is not None:
    if rows is not None and rows >= 1 and columns is not None:
        fig, axs = plt.subplots(
            rows, columns, figsize=(size * columns, size * rows), dpi=dpi)
        for i in range(rows):
            for j in range(columns):
                if columns == 1:
                    axss = axs[i]
                elif rows == 1:
                    axss = axs[j]
                else:
                    axss = axs[i][j]
                plt.sca(axss)
                if subplot_titles is not None and i == 0:
                    # axs[i][j].title.set_text(subplot_titles[j], size=16)
                    axss.set_title(subplot_titles[j], fontsize=20)

                if j == 0:
                    axss.set_title(idx[i], fontsize=20)

                axss.axis('off')
                im = axss.imshow(img_arrs[i][j], cmap=cmap)
    else:
        if len(img_arrs) == 1:
            plt.figure(figsize=(size * len(img_arrs), size), dpi=dpi)
            plt.axis('off')
            im = plt.imshow(img_arrs[0], cmap=cmap)
        else:
            fig, axs = plt.subplots(
                1, len(img_arrs), figsize=(size * len(img_arrs), size), dpi=dpi)
            for i in range(len(img_arrs)):
                plt.sca(axs[i])
                if subplot_titles is not None:
                    axs[i].set_title(subplot_titles[i], fontsize=20)
                axs[i].axis('off')
                im = axs[i].imshow(img_arrs[i], cmap=cmap)
    if cb:
        plt.colorbar(im, ax=axs.ravel().tolist(), shrink=0.7)
    plt.box(False)
    plt.show()


def plot_3d(arr, ori, su_arg=def_su_arg, level=[], contour=False, title=None, cmap=ccmap):
    ls = LightSource(270, 45)
    arr = np.array(arr)
    xl = arr.shape[1]
    yl = arr.shape[0]
    x = np.array([i for i in range(xl)])
    y = np.array([i for i in range(yl)])
    x, y = np.meshgrid(x, y)
    z = np.transpose(arr, (1, 0))
    z = np.rot90(z)
    size = 5
    total_n = len(level) + 2
    fig, axs = plt.subplots(1, total_n, figsize=(total_n * size, size),
                            dpi=130, gridspec_kw={'width_ratios': [1 for i in range(total_n)]})
    if title is not None:
        print(title)
        # fig.suptitle(title, fontsize=14)

    def p3d(x, y, z, ax):
        ax.set_zlim(0, 1)
        rgb = ls.shade(z, cmap=ccmap, vert_exag=0.1, blend_mode='soft')
        if not contour:
            surf = ax.plot_surface(x, y, z, facecolors=rgb, **su_arg)
        else:
            surf = ax.contourf(x, y, z, cmap=ccmap)
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        ax.set_xticks([])
        ax.set_yticks([])

        return surf

    # original 3d
    plt.clf()
    ax0 = plt.subplot(1, total_n, 1, projection='3d')
    p3d(x, y, z, ax0)
    # fig.colorbar(cm.ScalarMappable(norm=None, cmap=ccmap), ax=ax0, shrink= 0.5, anchor=(0.5, 0.5))

    # 3d with view
    m = {1: (0, -90, 0), 2: (0, 180, 0), 3: (90, -90, 0)}
    for i in level:
        ax_view = m[i]
        ax = plt.subplot(1, total_n, i + 1, projection='3d')
        if i == 1:
            ax.set_xlabel('Front View')
        if i == 2:
            ax.set_ylabel('Side View')
        if i == 3:
            ax.set_xlabel('Overlook View')
        surf = p3d(x, y, z, ax)
        ax.view_init(*ax_view)
        if ax_view == (90, -90, 0):
            ax.set_zticks([])
        else:
            ax.set_zlabel('heat')

    axe = plt.subplot(1, total_n, total_n)
    axe.imshow(data_utils.mask(ori, arr))
    axe.axis(False)

    plt.draw()
    plt.pause(.1)
    plt.show()


def plot_43d(maps, ori, level=[1, 2, 3]):
    size = 4
    level = [] if level is None else level
    for m in maps:
        plot_3d(m[0], ori, level=level,
                title=f"3D surface for {m[1]}")


def clp(img):
    return np.clip(np.transpose(img, (1, 2, 0)) * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)


def plot_cam(model, images, targets):
    cam = GradCAM(model=model,
                  target_layers=[model.layer4],
                  use_cuda=torch.cuda.is_available())
    grayscale_cam = cam(
        input_tensor=images,
        targets=[ClassifierOutputTarget(
            t) for t in targets],
        aug_smooth=False,
        eigen_smooth=True)

    return grayscale_cam


def ttonp(t):
    return t.cpu().detach().numpy()


def transpose_sa(row, sa_arrs, sa_names=[], idx=[]):
    col = len(sa_arrs)
    sa_arrs_transposed = list(zip(*sa_arrs))
    plot_hor(sa_arrs_transposed,
             rows=row, columns=col, subplot_titles=sa_names, idx=idx)


def display_df(data, exclude=[], only=None, caption='', custom_order=None,
               columns=[], sort_key=None, save_path=None, ascending=True):
    with option_context('display.max_colwidth', 80):
        rename_method = {
            'GAG-3': "Guided AbsoluatGrad",
            'IG': "Integrated Gradients",
            'IG2': "Integrated Gradients",
            'GB': "Guided Backpropagation",
            'GuidedIG': "Guided IG",
            'BlurIG': "Blur IG",
            'Vanilla': "Vanilla Gradient",
        }
        for dd in data:
            if dd[-1] in rename_method.keys():
                dd[-1] = rename_method[dd[-1]]
        df = pd.DataFrame(data=data,
                          columns=columns)

        rename_column = {
            "RCAP2": "RCAP↑",
            "DAUC": "DAUC↓",
            "IAUC": "IAUC↑",
            "MAE": "MAE↓",
            "Log-Cosh Dice Loss": "Log-Cosh Dice Loss↓",
            # "RCAP2": "RCAP↑",
        }
        df = df.drop(['Key', 'Params'], axis=1).rename(
            columns=rename_column)

        # df['Prob: M1'] = df['Prob: M1'].map(lambda name: name * 100)
        # df['RCAP'] = df['RCAP'].map(lambda name: name * 100)
        # df['RCAP2'] = df['RCAP2'].map(lambda name: name * 100)
        order = []
        if sort_key is not None:
            df = df.sort_values(by=sort_key, ascending=ascending)
            order = df.index.to_list()
        if custom_order is not None:
            custom_dict = {}
            for i, n in enumerate(custom_order):
                custom_dict[n] = i
            df = df.sort_values(by=['Name'], key=lambda x: x.map(custom_dict))
            order = df.index.to_list()

        if len(exclude) > 0 and only is None:
            df = df.query(f"Key not in {exclude}")
        elif only is not None:
            df = df.query(f"Key in {only}")
        s = df.style.set_properties(**{
            'font-size': '10pt', 'border-color': 'black'
        }).background_gradient(cmap='YlGn').set_table_styles([
            {'selector': 'th.col_heading', 'props': 'text-align: center;'},
            {'selector': 'td', 'props': 'text-align: center; font-weight: bold;'},
            {
                'selector': 'caption',
                'props': 'caption-side: Top; font-size:14pt; margin:10pt 0pt;'
            },
            {  # for row hover use <tr> instead of <td>
                'selector': 'td:hover',
                'props': [('background-color', '#986fc2')]
            },
            {  # for row hover use <tr> instead of <td>
                'selector': 'tr:hover',
                'props': [('background-color', '#325366')]
            },
        ], overwrite=True).set_caption(caption)
        print([(i + 1, v) for i, v in enumerate(np.array(df['Name']))])
        if save_path is not None:
            dfi.export(s, save_path, dpi=200)
        s.hide()
        display(s)
        return order


def display_df_df(df, exclude=[], only=None, caption='',
                  sort_key=None, save_path=None, ascending=True):
    with option_context('display.max_colwidth', 80):
        rename_column = {
            "RCAP2": "RCAP↑",
            "DAUC": "DAUC↓",
            "IAUC": "IAUC↑",
            "MAE": "MAE↓",
            "Log-Cosh Dice Loss": "Log-Cosh Dice Loss↓",
            # "RCAP2": "RCAP↑",
        }

        order = []
        if sort_key is not None:
            df = df.sort_values(by=sort_key, ascending=ascending)
            order = df.index.to_list()

        if len(exclude) > 0 and only is None:
            df = df.query(f"Key not in {exclude}")
        elif only is not None:
            df = df.query(f"Key in {only}")
        s = df.style.set_properties(**{
            'font-size': '10pt', 'border-color': 'black'
        }).background_gradient(cmap='YlGn', vmin=1.0).set_table_styles([
            {'selector': 'th.col_heading', 'props': 'text-align: center;'},
            {'selector': 'td', 'props': 'text-align: center; font-weight: bold;'},
            {
                'selector': 'caption',
                'props': 'caption-side: Top; font-size:14pt; margin:10pt 0pt;'
            },
            {  # for row hover use <tr> instead of <td>
                'selector': 'td:hover',
                'props': [('background-color', '#986fc2')]
            },
            {  # for row hover use <tr> instead of <td>
                'selector': 'tr:hover',
                'props': [('background-color', '#325366')]
            },
        ], overwrite=True).set_caption(caption)
        display(s)
        return order


def plot_sanity_check(sanity_check_rs_map: dict):
    fig, ax = plt.subplots(dpi=200, figsize=(6, 3))
    ax.set_ylim([-0.05, 1.02])
    data = []
    new_sanity_check_rs_map = {}
    for setting_key, sa_rss in sanity_check_rs_map.items():
        key, name, des, stat, eval_rs, \
            entropy, setting, instance_path = sa_rss

        mprt = eval_rs['mprt']
        if mprt is not None:
            new_sanity_check_rs_map[name] = mprt

    for method, sanity_check_rs in new_sanity_check_rs_map.items():
        for img_number, sanity_check_rs_per_img in enumerate(sanity_check_rs):
            for layer_number, sanity_check_value in enumerate(sanity_check_rs_per_img):
                data.append([method, sanity_check_value,
                            img_number, layer_number])

    df = pd.DataFrame(data, columns=["Method", 'Sanity', 'Image', 'Layer'])
    sns.lineplot(x="Layer", y="Sanity", style='Method',
                 hue="Method", markers=True, dashes=False, legend='brief',
                 data=df)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.set_xticks([])
    ax.set_ylabel('Spearman Rank Correlation')
    ax.set_xlabel('Removed Layer (from Output to Input)')
    plt.show()


def radar(aggregated_quantus_rs_map, write_path=None, range=[-0.2, 1.1]):
    # method_name:score for all metrics
    data = []
    for k, v in aggregated_quantus_rs_map.items():
        gen_eval_rs = v[5]
        for e in gen_eval_rs:
            method_name, eval_rs = e
            data.append([method_name, v[0], eval_rs])

    df = pd.DataFrame(data=data, columns=[
                      'XAI Method', 'metric_name', 'gen_value'])

    fig = px.line_polar(df, r="gen_value",
                        theta="metric_name", color="XAI Method",
                        line_close=True,
                        template='plotly',
                        color_discrete_sequence=px.colors.qualitative.Vivid,
                        range_r=range)
    # fig.update_traces(fill='toself')
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                gridcolor='#c7c4e4'
            )
        ),
        showlegend=True,
        width=800,
        height=450,
        margin={
            't': 30,
            'b': 30,
            'l': 150,
        },
        legend=dict(
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.15
        )
    )
    if write_path is not None:
        fig.write_image(write_path, scale=3)
    fig.show()


def plot_eval_rs(
        aggregated_eval_rs_map, aggregated_quantus_rs_map, custom_order=None,
        # sort_key="Score: M1", ascending=True, table_name="Evaluation Score"):
        sort_key=None, ascending=True, table_name="Evaluation Score"):
    # if aggregated_quantus_rs_map is not None and len(aggregated_quantus_rs_map.keys()) > 0:
    #     radar(aggregated_quantus_rs_map)
    data_map = {}
    extra_info = {}
    has_extra_info = False
    # if aggregated_quantus_rs_map is not None:
    #     for k, v in aggregated_quantus_rs_map.items():
    #         aggregated_eval_rs_map[k] = v
    for metrics_name, v in aggregated_eval_rs_map.items():
        eval_rs = v[2]
        for e_rs in eval_rs:
            if data_map.get(e_rs[0]) is None:
                data_map[e_rs[0]] = []

            if len(e_rs) > 2:
                has_extra_info = True
                extra_info[e_rs[0]] = e_rs[2:]
            data_map[e_rs[0]].append(e_rs[1])
    for kk, vv in extra_info.items():
        data_map[kk].extend(vv)

    data = []
    for k, v in data_map.items():
        data.append([*v, k])
    # print(data)

    metrics_title = [v[0] for k, v in aggregated_eval_rs_map.items()]

    col = [*metrics_title, "Key", "Params", "Name"] if has_extra_info else [
        *metrics_title, 'Name'
    ]
    display_df(
        data=data, columns=col,
        caption=table_name, custom_order=custom_order, sort_key=sort_key, ascending=ascending)

    return aggregated_eval_rs_map


def save_pic(exp_key, ran_idx, unary_rs, settings, cmap=ccmap, save_recovered=False):
    original_imgs = unary_rs[0]
    saliency_maps = unary_rs[2]
    recovered_imgs = unary_rs[3]
    for i, image_idx in enumerate(ran_idx):
        plt.imshow(clp(
            original_imgs[i].cpu().detach().numpy()))
        plt.grid(False)
        plt.axis('off')
        plt.savefig(f'./paper_resources/{exp_key}/{image_idx}_original.png',
                    bbox_inches='tight', pad_inches=0, dpi=300)
        plt.show()
        for j, setting_key in enumerate(list(settings.keys())):
            print(setting_key)
            current_saliency_map = saliency_maps[setting_key][i].cpu(
            ).detach().numpy()
            plt.imshow(current_saliency_map, cmap=cmap,
                       interpolation='nearest')
            plt.grid(False)
            plt.axis('off')
            plt.savefig(f'./paper_resources/{exp_key}/{image_idx}_{setting_key}.png',
                        bbox_inches='tight', pad_inches=0, dpi=300)
            plt.show()

            if save_recovered:
                current_recovered_imgs = recovered_imgs[j][i]
                for k, recovered_img in enumerate(current_recovered_imgs):
                    plt.imshow(recovered_img.transpose(1, 2, 0),
                               cmap=cmap, interpolation='nearest')
                    plt.grid(False)
                    plt.axis('off')
                    plt.savefig(
                        f'./paper_resources/{exp_key}/{image_idx}_{setting_key}_recovered_{k}.png', bbox_inches='tight', pad_inches=0, dpi=300)
                    plt.show()


def plot_metric_for_gag(data):
    plt.figure(figsize=(6, 3), dpi=160)
    # plt.plot(data[2])
    plt.plot(data[3])
    # plt.plot(data[1])
    plt.show()
