import os
import numpy as np
from copy import deepcopy

from abc import ABC, abstractmethod

import source.evaluation.namespace as eva_ns

from source.utils.vis_utils import mix_colors


class BasePlotter(ABC):

    def __init__(self, methods_list):
        self.aliases_list = [[ms[eva_ns.ALIAS] for ms in msl[eva_ns.METHODS].values()] for msl in methods_list]
        self.colors_list = [[ms[eva_ns.COLOR] for ms in msl[eva_ns.METHODS].values()] for msl in methods_list]
        self.plot_params_list = [msl[eva_ns.PLOT_PARAMS] for msl in methods_list]

    def plot(self, idx, save):
        fig, axes = self._get_figure_and_axes()

        base_plot_params = self.plot_params_list[idx]
        plot_params = base_plot_params.get(self._get_name(), base_plot_params)

        plot_title = base_plot_params[eva_ns.PLOT_TITLE]

        if plot_title is not None and not save:
            fig.suptitle(plot_title, fontsize=plot_params.get(eva_ns.TITLE_FONT_SIZE, 20))

        self._plot_impl(fig, axes, idx, save)

        try:
            iter(axes)

            legend = plot_params.get(eva_ns.LEGEND_LOC, np.full(axes.shape, eva_ns.BEST))

            for axi, legendi in zip(axes, legend):
                axi.legend(loc=legendi)

        except TypeError:
            legend = plot_params.get(eva_ns.LEGEND_LOC, eva_ns.BEST)
            axes.legend(loc=legend)

        if save:
            save_dir = plot_params[eva_ns.SAVE_DIR]

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            fig.savefig(os.path.join(save_dir, plot_params[eva_ns.SAVE_NAME]), bbox_inches='tight')

    def plot_all(self, save):
        for idx in range(len(self.aliases_list)):
            self.plot(idx, save)

    @abstractmethod
    def _get_figure_and_axes(self):
        ...

    @abstractmethod
    def _plot_impl(self, fig, axes, idx, save):
        ...

    @abstractmethod
    def _get_name(self):
        ...


class LinePlotter(BasePlotter):

    def __init__(self, methods_list):
        super().__init__(methods_list)
        self.line_styles_list = [[ms[eva_ns.LINE_STYLE] for ms in msl[eva_ns.METHODS].values()] for msl in methods_list]


def prepare_methods_list(methods_list, select=None):
    if select is not None:
        return [methods_list[idx] for idx in select]

    else:
        return methods_list


def prepare_ablation_methods_list(base_methods_dict, base_methods_setup,
                                  test_dir, evaluation_task, ablation,
                                  ablation_title):
    ablation_methods_list = []

    for model_name in base_methods_dict.keys():
        abl_cats = sorted(os.listdir(os.path.join(test_dir, evaluation_task, model_name, ablation)))

        color = base_methods_dict[model_name][eva_ns.COLOR]

        ablation_method_dict = {}

        for i, abl_cat in enumerate(abl_cats):
            abl_name = os.path.join(model_name, ablation, abl_cat)
            mix = float(i) / len(abl_cats)

            ablation_method_dict[abl_name] = deepcopy(base_methods_dict[model_name])
            ablation_method_dict[abl_name][eva_ns.ALIAS] += f"/{abl_cat}"
            ablation_method_dict[abl_name][eva_ns.COLOR] = mix_colors(color, 'red', mix)

        ablation_method_setup = deepcopy(base_methods_setup)
        ablation_method_setup[eva_ns.METHODS] = ablation_method_dict

        plot_params = ablation_method_setup[eva_ns.PLOT_PARAMS]
        plot_params[eva_ns.PLOT_TITLE] = ablation_title

        format_save_variables(plot_params, ablation, model_name)

        for v in plot_params.values():
            if isinstance(v, dict):
                format_save_variables(v, ablation, model_name)

        ablation_methods_list.append(ablation_method_setup)

    return ablation_methods_list


"""
Support utils
"""

def format_save_variables(_dict, ablation, model_name):
    if eva_ns.SAVE_DIR in _dict:
        _dict[eva_ns.SAVE_DIR] = _dict[eva_ns.SAVE_DIR].format(ablation)

    if eva_ns.SAVE_NAME in _dict:
        _dict[eva_ns.SAVE_NAME] = _dict[eva_ns.SAVE_NAME].format(model_name)
