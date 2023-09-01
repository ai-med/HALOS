# This file is part of Hallucination-free Organ Segmentation after Organ Resection Surgery (HALOS).
#
# HALOS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HALOS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HALOS. If not, see <https://www.gnu.org/licenses/>.
import io
import os
import pickle
from datetime import datetime
from math import log, pow, ceil, floor
from pathlib import Path
from typing import Optional, Any, OrderedDict

import numpy as np
import pandas as pd
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter


def get_missing_organ_df(source_path: str, datafield: int, save_csv: bool, target_path: Optional[str]) -> pd.DataFrame:
    """
    Extract patient information that without a certain organ. Criterion: exclude patients who only reported the
    corresponding operation in the First Repeat Imaging Visit (instance 3).
    Args:
        source_path: filepath of the original csv table.
        datafield: target datafield number that specifies the operation.
        save_csv: whether to save the extracted information or not.
        target_path: filepath of the new csv table.

    Returns:
        A DataFrame that contains eligible patient information.
    """
    df = pd.read_csv(source_path)
    df_operation_only = df.loc[df[(df.iloc[:, 603:] == datafield).any(1)].index.to_list()]
    sum_prev = df_operation_only.iloc[:, 603:699].isin({datafield}).sum(1)
    sum_last = df_operation_only.iloc[:, 699:].isin({datafield}).sum(1)
    check_last = sum_prev.loc[sum_last[sum_last == 1].index.to_list()]
    check_prev = check_last[check_last == 0].index.to_list()
    df_extracted = df_operation_only.drop(check_prev).drop(columns=["Unnamed: 0"]).reset_index(drop=True)
    if save_csv:
        df_extracted.to_csv(target_path)
    return df_extracted


def export_filenames_to_txt(source_path: str, target_path: str, path_prefix: str, path_postfix: str) -> None:
    """
    Generate a txt file containing the selected cases' paths. This function can help to download the data needed.
    Args:
        source_path: filepath of the csv table (selected cases).
        target_path: filepath of the new txt.
        path_prefix: directory prefix of the actual data, e.g, "home/xxx/".
        path_postfix: directory postfix of the actual data, e.g, "_20201_2_0" for N4_cropped images.
    """
    df = pd.read_csv(source_path)
    ids = df["eid"].to_list()
    with open(target_path, mode='wt', encoding='utf-8') as f:
        for id in ids:
            f.write(os.path.join(path_prefix, str(id)) + path_postfix)
            f.write('\n')


def export_datafields_to_csv(df: pd.DataFrame, ids: np.ndarray, target_path: str) -> None:
    """
    Create a csv table for selected cases and their datafields.
    Args:
        df: original dataframe.
        ids: case ids.
        target_path: filepath of the new csv table.
    """
    df_new = df[df["eid"].isin(ids)].drop(columns=["Unnamed: 0"]).reset_index(drop=True)
    df_new.to_csv(target_path)


def find_closest_powers_of_two(x: Any) -> int:
    """
    Given an input value, find the closest value to it that is divisible by powers of two.
    Args:
        x: input value.
    Returns:
        Its closest integer value which is divisible by powers of two.
    """
    orders = floor(log(x, 2)), ceil(log(x, 2))
    closest_order = min(orders, key=lambda z: abs(x - 2 ** z))
    return int(pow(2, closest_order))


def find_closest_n_powers_of_two(x: Any, n: int) -> int:
    """
    Given an input value, find the closest upper value to it that is divisible by n powers of two.
    Args:
        x: input value.
        n: divide by 2.^n continuously.
    Returns:
        Its closest upper integer value which is divisible by n powers of two.
    """
    divisor = pow(2, n)
    remain = x % divisor
    return int(x + (divisor - remain))


def create_run_folder(path: str) -> str:
    """
    Create a folder with current timestamp.
    Args:
        path: target path.
    Returns:
        The created path.
    """
    now = datetime.now()
    time = now.strftime("%Y%m%d_%H%M%S")
    path_target = os.path.join(path, time)
    Path(path_target).mkdir(parents=True, exist_ok=True)
    return path_target


def extract_state_dict_encoder_only(model_dict: OrderedDict) -> dict:
    """
    Load state dictionary for only encoder parts. Can be used after pre-training to remove the projection layers.
    Args:
        model_dict: the original pre-trained model's state dictionary.
    Returns:
        New state dictionary of encoder.
    """
    keys = list(model_dict.keys())
    keys_enc = [key for key in keys if key.startswith("enc_blocks")]
    encoder_dict = {k: v for k, v in model_dict.items() if k in keys_enc}
    return encoder_dict


class CPU_Unpickler(pickle.Unpickler):
    """
    Alternative way to load pickle file that is obtained from GPU on CPU device.
    Usage: contents = CPU_Unpickler(f).load()
    """

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def record_metric_results(writer: SummaryWriter, data: str, dict_metrics: dict, epoch: int):
    """
    Write metric summary of current epoch to TensorBoard.
    Args:
        writer: the TensorBoard SummaryWriter.
        data: "train" or "val"
        dict_metrics: a dictionary of saved metric results.
        epoch: current epoch number.
    """
    for key, value in dict_metrics.items():
        writer.add_scalar(key + "/" + data, value, epoch + 1)


def record_metric_results_wandb(run: wandb, data: str, dict_metrics: dict, epoch: int):
    """
    Write metric summary of current epoch to wandb.
    Args:
        run: the wandb object.
        data: "train" or "val"
        dict_metrics: a dictionary of saved metric results.
        epoch: current epoch number.
    """
    d = {}
    for key, value in dict_metrics.items():
        name = key + "/" + data
        d[name] = value[-1]
    d["epoch"] = epoch
    run.log(d)


def create_metric_bar_chart_wandb(run: wandb, metric_name: str, dataset: str, dict_metrics: dict):
    """
    Log a custom bar chart to wandb.
    Args:
        run: the wandb object.
        metric_name: the name of metric to be logged.
        dataset: "train" or "val"
        dict_metrics: a dictionary of saved metric results.
    """
    data = []
    for key, value in dict_metrics.items():
        data.append([key, value[-1]])
    data.pop()
    table = wandb.Table(data=data, columns=["organ", metric_name])
    run.log({"comparison/" + dataset: wandb.plot.bar(table, "organ", metric_name,
                                                     title="classwise " + metric_name + ": " + dataset)})
