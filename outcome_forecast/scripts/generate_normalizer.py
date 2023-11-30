"""General testing of dataloading"""
import os
from pathlib import Path

from src.data.replayFolder import SC2Replay, Split
import torch
import matplotlib.pyplot as plt


def generate():
    dataset = SC2Replay(
        Path(os.environ["DATAPATH"]),
        Split.TRAIN,
        0.8,
        {"minimap_features", "scalar_features"},
        2,
        300,
    )
    len_ = 5000
    all_data = torch.zeros(len_, 150, 28)

    for idx, sample in enumerate(dataset):
        if idx == len_:
            break
        all_data[idx, :, :] = sample["scalar_features"]

    keys = [
        "score_float",
        "idle_production_time",
        "idle_worker_time",
        "total_value_units",
        "total_value_structures",
        "killed_value_units",
        "killed_value_structures",
        "collected_minerals",
        "collected_vespene",
        "collection_rate_minerals",
        "collection_rate_vespene",
        "spent_minerals",
        "spent_vespene",
        "total_damage_dealt_life",
        "total_damage_dealt_shields",
        "total_damage_dealt_energy",
        "total_damage_taken_life",
        "total_damage_taken_shields",
        "total_damage_taken_energy",
        "total_healed_life",
        "total_healed_shields",
        "total_healed_energy",
        "minerals",
        "vespere",
        "popMax",
        "popArmy",
        "popWorkers",
        "gameStep",
    ]

    non_zero_mask = all_data != 0
    average_data = torch.sum(all_data * non_zero_mask, dim=0) / non_zero_mask.sum(dim=0)

    out = open("../src/data/normalize.py", "w")
    out.write("import numpy as np\nnormalizer = np.array([")
    for i in range(28):
        if torch.isnan(average_data[0, i]) or average_data[0, i] == 0:
            average_data[0, i] = torch.tensor(1)
        for j in range(1, 150):
            if torch.isnan(average_data[j, i]):
                average_data[j, i] = average_data[j - 1, i]
            if average_data[j, i] == 0:
                average_data[j, i] = torch.tensor(1)

        out.write("\t")
        out.write(str(list([x.item() for x in average_data[:, i]])))
        out.write(",")
        out.write("\n")
    out.write("]).T")
    out.close()

    std_dev_data = torch.sqrt(
        torch.sum((all_data - average_data) ** 2 * non_zero_mask, dim=0)
        / non_zero_mask.sum(dim=0)
    )

    for feature_index, k in enumerate(keys):
        feature_name = f"Feature: {k}"

        # Plotting
        plt.figure(figsize=(10, 6))

        # Plot average
        plt.plot(
            average_data[:, feature_index],
            label=f"{feature_name} - Average",
            color="blue",
        )

        # Plot standard deviation as shaded area
        plt.fill_between(
            range(average_data.size(0)),
            average_data[:, feature_index] - std_dev_data[:, feature_index],
            average_data[:, feature_index] + std_dev_data[:, feature_index],
            color="lightblue",
            alpha=0.4,
            label=f"{feature_name} - Standard Deviation",
        )

        # Add labels and title
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title(f"Average and Standard Deviation Over Time for {feature_name}")
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt.savefig(f"image_{feature_index}.png")


if __name__ == "__main__":
    generate()
