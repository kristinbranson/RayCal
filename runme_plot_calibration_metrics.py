import yaml
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

calibration_logs_dir = './calibration_scripts_logs'
distance_tolerance = 2 * 1 # micron

files = [os.path.join(calibration_logs_dir, f) for f in os.listdir(calibration_logs_dir) if 'yaml' in f]
repr_errors = {}
pairwise_distance_errors = {}
triangulation_errors = {}

for file in files:
    with open(file, 'r') as f:
        calibration_yaml = yaml.safe_load(f)
    if calibration_yaml['training']['status'] == 'failed':
        continue
    base_filename = os.path.splitext(os.path.basename(file))[0]
    repr_error = {}
    repr_error['cam_0_real'] = calibration_yaml['training']['repr_error_real_cam_0']
    repr_error['cam_1_real'] = calibration_yaml['training']['repr_error_real_cam_1']
    repr_error['cam_0_virtual'] = calibration_yaml['training']['repr_error_virtual_cam_0']
    repr_error['cam_1_virtual'] = calibration_yaml['training']['repr_error_virtual_cam_1']
    pairwise_distance_error = calibration_yaml['training']['pairwise_distance_error']
    triangulation_error = calibration_yaml['training']['triangulation_error']
    repr_errors[base_filename] = repr_error
    pairwise_distance_errors[base_filename] = pairwise_distance_error * 1e3
    triangulation_errors[base_filename] = triangulation_error * 1e3


df_rep = pd.DataFrame(repr_errors).T.reset_index().melt(
id_vars="index", var_name="error_type", value_name="error"
)
df_rep = df_rep.rename(columns={"index": "experiment"})

# Style
sns.set_theme(style="whitegrid")

# Plot grouped bars
plt.figure(figsize=(10, 6))
ax = sns.barplot(
    data=df_rep,
    x="experiment",
    y="error",
    hue="error_type",
    palette="deep"
)
plt.ylabel('Mean reprojection error (pixel)')
plt.xticks(rotation=45)
plt.savefig(os.path.join(
    calibration_logs_dir,
    'repr_errors.png'
    )
    )
plt.savefig(os.path.join(
    calibration_logs_dir,
    'repr_errors.svg'
    )
    )

df_tri = pd.DataFrame(
    list(triangulation_errors.items()),
    columns=["experiment", "triangulation_error"]
)

sns.set_theme(style="whitegrid")

# Plot bars
plt.figure(figsize=(10, 6))
ax = sns.barplot(
    data=df_tri,
    x="experiment",
    y="triangulation_error",
    color="steelblue"
)

# Rotate x labels
plt.xticks(rotation=45)

# Labels
plt.xlabel("Experiment")
plt.ylabel("Mean triangulation Error ($\mu$m)")
plt.title("Triangulation Errors")

# Save
plt.savefig(os.path.join(
    calibration_logs_dir,
    'triangulation_errors.png'
))
plt.savefig(os.path.join(
    calibration_logs_dir,
    'triangulation_errors.svg'
))
plt.close()

# Plot pairwise distance error
df_pair = pd.DataFrame(
    list(pairwise_distance_errors.items()),
    columns=["experiment", "pairwise_distance_error"]
)

sns.set_theme(style="whitegrid")

# Plot bars
plt.figure(figsize=(10, 6))
ax = sns.barplot(
    data=df_pair,
    x="experiment",
    y="pairwise_distance_error",
    color="steelblue"
)


plt.axhline(
    y=distance_tolerance,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Manufacturer Tolerance ($\mu$m)"
)

# Rotate x labels
plt.xticks(rotation=45)

# Labels
plt.xlabel("Experiment")
plt.ylabel("Mean pairwise Distance Error ($\mu$m)")
plt.title("Pairwise Distance Errors")

# Save
plt.legend()
plt.savefig(os.path.join(
    calibration_logs_dir,
    'pairwise_distance_errors.png'
))
plt.savefig(os.path.join(
    calibration_logs_dir,
    'pairwise_distance_errors.svg'
))
plt.close()

# Pairwise distance histogram
sns.set_theme(style="whitegrid")

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(
    df_pair["pairwise_distance_error"],
    bins=10,  # Adjust number of bins as needed
    color="steelblue",
    edgecolor="black",
    alpha=0.7
)

plt.axvline(
    x=distance_tolerance,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Manufacturer Tolerance ($\mu$m)"
)

# Labels
plt.xlabel("Pairwise Distance Error ($\mu$m)")
plt.ylabel("Frequency (Number of Experiments)")
plt.title("Distribution of Pairwise Distance Errors")

# Save
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(
    calibration_logs_dir,
    'pairwise_distance_errors_histogram.png'
))
plt.savefig(os.path.join(
    calibration_logs_dir,
    'pairwise_distance_errors_histogram.svg'
))
