import os
import matplotlib.pyplot as plt
import numpy as np


def plot_results(diameters_nm, image_path, df=None):
    """
    Generate histograms: overall + morphology-separated.

    PRESERVES: Original overall histogram
    ADDS: Morphology breakdown plots (if df provided)
    """
    base = os.path.splitext(os.path.basename(image_path))[0]

    # === EXISTING: Overall histogram (UNCHANGED) ===
    plt.figure(figsize=(10, 4))
    plt.hist(diameters_nm, bins=30, color="skyblue", edgecolor="black")
    plt.xlabel("Diameter (nm)")
    plt.ylabel("Count")
    plt.title(f"Histogram of Nanoparticle Diameters: {base}")
    plt.grid(True)
    plt.tight_layout()
    out_path = f"outputs/figures/{base}_diameter_histogram.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

    # === NEW: Morphology-specific plots ===
    if df is not None and "Morphology" in df.columns:
        # Figure 2: 4-panel morphology histograms
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))


def _generate_batch_report(self) -> None:
    """
    Generate aggregate outputs for batch processing.

    Creates:
    - Combined CSV with all particles
    - Summary statistics table
    - Comparison visualizations
    """
    import pandas as pd
    from scripts.visualization.plotting import plot_batch_comparison

    logging.info(f"\n{'='*60}")
    logging.info("GENERATING BATCH REPORT")
    logging.info(f"{'='*60}")

    # Combine all dataframes
    df_all = pd.concat(self.batch_results, ignore_index=True)

    # Save combined CSV
    combined_csv = "outputs/results/batch_all_particles.csv"
    df_all.to_csv(combined_csv, index=False)
    logging.info(f"Saved combined CSV: {combined_csv}")

    # Generate summary statistics per image
    summary_data = []
    for img_name in df_all["Image"].unique():
        img_df = df_all[df_all["Image"] == img_name]

        summary_data.append(
            {
                "Image": img_name,
                "Total_Particles": len(img_df),
                "Mean_Diameter_nm": img_df["Diameter (nm)"].mean(),
                "Std_Diameter_nm": img_df["Diameter (nm)"].std(),
                "Min_Diameter_nm": img_df["Diameter (nm)"].min(),
                "Max_Diameter_nm": img_df["Diameter (nm)"].max(),
                "Spherical_Count": len(img_df[img_df["Morphology"] == "spherical"]),
                "RodLike_Count": len(img_df[img_df["Morphology"] == "rod-like"]),
                "Aggregate_Count": len(img_df[img_df["Morphology"] == "aggregate"]),
            }
        )

    df_summary = pd.DataFrame(summary_data)
    summary_csv = "outputs/results/batch_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    logging.info(f"Saved summary statistics: {summary_csv}")

    # Print summary table
    print(f"\n{'='*80}")
    print("BATCH SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(df_summary.to_string(index=False))
    print(f"{'='*80}")
    print(f"Total Images Processed: {len(df_summary)}")
    print(f"Total Particles Detected: {len(df_all)}")
    print(f"Overall Mean Diameter: {df_all['Diameter (nm)'].mean():.2f} nm")
    print(f"Overall Std Deviation: {df_all['Diameter (nm)'].std():.2f} nm")
    print(f"{'='*80}\n")

    # Generate comparison plots
    plot_batch_comparison(df_all, df_summary)

    logging.info("Batch report generation complete")
    fig.suptitle(f"Morphology Analysis - {base}", fontsize=16)

    colors = {"spherical": "green", "rod-like": "blue", "aggregate": "red"}

    # Panel 1: Overlay all types
    ax = axes[0, 0]
    for morph in ["spherical", "rod-like", "aggregate"]:
        data = df[df["Morphology"] == morph]["Diameter (nm)"]
        if len(data) > 0:
            ax.hist(
                data,
                bins=20,
                alpha=0.6,
                label=morph.capitalize(),
                color=colors[morph],
                edgecolor="black",
            )
    ax.set_xlabel("Diameter (nm)")
    ax.set_ylabel("Count")
    ax.set_title("All Particles (Color-coded)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: Spherical only
    ax = axes[0, 1]
    spherical_data = df[df["Morphology"] == "spherical"]["Diameter (nm)"]
    if len(spherical_data) > 0:
        ax.hist(spherical_data, bins=20, color="green", edgecolor="black", alpha=0.7)
        ax.set_title(f"Spherical (n={len(spherical_data)})")
    else:
        ax.text(0.5, 0.5, "No spherical particles", ha="center", va="center")
    ax.set_xlabel("Diameter (nm)")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: Rod-like only
    ax = axes[1, 0]
    rod_data = df[df["Morphology"] == "rod-like"]["Diameter (nm)"]
    if len(rod_data) > 0:
        ax.hist(rod_data, bins=20, color="blue", edgecolor="black", alpha=0.7)
        ax.set_title(f"Rod-like (n={len(rod_data)})")
    else:
        ax.text(0.5, 0.5, "No rod-like particles", ha="center", va="center")
    ax.set_xlabel("Diameter (nm)")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)

    # Panel 4: Aggregate only
    ax = axes[1, 1]
    agg_data = df[df["Morphology"] == "aggregate"]["Diameter (nm)"]
    if len(agg_data) > 0:
        ax.hist(agg_data, bins=20, color="red", edgecolor="black", alpha=0.7)
        ax.set_title(f"Aggregate (n={len(agg_data)})")
    else:
        ax.text(0.5, 0.5, "No aggregate particles", ha="center", va="center")
    ax.set_xlabel("Diameter (nm)")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = f"outputs/figures/{base}_morphology_histograms.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

    # Figure 3: Pie chart
    plt.figure(figsize=(8, 8))
    morph_counts = df["Morphology"].value_counts()
    colors_list = [colors[m] for m in morph_counts.index]
    plt.pie(
        morph_counts,
        labels=[m.capitalize() for m in morph_counts.index],
        autopct="%1.1f%%",
        colors=colors_list,
        startangle=90,
    )
    plt.title(f"Morphology Distribution - {base}", fontsize=14)

    out_path = f"outputs/figures/{base}_morphology_pie.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_batch_comparison(df_all, df_summary):
    """
    Generate comparison visualizations for batch processing.

    Creates:
    - Overlaid histograms
    - Morphology comparison bar chart
    - Summary statistics table plot
    """
    import matplotlib.pyplot as plt
    import numpy as np

    images = df_all["Image"].unique()
    n_images = len(images)

    # Figure 1: Overlaid histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # All particles overlaid
    colors = plt.cm.tab10(np.linspace(0, 1, n_images))
    for i, img in enumerate(images):
        data = df_all[df_all["Image"] == img]["Diameter (nm)"]
        ax1.hist(
            data, bins=20, alpha=0.5, label=img, color=colors[i], edgecolor="black"
        )
    ax1.set_xlabel("Diameter (nm)", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title("Size Distribution Comparison (All Images)", fontsize=14)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # Box plot comparison
    data_by_image = [
        df_all[df_all["Image"] == img]["Diameter (nm)"].values for img in images
    ]
    ax2.boxplot(
        data_by_image, labels=[img[:15] for img in images]
    )  # Truncate long names
    ax2.set_xlabel("Image", fontsize=12)
    ax2.set_ylabel("Diameter (nm)", fontsize=12)
    ax2.set_title("Size Distribution Box Plots", fontsize=14)
    ax2.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    out_path = "outputs/figures/batch_histogram_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # Figure 2: Morphology comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Stacked bar chart
    x = np.arange(n_images)
    width = 0.6

    spherical = [df_summary.iloc[i]["Spherical_Count"] for i in range(n_images)]
    rodlike = [df_summary.iloc[i]["RodLike_Count"] for i in range(n_images)]
    aggregate = [df_summary.iloc[i]["Aggregate_Count"] for i in range(n_images)]

    ax1.bar(x, spherical, width, label="Spherical", color="green", alpha=0.7)
    ax1.bar(
        x, rodlike, width, bottom=spherical, label="Rod-like", color="blue", alpha=0.7
    )
    bottom = np.array(spherical) + np.array(rodlike)
    ax1.bar(
        x, aggregate, width, bottom=bottom, label="Aggregate", color="red", alpha=0.7
    )

    ax1.set_xlabel("Image", fontsize=12)
    ax1.set_ylabel("Particle Count", fontsize=12)
    ax1.set_title("Morphology Distribution by Image", fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels([img[:15] for img in images], rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Overall morphology pie chart
    total_spherical = df_all[df_all["Morphology"] == "spherical"].shape[0]
    total_rodlike = df_all[df_all["Morphology"] == "rod-like"].shape[0]
    total_aggregate = df_all[df_all["Morphology"] == "aggregate"].shape[0]

    ax2.pie(
        [total_spherical, total_rodlike, total_aggregate],
        labels=["Spherical", "Rod-like", "Aggregate"],
        autopct="%1.1f%%",
        colors=["green", "blue", "red"],
        startangle=90,
    )
    ax2.set_title("Overall Morphology Distribution (All Images)", fontsize=14)

    plt.tight_layout()
    out_path = "outputs/figures/batch_morphology_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # Figure 3: Statistics table visualization
    fig, ax = plt.subplots(figsize=(14, max(4, n_images * 0.5)))
    ax.axis("tight")
    ax.axis("off")

    # Format summary table for display
    table_data = df_summary[
        [
            "Image",
            "Total_Particles",
            "Mean_Diameter_nm",
            "Std_Diameter_nm",
            "Spherical_Count",
            "RodLike_Count",
            "Aggregate_Count",
        ]
    ].copy()
    table_data["Mean_Diameter_nm"] = table_data["Mean_Diameter_nm"].round(2)
    table_data["Std_Diameter_nm"] = table_data["Std_Diameter_nm"].round(2)

    table = ax.table(
        cellText=table_data.values,
        colLabels=[
            "Image",
            "Total",
            "Mean (nm)",
            "Std (nm)",
            "Spherical",
            "Rod-like",
            "Aggregate",
        ],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Color header
    for i in range(len(table_data.columns)):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    plt.title("Batch Processing Summary Statistics", fontsize=16, pad=20)
    out_path = "outputs/figures/batch_summary_table.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
