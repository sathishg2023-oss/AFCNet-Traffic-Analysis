# AFCNet: Attention Fusion Counting Network for Multistage Traffic Scene Analysis

This repository packages the AFCNet implementation notebook for version control and reproducibility review. The notebook's code-cell source is preserved from the supplied `Final_AFCNet(Kaggle).ipynb`; saved outputs and execution counts were removed to keep the repository small and avoid publishing bulky generated artifacts.

## What is included

- `notebooks/AFCNet_Implementation.ipynb` — source-preserving notebook, including vehicle-count regression, density/severity classification, scene-change analysis, and Grad-CAM experiments.
- `requirements.txt` — Python packages used or imported by the notebook.
- `docs/REPRODUCIBILITY_AUDIT.md` — important setup and validation notes identified during repository preparation.
- `.gitignore` — excludes datasets, checkpoints, caches, and generated outputs.

## Quick start (Google Colab recommended)

1. Create a new GitHub repository, for example `AFCNet-Traffic-Analysis`.
2. Upload the files in this repository, or push the extracted directory using Git.
3. Open `notebooks/AFCNet_Implementation.ipynb` in Google Colab.
4. Select **Runtime → Change runtime type → T4 GPU** (or another supported GPU).
5. Run the Kaggle dataset download cell. If Kaggle authentication or dataset access is required, configure it in Colab using your own Kaggle credentials. Do not commit Kaggle tokens or credentials.
6. Inspect the dataset directory structure printed by the download cells. The notebook contains more than one assumed directory layout; adjust the dataset paths to match the actual downloaded files before running later cells.
7. Execute cells in order, checking each dataset path and output before proceeding. Do not assume that a successful download means the subsequent path configuration is correct.
8. Save model checkpoints, CSVs, and visualizations to a local output directory or Google Drive. Do not commit the full UA-DETRAC dataset, private credentials, or large generated files to GitHub.

## Dataset and labels

The notebook downloads `dtrnngc/ua-detrac-dataset` through `kagglehub`. In `VehicleDataset`, the target count is calculated as the number of lines in each corresponding `.txt` annotation file. This is a valid vehicle count only when each non-empty line represents one vehicle instance. Verify annotation format and image-to-label pairing before training or reporting metrics.

The UA-DETRAC data itself is not included in this repository. Follow the dataset provider's terms and access conditions.

## Model scope

The notebook includes a MobileNetV3-based AFCNet counting model with attention/fusion components, count-derived density/severity analysis, scene-change experiments using L2 embedding distance and SSIM-based dissimilarity, and Grad-CAM visualization. The notebook reflects the supplied implementation; this packaging step does not claim that every cell has been refactored or that the entire notebook has been re-executed from a clean runtime.

## Reported results

Use only results regenerated from a documented run when preparing a paper or release. Record the dataset subset, exact frame/image IDs, split protocol, checkpoint selection, random seed, hardware, input size, batch size, and whether inference timing includes preprocessing and all branches. Do not treat previously saved notebook outputs as independently reproduced results.

## Suggested GitHub release checklist

- [ ] Confirm the notebook executes top-to-bottom in a fresh Colab runtime.
- [ ] Confirm image and annotation filenames are paired by basename, not only by sorted list position.
- [ ] Confirm the annotation file format and count derivation.
- [ ] Record the exact dataset subset and train/validation/test split.
- [ ] Save a final checkpoint and machine-readable evaluation CSV outside the Git repository unless small and intentionally public.
- [ ] Re-run MAE/RMSE, classification metrics, scene-change metrics, parameter count, FLOPs, and FPS from the same documented checkpoint/configuration.
- [ ] Add a license only after deciding what license applies to your code and confirming rights to all included material.
