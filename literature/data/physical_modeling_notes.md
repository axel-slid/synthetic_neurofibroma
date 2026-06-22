# Physical Modeling Notes

These notes translate the literature package into model components.

| Component | Representation | Data to use | Key sources |
|---|---|---|---|
| NF1-deficient Schwann lineage | Initiating/proliferating agents with Ras/MAPK-active state | Cell-origin studies, iPSC models | 11988578, 21551250, 25446898, 30348677, 33108355 |
| Nerve topology | Anisotropic graph/tube scaffold, not isotropic free space | MRI segmentation, anatomical nerve roots/plexus geometry | 17215493, 23035791, 18559970 |
| Growth kinetics | Tumor-specific growth-rate distribution with age dependence | Volumetric MRI and WBMRI cohorts | 17215493, 23035791, 36332985, 39497113 |
| Fibroblasts | Nf1+/- stromal agents secreting growth and ECM-modifying factors | 3D spheroid secretome and microenvironment studies | 16835260, 39061138 |
| Mast cells and macrophages | Recruited immune/stromal agents or density fields | c-kit/mast cell and macrophage inhibition studies | 18984156, 20233971, 23099891, 29596064 |
| ECM composition | Collagen/basement-membrane/hyaluronan fields | Matrisome and ECM-response papers | 33413690, 37140985 |
| ECM mechanics | Stiffness field coupled to proliferation, invasion, and therapy response | 3D matrix stiffness experiments | 10.3390/cells15100877 |
| Neuronal activity | Local nerve activity/paracrine factor field | NF1 neuronal hyperexcitability and COL1A2 work | 35589737 |
| Malignant transition | Optional state change through atypical neurofibroma/nodular lesion | Pathology/genomic cohorts and tumor burden risk | 24166582, 21987445, 29409029, 28592921 |

Recommended first-pass model:

1. Use a nerve-graph scaffold with local tissue radius.
2. Seed NF1-deficient Schwann-lineage cells on susceptible nerve-root/branch regions.
3. Let tumor-cell proliferation depend on intrinsic NF1/RAS state, age/developmental factor, local fibroblast/immune support, ECM stiffness, and nerve activity.
4. Let ECM deposition and stiffness increase through fibroblast/mast-cell/macrophage signals.
5. Calibrate macro growth to volumetric MRI cohorts and micro growth to 3D culture/organoid assays.
