![fhe](fhe.png)

# Reproduction code for: How much research shared on Facebook is hidden from public view?

> A comparison of public and private online activity around PLOS ONE papers

This repository contains all figures and tables present in the manuscript for "How much research shared on Facebook is hidden from public view?".

Furthermore, all the input data and code required to reproduce results are provided with instructions.

| Resource | Link |
|-|-|
| Preprint | TBD |
| Article | TBD |
| Code | [GitHub](https://github.com/ScholCommLab/fhe-plos-paper)|
| Data | [Dataverse](https://dataverse.harvard.edu/privateurl.xhtml?token=58246dfc-bdf8-454d-8edc-60d5918dedfc) |

This article is part of a broader investigation of the hidden engagement on Facebook. More information about the project can be found [here](https://github.com/ScholCommLab/facebook-hidden-engagement).

## Instructions

All scripts have been written with Python 3.x. To explore results interactively a working instance of Jupyter Notebooks/Labs is required.

Packages specified in `requirements.txt` can be installed via

```pip install -r requirements.txt```

### Reproduce results

1. Clone this repository and cd into it

    ```
    git clone git@github.com:ScholCommLab/fhe-plos-paper.git
    cd fhe-plos-paper
    ```

2. Download data from Dataverse.

    All the data is hosted on dataverse: [Dataverse repository](https://dataverse.harvard.edu/privateurl.xhtml?token=58246dfc-bdf8-454d-8edc-60d5918dedfc)

    Using the helper script provided, you can download all files into the respective locations.

    ```download_data.sh```

3. Preprocess data

    Run the preprocessing script to apply transformations on the input dataset.

    ```python process_data.py```

4. (Re)produce results

    Run the analysis script to produce figures and tables.

    ```python analysis.py```

    Optionally, you can also open the analysis notebook with Jupyter to explore the dataset.
