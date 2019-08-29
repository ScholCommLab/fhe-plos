# Reproduction code for: How much research shared on Facebook is hidden from public view?

This repository contains all figures and tables present in the article. Furthermore, all the input data and code are provided to reproduce the results.

## Instructions



### Requirements

To reproduce results Python 3.5 is required. To explore results interactively Jupyter Notebook is also required.

Packages specified in `requirements.txt` need to be installed via

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

## Other links and resources

| Resource | Link |
|-|-|
| Preprint | TBA |
| Article | TBA |
| Code | [GitHub Repository](https://github.com/ScholCommLab/fhe-plos-paper)|
| Data | [Dataverse]() |
