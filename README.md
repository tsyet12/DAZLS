# DAZLS
## Domain adaptation zero shot learning for near real-time prediction of renewable energy
[![DOI](https://zenodo.org/badge/681149311.svg)](https://zenodo.org/badge/latestdoi/681149311)

This repository demonstrates the main algorithm DAZLS in the most basic implementation. It compliments the paper (below) to demonstrate the basic DAZLS algorithm:

<br>
S.Y. Teng et al. (2023). Near real-time predictions of renewable electricity production at substation level via domain adaptation zero-shot learning in sequence. Renewable and Sustainable Energy Reviews.


# Instructions
1. The "combined_data" folder contains the raw data. (Example data is provided)
2. Use PrepData.py to preprocess data from "combined_data" and save it in "prep_data" folder.
3. After running PrepData.py the preprocessed data with metadata can be found in prep_data folder.
4. Run Model.py to test the DAZLS model.



# Acknowledgements
The research contribution from S.Y. Teng is supported by the European Union's Horizon Europe Research and Innovation Program, under Marie Skłodowska-Curie Actions grant agreement no. 101064585 (MoCEGS).
