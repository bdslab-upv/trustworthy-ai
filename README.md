# Trustworthy AI development framework (TAIDEV)
TAIDEV is a reference methodology and semi-automated code pipelines for developing trustworthy AI systems in health. This framework aligns with the European Ethics Guidelines for Trustworthy AI and supports compliance with the EU AI Act.

## Trustworthy AI Matrix (TAI Matrix)
The methodology core is a TAI matrix that classifies technical methods addressing EU guideline for Trustworthy AI requirements (privacy and data governance; diversity, non-discrimination and fairness; transparency; and technical robustness and safety) across the different AI lifecycle stages (data preparation; model development, deployment and use, and model management)

<img src="https://github.com/bdslab-upv/trustworthy-ai/blob/main/data/TAI_Matrix.png">

*Methods for achieving the trustworthy AI requirements across all lifecycle phases.*

## How to use the example code pipelines
The "Diabetes Dataset" and "Heart Disease Dataset" folders contain an example with different datasets of the methodology. For each dataset, there is a notebook per requirement so that anyone can consult the practical application of the methodology.

1. Download the whole repository.
2. Choose one dataset.
3. Look at the methodology proposed for each requirement in its single notebook `.ipynb`.
4. Apply the methods to your specific dataset.

It is only necessary to modify and adapt the Data Collection and Metadata script in order to incorporate specific information.

## Checklist
The "Checklist" folder includes a checklist to evaluate the implementation of the proposed technical requirements in health AI projects, which, as the TAI matrix, remains extensible.

## Citation
If you use this methodology and/or its code please cite:

<blockquote style='font-size:14px'> Carlos de-Manuel-Vicente, David Fernández-Narro, Vicent Blanes-Selva, Juan M García-Gómez, Carlos Sáez (2025). A Trustworthy Health AI Development Framework with Example Code Pipelines. Studies in Health Technology and Informatics, 332, 180–184. https://doi.org/10.3233/SHTI251522 </blockquote>

Preprint originally published in medRxiv:

<blockquote style='font-size:14px'> Carlos de-Manuel-Vicente, David Fernández-Narro, Vicent Blanes-Selva, Juan M García-Gómez, Carlos Sáez (2024). A Development Framework for Trustworthy Artificial Intelligence in Health with Example Code Pipelines. medRxiv 2024.07.17.24310418; doi: https://doi.org/10.1101/2024.07.17.24310418 </blockquote>

## Credits
- **Version**: 1.0.1
- **Authors**: Carlos de Manuel Vicente (UPV), David Fernández-Narro (UPV), Vicent Blanes-Selva (UPV), Juan M García-Gómez (UPV), [Carlos Sáez](mailto:carsaesi@upv.es) (UPV, Principal Investigator).

Work funded by Spanish *Agencia Estatal de Investigación, Proyectos de Generación de Conocimiento (PID2022-138636OA-I00), KINEMAI* and *Ministerio de Educación, Beca de colaboración (2023/12/00002)*.

Copyright: 2024 - The Authors. Biomedical Data Science Lab, Institute of Information and Communication Technologies (ITACA), Universitat Politècnica de València, Spain (UPV)

If you are interested in collaborating in this work please [contact us](mailto:carsaesi@upv.es).


