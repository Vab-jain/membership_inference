## Membership Inference Attack Implementation

This project implements a membership inference attack on target models.

### Project Overview

This project explores the concept of membership inference attacks in machine learning. The code implements an attack to determine if a data point was used to train a target model.

### Files

* **Instructions.md:** Defines the project task, submission guidelines, and instructions on downloading shadow datasets.
* **config.py:** Configuration file containing dataset names and other relevant parameters.
* **pipeline.py:** The main script to run the attack pipeline. Takes a command-line argument (`eval` or `test`) to control execution mode.
    * `eval`: Runs the entire pipeline, including training shadow and attack models, evaluating performance.
    * `test`: Uses pre-trained models to generate submission files.
* **create_attack_dataset.py:** Creates the attack dataset used to evaluate the attack model.
* **train_shadow_models.py:** Trains the shadow models used in the attack.
* **train_attack_models.py:** Trains the attack models based on predictions from shadow models.
* **membership_inference.py:** Implements the core logic of the membership inference attack.
* **attack_models.py:** Contains code for different attack model architectures.
* **saved_shadow_models/ (directory):** Stores trained shadow models (potentially empty if not run in `eval` mode).
* **saved_attack_models/ (directory):** Stores trained attack models (potentially empty if not run in `eval` mode).
* **attack_dataset/ (directory):** Stores the generated attack dataset (potentially empty if not run in `eval` mode).
* **submission.py:** Generates submission files in the required format.


**Important Note:**

* A directory named "datasets" needs to be created to store the shadow datasets before running the code. Download the datasets [here](https://drive.google.com/drive/folders/1LZhRnyw9aJ2NzKpIRqJdZACAQpN5ulTY). More details on datasets can be found in [`Instructions.md`](https://github.com/Vab-jain/membership_inference/blob/main/Instructions.md)
`

### Running the Project

1. Create the "datasets" directory and download the shadow datasets as instructed in `Instructions.md`.
2. Run the pipeline script:

   ```bash
   python pipeline.py --mode <mode>
   ```

   Replace `<mode>` with either `eval` or `test`.
     * `eval`: Runs the entire attack pipeline (training, evaluation).
     * `test`: Uses pre-trained models to generate submission files (assuming models are already trained).

This project provides a basic implementation of a membership inference attack. You can explore further by modifying the attack model architecture, experimenting with different datasets, or analyzing the attack success rate under various conditions.
