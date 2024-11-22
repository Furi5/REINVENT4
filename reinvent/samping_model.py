import os
import logging
from typing import List, Dict, Optional
import pandas as pd
from reinvent.runmodes.samplers.run_sampling import run_sampling
from reinvent.runmodes.utils.helpers import set_torch_device
from reinvent.config_parse import read_smiles_csv_file

# 设置日志
logging.basicConfig(level=logging.INFO)

MODELS_PATH = os.path.join(os.getcwd(), 'priors')

MODEL_TYPES = {
    "De_novo_design": f"{MODELS_PATH}/reinvent.prior",
    "Scaffold_design": f"{MODELS_PATH}/libinvent.prior",
    "Linker_design": f"{MODELS_PATH}/linkinvent.prior",
    "Molecule_optimization": {
        'low_similarity': f"{MODELS_PATH}/mol2mol_similarity.prior",
        'high_similarity': f"{MODELS_PATH}/mol2mol_high_similarity.prior",
        'medium_similarity': f"{MODELS_PATH}/mol2mol_medium_similarity.prior",
        'scaffold': f"{MODELS_PATH}/mol2mol_scaffold.prior",
        'generic_scaffold': f"{MODELS_PATH}/mol2mol_scaffold_generic.prior",
    },
}


class REINVENT4Error(Exception):
    """Custom exception for REINVENT4-related errors."""
    pass


def get_model_parameters(
    model_type: str, 
    mol2mol_priors: Optional[str] = None, 
    sample_strategy: Optional[str] = None, 
    temperature: Optional[float] = None
) -> Dict[str, Optional[str]]:
    """
    Generate configuration parameters based on model type and settings.
    
    Args:
        model_type (str): Type of the model to use. Supported values:
            - 'De_novo_design'
            - 'Scaffold_design'
            - 'Linker_design'
            - 'Molecule_optimization'
        mol2mol_priors (str, optional): Prior choice for 'Molecule_optimization' mode. Supported values:
            - 'low_similarity'
            - 'medium_similarity'
            - 'high_similarity'
            - 'scaffold'
            - 'generic_scaffold'
        sample_strategy (str, optional): Sampling strategy for 'Molecule_optimization' mode. Supported values:
            - 'beamsearch'
            - 'multinomial'
        temperature (float, optional): Sampling randomness parameter for 'multinomial' strategy (0.0 to 1.0).
    
    Returns:
        Dict[str, Optional[str]]: Configuration dictionary for the selected model.
    
    Raises:
        ValueError: If invalid parameters are provided.
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Invalid model_type '{model_type}'. Supported types: {list(MODEL_TYPES.keys())}")

    params = {}
    if model_type == "Molecule_optimization":
        if mol2mol_priors not in MODEL_TYPES[model_type]:
            raise ValueError(f"Invalid mol2mol_priors '{mol2mol_priors}'. Supported values: {list(MODEL_TYPES[model_type].keys())}")
        params['model_file'] = MODEL_TYPES[model_type][mol2mol_priors]
        if sample_strategy not in {"beamsearch", "multinomial"}:
            raise ValueError("Invalid sample_strategy. Supported values: 'beamsearch', 'multinomial'")
        params['sample_strategy'] = sample_strategy
        if sample_strategy == "multinomial" and (temperature is None or not (0.0 < temperature <= 1.0)):
            raise ValueError("For 'multinomial' strategy, 'temperature' must be a float between 0.0 and 1.0")
        if temperature is not None:
            params['temperature'] = float(temperature)
    else:
        params['model_file'] = MODEL_TYPES[model_type]
    return params


def sampling_fun(
    model_type: str,
    input_smiles: Optional[List[str]] = None,
    num_molecules: int = 100,
    unique_molecules: bool = True,
    randomize_smiles: bool = True,
    mol2mol_priors: Optional[str] = None,
    sample_strategy: Optional[str] = None,
    temperature: Optional[float] = None,
    use_cuda: bool = True,
) -> pd.DataFrame:
    """
    Perform molecular sampling based on the specified model type and parameters.
    
    Args:
        model_type (str): Type of the model to use. Refer to `get_model_parameters` for supported values.
        input_smiles (list, optional): List of input SMILES strings. Not required for 'De_novo_design'.
        num_molecules (int): Number of molecules to generate. Defaults to 100.
        unique_molecules (bool): Whether to filter unique molecules. Defaults to True.
        randomize_smiles (bool): Whether to randomize SMILES strings. Defaults to True.
        mol2mol_priors (str, optional): Prior type for 'Molecule_optimization' mode. Refer to `get_model_parameters`.
        sample_strategy (str, optional): Sampling strategy. Refer to `get_model_parameters`.
        temperature (float, optional): Sampling randomness for 'multinomial' strategy. Defaults to None.
        use_cuda (bool): Whether to use CUDA for computation. Defaults to True.
    
    Returns:
        pd.DataFrame: DataFrame containing the sampled molecules.
    
    Raises:
        REINVENT4Error: If the sampling process fails.
    """
    device = "cuda" if use_cuda else "cpu"
    set_torch_device(device)

    try:
        # Generate configuration
        params = get_model_parameters(model_type, mol2mol_priors, sample_strategy, temperature)
        # if model_type == 'Molecule_optimization':
        #     randomize_smiles = False
        params.update({
            'num_smiles': num_molecules,
            'unique_molecules': unique_molecules,
            'randomize_smiles': randomize_smiles,
        })
        config = {'parameters': params}

        # Run sampling
        return run_sampling(config, device, input_smiles)
    except Exception as exc:
        raise REINVENT4Error(f"Sampling failed: {exc}") from exc


if __name__ == "__main__":
    try:
        # Input file 
        small_molecule_file = '/home/jovyan/my_code/REINVENT4/example_file/Molecule_optimization_example.smi'
        input_smiles = read_smiles_csv_file(small_molecule_file, columns=0)

        # Run sampling
        output_df = sampling_fun(
            model_type="Molecule_optimization",
            input_smiles=input_smiles,
            num_molecules=100,
            unique_molecules=True,
            randomize_smiles=True,
            mol2mol_priors="medium_similarity",
            sample_strategy="multinomial",
            temperature=1.0,
            use_cuda=True,
        )

        # Output
        print(output_df.head())
        print(f"Total molecules generated: {len(output_df)}")
        print(f"Columns: {output_df.columns.tolist()}")

    except REINVENT4Error as exc:
        logging.error(f"Error during REINVENT4 sampling: {exc}")
    except FileNotFoundError as exc:
        logging.error(f"File not found: {exc}")
    except Exception as exc:
        logging.error(f"Unexpected error: {exc}")