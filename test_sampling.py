import pytest 
from reinvent import sampling_fun
from reinvent.config_parse import read_smiles_csv_file

def test_De_novo_design():
    uni_molecules = [True, False]
    rand_smiles = [True, False]
    use_cuda_list  = [True, False]
    num_molecules_list = [10, 100, 1000]
    for unique_molecule in uni_molecules:
        for randomize_smile in rand_smiles:
            for use_cuda_ in use_cuda_list:
                for num_molecule in num_molecules_list:
                    output_df = sampling_fun(
                        model_type = "De_novo_design",
                        input_smiles = None,
                        num_molecules = num_molecule,
                        unique_molecules = unique_molecule,
                        randomize_smiles = randomize_smile,
                        use_cuda = use_cuda_
                    )
                    if unique_molecule:
                        assert list(output_df.columns) == ["SMILES", "NLL"]
                        assert len(output_df) <= num_molecule
                    else:
                        assert list(output_df.columns) == ["SMILES", "NLL"]
                        assert len(output_df) == num_molecule


def test_Scaffold_design():
    uni_molecules = [True, False]
    rand_smiles = [True, False]
    use_cuda_list  = [True, False]
    num_molecules_list = [10, 100, 1000]
    input_smiles = read_smiles_csv_file('/home/jovyan/my_code/REINVENT4/example_file/Scaffold_design_example.smi', columns=0)
    for unique_molecule in uni_molecules:
        for randomize_smile in rand_smiles:
            for use_cuda_ in use_cuda_list:
                for num_molecule in num_molecules_list:
                    output_df = sampling_fun(
                        model_type = "Scaffold_design",
                        input_smiles = input_smiles,
                        num_molecules = num_molecule,
                        unique_molecules = unique_molecule,
                        randomize_smiles = randomize_smile,
                        use_cuda = use_cuda_
                    )
                    if unique_molecule:
                        assert list(output_df.columns) == ["SMILES", "Scaffold", "R-groups", "NLL"]
                        assert len(output_df) <= len(input_smiles)*num_molecule
                    else:
                        assert list(output_df.columns) == ["SMILES", "Scaffold", "R-groups", "NLL"]
                        assert len(output_df) == len(input_smiles)*num_molecule  

def test_Linker_design():
    uni_molecules = [True, False]
    rand_smiles = [True, False]
    use_cuda_list  = [True, False]
    num_molecules_list = [10, 100, 1000]
    input_smiles = read_smiles_csv_file('/home/jovyan/my_code/REINVENT4/example_file/Linker_design_example.smi', columns=0)
    for unique_molecule in uni_molecules:
        for randomize_smile in rand_smiles:
            for use_cuda_ in use_cuda_list:
                for num_molecule in num_molecules_list:
                    output_df = sampling_fun(
                        model_type = "Linker_design",
                        input_smiles = input_smiles,
                        num_molecules = num_molecule,
                        unique_molecules = unique_molecule,
                        randomize_smiles = randomize_smile,
                        use_cuda = use_cuda_
                    )
                    if unique_molecule:
                        assert list(output_df.columns) == ["SMILES", "Warheads", "Linker", "NLL"]
                        assert len(output_df) <= len(input_smiles)*num_molecule
                    else:
                        assert list(output_df.columns) == ["SMILES", "Warheads", "Linker", "NLL"]
                        assert len(output_df) == len(input_smiles)*num_molecule  


def test_Molecule_optimization():
    uni_molecules = [True, False]
    Rand_smiles_list = [True, False]
    use_cuda_list  = [True, False]
    num_molecules_list = [10, 100, 1000]
    Mol2Mol_Priors_list = ["low_similarity", "high_similarity", "medium_similarity", "scaffold", "generic_scaffold"]
    sample_strategy_list = ["beamsearch", "multinomial"]
    Temperature_list = [0.1, 0.5, 1.0]
    input_smiles = read_smiles_csv_file('/home/jovyan/my_code/REINVENT4/example_file/Molecule_optimization_example.smi', columns=0)
    for unique_molecule in uni_molecules:
        for num in num_molecules_list:
            for cuda_use in use_cuda_list:
                for rand_smiles in Rand_smiles_list:
                    for Mol2Mol_Priors in Mol2Mol_Priors_list:
                        for sample_strategy in sample_strategy_list:
                            if sample_strategy == "multinomial":
                                for temp in Temperature_list:
                                    output_df = sampling_fun(
                                        model_type = "Molecule_optimization",
                                        input_smiles = input_smiles,
                                        num_molecules = num,
                                        unique_molecules = unique_molecule,
                                        randomize_smiles = rand_smiles,
                                        mol2mol_priors = Mol2Mol_Priors,
                                        sample_strategy = sample_strategy,
                                        temperature = temp,
                                        use_cuda = cuda_use,
                                    )
                                    if unique_molecule:
                                        assert list(output_df.columns) == ["SMILES", "Input_SMILES", "Tanimoto", "NLL"]
                                        assert len(output_df) <= len(input_smiles)*num
                                    else:
                                        assert list(output_df.columns) == ["SMILES", "Input_SMILES", "Tanimoto", "NLL"]
                                        assert len(output_df) == len(input_smiles)*num
                                else:
                                    output_df = sampling_fun(
                                        model_type = "Molecule_optimization",
                                        input_smiles = input_smiles,
                                        num_molecules = num,
                                        unique_molecules = unique_molecule,
                                        randomize_smiles = rand_smiles,
                                        mol2mol_priors = Mol2Mol_Priors,
                                        sample_strategy = sample_strategy,
                                        temperature = 1.0,
                                        use_cuda = cuda_use,
                                    )
                                    if unique_molecule:
                                        assert list(output_df.columns) == ["SMILES", "Input_SMILES", "Tanimoto", "NLL"]
                                        assert len(output_df) <= len(input_smiles)*num
                                    else:
                                        assert list(output_df.columns) == ["SMILES", "Input_SMILES", "Tanimoto", "NLL"]
                                        assert len(output_df) == len(input_smiles)*num
