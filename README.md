REINVENT 4
==========

REINVENT4 是一个用于小分子设计的工具，适用于全新分子设计（de novo design）、骨架跃迁（scaffold hopping）、linker设计（linker design）、分子优化(molecule optimization)。

环境
------------

- 该代码使用 Python 3（版本 ≥ 3.10）编写。
- 对于大多数任务，约 8 GiB 的 CPU 主存储器和 GPU 显存通常足够。

    ```shell
    # 创建并激活新的 Conda 环境
    conda create --name reinvent4 python=3.10
    conda activate reinvent4
    pip install -r requirements-linux-64.lock

    # 若有 GPU，则需要手动安装CUDA对应的 torch 版本
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

    > **Note**:  请在 REINVENT4/reinvent/sampling_model.py 文件中，将 MODELS_PATH 设置为本地 priors 文件夹的绝对路径。

安装
------------

```shell
pip install --no-deps .
```

示例
------------

1. 示例

    ```python
    from reinvent.config_parse import read_smiles_csv_file
    from reinvent import sampling_fun

    small_molecule_file = 'example_file/Molecule_optimization_example.smi'
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

    ```

文件结构
------------

- 'reinvent/': 包含主要代码
  'samping_model.py':基于 samping 模式做分子生成和优化的主要代码
- 'priors': 分子生成的模型文件
- 'example_file': 运行的不同分子生成方法的示例文件
- 'test_samping.py': 测试文件
  
测试
------------

```python
pytest test_samping.py
```

参考文献：
[Reinvent 4: Modern AI–driven generative molecule design](https://link.springer.com/article/10.1186/s13321-024-00812-5?utm_source=rct_congratemailt&utm_medium=email&utm_campaign=oa_20240221&utm_content=10.1186/s13321-024-00812-5).
