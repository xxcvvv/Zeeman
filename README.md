## Model Description

Zeeman is a deep learning atmospheric chemistry model designed to predict atmospheric concentrations of multiple pollutants (e.g., `no2`, `o3`, `co`, etc.). It is provided in the form of ONNX. It processes multi-dimensional input data—including 3D concentration fields (`conc3d`), meteorological data (`meteo3d`, `meteo2d`), emissions (`emis2d`), and boundary conditions (`boundary`)—to generate predictions on a 40x50 spatial grid over Netherlands. The model supports auto-regressive forecasting, where previous outputs feed into subsequent predictions, and can run on either CPU or GPU using ONNX Runtime.

Two implementations exist:
1. **Script-Based Model**: A standalone script with command-line argument parsing.
2. **Zeeman Class**: An object-oriented version encapsulating the same functionality.

### Key Features
- **Input**: `conc3d`, `meteo3d`, `meteo2d`, `emis2d`, `boundary`
- **Output**: 187 channels representing 17 atmospheric variables, reshaped to (timesteps, 187, 40, 50)
- **Variables Predicted**: `no2`, `o3`, `co`, `so2`, `nh3`, `nh4a_f`, `pan`, `so4a_f`, `no3a_f`, `no3a_c`, `ec`, `pom`, `ppm`, `tnmvoc`, `tpm25`, `tpm10`, `tss`
- **Inference Device**: CPU or GPU
- **Output Format**: NetCDF files (`inference.nc` for predictions, optionally `target.nc` for targets)
---

## How to Use the Model

### Prerequisites
- **Python Environment**: Python >=3.9.
- **Packages**: Check `requirements.txt`
- **ONNX Model File**: A pre-trained `.onnx` file
- **Input Data**: Samples are provided and managed by the `MyDataset` class

### File structure
Onnx model and data samples are provided separately. These files should be organized as below:
```
├── inference.py
├── requirements.txt
├── utils
│   └── ...
│
├── data
│   ├── conc-3d
│   │   ├── LE_MT_conc-3d_20220101.nc
│   │   └── ...
│   ├── conc-halo
│   │   ├── LE_MT_conc-halo_20220101.nc
│   │   └── ...
│   ├── emis
│   │   ├── LE_MT_emis_20220101.nc
│   │   └── ...
│   └── meteo
│       ├── LE_MT_meteo-2d_20220101.nc
│       ├── LE_MT_meteo-3d_20220101.nc
│       └── ...
├── model
│    ├── Zeeman.onnx 
│    └── ...
└── inference
    └── test
        └── 20220101_0000
            ├── inference.nc
            └── target.nc
```

### Option 1: Using the Script-Based Model

#### Steps
1. **Run the Script**:
   Execute via the command line with arguments:
   ```bash
   python inference.py -start "2023-11-20 00:00" -step 120 -exp_name test -file path-to-model.onnx -device gpu
   ```
   - `-start`: Start time (e.g., `2023-11-20 00:00`, format: `YYYY-MM-DD HH:MM`)
   - `-step`: Number of hourly timesteps (e.g., `120`)
   - `-exp_name`: Experiment name (e.g., `test_run`)
   - `-file`: Path to ONNX file (e.g., `model.onnx`)
   - `-device`: `cpu` or `gpu`

2. **Output**:
   - Results are saved in `inference/test/20231120_0000/` as `inference.nc` and `target.nc`.

#### Notes
- Configuration is handled via `argparse` at runtime.

---

### Option 2: Using the Zeeman Class

#### Description
The `Zeeman` class encapsulates the inference process in an object-oriented manner, offering the same core functionality as the script with added flexibility. It allows instantiation of a model instance, and programmatic control over inference.

#### Key Methods
- **`__init__(file, device='gpu', log_level='INFO')`**: Initializes the model with an ONNX file, device, and logging level.
- **`inference(start, step, exp_id='standard')`**: Runs inference for the specified time range and saves results.
- **`_load_model()`**: Internal method to load and configure the ONNX model.
- **`_pack_output(output_dir, output, target, time_range)`**: Saves results to NetCDF files.
- 
#### Steps
1. **Instantiate the Zeeman Class**:
   ```python
   from inference_lib import Zeeman

   zeeman = Zeeman(
       file="/path/to/model.onnx",
       device="gpu",
       log_level="INFO"
   )
   ```

2. **Run Inference**:
   ```python
   zeeman.inference(
       start="2023-01-01 00:00",
       step=120,
       exp_id="test_run"
   )
   ```
   - `start`: Start time (format: `YYYY-MM-DD HH:MM`)
   - `step`: Number of hourly timesteps
   - `exp_id`: Experiment identifier (e.g., `test_run`)

3. **Output**:
   - Results are saved in `inference/test_run/20230101_0000/inference.nc`.

#### Example
```python
zeeman = Zeeman("/home/pangmj/model.onnx")
zeeman.inference("2023-01-01 00:00", 120, "experiment")
```

#### Notes
- **Flexibility**: The class supports reuse across multiple inference runs without re-parsing arguments.

---

## Comparison
| Feature                | Script-Based Model          | Zeeman Class               |
|------------------------|-----------------------------|----------------------------|
| **Configuration**      | Command-line arguments      | Constructor + method calls |
| **Reusability**        | Single run per execution    | Reusable instance          |
| **Output Control**     | Fixed NetCDF output         | Customizable via methods   |
| **Ease of Use**        | Simple for one-off tasks    | Better for programmatic use|

---

Choose the script for quick, one-time runs or the `Zeeman` class for more complex workflows requiring perturbation or multiple inferences. Both save predictions in NetCDF format for downstream analysis.
