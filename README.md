# [NTIRE 2026 Challenge on Mobile Real-World Image Super-Resolution](https://cvlai.net/ntire/2026/) @ [CVPR 2026](https://cvpr.thecvf.com/)

## Environments

```sh
conda create -n NTIRE-SR python=3.8
conda activate NTIRE-SR
pip install -r requirements.txt
```

## Testing commands

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --valid_dir [path to val data dir] --test_dir [path to test data dir] --save_dir [path to your save dir] --model_id 9
```

You can use either `--valid_dir`, or `--test_dir`, or both of them. Be sure the change the directories `--valid_dir`/`--test_dir` and `--save_dir`.

## How to eval images using IQA metrics?

### Folder Structure

```
test_dir
├── HR
│   ├── 0901.png
│   ├── 0902.png
│   ├── ...
├── LQ
│   ├── 0901x4.png
│   ├── 0902x4.png
│   ├── ...
    
output_dir
├── 0901x4.png
├── 0902x4.png
├──...

```

### Command to calculate metrics

```sh
python eval.py --output_folder "/path/to/your/output_dir" --target_folder "/path/to/test_dir/HR" --metrics_save_path "./IQA_results" --gpu_ids 0
```

The `eval.py` file accepts the following 4 parameters:

- `output_folder`: Path where the restored images are saved.
- `target_folder`: Path to the HR images in the `test` dataset. This is used to calculate FR-IQA metrics.
- `metrics_save_path`: Directory where the evaluation metrics will be saved.
- `device`: Computation devices. For multi-GPU setups, use the format `0,1,2,3`.

## License and Acknowledgement
This code repository is release under [MIT License](LICENSE). 
