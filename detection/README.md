# Detection 

## How to run

virtual environment located at /imatge/ppalau/virtualenvs/fishes_venv, to activate it:
```
$source /imatge/ppalau/virtualenvs/fishes_venv/bin/activate
```
### Modules to load

python/3.6.2

cuda/8.0

opencv/3.4.1

cudnn/v6.0

### Usage
```
$srun --mem 8G --gres=gpu:1,gmem:10G python visualize.py --dataset csv --csv_classes /imatge/ppalau/work/Fishes/classes_mappings.csv --csv_val /imatge/ppalau/work/Fishes/test_image.csv --model /imatge/ppalau/work/Fishes/coco_resnet_50_map_0_335.pt
```