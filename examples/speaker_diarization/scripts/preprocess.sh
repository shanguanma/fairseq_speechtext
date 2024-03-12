data_path=/mlx_devbox/users/ajy/playground/junyi-nas2/alimeeting_eval

echo " Process dataset: Train/Eval dataset, get json files"
python prepare_data.py \
    --data_path ${data_path} \
    --type Eval \

python prepare_data.py \
    --data_path ${data_path} \
    --type Train \
