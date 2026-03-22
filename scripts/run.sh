mode=train
eval_args=""

function run_neus() {
	exp_name=$1
	conf_name=$2
	base_dir=$3
	view=$4
	offset=$5
	other_args=${@:6:($#-5)}
	if [ ${mode} = "train" ]; then
		python3 exp_runner.py --mode train --exp_name ${exp_name} --conf ${conf_name} --case ${base_dir} --num_images ${view} --image_ind_offset ${offset} ${other_args}

	elif [ ${mode} = "validate_mesh_all" ]; then
		python3 exp_runner.py --mode validate_mesh_multi_0_${view}_$((view / 2)) --exp_name ${exp_name} --conf ${conf_name} --case ${base_dir} --num_images ${view} ${other_args} --resume

	elif [ ${mode} = "rendering" ]; then
		python3 exp_runner.py --mode rendering --exp_name ${exp_name} --conf ${conf_name} --case ${base_dir} --num_images ${view} ${other_args} --resume

	elif [ ${mode} = "render_interp" ]; then
		python3 exp_runner.py --mode rendering_interp_${offset}_$((${offset}+1))_10 --exp_name ${exp_name} --conf ${conf_name} --case ${base_dir} ${other_args} --resume

	fi
}


base_dir=sample_scene
run_neus activeneus ./confs/oneshot_active.conf ${base_dir} 10 0 --estimate_illum --resume
