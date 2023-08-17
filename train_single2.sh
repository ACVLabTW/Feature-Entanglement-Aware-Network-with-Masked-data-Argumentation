#dataset = ['NeuralTextures', 'Deepfakes', 'Face2Face', 'FaceSwap']

#CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node=1 main_single.py #--lm var
# CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node=1 main_single.py --percent 0.0 --model_path "checkpoint/ViTSingle-rm0-h16-LD2-frr16-cspdarknet53" --log_name "simclr-single-rm0-h16-LD2-frr16-cspdarknet53" 
# CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node=1 main_single.py --percent 0.1 --model_path "checkpoint/ViTSingle-rm1-h16-LD2-frr16-cspdarknet53" --log_name "simclr-single-rm1-h16-LD2-frr16-cspdarknet53" 
# CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node=1 main_single.py --percent 0.2 --model_path "checkpoint/ViTSingle-rm2-h16-LD2-frr16-cspdarknet53" --log_name "simclr-single-rm2-h16-LD2-frr16-cspdarknet53" 
# CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node=1 main_single.py --percent 0.3 --model_path "checkpoint/ViTSingle-rm3-h16-LD2-frr16-cspdarknet53" --log_name "simclr-single-rm3-h16-LD2-frr16-cspdarknet53" 
# CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node=1 main_single.py --percent 0.4 --model_path "checkpoint/ViTSingle-rm4-h16-LD2-frr16-cspdarknet53" --log_name "simclr-single-rm4-h16-LD2-frr16-cspdarknet53" 
# CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --nproc_per_node=1 main_single.py --percent 0.5 --model_path "checkpoint/ViTSingle-rm5-h16-LD2-frr16-cspdarknet53" --log_name "simclr-single-rm5-h16-LD2-frr16-cspdarknet53"
CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --nproc_per_node=1 main_single.py --percent 0.6 --model_path "checkpoint/ViTSingle-rm6-h16-LD2-frr16-cspdarknet53" --log_name "simclr-single-rm6-h16-LD2-frr16-cspdarknet53" --epoch_num 39
# CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --nproc_per_node=1 main_single.py --percent 0.7 --model_path "checkpoint/ViTSingle-rm7-h16-LD2-frr16-cspdarknet53" --log_name "simclr-single-rm7-h16-LD2-frr16-cspdarknet53" 
# CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --nproc_per_node=1 main_single.py --percent 0.8 --model_path "checkpoint/ViTSingle-rm8-h16-LD2-frr16-cspdarknet53" --log_name "simclr-single-rm8-h16-LD2-frr16-cspdarknet53" 
 
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 main_single.py --lm default
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 main_single.py --lm nri_uniform