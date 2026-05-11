# 先训练 FP32
# python train.py
# python train.py --width_mult 0.75

# 测试集抽一张图推理
# python predict.py --weight_path ./runs/cifar10_repopt_vgglike_fp32/best.pth

# 做 QAT
# python qat.py --fp32_ckpt ./runs/cifar10_repopt_vgglike_fp32/best.pth
# python qat.py --fp32_ckpt ./runs/cifar10_repopt_vgglike_fp32/best.pth --width_mult 0.75

