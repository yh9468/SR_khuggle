python main_test_swinir.py --task real_sr --scale 4 --model_path experiments/pretrained_models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_PSNR.pth --folder_lq ../dataset/test_hor --output hor --large_model

python main_test_swinir.py --task real_sr --scale 4 --model_path experiments/pretrained_models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_PSNR.pth --folder_lq ../dataset/test_ver --output ver --large_model

python main_test_swinir.py --task real_sr --scale 4 --model_path experiments/pretrained_models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_PSNR.pth --folder_lq ../dataset/test --output origin --large_model

python main_test_swinir.py --task real_sr --scale 4 --model_path experiments/pretrained_models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_PSNR.pth --folder_lq ../dataset/test_90 --output 90 --large_model

python main_test_swinir.py --task real_sr --scale 4 --model_path experiments/pretrained_models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_PSNR.pth --folder_lq ../dataset/test_180 --output 180 --large_model

python main_test_swinir.py --task real_sr --scale 4 --model_path experiments/pretrained_models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_PSNR.pth --folder_lq ../dataset/test_270 --output 270 --large_model
