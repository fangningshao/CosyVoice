@echo off
setlocal
SET USE_LIBUV=0
SET PYTHONPATH=%PYTHONPATH%;\PATH\TO\CosyVoice
SET MASTER_ADDR=localhost
SET MASTER_PORT=29500
SET WORLD_SIZE=1
SET RANK=0
SET LOCAL_RANK=0

call conda activate cosyvoice
set EXPNAME=20250501_llm_exp01
set TRAINDATA=\PATH\TO\output_cv_data-train\parquet\data.list
set DEVDATA=\PATH\TO\output_cv_data-dev\parquet\data.list
echo Environment settings:
echo EXPNAME=%EXPNAME%
echo TRAINDATA=%TRAINDATA%
echo DEVDATA=%DEVDATA%
echo USE_LIBUV=%USE_LIBUV%
echo PYTHONPATH=%PYTHONPATH%
echo MASTER_ADDR=%MASTER_ADDR%
echo MASTER_PORT=%MASTER_PORT%

python bin\train.py ^
    --train_engine=torch_ddp ^
    --model=llm ^
    --config=\PATH\TO\models\cosyvoice_models\CosyVoice2-0.5B\cosyvoice2_exp04_llm_lr1e-6.yaml ^
    --checkpoint=\PATH\TO\models\cosyvoice_models\CosyVoice2-0.5B\llm.pt ^
    --model_dir=\PATH\TO\models\cosyvoice_models\%EXPNAME% ^
    --train_data=%TRAINDATA% ^
    --cv_data=%DEVDATA% ^
    --num_workers=8 ^
    --qwen_pretrain_path=\PATH\TO\models\QWen2.5-0.5B ^
    --tensorboard_dir \PATH\TO\models\cosyvoice_models\tensorboard_llm\%EXPNAME%\ ^
    --ddp.dist_backend=gloo

endlocal
