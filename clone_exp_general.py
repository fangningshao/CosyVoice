import os
import argh
import time
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio # type: ignore

def main(input_file=None, output_dir=None, flow_ckpt=None, llm_ckpt=None, prompt_name='cvyo_mixed_000119', language=None, filename_prefix='cvyo', base_model_path='D:\\models\\cosyvoice_models\\CosyVoice2-0.5B-exp01'):
    """Based on each line in input_file, clone the voice and save the result to output_dir.
    """
    if not output_dir:
        output_dir = 'tmp_output'
    
    cosyvoice = CosyVoice2(base_model_path, load_jit=False, load_trt=False, fp16=False, use_flow_cache=False, strict_load=False, flow_ckpt=flow_ckpt, llm_ckpt=llm_ckpt)

    zeroshot_seed_name = prompt_name
    prompt_speech_16k = load_wav(os.path.join('asset', zeroshot_seed_name + '.wav'), 16000)
    prompt_text = open(os.path.join('asset', zeroshot_seed_name + '.txt'), 'r', encoding='utf-8').readline().strip()
    print("Using prompt:", zeroshot_seed_name, '||', prompt_text)

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        start_time = time.time()
        if not language:
            for i, j in enumerate(cosyvoice.inference_zero_shot(line, prompt_text, prompt_speech_16k, stream=False)):
                # fill filenames such as cvyo_0000.wav, cvyo_0001.wav, etc.
                if i == 0:
                    output_path = os.path.join(output_dir, f'{filename_prefix}_{str(idx).zfill(4)}.wav')
                else:
                    output_path = os.path.join(output_dir, f'{filename_prefix}_{str(idx).zfill(4)}_{i}.wav')
                torchaudio.save(output_path, j['tts_speech'], cosyvoice.sample_rate)

            print('inference_zero_shot time:', time.time() - start_time)
        else:  # with language

            for i, j in enumerate(cosyvoice.inference_instruct2(line, f'用{language}说这句话', prompt_speech_16k, stream=False)):
                # fill filenames such as cvyo_0000.wav, cvyo_0001.wav, etc.
                if i == 0:
                    output_path = os.path.join(output_dir, f'{filename_prefix}_{str(idx).zfill(4)}.wav')
                else:
                    output_path = os.path.join(output_dir, f'{filename_prefix}_{str(idx).zfill(4)}_{i}.wav')
                torchaudio.save(output_path, j['tts_speech'], cosyvoice.sample_rate)

            print('inference_zero_shot time:', time.time() - start_time)


if __name__ == "__main__":
    argh.dispatch_command(main)


"""
Usage:

python clone_exp_general.py -i test_mixed_scripts.txt -o test_mixed_exp01 --llm-ckpt 20250501_llm_exp01_zh\epoch_1234_whole.pt --flow-ckpt 20250501_flow_exp02\epoch_5678_whole.pt
"""
