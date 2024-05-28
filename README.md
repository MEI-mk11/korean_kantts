# KAN-TTS

This is a model built based on modelscope kantts/a finetune version of korean tts

# Training Process For Sambert

Use modify_sr to change the sampling rate to 16000hz

Use korean_recognition to auto recognize korean

Use text2phone to change text to required form

python kantts/preprocess/data_process_korean.py --voice_input_dir data/hanyu2h_label --voice_output_dir data/hanyu2h_labeled --audio_config kantts/configs/audio_config_se_16k.yaml --speaker F7 --se_model speech_personal_sambert-hifigan_nsf_tts_zh-cn_pretrain_16k/basemodel_16k/speaker_embedding/se.model    to extract the relevant features

cp the raw_metafile into the labeled file





# Training Process For Hifigan

Preprocess korean dataset as shown in training process for sambert, run the following command to train hifigan for your own voice

CUDA_VISIBLE_DEVICES=2 python kantts/bin/train_hifigan.py --model_config speech_sambert-hifigan_tts_kyong_Korean_16k/basemodel_16k/hifigan/config.yaml --root_dir data/korean_labeled --stage_dir training_stage_hifigan --resume_path speech_sambert-hifigan_tts_kyong_Korean_16k/basemodel_16k/hifigan/ckpt/checkpoint_160000.pth


# Inference Process

You can download the pretrain korean model from modelscope (https://modelscope.cn/models/speech_tts/speech_sambert-hifigan_tts_kyong_Korean_16k/files)

Then run the following command:

CUDA_VISIBLE_DEVICES=0 python kantts/bin/text_to_wav.py --txt test.txt --output_dir res/test_male_ptts_syn --res_zip speech_sambert-hifigan_tts_kyong_Korean_16k/resource.zip --am_ckpt speech_sambert-hifigan_tts_kyong_Korean_16k/basemodel_16k/sambert/ckpt/checkpoint_630000.pth --voc_ckpt speech_sambert-hifigan_tts_kyong_Korean_16k/basemodel_16k/hifigan/ckpt/checkpoint_160000.pth

