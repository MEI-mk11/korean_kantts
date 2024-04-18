import logging
import os
import sys
import argparse
import yaml
import time
import codecs

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # NOQA: E402
sys.path.insert(0, os.path.dirname(ROOT_PATH))  # NOQA: E402

try:
    from kantts.preprocess.audio_processor.audio_processor import AudioProcessor
    from kantts.preprocess.se_processor.se_processor import SpeakerEmbeddingProcessor
    from kantts.preprocess.script_convertor.TextScriptConvertor import (
        TextScriptConvertor,
    )
    from kantts.preprocess.fp_processor import FpProcessor, is_fp_line
    from kantts.preprocess.languages import languages
    from kantts.datasets.dataset import AM_Dataset, Voc_Dataset
    from kantts.utils.log import logging_to_file, get_git_revision_hash
except ImportError:
    raise ImportError("Please install kantts.")

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)

LANGUAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "languages")


def gen_metafile(
    voice_output_dir,
    fp_enable=False,
    badlist=None,
    split_ratio=0.98,
):

    voc_train_meta = os.path.join(voice_output_dir, "train.lst")
    voc_valid_meta = os.path.join(voice_output_dir, "valid.lst")
    if not os.path.exists(voc_train_meta) or not os.path.exists(voc_valid_meta):
        Voc_Dataset.gen_metafile(
            os.path.join(voice_output_dir, "wav"),
            voice_output_dir,
            split_ratio,
        )
        logging.info("Voc metafile generated.")

    raw_metafile = os.path.join(voice_output_dir, "raw_metafile.txt")
    am_train_meta = os.path.join(voice_output_dir, "am_train.lst")
    am_valid_meta = os.path.join(voice_output_dir, "am_valid.lst")
    if not os.path.exists(am_train_meta) or not os.path.exists(am_valid_meta):
        AM_Dataset.gen_metafile(
            raw_metafile,
            voice_output_dir,
            am_train_meta,
            am_valid_meta,
            badlist,
            split_ratio,
        )
        logging.info("AM metafile generated.")

    if fp_enable:
        fpadd_metafile = os.path.join(voice_output_dir, "fpadd_metafile.txt")
        am_train_meta = os.path.join(voice_output_dir, "am_fpadd_train.lst")
        am_valid_meta = os.path.join(voice_output_dir, "am_fpadd_valid.lst")
        if not os.path.exists(am_train_meta) or not os.path.exists(am_valid_meta):
            AM_Dataset.gen_metafile(
                fpadd_metafile,
                voice_output_dir,
                am_train_meta,
                am_valid_meta,
                badlist,
                split_ratio,
            )
            logging.info("AM fpaddmetafile generated.")

        fprm_metafile = os.path.join(voice_output_dir, "fprm_metafile.txt")
        am_train_meta = os.path.join(voice_output_dir, "am_fprm_train.lst")
        am_valid_meta = os.path.join(voice_output_dir, "am_fprm_valid.lst")
        if not os.path.exists(am_train_meta) or not os.path.exists(am_valid_meta):
            AM_Dataset.gen_metafile(
                fprm_metafile,
                voice_output_dir,
                am_train_meta,
                am_valid_meta,
                badlist,
                split_ratio,
            )
            logging.info("AM fprmmetafile generated.")


#  TODO: Zh-CN as default
def process_data(
    voice_input_dir,
    voice_output_dir,
    audio_config,
    speaker_name=None,
    targetLang="Korean",
    skip_script=False,
    se_model=None,
):
    foreignLang = "EnUS"

    fp_enable=False

    if audio_config is not None:
        with open(audio_config, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)

    raw_metafile = "/nfs/meizhengkun/KAN-TTS/data/raw_metafile.txt"
    #  Audio processor
    ap = AudioProcessor(config["audio_config"])
    ap.process(
        voice_input_dir,
        voice_output_dir,
        raw_metafile,
    )
    logging.info("Processing audio done.")

     # SpeakerEmbedding processor
    if True:
        sep = SpeakerEmbeddingProcessor()
        sep.process(
            voice_output_dir,
            se_model,
        )
        logging.info("Processing speaker embedding done.")

    logging.info("Processing done.")

    # Generate Voc&AM metafile
    # TODO: train/valid ratio setting
    gen_metafile(voice_output_dir, fp_enable, ap.badcase_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset preprocessor")
    parser.add_argument("--voice_input_dir", type=str, required=True)
    parser.add_argument("--voice_output_dir", type=str, required=True)
    parser.add_argument("--audio_config", type=str, required=True)
    parser.add_argument("--speaker", type=str, default=None, help="speaker")
    parser.add_argument("--lang", type=str, default="PinYin", help="target language")
    parser.add_argument(
        "--se_model",
        type=str,
        default="../pre_data/speaker_embeddding/se.*",
        help="speaker embedding extractor model",
    )
    parser.add_argument(
        "--skip_script", action="store_true", help="skip script converting"
    )
    args = parser.parse_args()

    os.makedirs(args.voice_output_dir, exist_ok=True)
    logging_to_file(os.path.join(args.voice_output_dir, "data_process_stdout.log"))

    try:
        process_data(
            args.voice_input_dir,
            args.voice_output_dir,
            args.audio_config,
            args.speaker,
            'Korean',
            args.skip_script,
            args.se_model,
        )
    except (Exception, KeyboardInterrupt) as e:
        logging.error(e, exc_info=True)
