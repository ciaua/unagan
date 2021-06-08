import os
# import numpy as np
import librosa
from pydub import AudioSegment
# from shutil import copyfile
from multiprocessing import Pool


def process_one(args):
    fn, out_dir = args

    in_fp = os.path.join(audio_dir, f'{fn}{ext}')
    if not os.path.exists(in_fp):
        print('Not exists')
        return

    duration = librosa.get_duration(filename=in_fp)

    # duration = song.duration_seconds
    num_subclips = int(duration // subclip_duration)
    # num_subclips = int(np.ceil(duration / subclip_duration))

    try:
        song = AudioSegment.from_mp3(in_fp)
    except Exception:
        print('Error in loading')
        return

    for ii in range(num_subclips):
        start = ii*subclip_duration
        end = (ii+1)*subclip_duration
        print(fn, start, end)

        out_fp = os.path.join(out_dir, f'{fn}.{start}_{end}.mp3')
        if os.path.exists(out_fp):
            print('Done before')
            continue

        subclip = song[start*1000:end*1000]
        subclip.export(out_fp, format='mp3')


if __name__ == '__main__':
    audio_dir = './path/to/audios/'  # Clean audios or separated audios from mixture
    out_dir = './training_data/clips/'

    subclip_duration = 10
    sr = 22050
    ext = '.wav'

    # ### Process ###
    num_samples = int(round(subclip_duration * sr))
    os.makedirs(out_dir, exist_ok=True)

    fns = [fn.replace(ext, '') for fn in os.listdir(audio_dir)]

    pool = Pool(10)

    args_list = []

    for fn in fns:
        args_list.append((fn, out_dir))

    pool.map(process_one, args_list)
