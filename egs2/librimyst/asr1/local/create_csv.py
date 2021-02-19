import os
import pandas as pd
import argparse
import csv
import glob
import librosa
from pathlib import Path

def create_csv(data_root):
    output_csv = os.path.join(data_root, 'myst.csv')
    with open(output_csv, 'w', newline='', encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #word, word_fname, duration, reader_id, chapter_id, id
        csv_writer.writerow(('fname', 'transcript', 'reader_id', 'session_id'))

        for filename in glob.iglob(os.path.join(os.path.join(data_root, 'data'), '**'), recursive=True):
            filename = Path(filename)
            if filename.is_file() and filename.suffix == '.trn': # filter dirs
                with open(filename, 'r+', encoding="utf-8") as f:
                    transcript = f.read().rstrip()
                
                wav_fpath = filename.parent.joinpath(filename.stem + ".wav")
                if not wav_fpath.exists():
                    continue
                # don't include files longer than 25 sec
                try:
                    duration = librosa.get_duration(filename=str(wav_fpath))
                        
                except:
                    print(wav_fpath)
                    pass
                else:
                    if 2.0 < duration < 30.0:
                        relative_wav_fpath = wav_fpath.relative_to(data_root)
                        reader_id = relative_wav_fpath.parent.parent.stem
                        session_id = relative_wav_fpath.parent.stem
                        
                        csv_writer.writerow((str(relative_wav_fpath), transcript, reader_id, session_id))
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-root_dir', '-R', help="path to the dataset root directory")
    args = parser.parse_args()

    create_csv(args.root_dir)
