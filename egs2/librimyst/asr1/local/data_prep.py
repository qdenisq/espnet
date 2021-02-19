#!/usr/bin/env python3

import os
import re
import sys
import csv

if len(sys.argv) != 2:
    print("Usage: python data_prep.py [myst_root]")
    sys.exit(1)
myst_root = sys.argv[1]

for x in ["train", "dev", "test"]:
    os.makedirs(os.path.join("data", x+"_myst"), exist_ok=True)
    with open(os.path.join(myst_root, "myst_clean_" + x + ".csv"), newline='') as data_csv, open(
        os.path.join("data", x+"_myst", "text"), "w") as text_f, open(
         os.path.join("data", x+"_myst", "wav.scp"), "w") as wav_scp_f, open(
             os.path.join("data", x+"_myst", "utt2spk"), "w") as utt2spk_f:

        csv_reader = csv.DictReader(data_csv)

        text_f.truncate()
        wav_scp_f.truncate()
        utt2spk_f.truncate()

        for row in csv_reader:
            fname = row['fname']
            transcript = row['transcript'].upper() # uppercase all trancsripts
            reader_id = f"{int(row['reader_id']):06d}"
            utt_id = reader_id + "_" + os.path.splitext(os.path.basename(fname))[0]

            text_f.write(utt_id + " " + transcript + "\n")
            wav_scp_f.write(utt_id + " " + os.path.join(myst_root, fname) + "\n")
            utt2spk_f.write(utt_id + " " + reader_id + "\n")

