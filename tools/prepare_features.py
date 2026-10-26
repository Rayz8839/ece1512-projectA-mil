"""Place the pre-extracted HDF5 feature files where the starter YAML configs expect them.

The Zenodo record (https://zenodo.org/records/23130543) distributes the features as archives.
This script takes a downloaded archive (or an already-extracted .h5 file) for one dataset and
copies the HDF5 file to

    starter/feature/<DATASET>/patch_feats_pretrain_medical_ssl.h5

which is the path given by data_dir in starter/config/*.yml.

Usage (from the repository root):

    python tools/prepare_features.py C16   path/to/<downloaded>.zip
    python tools/prepare_features.py C17   path/to/<downloaded>.zip
    python tools/prepare_features.py BRACS path/to/<downloaded>.zip

Inside an archive the script looks for a member named patch_feats_pretrain_medical_ssl.h5; if there
is none, it uses the archive's only .h5 member and reports which one it used. Use --force to overwrite.
"""
import argparse
import os
import shutil
import sys
import zipfile

H5_NAME = "patch_feats_pretrain_medical_ssl.h5"
DATASETS = ("C16", "C17", "BRACS")
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def target_path(dataset):
    return os.path.join(REPO_ROOT, "starter", "feature", dataset, H5_NAME)


def pick_member(zf):
    h5 = [i for i in zf.infolist() if i.filename.lower().endswith(".h5") and not i.is_dir()]
    exact = [i for i in h5 if os.path.basename(i.filename) == H5_NAME]
    if exact:
        return exact[0]
    if len(h5) == 1:
        print(f"note: using the only .h5 member '{h5[0].filename}' (name differs from {H5_NAME})")
        return h5[0]
    names = [i.filename for i in h5] or ["(none)"]
    sys.exit("error: could not choose one .h5 member in the archive; candidates: " + ", ".join(names))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=DATASETS)
    ap.add_argument("source", help="downloaded .zip archive or an extracted .h5 file")
    ap.add_argument("--force", action="store_true", help="overwrite an existing target file")
    args = ap.parse_args()

    dst = target_path(args.dataset)
    if os.path.exists(dst) and not args.force:
        sys.exit(f"exists: {dst} (use --force to overwrite)")
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    if zipfile.is_zipfile(args.source):
        with zipfile.ZipFile(args.source) as zf:
            member = pick_member(zf)
            with zf.open(member) as fin, open(dst + ".part", "wb") as fout:
                shutil.copyfileobj(fin, fout, length=16 * 1024 * 1024)
    elif args.source.lower().endswith(".h5"):
        with open(args.source, "rb") as fin, open(dst + ".part", "wb") as fout:
            shutil.copyfileobj(fin, fout, length=16 * 1024 * 1024)
    else:
        sys.exit("error: source must be a .zip archive or an .h5 file")

    with open(dst + ".part", "rb") as f:
        if f.read(8) != HDF5_SIGNATURE:
            os.remove(dst + ".part")
            sys.exit("error: the extracted file is not an HDF5 file")
    os.replace(dst + ".part", dst)
    print(f"{args.dataset}: wrote {dst} ({os.path.getsize(dst) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
