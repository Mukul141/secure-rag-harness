import os
import requests
import zipfile
import tarfile
import io
import gdown
from datasets import load_dataset


BASE_DIR = "data/corpus"
os.makedirs(BASE_DIR, exist_ok=True)


def save_hf_dataset(dataset_name, subset, output_dir):
    """
    Downloads a Hugging Face dataset and saves each split as JSONL.
    """
    print(f"Downloading dataset: {dataset_name} ({subset})")

    try:
        dataset = load_dataset(dataset_name, subset, trust_remote_code=True)
        os.makedirs(output_dir, exist_ok=True)

        for split in dataset.keys():
            file_path = os.path.join(output_dir, f"{split}.jsonl")
            dataset[split].to_json(file_path, orient="records", lines=True)
            print(f"Saved {split} to {file_path}")

    except Exception as exc:
        print(f"Failed to download {dataset_name}: {exc}")


def download_file(url, output_path):
    """
    Downloads a file if it does not already exist locally.
    """
    if os.path.exists(output_path):
        print(f"File already exists: {output_path}")
        return

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Saved file to {output_path}")

    except Exception as exc:
        print(f"Failed to download {url}: {exc}")


def download_jfleg():
    """
    Grammar correction dataset (JFLEG).
    """
    print("Processing JFLEG dataset")
    task_dir = os.path.join(BASE_DIR, "jfleg")
    os.makedirs(task_dir, exist_ok=True)

    base_url = "https://raw.githubusercontent.com/keisks/jfleg/master"
    files = [
        "dev/dev.src",
        "dev/dev.ref0",
        "dev/dev.ref1",
        "dev/dev.ref2",
        "dev/dev.ref3",
        "test/test.src",
        "test/test.ref0",
        "test/test.ref1",
        "test/test.ref2",
        "test/test.ref3",
    ]

    for file_path in files:
        filename = os.path.basename(file_path)
        url = f"{base_url}/{file_path}"
        download_file(url, os.path.join(task_dir, filename))


def download_hsol():
    """
    Hate speech and offensive language dataset.
    """
    print("Processing HSOL dataset")
    task_dir = os.path.join(BASE_DIR, "hsol")
    os.makedirs(task_dir, exist_ok=True)

    url = (
        "https://raw.githubusercontent.com/t-davidson/"
        "hate-speech-and-offensive-language/master/data/labeled_data.csv"
    )
    download_file(url, os.path.join(task_dir, "labeled_data.csv"))


def download_sms_spam():
    """
    SMS spam detection dataset.
    """
    print("Processing SMS Spam Collection")
    task_dir = os.path.join(BASE_DIR, "sms_spam")
    os.makedirs(task_dir, exist_ok=True)

    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = os.path.join(task_dir, "sms_spam.zip")

    download_file(url, zip_path)

    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(task_dir)
        print("SMS Spam archive extracted")

    except Exception as exc:
        print(f"Failed to extract SMS Spam dataset: {exc}")


def download_gigaword():
    """
    Summarization dataset (Gigaword) using a local builder with a manual
    download fallback for Google Drive.
    """
    print("Processing Gigaword dataset")
    task_dir = os.path.join(BASE_DIR, "gigaword")
    os.makedirs(task_dir, exist_ok=True)

    file_id = "1USoQ8lJgN8kAWnUnRrupMGrPMLlDVqlV"
    archive_path = os.path.join(task_dir, "gigaword_data.tar.gz")

    if not os.path.exists(archive_path) or os.path.getsize(archive_path) < 10000:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, archive_path, quiet=False)
    else:
        print("Gigaword archive already present")

    try:
        builder_script = os.path.abspath("data/scripts/gigaword_builder.py")
        dataset = load_dataset(builder_script, trust_remote_code=True)

        for split in dataset.keys():
            file_path = os.path.join(task_dir, f"{split}.jsonl")
            dataset[split].to_json(file_path, orient="records", lines=True)
            print(f"Saved {split} to {file_path}")

    except Exception as exc:
        print(f"Failed to process Gigaword dataset: {exc}")


def main():
    print("Starting unified dataset download")

    save_hf_dataset("glue", "mrpc", os.path.join(BASE_DIR, "mrpc"))
    download_jfleg()
    download_hsol()
    save_hf_dataset("glue", "rte", os.path.join(BASE_DIR, "rte"))
    save_hf_dataset("glue", "sst2", os.path.join(BASE_DIR, "sst2"))
    download_sms_spam()
    download_gigaword()

    print("Dataset download completed")


if __name__ == "__main__":
    main()
