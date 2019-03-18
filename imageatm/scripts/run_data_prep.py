from imageatm.components import DataPrep


def run_data_prep(
    image_dir: str, samples_file: str, job_dir: str, resize: bool = False, **kwargs
) -> DataPrep:
    dp = DataPrep(job_dir=job_dir, image_dir=image_dir, samples_file=samples_file, **kwargs)
    dp.run(resize=resize)

    return dp
