from imageatm.components import Evaluation


def run_evaluation(image_dir: str, job_dir: str, **kwargs):
    eval = Evaluation(image_dir=image_dir, job_dir=job_dir, **kwargs)
    eval.run()
